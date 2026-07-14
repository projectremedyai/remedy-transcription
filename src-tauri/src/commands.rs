use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, State};
use tauri_plugin_shell::process::CommandEvent;

use crate::events::HealthStatus;
use crate::sidecar::spawn_sidecar;
use crate::store::{Job, JobStatus, JobUpdate, SourceType, TaskType, TranscriptionSegment};
use crate::AppState;

/// The frontend's model config, embedded at COMPILE time.
///
/// The model IDs used to be typed out again here, and they drifted the moment
/// the frontend renamed them: `list_models` kept answering for the old IDs, the
/// frontend gates the Transcribe button on an EXACT id match against that
/// answer, and every entry point to transcription went dead with no error. The
/// backend has no independent knowledge of these models — they stream from
/// huggingface.co on demand — so it has no business holding a second copy of the
/// list. It reads the frontend's.
///
/// `include_str!` is a compile-time dependency that rustc records in its
/// dep-info, so editing the TS config rebuilds this crate.
const TRANSCRIPTION_CONFIG_TS: &str = include_str!("../../frontend/src/config/transcription.ts");

static MODEL_ID_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"modelId:\s*"([^"]+)""#).expect("valid regex"));

/// Every model the frontend can actually resolve, derived from the single source
/// of truth above. `"__auto__"` is a sentinel the frontend resolves to one of the
/// real presets, not a model.
static REQUIRED_MODEL_IDS: Lazy<Vec<String>> = Lazy::new(|| {
    let ids: Vec<String> = MODEL_ID_RE
        .captures_iter(TRANSCRIPTION_CONFIG_TS)
        .map(|caps| caps[1].to_string())
        .filter(|id| id != "__auto__")
        .collect();

    // Fail LOUDLY, in the running app and not only under `cargo test`.
    //
    // A parse miss yields a short or empty Vec, `list_models` answers with fewer
    // models than the frontend offers, and the frontend — which gates the
    // Transcribe button on an exact id match against that answer — goes dead with
    // no error anywhere. That is the exact bug this derivation exists to prevent,
    // and leaving it to a test means it comes back silently for anyone who builds
    // without running the suite. A panic here is a crash on startup with a message
    // that names the cause; a dead button is a bug report six weeks later.
    assert!(
        !ids.is_empty(),
        "MODEL_ID_RE matched no models in frontend/src/config/transcription.ts. \
         list_models would answer with an EMPTY list and the frontend would disable \
         every entry point to transcription with no error. The config's format has \
         changed (quoting? a rename? moved file?) — fix MODEL_ID_RE. Do NOT retype \
         the model list here: that is how it drifted last time."
    );

    ids
});

static YOUTUBE_ID_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[A-Za-z0-9_-]{11}$").expect("valid regex"));

static FILE_HASH_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[0-9a-f]{64}$").expect("valid regex"));

#[derive(Debug, Deserialize)]
pub struct YouTubeJobRequest {
    pub url: String,
    pub model_id: String,
    #[serde(default = "default_task")]
    pub task: TaskType,
    #[serde(default = "default_language")]
    pub language: String,
}

#[derive(Debug, Deserialize)]
pub struct FileJobRequest {
    pub file_hash: String,
    pub filename: String,
    pub size_bytes: i64,
    pub model_id: String,
    #[serde(default = "default_task")]
    pub task: TaskType,
    #[serde(default = "default_language")]
    pub language: String,
}

#[derive(Debug, Deserialize)]
pub struct PersistTranscriptRequest {
    pub model_id: String,
    pub task: TaskType,
    #[serde(default = "default_language")]
    pub language: String,
    pub full_text: String,
    #[serde(default)]
    pub segments: Vec<TranscriptionSegment>,
}

#[derive(Debug, Deserialize)]
pub struct ExportRequest {
    pub job_id: String,
    pub format: ExportFormat,
    pub destination: String,
    pub content: String,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Srt,
    Vtt,
    Txt,
    Json,
}

#[derive(Debug, Serialize)]
pub struct QueueStatusResponse {
    pub position: i32,
    pub total_in_queue: usize,
    pub estimated_wait_minutes: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct ModelStatusItem {
    pub model_id: String,
    pub ready: bool,
    pub path: String,
}

#[derive(Debug, Serialize)]
pub struct ModelStatusResponse {
    pub models_ready: bool,
    pub missing_models: Vec<String>,
    pub items: Vec<ModelStatusItem>,
}

fn default_task() -> TaskType {
    TaskType::Transcribe
}

fn default_language() -> String {
    "auto".to_string()
}

fn normalize_language(language: &str) -> String {
    let trimmed = language.trim();
    if trimmed.is_empty() {
        "auto".to_string()
    } else {
        trimmed.to_lowercase()
    }
}

fn extract_youtube_id(url: &str) -> Result<String, String> {
    let parsed = url::Url::parse(url).map_err(|_| "Invalid YouTube URL".to_string())?;
    let host = parsed.host_str().unwrap_or("").to_lowercase();
    let path = parsed.path();

    let id = if host == "youtu.be" || host == "www.youtu.be" {
        path.trim_start_matches('/').split('/').next().unwrap_or("").to_string()
    } else if host.contains("youtube.com") {
        if path == "/watch" {
            parsed
                .query_pairs()
                .find(|(k, _)| k == "v")
                .map(|(_, v)| v.into_owned())
                .unwrap_or_default()
        } else if path.starts_with("/shorts/") || path.starts_with("/embed/") {
            path.split('/').nth(2).unwrap_or("").to_string()
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    if !YOUTUBE_ID_RE.is_match(&id) {
        return Err("Invalid YouTube URL".to_string());
    }
    Ok(id)
}

fn audio_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    Ok(dir.join("audio"))
}

fn downloads_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    Ok(dir.join("downloads"))
}

#[tauri::command]
pub async fn create_youtube_job(
    request: YouTubeJobRequest,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<Job, String> {
    let audio_root = audio_dir(&app)?;
    state
        .store
        .cleanup_expired_audio(crate::paths::PREPARED_AUDIO_TTL_HOURS, &audio_root);

    let source_key = extract_youtube_id(&request.url)?;
    let language = normalize_language(&request.language);
    let canonical_url = format!("https://www.youtube.com/watch?v={}", source_key);

    let source = state
        .store
        .get_or_create_source(
            SourceType::Youtube,
            &source_key,
            Some(&canonical_url),
            None,
            None,
        )
        .map_err(|e| e.to_string())?;

    if let Some(transcript) = state
        .store
        .find_transcript(source.id, &request.model_id, request.task, &language)
        .map_err(|e| e.to_string())?
    {
        let job = state
            .store
            .create_job_from_cache(
                source.id,
                SourceType::Youtube,
                &source_key,
                source.filename.as_deref(),
                source.audio_mime_type.as_deref(),
                &request.model_id,
                request.task,
                &language,
                &transcript.full_text,
                &transcript.segments,
            )
            .map_err(|e| e.to_string())?;
        return Ok(job);
    }

    if let Some(audio_path) = source.audio_path.as_deref() {
        if Path::new(audio_path).exists() {
            state
                .store
                .touch_source(source.id)
                .map_err(|e| e.to_string())?;
            return state
                .store
                .create_pending_job(
                    source.id,
                    SourceType::Youtube,
                    &source_key,
                    JobStatus::Ready,
                    source.filename.as_deref(),
                    source.audio_mime_type.as_deref(),
                    &request.model_id,
                    request.task,
                    &language,
                )
                .map_err(|e| e.to_string());
        }
    }

    let job = state
        .store
        .create_pending_job(
            source.id,
            SourceType::Youtube,
            &source_key,
            JobStatus::Downloading,
            source.filename.as_deref(),
            source.audio_mime_type.as_deref(),
            &request.model_id,
            request.task,
            &language,
        )
        .map_err(|e| e.to_string())?;

    state.events.publish(&job);

    let job_id = job.id.clone();
    let source_id = source.id;
    let url_clone = canonical_url.clone();
    let app_clone = app.clone();

    tauri::async_runtime::spawn(async move {
        prepare_youtube_audio(app_clone, job_id, source_id, source_key, url_clone).await;
    });

    Ok(job)
}

async fn prepare_youtube_audio(
    app: AppHandle,
    job_id: String,
    source_id: i64,
    source_key: String,
    url: String,
) {
    let state = app.state::<AppState>();
    let _permit = match state.download_semaphore.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return,
    };

    state.events.track_download_start();

    let audio_root = match audio_dir(&app) {
        Ok(p) => p,
        Err(_) => {
            state.events.track_download_end();
            return;
        }
    };
    let downloads_root = match downloads_dir(&app) {
        Ok(p) => p,
        Err(_) => {
            state.events.track_download_end();
            return;
        }
    };
    let _ = std::fs::create_dir_all(&audio_root);
    let _ = std::fs::create_dir_all(&downloads_root);

    let temp_audio = audio_root.join(format!("{}.wav", job_id));
    let cached_audio = audio_root.join(format!("youtube-{}.wav", source_key));
    let download_template = downloads_root.join(format!("{}.%(ext)s", job_id));

    let result: Result<(), String> = async {
        let info = run_yt_dlp_dump_json(&app, &url).await.ok();
        let title = info
            .as_ref()
            .and_then(|v| v.get("title"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let duration = info
            .as_ref()
            .and_then(|v| v.get("duration"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as u64;

        let max_seconds = crate::paths::MAX_DURATION_HOURS * 3600;
        if duration > max_seconds {
            let err = format!(
                "Video exceeds maximum duration of {} hours",
                crate::paths::MAX_DURATION_HOURS
            );
            return Err(err);
        }

        if let Some(ref t) = title {
            let _ = state.store.update_source_filename(source_id, t);
            let _ = state.store.update_job(
                &job_id,
                JobUpdate {
                    filename: Some(t.clone()),
                    ..Default::default()
                },
            );
            if let Ok(Some(j)) = state.store.get_job(&job_id) {
                state.events.publish(&j);
            }
        }

        run_yt_dlp_download(&app, &url, &download_template).await?;

        let downloaded = find_downloaded_file(&downloads_root, &job_id)
            .ok_or_else(|| "Downloaded file not found".to_string())?;

        let _ = state.store.update_job(
            &job_id,
            JobUpdate {
                status: Some(JobStatus::Extracting),
                progress: Some(0.65),
                filename: title.clone(),
                ..Default::default()
            },
        );
        if let Ok(Some(j)) = state.store.get_job(&job_id) {
            state.events.publish(&j);
        }

        run_ffmpeg_extract(&app, &downloaded, &temp_audio).await?;
        if let Err(e) = std::fs::rename(&temp_audio, &cached_audio) {
            return Err(format!("rename audio failed: {}", e));
        }
        let _ = std::fs::remove_file(&downloaded);

        let _ = state.store.update_source_audio(
            source_id,
            cached_audio.to_string_lossy().as_ref(),
            "audio/wav",
            title.as_deref(),
        );
        let _ = state.store.update_job(
            &job_id,
            JobUpdate {
                status: Some(JobStatus::Ready),
                progress: Some(1.0),
                filename: title.clone(),
                audio_mime_type: Some("audio/wav".to_string()),
                ..Default::default()
            },
        );
        if let Ok(Some(j)) = state.store.get_job(&job_id) {
            state.events.publish(&j);
        }

        Ok(())
    }
    .await;

    if let Err(message) = result {
        let _ = state.store.update_job(
            &job_id,
            JobUpdate {
                status: Some(JobStatus::Failed),
                error: Some(message),
                ..Default::default()
            },
        );
        if let Ok(Some(j)) = state.store.get_job(&job_id) {
            state.events.publish(&j);
        }
    }

    let _ = std::fs::remove_file(&temp_audio);
    state.events.track_download_end();
}

async fn run_yt_dlp_dump_json(app: &AppHandle, url: &str) -> Result<serde_json::Value, String> {
    let handle = spawn_sidecar(app, "yt-dlp", &["-J", "--no-warnings", "--quiet", url], None)
        .map_err(|e| e.to_string())?;
    collect_json_output(handle).await
}

async fn run_yt_dlp_download(
    app: &AppHandle,
    url: &str,
    template: &Path,
) -> Result<(), String> {
    let template_str = template.to_string_lossy().to_string();
    let handle = spawn_sidecar(
        app,
        "yt-dlp",
        &[
            "-f",
            "bestaudio/best",
            "-o",
            &template_str,
            "--no-playlist",
            "--no-warnings",
            "--quiet",
            "--retries",
            "3",
            "--fragment-retries",
            "3",
            url,
        ],
        None,
    )
    .map_err(|e| e.to_string())?;
    drain_until_exit(handle).await
}

async fn run_ffmpeg_extract(
    app: &AppHandle,
    input: &Path,
    output: &Path,
) -> Result<(), String> {
    let input_str = input.to_string_lossy().to_string();
    let output_str = output.to_string_lossy().to_string();
    let handle = spawn_sidecar(
        app,
        "ffmpeg",
        &[
            "-i",
            &input_str,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            &output_str,
        ],
        None,
    )
    .map_err(|e| e.to_string())?;
    drain_until_exit(handle).await
}

async fn collect_json_output(
    mut handle: crate::sidecar::SidecarHandle,
) -> Result<serde_json::Value, String> {
    let mut stdout_bytes: Vec<u8> = Vec::new();
    let mut stderr_text = String::new();
    let mut code: Option<i32> = None;

    while let Some(event) = handle.rx.recv().await {
        match event {
            CommandEvent::Stdout(line) => stdout_bytes.extend_from_slice(&line),
            CommandEvent::Stderr(line) => {
                stderr_text.push_str(&String::from_utf8_lossy(&line));
                stderr_text.push('\n');
            }
            CommandEvent::Terminated(payload) => {
                code = payload.code;
                break;
            }
            CommandEvent::Error(err) => return Err(err),
            _ => {}
        }
    }

    if !matches!(code, Some(0)) {
        return Err(format!(
            "yt-dlp exited with code {:?}: {}",
            code,
            stderr_text.trim()
        ));
    }
    serde_json::from_slice(&stdout_bytes).map_err(|e| e.to_string())
}

async fn drain_until_exit(mut handle: crate::sidecar::SidecarHandle) -> Result<(), String> {
    let mut stderr_text = String::new();
    let mut code: Option<i32> = None;

    while let Some(event) = handle.rx.recv().await {
        match event {
            CommandEvent::Stderr(line) => {
                stderr_text.push_str(&String::from_utf8_lossy(&line));
                stderr_text.push('\n');
            }
            CommandEvent::Terminated(payload) => {
                code = payload.code;
                break;
            }
            CommandEvent::Error(err) => return Err(err),
            _ => {}
        }
    }
    if !matches!(code, Some(0)) {
        return Err(format!(
            "sidecar exited with code {:?}: {}",
            code,
            stderr_text.trim()
        ));
    }
    Ok(())
}

fn find_downloaded_file(downloads_dir: &Path, stem: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(downloads_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s == stem)
            .unwrap_or(false)
            && !path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.ends_with("part"))
                .unwrap_or(false)
        {
            return Some(path);
        }
    }
    None
}

#[tauri::command]
pub async fn create_file_job(
    request: FileJobRequest,
    state: State<'_, AppState>,
) -> Result<Job, String> {
    if !FILE_HASH_RE.is_match(&request.file_hash) {
        return Err("Invalid file hash".to_string());
    }
    let language = normalize_language(&request.language);

    let source = state
        .store
        .get_or_create_source(
            SourceType::File,
            &request.file_hash,
            None,
            Some(&request.filename),
            Some(request.size_bytes),
        )
        .map_err(|e| e.to_string())?;

    if let Some(transcript) = state
        .store
        .find_transcript(source.id, &request.model_id, request.task, &language)
        .map_err(|e| e.to_string())?
    {
        return state
            .store
            .create_job_from_cache(
                source.id,
                SourceType::File,
                &request.file_hash,
                Some(&request.filename),
                None,
                &request.model_id,
                request.task,
                &language,
                &transcript.full_text,
                &transcript.segments,
            )
            .map_err(|e| e.to_string());
    }

    state
        .store
        .create_pending_job(
            source.id,
            SourceType::File,
            &request.file_hash,
            JobStatus::Ready,
            Some(&request.filename),
            None,
            &request.model_id,
            request.task,
            &language,
        )
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_job(job_id: String, state: State<'_, AppState>) -> Result<Job, String> {
    state
        .store
        .get_job(&job_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "Job not found".to_string())
}

#[tauri::command]
pub async fn get_prepared_audio_path(
    job_id: String,
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<String, String> {
    let job = state
        .store
        .get_job(&job_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "Job not found".to_string())?;

    if !matches!(job.source_type, SourceType::Youtube) {
        return Err("This job does not have prepared audio".to_string());
    }

    let audio_root = audio_dir(&app)?;
    let audio_path = audio_root.join(format!("youtube-{}.wav", job.source_key));
    if !audio_path.exists() {
        return Err("Prepared audio not found".to_string());
    }

    Ok(audio_path.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn persist_transcript(
    job_id: String,
    request: PersistTranscriptRequest,
    state: State<'_, AppState>,
) -> Result<Job, String> {
    let language = normalize_language(&request.language);
    let job = state
        .store
        .get_job(&job_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "Job not found".to_string())?;

    if matches!(job.status, JobStatus::Failed) {
        return Err("Cannot persist transcript for a failed job".to_string());
    }
    if request.model_id != job.model_id || request.task != job.task || language != job.language {
        return Err("Transcript payload does not match the job configuration".to_string());
    }

    let updated = state
        .store
        .persist_transcript(
            &job_id,
            &request.model_id,
            request.task,
            &language,
            &request.full_text,
            &request.segments,
        )
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "Job not found".to_string())?;

    state.events.publish(&updated);
    Ok(updated)
}

#[tauri::command]
pub async fn queue_status(state: State<'_, AppState>) -> Result<QueueStatusResponse, String> {
    let health = state.events.health();
    Ok(QueueStatusResponse {
        position: 0,
        total_in_queue: health.queue_length,
        estimated_wait_minutes: None,
    })
}

#[tauri::command]
pub async fn health(state: State<'_, AppState>) -> Result<HealthStatus, String> {
    Ok(state.events.health())
}

#[tauri::command]
pub async fn list_models(_app: AppHandle) -> Result<ModelStatusResponse, String> {
    // Models are loaded on demand from huggingface.co (cached in IndexedDB).
    // Treat all required models as ready so the UI doesn't gate on a local
    // bootstrap step that no longer exists.
    let items: Vec<ModelStatusItem> = REQUIRED_MODEL_IDS
        .iter()
        .map(|id| ModelStatusItem {
            model_id: id.clone(),
            ready: true,
            path: String::new(),
        })
        .collect();

    Ok(ModelStatusResponse {
        models_ready: true,
        missing_models: Vec::new(),
        items,
    })
}

#[tauri::command]
pub async fn resolve_models_dir(app: AppHandle) -> Result<String, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("models");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn export_transcript(request: ExportRequest) -> Result<(), String> {
    std::fs::write(&request.destination, request.content.as_bytes()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The body of the frontend's `MODEL_PRESETS` array, located WITHOUT the
    /// regex the production code uses — so it can count the presets independently
    /// of whether that regex still matches anything.
    fn model_presets_block() -> &'static str {
        let start = TRANSCRIPTION_CONFIG_TS
            .find("export const MODEL_PRESETS")
            .expect(
                "frontend/src/config/transcription.ts no longer exports MODEL_PRESETS — \
                 TRANSCRIPTION_CONFIG_TS is pointing at the wrong file",
            );
        let rest = &TRANSCRIPTION_CONFIG_TS[start..];
        // `\n]` and not `\n];`: the array closes with
        // `] as const satisfies readonly ModelPreset[];` — the `as const` is what
        // derives `ModelPresetId` from the array, so a deleted preset is a compile
        // error at every site that names it. A locator pinned to `];` silently ran
        // PAST the array into the rest of the file and counted the `"__auto__"`
        // comparisons in `modelIdForPreset` as if they were presets.
        let end = rest
            .find("\n]")
            .expect("MODEL_PRESETS array is not terminated by a closing bracket");
        &rest[..end]
    }

    /// The guard on the bug that killed the whole app.
    ///
    /// `REQUIRED_MODEL_IDS` is now DERIVED from the frontend's config, so the two
    /// cannot hold DIFFERENT lists — the class of bug that renamed four models on
    /// one side only, left `list_models` answering for ids the frontend no longer
    /// asks for, and disabled every entry point to transcription with no error
    /// anywhere.
    ///
    /// What can still go wrong is the derivation itself: if `MODEL_ID_RE` stops
    /// matching (the config reformatted, switched to single quotes, moved), it
    /// silently yields a SHORT or EMPTY list and the button goes dead exactly as
    /// before. So count the presets a second way — textually, without the regex —
    /// and demand the parser found all of them.
    #[test]
    fn model_ids_are_parsed_from_the_frontend_config() {
        let block = model_presets_block();
        let declared = block.matches("modelId:").count();

        assert!(
            declared > 1,
            "no model presets found in MODEL_PRESETS — the parser is looking at the wrong thing"
        );
        assert_eq!(
            block.matches("__auto__").count(),
            1,
            "expected exactly one `__auto__` sentinel preset in MODEL_PRESETS"
        );

        // Every preset declares a `modelId`; exactly one is the `__auto__`
        // sentinel, which the frontend resolves to one of the real presets.
        assert_eq!(
            REQUIRED_MODEL_IDS.len(),
            declared - 1,
            "the model-id parser missed a preset in frontend/src/config/transcription.ts. \
             list_models must offer EXACTLY the ids the frontend can select — the frontend \
             gates the Transcribe button on an exact id match against this answer — so fix \
             MODEL_ID_RE. Do NOT retype the list here: that is how it drifted last time. \
             parsed: {:?}",
            *REQUIRED_MODEL_IDS
        );

        for id in REQUIRED_MODEL_IDS.iter() {
            assert!(
                id.contains('/') && !id.contains("__auto__"),
                "{id:?} does not look like a HuggingFace repo id"
            );
            assert!(
                block.contains(&format!("\"{id}\"")),
                "{id:?} is not one of the frontend's presets"
            );
        }
    }

    /// The frontend asks for word timestamps (`return_timestamps: 'word'`), which
    /// only the `_timestamped` exports can serve — they are the ones carrying the
    /// `cross_attentions` outputs and `alignment_heads` that DTW needs. A plain
    /// export throws "Model outputs must contain cross attentions to extract
    /// timestamps" at transcribe time.
    #[test]
    fn every_model_is_a_timestamped_export() {
        for id in REQUIRED_MODEL_IDS.iter() {
            assert!(
                id.ends_with("_timestamped"),
                "{id:?} is not a _timestamped export — word timestamps will throw at runtime"
            );
        }
    }
}
