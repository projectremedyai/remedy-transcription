use std::io::Read;
use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
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

/// The body of the frontend's `MODEL_PRESETS` array, located WITHOUT
/// `MODEL_ID_RE` — so the regex's match count can be checked against an
/// independent count instead of trusting whatever it happens to match.
///
/// Shared by `REQUIRED_MODEL_IDS` below (a partial regex miss must fail loudly
/// at runtime, not only under `cargo test`) and by the guard test in `mod
/// tests`, which is exactly where this locator originated.
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
    // A TOTAL parse miss yields an empty Vec, but a PARTIAL miss is just as
    // dangerous and easy to overlook: if someone switches one preset to single
    // quotes, `MODEL_ID_RE` matches the other three, `ids` is non-empty, and
    // `list_models` quietly answers with three models instead of four — the
    // fourth preset's Transcribe button goes dead with no error anywhere. That
    // is the exact bug this derivation exists to prevent, just narrower than an
    // empty-list check catches.
    //
    // So compare the regex's match count against an independent, regex-free
    // count of the presets declared in the same `include_str!`'d source
    // (`model_presets_block()`, shared with the guard test below) rather than
    // merely checking the regex matched *something*. Leaving this to a test
    // means it comes back silently for anyone who builds without running the
    // suite. A panic here is a crash on the first `list_models` call with a
    // message that names the cause; a dead button is a bug report six weeks
    // later.
    let block = model_presets_block();
    let declared = block.matches("modelId:").count();

    assert!(
        declared > 1,
        "no model presets found in frontend/src/config/transcription.ts's \
         MODEL_PRESETS block — model_presets_block() is looking at the wrong thing."
    );

    assert_eq!(
        ids.len(),
        declared - 1, // one `modelId:` belongs to the `__auto__` sentinel, not a real model
        "MODEL_ID_RE parsed {} model id(s) but frontend/src/config/transcription.ts's \
         MODEL_PRESETS block declares {} `modelId:` entries (one of which is the \
         `__auto__` sentinel) — a PARTIAL parse miss. list_models must offer EXACTLY \
         the ids the frontend can select — the frontend gates the Transcribe button on \
         an exact id match against this answer — so any preset MODEL_ID_RE missed gets \
         a dead Transcribe button with no error anywhere. The config's format has \
         changed (quoting? a rename? moved file?) — fix MODEL_ID_RE. Do NOT retype the \
         model list here: that is how it drifted last time. parsed: {:?}",
        ids.len(),
        declared,
        ids
    );

    ids
});

static YOUTUBE_ID_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[A-Za-z0-9_-]{11}$").expect("valid regex"));

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
    /// A real filesystem path.
    ///
    /// This used to be a `file_hash` + `filename` + `size_bytes` triple, because
    /// the webview held a browser `File` — which has no path — read it into an
    /// `ArrayBuffer`, hashed THAT, and decoded the audio itself. Rust never saw
    /// the file, so a Rust-side diarizer had no audio for it. Local files now
    /// arrive as a path (from the Tauri dialog or a file drop) and Rust derives
    /// the name, the size and the content hash itself.
    pub path: String,
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

/// Where the ONE canonical audio artifact for a source lives, inside `audio_dir`.
///
/// Every job — YouTube or local file — has one of these, and it is always a
/// 16 kHz mono `pcm_s16le` WAV (see `ffmpeg_canonical_wav_args`). That invariant
/// is what lets anything on the Rust side (diarization, in particular) assume
/// there is audio on disk for ANY job, rather than only for the YouTube ones.
///
/// The `youtube-{id}.wav` shape is the one the YouTube path already used, so
/// WAVs already sitting in a user's cache keep resolving; `file-{sha256}.wav` is
/// the new twin. Prefixing by source type also keeps the two key spaces from
/// ever colliding.
fn prepared_audio_filename(source_type: SourceType, source_key: &str) -> String {
    format!("{}-{}.wav", source_type.as_str(), source_key)
}

/// The source key for a local file is a hash of its CONTENTS, not of its path.
///
/// That is what the webview's old `sha256Hex(await file.arrayBuffer())` gave us,
/// and the transcript cache is keyed on it: renaming or moving a file must still
/// hit its existing transcript, and two copies of the same recording must not be
/// transcribed twice. Streamed in chunks rather than read whole, because not
/// pulling entire media files into memory is half the point of this change.
fn hash_file(path: &Path) -> Result<String, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 64 * 1024];

    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// The ffmpeg invocation that DEFINES "canonical audio" for this app.
///
/// 16 kHz, mono, signed 16-bit PCM: exactly what sherpa-onnx wants for
/// diarization, and exactly what the webview's `AudioContext({sampleRate:
/// 16000})` decodes without resampling. Both source types go through this one
/// argument list on purpose — a second ffmpeg invocation elsewhere is a second
/// thing to drift, and a drifted sample rate is a silent wrong-answer bug in the
/// diarizer, not a crash.
fn ffmpeg_canonical_wav_args<'a>(input: &'a str, output: &'a str) -> Vec<&'a str> {
    vec![
        "-i", input, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", output,
    ]
}

/// Called with the fraction of the extraction ffmpeg has completed, 0.0..=1.0.
///
/// `Send + Sync` because the futures these run inside are handed to
/// `tauri::async_runtime::spawn`, which requires a `Send` future.
type ProgressSink<'a> = &'a (dyn Fn(f32) + Send + Sync);

/// ffmpeg's `HH:MM:SS.frac` — used for both its `Duration:` banner and the
/// `out_time=` lines `-progress` writes.
fn parse_ffmpeg_timestamp(value: &str) -> Option<f64> {
    let mut seconds = 0.0f64;
    let mut parts = 0;
    for field in value.trim().split(':') {
        let number: f64 = field.parse().ok()?;
        seconds = seconds * 60.0 + number;
        parts += 1;
    }
    if parts != 3 {
        return None;
    }
    Some(seconds)
}

/// The total duration, out of ffmpeg's stderr banner:
/// `  Duration: 00:07:30.05, start: 0.000000, bitrate: 128 kb/s`
///
/// A live stream or a malformed container reports `N/A`, which parses to `None`
/// — progress then simply never advances, rather than dividing by zero.
fn parse_ffmpeg_duration_line(line: &str) -> Option<f64> {
    let rest = line.trim_start().strip_prefix("Duration:")?;
    let value = rest.split(',').next()?;
    parse_ffmpeg_timestamp(value)
}

/// A position, out of the `key=value` stream `-progress pipe:1` writes to stdout.
fn parse_ffmpeg_progress_line(line: &str) -> Option<f64> {
    let value = line.trim().strip_prefix("out_time=")?;
    parse_ffmpeg_timestamp(value)
}

/// Any media file in, the canonical WAV out — the step both source types share.
///
/// ffmpeg writes to `scratch` and the result is renamed into `destination`, so a
/// crashed or half-finished extraction never leaves a TRUNCATED WAV sitting at
/// the path `get_prepared_audio_path` hands out. The rename is what makes the
/// destination's existence mean "this is complete", which is the check both the
/// audio cache and the transcriber rely on.
async fn produce_canonical_wav(
    app: &AppHandle,
    input: &Path,
    scratch: &Path,
    destination: &Path,
    on_progress: ProgressSink<'_>,
) -> Result<(), String> {
    run_ffmpeg_extract(app, input, scratch, on_progress).await?;
    std::fs::rename(scratch, destination).map_err(|e| format!("rename audio failed: {}", e))
}

/// Move a job's progress bar and tell the frontend. Status is left alone.
fn report_progress(state: &AppState, job_id: &str, progress: f64) {
    let _ = state.store.update_job(
        job_id,
        JobUpdate {
            progress: Some(progress.clamp(0.0, 1.0)),
            ..Default::default()
        },
    );
    if let Ok(Some(j)) = state.store.get_job(job_id) {
        state.events.publish(&j);
    }
}

/// Mark a job failed and tell the frontend. Both `prepare_*` paths end here when
/// something goes wrong; without the publish, the UI waits on `Ready` forever.
fn fail_job(state: &AppState, job_id: &str, message: String) {
    let _ = state.store.update_job(
        job_id,
        JobUpdate {
            status: Some(JobStatus::Failed),
            error: Some(message),
            ..Default::default()
        },
    );
    if let Ok(Some(j)) = state.store.get_job(job_id) {
        state.events.publish(&j);
    }
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

    // Every exit from here on MUST resolve the job. Returning without publishing
    // anything strands it in `Downloading`/`Extracting` with no event ever coming,
    // and the frontend's 300 ms poll then spins on it forever while both entry
    // points sit locked behind `isBusy` — a hang with no way out but quitting.
    // Same rule as `prepare_file_audio`; these three returns were the twin that
    // never got it.
    let _permit = match state.download_semaphore.clone().acquire_owned().await {
        Ok(p) => p,
        Err(e) => {
            fail_job(&state, &job_id, format!("Could not queue download: {}", e));
            return;
        }
    };

    state.events.track_download_start();

    let audio_root = match audio_dir(&app) {
        Ok(p) => p,
        Err(e) => {
            fail_job(&state, &job_id, e);
            state.events.track_download_end();
            return;
        }
    };
    let downloads_root = match downloads_dir(&app) {
        Ok(p) => p,
        Err(e) => {
            fail_job(&state, &job_id, e);
            state.events.track_download_end();
            return;
        }
    };
    let _ = std::fs::create_dir_all(&audio_root);
    let _ = std::fs::create_dir_all(&downloads_root);

    let temp_audio = audio_root.join(format!("{}.wav", job_id));
    let cached_audio =
        audio_root.join(prepared_audio_filename(SourceType::Youtube, &source_key));
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

        // The download owns 0.0..0.65; extraction owns the rest, up to the 1.0
        // that `Ready` sets.
        let report = |fraction: f32| {
            report_progress(&state, &job_id, 0.65 + f64::from(fraction) * 0.30);
        };
        produce_canonical_wav(&app, &downloaded, &temp_audio, &cached_audio, &report).await?;
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
        fail_job(&state, &job_id, message);
    }

    let _ = std::fs::remove_file(&temp_audio);
    state.events.track_download_end();
}

/// The local-file twin of `prepare_youtube_audio`.
///
/// Same `tauri::async_runtime::spawn`, same `JobEvents` publishing, same audio
/// cache directory, same canonical WAV. The ONLY difference is where the input
/// comes from: a path the user picked, rather than one yt-dlp downloaded. Local
/// files used to skip all of this — the webview decoded them with `AudioContext`
/// and posted the PCM straight to the worker, so no WAV ever hit the disk and
/// Rust never saw the audio at all.
async fn prepare_file_audio(
    app: AppHandle,
    job_id: String,
    source_id: i64,
    source_key: String,
    input: PathBuf,
) {
    let state = app.state::<AppState>();

    // The same concurrency bound the YouTube path respects. ffmpeg is the
    // expensive step in both, and it is the step being shared here.
    let _permit = match state.download_semaphore.clone().acquire_owned().await {
        Ok(p) => p,
        // Unreachable while nothing closes the semaphore — but returning here
        // would strand the job in `Extracting` with no event ever coming, and the
        // frontend's 300 ms poll would spin on it until the app was quit. A job
        // this function owns is a job it must resolve, one way or the other.
        Err(e) => {
            fail_job(&state, &job_id, format!("Could not queue extraction: {}", e));
            return;
        }
    };

    // A file extraction holds a download permit, so it MUST be counted like one.
    // It was not, so `health()` and `queue_status` reported 0 active downloads
    // while a permit was held — under-reporting exactly the contention that the
    // permit exists to express.
    state.events.track_download_start();

    let audio_root = match audio_dir(&app) {
        Ok(p) => p,
        Err(e) => {
            fail_job(&state, &job_id, e);
            state.events.track_download_end();
            return;
        }
    };
    let _ = std::fs::create_dir_all(&audio_root);

    let scratch = audio_root.join(format!("{}.wav", job_id));
    let destination = audio_root.join(prepared_audio_filename(SourceType::File, &source_key));

    let result: Result<(), String> = async {
        let _ = state.store.update_job(
            &job_id,
            JobUpdate {
                status: Some(JobStatus::Extracting),
                progress: Some(0.05),
                ..Default::default()
            },
        );
        if let Ok(Some(j)) = state.store.get_job(&job_id) {
            state.events.publish(&j);
        }

        // A local file's whole preparation IS the extraction — there is no
        // download phase to share the bar with — so it gets nearly all of it.
        let report = |fraction: f32| {
            report_progress(&state, &job_id, 0.05 + f64::from(fraction) * 0.90);
        };
        produce_canonical_wav(&app, &input, &scratch, &destination, &report).await?;

        let _ = state.store.update_source_audio(
            source_id,
            destination.to_string_lossy().as_ref(),
            "audio/wav",
            None,
        );
        let _ = state.store.update_job(
            &job_id,
            JobUpdate {
                status: Some(JobStatus::Ready),
                progress: Some(1.0),
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
        fail_job(&state, &job_id, message);
    }

    let _ = std::fs::remove_file(&scratch);
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
    on_progress: ProgressSink<'_>,
) -> Result<(), String> {
    let input_str = input.to_string_lossy().to_string();
    let output_str = output.to_string_lossy().to_string();

    // `-progress pipe:1` writes machine-readable `key=value` progress to STDOUT
    // (free, since the audio goes to a file), newline-terminated — unlike the
    // default `\r`-updated stats line, which the sidecar's line-oriented reader
    // would not surface until the process ended. `-nostats` drops that line.
    //
    // These are global options, so they precede `-i`. The canonical argument list
    // itself is untouched: it is the format contract, and it stays the one thing
    // both source types share.
    let mut args = vec!["-nostats", "-progress", "pipe:1"];
    args.extend(ffmpeg_canonical_wav_args(&input_str, &output_str));

    let handle = spawn_sidecar(app, "ffmpeg", &args, None).map_err(|e| e.to_string())?;
    drain_ffmpeg(handle, on_progress).await
}

/// `drain_until_exit`, plus ffmpeg's progress stream.
///
/// A local file used to sit at a flat "Extracting audio… 10%" for however long
/// ffmpeg took — a minute or more on a long recording, with nothing moving. The
/// total comes from the `Duration:` banner on stderr and the position from
/// `out_time=` on stdout; if the total is unknown (a live stream, a broken
/// container) no progress is reported and the bar simply holds, which is what it
/// did before.
async fn drain_ffmpeg(
    mut handle: crate::sidecar::SidecarHandle,
    on_progress: ProgressSink<'_>,
) -> Result<(), String> {
    let mut stderr_text = String::new();
    let mut code: Option<i32> = None;
    let mut total: Option<f64> = None;
    let mut last_reported = 0.0f32;

    while let Some(event) = handle.rx.recv().await {
        match event {
            CommandEvent::Stdout(line) => {
                let text = String::from_utf8_lossy(&line);
                for entry in text.lines() {
                    let Some(position) = parse_ffmpeg_progress_line(entry) else {
                        continue;
                    };
                    let Some(duration) = total.filter(|d| *d > 0.0) else {
                        continue;
                    };
                    let fraction = (position / duration).clamp(0.0, 1.0) as f32;
                    // Every tick is a DB write and an event to the webview, so
                    // only move on a visible change.
                    if fraction - last_reported >= 0.02 {
                        last_reported = fraction;
                        on_progress(fraction);
                    }
                }
            }
            CommandEvent::Stderr(line) => {
                let text = String::from_utf8_lossy(&line);
                if total.is_none() {
                    total = text.lines().find_map(parse_ffmpeg_duration_line);
                }
                stderr_text.push_str(&text);
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

/// A local file now takes exactly the same route as a YouTube video: a job, an
/// ffmpeg pass into the canonical WAV, `Ready`, and the frontend fetching that
/// WAV back. It used to return `Ready` immediately without touching the audio,
/// because the webview did the decoding — which left Rust with no audio on disk
/// for the one source type users are most likely to diarize.
#[tauri::command]
pub async fn create_file_job(
    request: FileJobRequest,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<Job, String> {
    let audio_root = audio_dir(&app)?;
    state
        .store
        .cleanup_expired_audio(crate::paths::PREPARED_AUDIO_TTL_HOURS, &audio_root);

    let input = PathBuf::from(&request.path);
    if !input.is_file() {
        return Err(format!("File not found: {}", request.path));
    }

    let filename = input
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("audio")
        .to_string();
    let size_bytes = std::fs::metadata(&input).map(|m| m.len() as i64).ok();
    let language = normalize_language(&request.language);

    // Hashing a multi-gigabyte file is not something to do on an async worker
    // thread; the whole runtime stalls behind it.
    let hash_input = input.clone();
    let source_key = tauri::async_runtime::spawn_blocking(move || hash_file(&hash_input))
        .await
        .map_err(|e| e.to_string())??;

    let source = state
        .store
        .get_or_create_source(
            SourceType::File,
            &source_key,
            Some(&request.path),
            Some(&filename),
            size_bytes,
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
                &source_key,
                Some(&filename),
                source.audio_mime_type.as_deref(),
                &request.model_id,
                request.task,
                &language,
                &transcript.full_text,
                &transcript.segments,
            )
            .map_err(|e| e.to_string());
    }

    // The canonical WAV may already be on disk from an earlier job for this same
    // file — the same audio-cache short circuit the YouTube path takes. The WAV
    // is content-addressed, so this hits even if the user moved or renamed the
    // file since.
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
                    SourceType::File,
                    &source_key,
                    JobStatus::Ready,
                    Some(&filename),
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
            SourceType::File,
            &source_key,
            JobStatus::Extracting,
            Some(&filename),
            None,
            &request.model_id,
            request.task,
            &language,
        )
        .map_err(|e| e.to_string())?;

    state.events.publish(&job);

    let job_id = job.id.clone();
    let source_id = source.id;
    let key = source_key.clone();
    let app_clone = app.clone();

    tauri::async_runtime::spawn(async move {
        prepare_file_audio(app_clone, job_id, source_id, key, input).await;
    });

    Ok(job)
}

#[tauri::command]
pub async fn get_job(job_id: String, state: State<'_, AppState>) -> Result<Job, String> {
    state
        .store
        .get_job(&job_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "Job not found".to_string())
}

/// Answers for EVERY job, not just the YouTube ones.
///
/// This used to reject anything that was not a YouTube job — "This job does not
/// have prepared audio" — which was true at the time: local files were decoded
/// in the webview and no WAV was ever written for them. Now both source types
/// produce one, so both can be fetched back, and anything Rust-side that wants
/// audio for a job (diarization) can rely on this.
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

    let audio_root = audio_dir(&app)?;
    let audio_path = audio_root.join(prepared_audio_filename(job.source_type, &job.source_key));
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

    /// The format contract, pinned.
    ///
    /// sherpa-onnx's diarization expects 16 kHz mono 16-bit PCM. ffmpeg will
    /// happily produce something else if these flags drift, and NOTHING downstream
    /// would throw: the webview resamples whatever it decodes, so transcription
    /// would look fine while the diarizer silently reads audio at the wrong rate
    /// and returns speaker turns at the wrong TIMES. That is a wrong-answer bug,
    /// not a crash, so the flags are asserted here rather than trusted.
    #[test]
    fn the_canonical_wav_is_16khz_mono_pcm_s16le() {
        let args = ffmpeg_canonical_wav_args("/in.mp4", "/out.wav");

        let flag = |name: &str| -> Option<String> {
            args.iter()
                .position(|a| *a == name)
                .and_then(|i| args.get(i + 1))
                .map(|v| v.to_string())
        };

        assert_eq!(flag("-ar").as_deref(), Some("16000"), "sample rate must be 16 kHz");
        assert_eq!(flag("-ac").as_deref(), Some("1"), "must be mono");
        assert_eq!(
            flag("-acodec").as_deref(),
            Some("pcm_s16le"),
            "must be signed 16-bit little-endian PCM"
        );
        assert_eq!(flag("-i").as_deref(), Some("/in.mp4"));
        assert!(args.contains(&"-vn"), "video streams must be dropped");
        assert_eq!(
            args.last().copied(),
            Some("/out.wav"),
            "ffmpeg takes the output path last"
        );
    }

    /// Both source types get a prepared WAV, and they cannot collide.
    ///
    /// The `youtube-{id}.wav` shape is load-bearing beyond neatness: it is the
    /// name the YouTube path has always written, so changing it would orphan
    /// every WAV already in a user's audio cache.
    #[test]
    fn every_source_type_has_a_prepared_wav_path() {
        assert_eq!(
            prepared_audio_filename(SourceType::Youtube, "dQw4w9WgXcQ"),
            "youtube-dQw4w9WgXcQ.wav"
        );
        assert_eq!(
            prepared_audio_filename(SourceType::File, "abc123"),
            "file-abc123.wav"
        );

        // The two key spaces are independent, so the prefix is what keeps a file
        // hash from ever shadowing a video id.
        assert_ne!(
            prepared_audio_filename(SourceType::File, "collide"),
            prepared_audio_filename(SourceType::Youtube, "collide")
        );
    }

    /// The source key for a local file is a SHA-256 of its BYTES.
    ///
    /// Keying on the path instead would re-transcribe a file the moment the user
    /// renamed it, and would treat two copies of the same recording as two
    /// sources. The webview's old `sha256Hex(await file.arrayBuffer())` was
    /// content-addressed, and the transcript cache is keyed on its output — so the
    /// Rust replacement must be content-addressed over the same bytes with the
    /// same algorithm, or every transcript users already made becomes unreachable.
    /// (That exact class of bug — a changed cache key orphaning existing
    /// transcripts — is what the `model_id_alias` machinery in `store.rs` exists to
    /// clean up after.)
    #[test]
    fn a_local_file_is_keyed_by_a_sha256_of_its_contents_not_its_path() {
        let dir = std::env::temp_dir().join(format!("remedy-hash-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir");

        let original = dir.join("lecture.mp3");
        let renamed = dir.join("lecture-final-v2.mp3");
        let different = dir.join("other.mp3");

        std::fs::write(&original, b"the same bytes").expect("write");
        std::fs::write(&renamed, b"the same bytes").expect("write");
        std::fs::write(&different, b"different bytes").expect("write");

        let key = hash_file(&original).expect("hash");

        assert_eq!(key.len(), 64, "a SHA-256 is 64 hex characters");
        assert!(
            key.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "must be lowercase hex, like the webview's sha256Hex: {key}"
        );

        assert_eq!(
            key,
            hash_file(&renamed).expect("hash"),
            "the same bytes under a different name must be the SAME source — \
             otherwise renaming a file silently re-transcribes it"
        );
        assert_ne!(
            key,
            hash_file(&different).expect("hash"),
            "different bytes must be different sources"
        );

        // It is really SHA-256, and not merely *a* stable digest: pinned against
        // the standard empty-input vector. This is what guarantees the Rust hash
        // and the webview's `crypto.subtle.digest("SHA-256", ...)` agree, so
        // transcripts cached under the old frontend-computed key still hit.
        let empty = dir.join("empty.mp3");
        std::fs::write(&empty, b"").expect("write");
        assert_eq!(
            hash_file(&empty).expect("hash"),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "not SHA-256 — every transcript cached under the old key would be orphaned"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The progress bar for a local file used to go 0.1 → 1.0 with NOTHING in
    /// between, so a long recording sat at a frozen "Extracting audio… 10%" for
    /// as long as ffmpeg took. These two lines are the whole source of movement:
    /// the total from ffmpeg's stderr banner, the position from the `-progress`
    /// stream on stdout. If either stops parsing, the bar silently freezes again
    /// rather than breaking — which is exactly why it is asserted.
    #[test]
    fn ffmpeg_progress_is_read_from_its_duration_banner_and_its_progress_stream() {
        assert_eq!(
            parse_ffmpeg_duration_line(
                "  Duration: 00:07:30.05, start: 0.000000, bitrate: 128 kb/s"
            ),
            Some(450.05)
        );
        assert_eq!(
            parse_ffmpeg_progress_line("out_time=00:00:12.500000"),
            Some(12.5)
        );

        // An hour-plus source: the fields are HH:MM:SS, so an unhandled hour
        // would put the bar at 60x the true position and peg it at 100%.
        assert_eq!(
            parse_ffmpeg_duration_line("  Duration: 01:30:00.00, start: 0.000000"),
            Some(5400.0)
        );

        // A live stream or a broken container reports no duration. That must
        // yield `None` — not 0.0, which would be a divide-by-zero — and the bar
        // then simply holds, as it did before.
        assert_eq!(
            parse_ffmpeg_duration_line("  Duration: N/A, start: 0.000000, bitrate: N/A"),
            None
        );

        // The `-progress` stream is many keys; only `out_time` is a position.
        assert_eq!(parse_ffmpeg_progress_line("out_time_us=12500000"), None);
        assert_eq!(parse_ffmpeg_progress_line("progress=continue"), None);
        assert_eq!(parse_ffmpeg_duration_line("Stream #0:0: Audio: aac"), None);
    }

    /// The canonical argument list is the format contract, and the progress flags
    /// must not disturb it: they are GLOBAL options, so they precede `-i`, and the
    /// output path stays last where ffmpeg expects it.
    #[test]
    fn the_progress_flags_do_not_disturb_the_canonical_arguments() {
        let canonical = ffmpeg_canonical_wav_args("/in.mp4", "/out.wav");
        let mut args = vec!["-nostats", "-progress", "pipe:1"];
        args.extend(canonical.clone());

        let input_at = args.iter().position(|a| *a == "-i").expect("-i");
        let progress_at = args.iter().position(|a| *a == "-progress").expect("-progress");
        assert!(
            progress_at < input_at,
            "-progress is a global option and must precede -i"
        );
        assert_eq!(args.last().copied(), Some("/out.wav"));
        assert!(
            args.windows(canonical.len())
                .any(|window| window == canonical.as_slice()),
            "the canonical argument list must survive intact"
        );
    }
}
