use std::collections::{BTreeSet, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Manager, State};
use tauri_plugin_shell::process::CommandEvent;

// `CANCELLED` is deliberately NOT imported here: cancellation is decided by
// asking the `CancelToken`, never by matching the engine's error text (a
// subprocess's stderr is not evidence about what the user did). The tests import
// it, to prove an engine that prints those very words still degrades.
use crate::diarize::{
    diarize_in_background, CancelToken, DiarizeOptions, Diarizer, SidecarDiarizer, SpeakerTurn,
};
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
    let path = prepared_audio_for_job(&job_id, &state, &app)?;
    Ok(path.to_string_lossy().to_string())
}

/// The canonical WAV for a job, or a REAL error.
///
/// Shared by `get_prepared_audio_path` (which the webview uses to fetch the
/// audio) and `diarize_job` (which hands it to the engine), so the two cannot
/// disagree about where a job's audio is.
///
/// Both failures here are genuine errors, not degradations: a job id nobody has
/// heard of, or a job whose audio has been cleaned up (see
/// `PREPARED_AUDIO_TTL_HOURS`). Neither is "diarization broke".
fn prepared_audio_for_job(
    job_id: &str,
    state: &AppState,
    app: &AppHandle,
) -> Result<PathBuf, String> {
    let job = state
        .store
        .get_job(job_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "Job not found".to_string())?;

    let audio_path =
        audio_dir(app)?.join(prepared_audio_filename(job.source_type, &job.source_key));
    if !audio_path.exists() {
        return Err("Prepared audio not found".to_string());
    }
    Ok(audio_path)
}

// ---------------------------------------------------------------------------
// Speaker diarization.
//
// The governing rule, and the reason this file is arranged the way it is:
// DIARIZATION FAILURE MUST NEVER FAIL TRANSCRIPTION. `Err` from `diarize_job`
// is reserved for things that are wrong with the REQUEST -- a job that does not
// exist, a job with no audio, a speaker count that is not a speaker count.
// Everything that can go wrong with the ENGINE -- a missing model, a sidecar
// killed by SIGABRT from inside ONNX Runtime, a timeout, garbage on stdout --
// comes back as `Ok(DiarizationOutcome::Degraded)`: no speaker labels, and the
// transcript is untouched.
// ---------------------------------------------------------------------------

/// Where the two ONNX models sit inside the bundle's resource directory.
///
/// This is the TARGET side of `tauri.conf.json`'s
/// `"resources": { "../models/diarization": "models/diarization" }` -- the models
/// live at the repo root (that is where `scripts/fetch-sidecars.sh` puts them and
/// where the `#[ignore]`d model tests read them from), and the map is what
/// relocates them to a stable, `..`-free path under `$RESOURCE`. A plain
/// `"../models/**/*"` list entry would land them under `_up_/models/...`, because
/// `tauri_utils::resources::resource_relpath` rewrites `..` to `_up_`.
///
/// `the_bundled_model_paths_are_the_ones_tauri_will_actually_produce` pins these
/// two strings against `tauri.conf.json` itself, so the config and this code
/// cannot drift into a bundle whose models the app cannot find.
const SEGMENTATION_MODEL: &str = "models/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx";
const EMBEDDING_MODEL: &str = "models/diarization/wespeaker_en_voxceleb_CAM++.onnx";

/// The `externalBin` name. Tauri strips the target triple when it stages the
/// binary, so `binaries/diarize-sidecar-aarch64-apple-darwin` is copied next to
/// the main executable as plain `diarize-sidecar` -- in `target/debug` under
/// `cargo run`, and in `Contents/MacOS` inside the `.app`.
const SIDECAR_EXE: &str = "diarize-sidecar";

/// The three files diarization needs, all of them present.
#[derive(Debug)]
struct DiarizationAssets {
    exe: PathBuf,
    segmentation_model: PathBuf,
    embedding_model: PathBuf,
}

/// Resolve the engine's files from a bundle layout, and say precisely which one
/// is missing if any is.
///
/// Split from [`diarization_assets`] so it can be tested against a FAKE BUNDLE
/// -- a temp dir shaped like `Foo.app/Contents/{MacOS,Resources}` -- rather than
/// only against the dev tree, where a repo-relative path would work by accident
/// and the bundled layout would be the thing that breaks in front of a user.
///
/// A missing file is NOT an error here in the `Err`-fails-the-job sense: the
/// caller turns it into a [`DiarizationOutcome::Degraded`].
fn diarization_assets_in(resource_dir: &Path, exe_dir: &Path) -> Result<DiarizationAssets, String> {
    let assets = DiarizationAssets {
        exe: exe_dir.join(format!("{SIDECAR_EXE}{}", std::env::consts::EXE_SUFFIX)),
        segmentation_model: resource_dir.join(SEGMENTATION_MODEL),
        embedding_model: resource_dir.join(EMBEDDING_MODEL),
    };

    for (what, path) in [
        ("the diarization sidecar", &assets.exe),
        ("the segmentation model", &assets.segmentation_model),
        ("the speaker embedding model", &assets.embedding_model),
    ] {
        if !path.is_file() {
            return Err(format!(
                "{what} is not installed at {} -- this build has no speaker diarization. \
                 Run ./scripts/fetch-sidecars.sh and ./scripts/build-diarize-sidecar.sh, \
                 then rebuild.",
                path.display()
            ));
        }
    }

    Ok(assets)
}

/// The same, for the running app.
///
/// `PathResolver::resource_dir()` -- the base `BaseDirectory::Resource` resolves
/// against -- is `Contents/Resources` in a bundled `.app` and the cargo output
/// directory in dev (`tauri_build` copies `resources` there for exactly this
/// reason), so one code path serves both. The sidecar is resolved from the
/// executable's own directory rather than from a hardcoded dev-tree path, because
/// that is where Tauri stages -- and code-signs -- it.
fn diarization_assets(app: &AppHandle) -> Result<DiarizationAssets, String> {
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("could not resolve the app's resource directory: {e}"))?;

    let exe = std::env::current_exe()
        .map_err(|e| format!("could not resolve the app's own executable: {e}"))?;
    let exe_dir = exe
        .parent()
        .ok_or_else(|| "the app's executable has no parent directory".to_string())?;

    diarization_assets_in(&resource_dir, exe_dir)
}

/// What a diarization attempt produced. **Three outcomes, not two.**
///
/// This exists because of one specific way to ship a lie: a transcript with no
/// speaker labels *because the engine crashed* must not be indistinguishable
/// from a transcript with no speaker labels *because one person was talking*.
/// Returning `Vec<SpeakerTurn>` cannot tell those apart -- both are "no labels
/// to draw" -- so the failure would arrive at the UI as a confident,
/// silent, wrong answer. The tag is the whole point of the type; the UI must
/// render `Degraded` as a warning and never as a speaker.
///
/// `Cancelled` is separated from `Degraded` for the same reason at one remove:
/// telling a user "speaker detection failed" for a thing they themselves stopped
/// is a lie in the other direction.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DiarizationOutcome {
    /// The engine ran.
    ///
    /// `turns` MAY BE EMPTY, and that is a SUCCESS: silence and zero-length audio
    /// have no speaker turns. `speaker_count` is then 0 -- do not divide by it.
    ///
    /// The ids in `turns` are DENSE (`0..speaker_count`), unlike the engine's own
    /// -- see [`densify`].
    Succeeded {
        turns: Vec<SpeakerTurn>,
        speaker_count: usize,
    },
    /// The engine did not run, or died trying. There are no speaker labels for
    /// this transcript, and the transcript is otherwise completely unaffected.
    ///
    /// `reason` is for the user: it already names the sidecar, the signal, or the
    /// missing file, and distinguishes a corrupt model from a bug in the sidecar.
    Degraded { reason: String },
    /// The user cancelled. The sidecar was killed; nothing is still running.
    Cancelled,
}

/// The engine's speaker ids are opaque and SPARSE. These are not.
///
/// sherpa hands back whatever cluster labels survived its duration-merging step,
/// and they come out with gaps: the two-speaker fixture returns `{0, 3}` --
/// measured, and pinned by `speaker_ids_are_sparse_not_contiguous` in
/// `diarize.rs`. Anything that indexes an array by a raw id, or renders
/// `Speaker {id + 1}`, is therefore wrong in a way that ships: a two-speaker file
/// would render "Speaker 1" and "Speaker 4", and a palette lookup would panic or
/// wrap.
///
/// Rather than publish that hazard to the frontend and hope every future caller
/// reads the doc comment, the ids are remapped here, at the boundary, to a dense
/// `0..n` -- so `speaker` IS a valid index on the other side, `speaker_count` IS
/// the number of speakers, and the hazard cannot escape this function.
///
/// The remap is by ascending original id, so it is deterministic and does not
/// depend on the turns being sorted by time. A dense id is a stable label WITHIN
/// one run and means nothing across runs; in particular it is not "who spoke
/// first".
fn densify(turns: Vec<SpeakerTurn>) -> (Vec<SpeakerTurn>, usize) {
    let distinct: BTreeSet<u32> = turns.iter().map(|t| t.speaker).collect();
    let dense: HashMap<u32, u32> = distinct
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index as u32))
        .collect();

    let remapped = turns
        .into_iter()
        .map(|t| SpeakerTurn {
            speaker: dense[&t.speaker],
            ..t
        })
        .collect();

    (remapped, distinct.len())
}

/// Every job that is currently diarizing, and the handle that kills it.
///
/// The token has to be reachable from OUTSIDE the run for Cancel to mean
/// anything: `diarize()` blocks on a CPU-bound ONNX child with a 30-minute
/// backstop timeout, so a cancelled run with no kill handle would leave that
/// child pinning a core for up to half an hour while the app sat there looking
/// idle. Abandoning it is not enough, the way abandoning an ffmpeg is.
pub type DiarizationRegistry = Mutex<HashMap<String, CancelToken>>;

/// Removes the job's token on every `return` from `diarize_job`, including the
/// early ones. A leaked entry would make the next diarization of that same job
/// fail as "already running", forever.
///
/// NOT on a panic, and it is worth being exact about that rather than claiming a
/// guarantee this build does not have: the release profile sets
/// `panic = "abort"`, so an unwinding-drop never happens -- the process dies with
/// the map in it. That is survivable precisely because there is no surviving
/// process to see the stale entry; it is the ordinary returns that matter, and
/// those are covered.
struct Registered<'a> {
    registry: &'a DiarizationRegistry,
    job_id: String,
}

impl Drop for Registered<'_> {
    fn drop(&mut self) {
        lock_registry(self.registry).remove(&self.job_id);
    }
}

/// A poisoned registry mutex means a thread panicked while holding it. The map
/// itself is fine, and refusing to hand out kill handles because of an unrelated
/// panic would be strictly worse than using it.
fn lock_registry(registry: &DiarizationRegistry) -> std::sync::MutexGuard<'_, HashMap<String, CancelToken>> {
    registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Claim the job for a diarization run, refusing a second concurrent one.
///
/// A vacant-only insert, not an overwrite. `CancelToken`'s contract is ONE TOKEN
/// PER RUN -- a shared one makes `adopt()` refuse the second child -- and an
/// overwrite would also strand the first run's token where no cancel could ever
/// reach it. So the second caller is told no, which is a REQUEST error (the UI
/// asked twice), not a degradation.
fn register<'a>(
    registry: &'a DiarizationRegistry,
    job_id: &str,
    cancel: CancelToken,
) -> Result<Registered<'a>, String> {
    let mut jobs = lock_registry(registry);
    if jobs.contains_key(job_id) {
        return Err("Diarization is already running for this job".to_string());
    }
    jobs.insert(job_id.to_string(), cancel);
    drop(jobs);

    Ok(Registered {
        registry,
        job_id: job_id.to_string(),
    })
}

/// The kill handle for a running diarization, if there is one.
///
/// The clone is taken and the registry lock released immediately: `cancel()`
/// kills and REAPS the child, which blocks, and the run's own `Drop` needs this
/// same lock to deregister. Holding it across the kill would have the canceller
/// and the cancellee waiting on each other.
fn registered_token(registry: &DiarizationRegistry, job_id: &str) -> Option<CancelToken> {
    lock_registry(registry).get(job_id).cloned()
}

/// Kill the sidecar for `job_id`, if one is running. Returns whether there was.
///
/// A no-op for a job that is not diarizing -- Cancel is pressed far more often
/// than diarization runs, and "you cancelled nothing" is not an error.
///
/// Synchronous, and therefore BLOCKING: see [`cancel_diarization`], which is what
/// the IPC calls and which keeps this off the async workers.
fn cancel_registered(registry: &DiarizationRegistry, job_id: &str) -> bool {
    match registered_token(registry, job_id) {
        Some(token) => {
            token.cancel();
            true
        }
        None => false,
    }
}

/// The requested speaker count, checked while the UI is still on screen.
///
/// It crosses a process boundary as a string and stops being checkable there, so
/// it is rejected here instead -- and rejected rather than cast: the engine takes
/// an `i32`, and `u32::MAX as i32` is `-1`, which the clusterer reads as
/// "auto-detect". A nonsense request would otherwise come back looking like a
/// perfectly good answer.
///
/// `None` and `Some(0)` both mean auto-detect, which is a real choice.
fn diarize_options(num_speakers: Option<u32>) -> Result<DiarizeOptions, String> {
    let opts = DiarizeOptions {
        num_speakers,
        ..Default::default()
    };
    opts.validate().map_err(|e| e.to_string())?;
    Ok(opts)
}

/// Run the engine and turn EVERY failure into a visible degradation.
///
/// This is the fail-soft boundary the whole task is about. Nothing below it can
/// produce an `Err` that reaches the frontend, because there is no failure of the
/// engine that should cost the user their transcript.
///
/// `diarize_in_background` (and not `Diarizer::diarize`) because `diarize()` is
/// synchronous and runs for minutes: awaiting it on a tokio worker would park
/// that worker for the whole job, and the app would stop answering IPC --
/// including the cancel command, which is the entire point of holding a token.
async fn run_diarization(
    diarizer: Arc<dyn Diarizer>,
    wav_path: PathBuf,
    opts: DiarizeOptions,
    cancel: CancelToken,
) -> DiarizationOutcome {
    match diarize_in_background(diarizer, wav_path, opts, cancel.clone()).await {
        Ok(turns) => {
            // An empty list is a SUCCESS (silence has no speakers), so it goes
            // down this arm, with speaker_count 0 and no division by it.
            let (turns, speaker_count) = densify(turns);
            DiarizationOutcome::Succeeded {
                turns,
                speaker_count,
            }
        }
        // ASK THE TOKEN, NEVER THE MESSAGE.
        //
        // This used to be `e.to_string().contains(CANCELLED)`, and that is a
        // forgeable test: the sidecar's own stderr is interpolated verbatim into
        // the non-zero-exit error, so a child that printed the words "diarization
        // was cancelled" on its way to dying would have been reported to the user
        // as *their own cancellation* -- a real engine failure, silently relabelled
        // as a thing the user did, and never shown as a degradation. The token is
        // the only authority on whether a cancel was actually requested, and it
        // cannot be spoofed by a subprocess.
        //
        // Every `CANCELLED` the engine returns is raised under a set flag
        // (`SidecarDiarizer` checks `is_cancelled()` before spawning, `adopt()`
        // refuses under the same lock `cancel()` takes, and an emptied child slot
        // means `cancel()` took it), so this is exactly as sensitive as the
        // substring was -- and no longer credulous.
        Err(_) if cancel.is_cancelled() => DiarizationOutcome::Cancelled,
        Err(e) => {
            let reason = format!("{e:#}");
            eprintln!("diarization failed, continuing without speaker labels: {reason}");
            DiarizationOutcome::Degraded { reason }
        }
    }
}

/// Speaker turns for a job's audio.
///
/// `num_speakers` is a HINT, not a guarantee: asking for 4 can return 3, and
/// leaving it unset (auto-detect) can over- or under-segment. It is still the
/// better mode whenever the user actually knows the count.
///
/// The returned `Ok` may be a [`DiarizationOutcome::Degraded`] -- the engine
/// broke and there are no speaker labels. That is not an error and must not fail
/// the transcript. `Err` means the REQUEST was wrong (unknown job, no audio, an
/// impossible speaker count), which is a bug on the calling side.
#[tauri::command]
pub async fn diarize_job(
    job_id: String,
    num_speakers: Option<u32>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<DiarizationOutcome, String> {
    // REGISTER FIRST. Nothing may precede this that can block or touch the disk.
    //
    // The token used to be registered last, after the store lookup and three
    // `is_file()` stats -- and that opened a window in which the job was ALREADY
    // ACCEPTED but NOT YET CANCELLABLE. A `cancel_diarization` landing in it found
    // no token, returned `false`, and was gone: the cancel is fire-and-forget on
    // the frontend and is never retried, so the run it failed to catch went on to
    // spawn an ONNX child that nothing would kill until the 30-minute backstop.
    // The user pressed Cancel, the UI cleared, and a core burned for half an hour.
    //
    // Registering before the checks closes it: from the first line of this command
    // there is a token, so a cancel either finds it (and the run below sees a
    // cancelled token before it spawns anything) or genuinely arrived before the
    // command did. The guard's `Drop` removes the entry on every return path
    // below, including the `?`s -- so a job that turns out not to exist does not
    // leave a token behind.
    let cancel = CancelToken::new();
    let _registered = register(&state.diarizations, &job_id, cancel.clone())?;

    // The real errors, all of them checked BEFORE anything is spawned.
    let wav_path = prepared_audio_for_job(&job_id, &state, &app)?;
    let opts = diarize_options(num_speakers)?;

    // Cap concurrent ONNX sidecars at `MAX_CONCURRENT_DIARIZATIONS`. Normal use
    // never contends this -- the frontend's `isBusy` gate allows only one run at
    // a time -- EXCEPT across a Cancel: `cancel()` clears that gate and fires
    // `cancel_diarization` as fire-and-forget before the old run's sidecar is
    // confirmed dead, so a Cancel immediately followed by a new run can have two
    // `diarize_job` calls alive together, each about to spawn its own
    // core-pegging child. The permit is held for the rest of this function, so
    // the second call queues behind the first rather than doubling up on cores.
    //
    // Taken AFTER registration (so a queued run is still cancellable) but BEFORE
    // anything is spawned, matching the "real errors first, degrade after"
    // shape below: `acquire_owned` only errors if the semaphore is closed, which
    // this app never does, but a queued run can still be cancelled while it
    // waits -- checked the moment the permit is granted.
    let _diarize_permit = match state.diarization_semaphore.clone().acquire_owned().await {
        Ok(permit) => permit,
        Err(_) if cancel.is_cancelled() => return Ok(DiarizationOutcome::Cancelled),
        Err(e) => {
            return Ok(DiarizationOutcome::Degraded {
                reason: format!("could not queue diarization: {e}"),
            })
        }
    };
    if cancel.is_cancelled() {
        // Cancelled while queued behind another diarization -- report it as the
        // user's own cancellation, not a degradation, same as every other
        // cancellation check in this function.
        return Ok(DiarizationOutcome::Cancelled);
    }

    // Everything from here on degrades rather than fails. A build with no models
    // (or no sidecar) is a build without speaker labels, not a broken app. That
    // build is reachable: `src-tauri/build.rs` deliberately lets a checkout with
    // no models compile, so this is not dead code.
    let assets = match diarization_assets(&app) {
        Ok(assets) => assets,
        // ...unless the user cancelled while we were stat'ing. Reporting "speaker
        // detection is not installed" for a run they stopped themselves is the
        // same lie `Cancelled` exists to prevent, just on a different path.
        Err(_) if cancel.is_cancelled() => return Ok(DiarizationOutcome::Cancelled),
        Err(reason) => return Ok(DiarizationOutcome::Degraded { reason }),
    };

    let diarizer: Arc<dyn Diarizer> = Arc::new(SidecarDiarizer::new(
        assets.exe,
        assets.segmentation_model,
        assets.embedding_model,
    ));

    Ok(run_diarization(diarizer, wav_path, opts, cancel).await)
}

/// Kill the diarization sidecar for a job, if one is running.
///
/// Wired to the Cancel button. Cancelling a transcription used to be purely a
/// frontend affair, and that was tolerable while the only abandoned child was an
/// ffmpeg that simply finishes. It is not tolerable for diarization: the child is
/// a CPU-bound ONNX process with a 30-minute backstop timeout.
///
/// Never an error: cancelling a job that is not diarizing is a no-op, and the UI
/// has no way to know whether one was in flight.
///
/// The kill runs on a BLOCKING pool, not on this async worker. `CancelToken::cancel`
/// is `kill()` + `wait()` -- two synchronous syscalls, the second of which sleeps
/// until the child is actually reaped, while holding the child mutex that the
/// engine's 20 ms poll loop is contending for. SIGKILL is uncatchable, so today
/// that wait is short and parking a worker on it is merely rude. It stops being
/// merely rude the moment the reap is slow for a reason nobody predicted (a child
/// wedged in an uninterruptible syscall, a signal the child does catch), and the
/// worker it parks is one of the handful serving ALL IPC -- including the next
/// Cancel. Blocking work belongs on the blocking pool; this is blocking work.
#[tauri::command]
pub async fn cancel_diarization(job_id: String, state: State<'_, AppState>) -> Result<bool, String> {
    // The lookup itself is a `Mutex` that is never held across a blocking call
    // (see `registered_token`), so it is safe to do here; only the kill moves.
    let Some(token) = registered_token(&state.diarizations, &job_id) else {
        return Ok(false);
    };

    tokio::task::spawn_blocking(move || token.cancel())
        .await
        .map_err(|e| format!("the diarization kill task did not complete: {e}"))?;

    Ok(true)
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

// ---------------------------------------------------------------------------
// Speaker display names.
//
// The segments keep their opaque `SPEAKER_00` keys FOREVER. A name is a row in
// `transcript_speakers`, joined on at render time. So renaming is a metadata
// write -- it never re-transcribes, never re-diarizes, and never rewrites a
// single segment. That is not an optimization; it is the only design in which
// renaming a speaker is possible at all once the audio has been cleaned up (see
// `PREPARED_AUDIO_TTL_HOURS`), which for any transcript older than a day it has.
//
// Names key on the SOURCE, because diarization is a property of the AUDIO: a
// name given to a speaker today must still be attached when the same file is
// re-transcribed under a different model preset -- which lands on a different
// transcript row but the same source. Both commands take a JOB id (the only
// handle the frontend has) and resolve it with `source_id_for_job`.
//
// Speaker ids are stable only within ONE diarization run, so re-diarizing the
// same audio drops the source's names -- see `Store::persist_transcript`. That
// clearing is the reason a name can never silently move onto the wrong voice.
// ---------------------------------------------------------------------------

/// A display name is a LABEL, not a document. SRT and VTT are positional,
/// blank-line-delimited formats -- an embedded newline in a name lets it
/// forge `\n\n<fake index>\n<fake timestamp> --> ...\n<fake text>` and inject
/// an entirely fabricated cue block into an exported file. Strip out embedded
/// newlines and every other C0/DEL control character so that junk can never
/// reach the database in the first place. (`srtGenerator.ts` strips/escapes
/// again at the export boundary -- defense in depth, because a serializer
/// must be safe on any input regardless of what validation ran upstream, and
/// because this store also predates this check for any row written before
/// it shipped.)
fn strip_control_chars(input: &str) -> String {
    input.chars().filter(|c| !c.is_control()).collect()
}

/// A label is short. 100 characters is generous for a person's name (or a
/// short role like "Interviewer") while still ruling out someone pasting a
/// paragraph into the rename box.
const MAX_SPEAKER_NAME_LEN: usize = 100;

/// A name, trimmed and sanitized, or a REFUSAL.
///
/// A blank name is rejected rather than stored because it would render as an
/// EMPTY speaker label -- `": Hello there."` in an SRT, a `<v >` in a WebVTT --
/// which is indistinguishable from a cue with NO speaker at all. Producing a
/// transcript in which "diarization found nobody here" and "the user pressed
/// enter on an empty box" look the same is the exact class of silent lie the rest
/// of this feature is built to avoid, so it is refused at the door.
///
/// Trimming, and not merely checking: a name of `"  "` is blank, and a name of
/// `" Alice "` is Alice. Split out from the command so it can be tested without a
/// Tauri `State`, which a unit test cannot construct.
///
/// Embedded newlines and other control characters are stripped (not merely
/// rejected) -- see `strip_control_chars` -- and the result is re-trimmed and
/// re-checked for blankness, since a name of nothing but control characters
/// (e.g. `"\t\n"`) strips down to empty. A name longer than
/// `MAX_SPEAKER_NAME_LEN` characters (after stripping) is refused outright.
fn validated_speaker_name<'a>(
    speaker_key: &'a str,
    display_name: &'a str,
) -> Result<(&'a str, String), String> {
    let speaker_key = speaker_key.trim();
    let display_name = strip_control_chars(display_name.trim());
    let display_name = display_name.trim().to_string();

    if speaker_key.is_empty() {
        return Err("A speaker key is required".to_string());
    }
    if display_name.is_empty() {
        return Err("A speaker name cannot be blank".to_string());
    }
    if display_name.chars().count() > MAX_SPEAKER_NAME_LEN {
        return Err(format!(
            "A speaker name cannot be longer than {MAX_SPEAKER_NAME_LEN} characters"
        ));
    }
    Ok((speaker_key, display_name))
}

/// Name a speaker: `SPEAKER_00` -> `"Alice"`. Renaming twice overwrites.
///
/// `Err` means the REQUEST was wrong: an unknown job, a job whose source was
/// deleted (nothing to attach a name to), a blank key, or a blank name.
#[tauri::command]
pub async fn set_speaker_name(
    job_id: String,
    speaker_key: String,
    display_name: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let (speaker_key, display_name) = validated_speaker_name(&speaker_key, &display_name)?;

    let source_id = state
        .store
        .source_id_for_job(&job_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "This job has no source to name speakers for".to_string())?;

    state
        .store
        .set_speaker_name(source_id, speaker_key, &display_name)
        .map_err(|e| e.to_string())
}

/// Every name the user has given this job's speakers, keyed by the opaque label
/// the segments carry: `{"SPEAKER_00": "Alice"}`.
///
/// A job whose source is gone, or a source with no names yet, answers with an
/// EMPTY MAP rather than an error -- the UI asks this on every load, and "nobody
/// has renamed anyone" is the normal answer, not a failure. Speakers with no name
/// simply have no entry; the caller falls back to the key.
#[tauri::command]
pub async fn get_speaker_names(
    job_id: String,
    state: State<'_, AppState>,
) -> Result<HashMap<String, String>, String> {
    let Some(source_id) = state
        .store
        .source_id_for_job(&job_id)
        .map_err(|e| e.to_string())?
    else {
        return Ok(HashMap::new());
    };

    state
        .store
        .get_speaker_names(source_id)
        .map_err(|e| e.to_string())
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
    use crate::diarize::mock::MockDiarizer;
    use crate::diarize::CANCELLED;

    fn turn(start: f32, end: f32, speaker: u32) -> SpeakerTurn {
        SpeakerTurn {
            start,
            end,
            speaker,
        }
    }

    async fn diarize_with(diarizer: MockDiarizer, cancel: CancelToken) -> DiarizationOutcome {
        run_diarization(
            Arc::new(diarizer),
            PathBuf::from("/any.wav"),
            DiarizeOptions::default(),
            cancel,
        )
        .await
    }

    // -- the fail-soft boundary ----------------------------------------------

    /// **The point of the whole task.**
    ///
    /// The engine died. The transcript is fine. `run_diarization` must not hand
    /// back an `Err` that could fail the job -- there is no failure of the
    /// diarizer that is worth a user's transcript.
    #[tokio::test]
    async fn a_broken_engine_degrades_and_never_fails_the_job() {
        let outcome = diarize_with(
            MockDiarizer::failing("the diarization sidecar was killed by signal 6 (SIGABRT)"),
            CancelToken::new(),
        )
        .await;

        match outcome {
            DiarizationOutcome::Degraded { reason } => {
                assert!(reason.contains("SIGABRT"), "the reason must survive: {reason}");
            }
            other => panic!("a dead sidecar must degrade, got {other:?}"),
        }
    }

    /// `Ok(vec![])` is a legitimate SUCCESS: silence, or a zero-length file, has
    /// no speaker turns. Treating empty as failure would report "speaker
    /// detection failed" for a perfectly good silent recording -- and dividing by
    /// the turn count would be worse.
    #[tokio::test]
    async fn silence_is_a_success_with_zero_speakers_not_a_failure() {
        let outcome = diarize_with(MockDiarizer::returning(vec![]), CancelToken::new()).await;

        assert_eq!(
            outcome,
            DiarizationOutcome::Succeeded {
                turns: vec![],
                speaker_count: 0,
            },
            "an empty turn list is a success, not a degradation"
        );
    }

    /// **The distinction the type exists for.**
    ///
    /// A transcript with no speaker labels because diarization BROKE must not be
    /// indistinguishable from one where diarization RAN and heard one person. A
    /// bare `Vec<SpeakerTurn>` cannot tell those apart; the tag can, and this
    /// test pins the fact that the frontend can see the difference -- on the wire,
    /// not merely in Rust.
    #[tokio::test]
    async fn a_broken_engine_is_distinguishable_from_one_speaker_and_from_silence() {
        let broken = diarize_with(MockDiarizer::failing("model missing"), CancelToken::new()).await;
        let one_speaker = diarize_with(
            MockDiarizer::returning(vec![turn(0.0, 9.0, 0)]),
            CancelToken::new(),
        )
        .await;
        let silence = diarize_with(MockDiarizer::returning(vec![]), CancelToken::new()).await;

        assert_ne!(broken, one_speaker);
        assert_ne!(broken, silence);
        assert_ne!(one_speaker, silence);

        let json = |o: &DiarizationOutcome| serde_json::to_value(o).unwrap();
        assert_eq!(json(&broken)["status"], "degraded");
        assert_eq!(json(&one_speaker)["status"], "succeeded");
        assert_eq!(json(&one_speaker)["speaker_count"], 1);
        assert_eq!(json(&silence)["status"], "succeeded");
        assert_eq!(json(&silence)["speaker_count"], 0);

        // A degradation carries no turns AT ALL, rather than an empty list that
        // could be mistaken for the silent case by a caller reading only `turns`.
        assert!(json(&broken).get("turns").is_none());
    }

    /// A cancelled run is not a broken one. Telling the user "speaker detection
    /// failed" for something they stopped themselves is a lie in the other
    /// direction.
    #[tokio::test]
    async fn a_cancelled_run_is_reported_as_cancelled_not_as_a_failure() {
        let cancel = CancelToken::new();
        cancel.cancel();

        let outcome = diarize_with(MockDiarizer::returning(vec![turn(0.0, 1.0, 0)]), cancel).await;
        assert_eq!(outcome, DiarizationOutcome::Cancelled);
    }

    /// **A subprocess cannot talk its way out of a degradation.**
    ///
    /// "Cancelled" used to be decided by `e.to_string().contains(CANCELLED)`, and
    /// the sidecar's own stderr is interpolated verbatim into the non-zero-exit
    /// error. So a broken engine that happened to print the words "diarization was
    /// cancelled" on its way down was reported to the user as THEIR OWN
    /// CANCELLATION: a genuine failure, relabelled as a deliberate act, and never
    /// shown as the degradation it was. Precisely the class of silent lie the
    /// three-outcome type exists to prevent, arriving through the error string.
    ///
    /// The token is now the only authority, and NOBODY CANCELLED THIS ONE.
    #[tokio::test]
    async fn an_engine_that_prints_the_word_cancelled_is_still_a_degradation() {
        let never_cancelled = CancelToken::new();

        let outcome = diarize_with(
            // The sidecar's stderr, verbatim, as the non-zero-exit path formats it.
            MockDiarizer::failing(&format!(
                "the diarization sidecar exited with status 1: {CANCELLED} (onnxruntime: bad alloc)"
            )),
            never_cancelled.clone(),
        )
        .await;

        assert!(
            !never_cancelled.is_cancelled(),
            "the premise of this test: nobody asked to cancel"
        );
        match outcome {
            DiarizationOutcome::Degraded { reason } => assert!(
                reason.contains("bad alloc"),
                "the real reason must survive: {reason}"
            ),
            other => panic!(
                "an engine failure that merely MENTIONS cancellation must degrade, not be \
                 reported as the user's own cancellation, got {other:?}"
            ),
        }
    }

    // -- the sparse-id hazard, closed at the boundary -------------------------

    /// The engine's ids are sparse: the two-speaker fixture really does come back
    /// as `{0, 3}` (pinned against the real model by
    /// `diarize::tests::speaker_ids_are_sparse_not_contiguous`). Anything that
    /// indexed a palette by that id, or rendered `Speaker {id + 1}`, would show
    /// "Speaker 4" for a two-speaker file. So the ids that leave this module are
    /// dense, and the hazard cannot reach the frontend at all.
    #[test]
    fn sparse_speaker_ids_are_densified_at_the_boundary() {
        let (turns, count) = densify(vec![
            turn(0.0, 7.0, 0),
            turn(7.0, 12.0, 3),
            turn(12.0, 18.0, 0),
            turn(18.0, 24.0, 3),
        ]);

        assert_eq!(count, 2);
        assert_eq!(
            turns.iter().map(|t| t.speaker).collect::<Vec<_>>(),
            vec![0, 1, 0, 1],
            "ids must be remapped to 0..n, preserving which turns share a speaker"
        );
        // The invariant every caller is now allowed to rely on: `speaker` is a
        // valid index into a list of `speaker_count` things.
        assert!(turns.iter().all(|t| (t.speaker as usize) < count));

        // The times are not touched.
        assert_eq!(turns[1].start, 7.0);
        assert_eq!(turns[1].end, 12.0);
    }

    #[test]
    fn densifying_is_stable_regardless_of_the_order_the_turns_arrive_in() {
        // Turns are not promised to be sorted by time, so the remap must not
        // depend on it: the same speaker must get the same dense id either way.
        let (a, _) = densify(vec![turn(9.0, 10.0, 7), turn(0.0, 1.0, 2)]);
        let (b, _) = densify(vec![turn(0.0, 1.0, 2), turn(9.0, 10.0, 7)]);
        assert_eq!(a[0].speaker, b[1].speaker);
        assert_eq!(a[1].speaker, b[0].speaker);
    }

    #[test]
    fn densifying_nothing_is_zero_speakers_not_a_panic() {
        let (turns, count) = densify(vec![]);
        assert!(turns.is_empty());
        assert_eq!(count, 0, "silence has zero speakers -- and nobody divides by it");
    }

    // -- the request errors, which ARE errors ---------------------------------

    /// An impossible speaker count is a bad REQUEST, caught before anything is
    /// spawned -- not a degradation. `u32::MAX as i32` is `-1`, which the
    /// clusterer reads as "auto-detect", so casting it would turn nonsense into a
    /// confident, plausible, unrequested answer.
    #[test]
    fn an_impossible_speaker_count_is_rejected_up_front() {
        let err = diarize_options(Some(u32::MAX)).expect_err("must be rejected");
        assert!(err.contains("num_speakers"), "{err}");

        // And the modes that are real choices survive.
        assert_eq!(diarize_options(None).unwrap().num_speakers, None);
        assert_eq!(diarize_options(Some(0)).unwrap().num_speakers, Some(0));
        assert_eq!(diarize_options(Some(2)).unwrap().num_speakers, Some(2));
    }

    // -- cancel really reaches the child --------------------------------------

    /// Cancel is only worth anything if it can reach a run that has already
    /// started. The registry is the thing that makes that possible, so it is
    /// tested for exactly that: a token handed to a run is the token `cancel`
    /// fires.
    #[test]
    fn a_registered_run_can_be_cancelled_from_outside_it() {
        let registry = DiarizationRegistry::default();
        let cancel = CancelToken::new();

        let registered = register(&registry, "job-1", cancel.clone()).expect("a free job");
        assert!(!cancel.is_cancelled());

        assert!(cancel_registered(&registry, "job-1"));
        assert!(
            cancel.is_cancelled(),
            "cancelling through the registry must fire the token the run is holding"
        );

        // Cancelling a job that is not diarizing is a no-op, not an error: the
        // Cancel button is pressed far more often than diarization runs.
        assert!(!cancel_registered(&registry, "job-2"));

        drop(registered);
        assert!(
            !cancel_registered(&registry, "job-1"),
            "a finished run must not leave its token behind"
        );
    }

    /// **A cancel that lands before the run starts must still kill the run.**
    ///
    /// `diarize_job` registers its token on its FIRST line, before the store
    /// lookup and the three `is_file()` stats. It used to register last, and a
    /// `cancel_diarization` arriving in that window found nothing, returned
    /// `false`, and was gone -- the frontend fires it and never retries -- leaving
    /// a run that had been cancelled by the user but was unkillable by that
    /// cancel, right up to the 30-minute backstop.
    ///
    /// This walks the command's real sequence at its real seam: register, THEN
    /// the racing cancel, THEN the run. The run must never reach the engine.
    #[tokio::test]
    async fn a_cancel_racing_the_start_of_a_run_still_kills_it() {
        let registry = DiarizationRegistry::default();
        let cancel = CancelToken::new();
        let _registered = register(&registry, "job-1", cancel.clone()).expect("a free job");

        // The cancel lands here: after the job is claimed, before the engine is
        // resolved or spawned. It must FIND the token.
        assert!(
            cancel_registered(&registry, "job-1"),
            "a cancel arriving before the run has spawned anything must still find its token"
        );

        // MockDiarizer, like SidecarDiarizer, refuses to run on a cancelled token.
        let outcome = diarize_with(MockDiarizer::returning(vec![turn(0.0, 1.0, 0)]), cancel).await;
        assert_eq!(
            outcome,
            DiarizationOutcome::Cancelled,
            "the run must observe the cancel that landed before it started"
        );
    }

    #[test]
    fn a_second_concurrent_diarization_of_the_same_job_is_refused() {
        // Two runs sharing one token is exactly the misuse `CancelToken` warns
        // about (the second child gets killed by `adopt`), and overwriting the
        // entry would strand the first run's token where no cancel could reach it.
        let registry = DiarizationRegistry::default();
        let first = register(&registry, "job-1", CancelToken::new()).expect("a free job");

        let err = match register(&registry, "job-1", CancelToken::new()) {
            Err(e) => e,
            Ok(_) => panic!("a second concurrent run must be refused"),
        };
        assert!(err.contains("already running"), "{err}");

        // ...and refusing it must not have unregistered the first one.
        assert!(cancel_registered(&registry, "job-1"));

        drop(first);
        register(&registry, "job-1", CancelToken::new()).expect("the job is free again");
    }

    /// **The diarization semaphore actually serializes.**
    ///
    /// `diarize_job` is reachable with two DIFFERENT job ids overlapping: Cancel
    /// clears the frontend's `isBusy` gate and fires `cancel_diarization`
    /// fire-and-forget before the cancelled run's sidecar is confirmed dead, so a
    /// Cancel immediately followed by a new run can have two `diarize_job` calls
    /// alive together (the registry's duplicate check above only refuses a
    /// SECOND run of the SAME job, which does not apply here). Each would spawn
    /// its own core-pegging ONNX child without a cap.
    ///
    /// This does not stand up a full `AppState`/`State`/`AppHandle` -- there is
    /// no tauri test harness in this crate, and the other `diarize_job` tests
    /// exercise its logic at the same seam (`register`, `run_diarization`, ...)
    /// rather than through the command itself. What is tested here is the
    /// mechanism `diarize_job` relies on: a `Semaphore::new(1)` (the same
    /// capacity as `paths::MAX_CONCURRENT_DIARIZATIONS`) genuinely makes a
    /// second acquire wait for the first permit to be released, rather than
    /// handing out two permits at once.
    #[tokio::test]
    async fn the_diarization_semaphore_serializes_two_overlapping_runs() {
        assert_eq!(
            crate::paths::MAX_CONCURRENT_DIARIZATIONS, 1,
            "this test's capacity-1 premise must match the real cap"
        );
        let semaphore = Arc::new(tokio::sync::Semaphore::new(
            crate::paths::MAX_CONCURRENT_DIARIZATIONS,
        ));

        // "job-1" takes the only permit, as `diarize_job` would.
        let first_permit = semaphore.clone().acquire_owned().await.expect("uncontended");

        // "job-2" (Cancel, then a new run) queues behind it instead of getting a
        // second permit.
        let contender = semaphore.clone();
        let mut second_acquire = tokio::spawn(async move { contender.acquire_owned().await });

        // Give the spawned task every chance to (wrongly) complete before the
        // first permit is released. `now_or_never`-style polling would be
        // cleaner, but this crate's other async tests do not depend on
        // `futures`, so a bounded `tokio::time::timeout` says the same thing
        // without a new dependency: the second acquire must NOT resolve yet.
        let raced = tokio::time::timeout(std::time::Duration::from_millis(50), &mut second_acquire)
            .await;
        assert!(
            raced.is_err(),
            "a second acquire must block while the cap's only permit is held"
        );

        // Releasing the first permit is what lets the second one through.
        drop(first_permit);
        let second_permit = second_acquire
            .await
            .expect("task did not panic")
            .expect("semaphore was not closed");
        drop(second_permit);
    }

    // -- the bundle ------------------------------------------------------------

    /// `tauri.conf.json` and this file must agree about where the models land, or
    /// the app ships a bundle whose models it cannot find -- a failure that
    /// `cargo run` cannot reproduce, because the dev tree happens to work.
    ///
    /// Read at COMPILE time from the config itself, so a rename on either side is
    /// a red test rather than a silent "speaker detection is unavailable" in a
    /// signed build.
    #[test]
    fn the_bundled_model_paths_are_the_ones_tauri_will_actually_produce() {
        const CONFIG: &str = include_str!("../tauri.conf.json");
        let config: serde_json::Value = serde_json::from_str(CONFIG).expect("valid tauri.conf.json");

        let resources = config["bundle"]["resources"]
            .as_object()
            .expect("bundle.resources must be a MAP, not a list: a list entry maps `../models/..` \
                     to `_up_/models/..` under $RESOURCE (tauri_utils::resources::resource_relpath \
                     rewrites `..`), and the paths in this file would not exist in the bundle");

        // Resolve each model through the config the way tauri_utils does, and
        // demand it come out at the exact path the app opens.
        //
        // Two mapping styles reach the same place, and the test accepts either, so
        // that the CHOICE between them stays free (it is not free: see build.rs --
        // the directory form is what lets a checkout with no models still build):
        //
        //   - a file entry, whose target IS the resource path; and
        //   - a directory entry, which walks the tree and preserves the relative
        //     path underneath the target.
        //
        // Either way the source must be `../<the same path>`: the repo-root layout
        // that scripts/fetch-sidecars.sh writes. That is the whole invariant --
        // where the script puts a model, where the bundle puts it, and where the
        // code opens it are one path.
        for model in [SEGMENTATION_MODEL, EMBEDDING_MODEL] {
            let source = resources
                .iter()
                .find_map(|(source, target)| {
                    let target = target.as_str()?;
                    if target == model {
                        Some(source.clone())
                    } else {
                        // A directory entry: `$RESOURCE/<target>/<rest>`.
                        let rest = model.strip_prefix(&format!("{target}/"))?;
                        Some(format!("{source}/{rest}"))
                    }
                })
                .unwrap_or_else(|| {
                    panic!(
                        "no bundle.resources entry puts {model:?} under $RESOURCE, so a packaged \
                         build would find no diarization models. Entries: {resources:?}"
                    )
                });

            assert_eq!(
                source,
                format!("../{model}"),
                "tauri.conf.json bundles {model:?} from {source:?}, but fetch-sidecars.sh writes \
                 it to ../{model}"
            );
        }

        // And build.rs -- which has to know the same two paths, to tell a
        // no-models build to warn instead of exploding -- must not drift from them.
        const BUILD_RS: &str = include_str!("../build.rs");
        for model in [SEGMENTATION_MODEL, EMBEDDING_MODEL] {
            let relative = model
                .strip_prefix("models/diarization/")
                .expect("the models live under models/diarization/");
            assert!(
                BUILD_RS.contains(&format!("\"{relative}\"")),
                "build.rs does not know about {relative:?}, so a build missing that model would \
                 not warn about it"
            );
        }

        // And the sidecar is staged next to the binary, stripped of its triple.
        let bins = config["bundle"]["externalBin"]
            .as_array()
            .expect("externalBin");
        assert!(
            bins.iter().any(|b| b == &format!("binaries/{SIDECAR_EXE}")),
            "the diarization sidecar must be an externalBin so Tauri stages and signs it"
        );
    }

    /// The resolution is exercised against a BUNDLE-shaped tree, not the dev tree.
    ///
    /// The dev tree would resolve for the wrong reason (a repo-relative path works
    /// there and nowhere else), which is precisely how the models came to be
    /// unbundled in the first place. So: a fake `.app`, with the sidecar in
    /// `Contents/MacOS` and the models in `Contents/Resources`, exactly where
    /// Tauri puts them.
    #[test]
    fn the_engine_resolves_inside_a_bundled_app_layout() {
        let bundle = tempfile::tempdir().unwrap();
        let macos = bundle.path().join("Remedy.app/Contents/MacOS");
        let resources = bundle.path().join("Remedy.app/Contents/Resources");
        std::fs::create_dir_all(&macos).unwrap();

        let seg = resources.join(SEGMENTATION_MODEL);
        let emb = resources.join(EMBEDDING_MODEL);
        std::fs::create_dir_all(seg.parent().unwrap()).unwrap();
        std::fs::create_dir_all(emb.parent().unwrap()).unwrap();

        let exe = macos.join(format!("{SIDECAR_EXE}{}", std::env::consts::EXE_SUFFIX));

        // Nothing is there yet: every missing piece must be NAMED, not shrugged at.
        let err = diarization_assets_in(&resources, &macos).expect_err("nothing is installed");
        assert!(err.contains("sidecar"), "{err}");

        std::fs::write(&exe, b"#!/bin/sh\n").unwrap();
        let err = diarization_assets_in(&resources, &macos).expect_err("no models yet");
        assert!(err.contains("segmentation model"), "{err}");

        std::fs::write(&seg, b"onnx").unwrap();
        let err = diarization_assets_in(&resources, &macos).expect_err("no embedding model yet");
        assert!(err.contains("embedding model"), "{err}");

        std::fs::write(&emb, b"onnx").unwrap();
        let assets = diarization_assets_in(&resources, &macos).expect("a complete bundle");
        assert_eq!(assets.exe, exe);
        assert_eq!(assets.segmentation_model, seg);
        assert_eq!(assets.embedding_model, emb);
    }

    /// A build with no models is a build without speaker labels, not a broken
    /// app: the resolution failure is a degradation with a reason a user can act
    /// on, never an `Err` that could fail a job.
    #[test]
    fn a_bundle_with_no_models_degrades_with_a_reason_that_says_what_to_do() {
        let empty = tempfile::tempdir().unwrap();
        let err = diarization_assets_in(empty.path(), empty.path()).expect_err("nothing installed");
        assert!(err.contains("fetch-sidecars.sh"), "{err}");
        assert!(err.contains("no speaker diarization"), "{err}");

        // This is the string the command turns into a degradation, so it must be
        // usable as one.
        let degraded = DiarizationOutcome::Degraded { reason: err };
        assert_ne!(
            degraded,
            DiarizationOutcome::Succeeded {
                turns: vec![],
                speaker_count: 0
            }
        );
    }

    // -- the real engine, through the command layer ----------------------------

    /// The one test that runs the REAL sidecar against the REAL models, through
    /// the same `run_diarization` seam the command uses -- so the degradation
    /// boundary, the `spawn_blocking` and the densification are all exercised by
    /// the thing that actually ships, not only by a mock.
    ///
    /// `#[ignore]` because it needs the 34 MB of models:
    ///     ./scripts/fetch-sidecars.sh --models-only
    ///     cargo test --workspace -- --ignored
    #[tokio::test]
    #[ignore = "requires the diarization models; run with: cargo test --workspace -- --ignored"]
    async fn the_real_engine_diarizes_the_fixture_through_the_command_layer() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let models = root.join("../models/diarization");

        // The test binary lives in `target/<profile>/deps`; the sidecar and the
        // app's own binary live one level up, which is the layout
        // `diarization_assets` sees in a real run.
        let mut exe_dir = std::env::current_exe().expect("a test binary has a path");
        exe_dir.pop();
        if exe_dir.ends_with("deps") {
            exe_dir.pop();
        }

        // The same two paths the bundle will hold, resolved here from the dev
        // tree: `models/diarization/...` under the resource root.
        let assets = diarization_assets_in(&root.join(".."), &exe_dir)
            .expect("run scripts/fetch-sidecars.sh and scripts/build-diarize-sidecar.sh first");
        assert!(assets.segmentation_model.starts_with(&models));

        let diarizer: Arc<dyn Diarizer> = Arc::new(SidecarDiarizer::new(
            assets.exe,
            assets.segmentation_model,
            assets.embedding_model,
        ));

        let outcome = run_diarization(
            diarizer,
            root.join("tests/fixtures/two_speakers.wav"),
            DiarizeOptions::default(),
            CancelToken::new(),
        )
        .await;

        match outcome {
            DiarizationOutcome::Succeeded {
                turns,
                speaker_count,
            } => {
                println!("--- diarize_job / two_speakers.wav ---");
                for t in &turns {
                    println!("{:6.2}s - {:6.2}s  speaker {}", t.start, t.end, t.speaker);
                }
                assert_eq!(speaker_count, 2, "the fixture has two speakers");

                // The engine's own ids for this file are {0, 3}. What comes out of
                // the command layer must be {0, 1} -- dense, and safe to index by.
                let ids: BTreeSet<u32> = turns.iter().map(|t| t.speaker).collect();
                assert_eq!(
                    ids,
                    BTreeSet::from([0, 1]),
                    "the sparse ids the engine returns must be densified before they \
                     reach the frontend"
                );
                assert!(turns[0].start < 0.5, "the first turn starts at the top of the file");
            }
            other => panic!("the real engine should diarize the fixture, got {other:?}"),
        }
    }

    /// A compile-proof, not a behaviour test.
    ///
    /// `MockDiarizer` used to live in a *private* `#[cfg(test)] mod tests` inside
    /// `diarize.rs`. `commands` is a sibling module, so it could not name it at
    /// all -- `error[E0603]: module tests is private`. The diarization command
    /// tests that need it therefore could not be written. It now lives in
    /// `#[cfg(test)] pub mod diarize::mock`, and this test exists to fail loudly
    /// if it is ever moved back somewhere unreachable.
    ///
    /// Note for whoever writes those command tests: `#[cfg(test)]` items are
    /// invisible to integration tests in `src-tauri/tests/`, which compile as a
    /// separate crate against the public library. Command tests that use this
    /// mock must be **in-crate unit tests**, right here, not files under `tests/`.
    #[test]
    fn the_mock_diarizer_is_reachable_from_the_command_tests() {
        use crate::diarize::mock::MockDiarizer;
        use crate::diarize::{CancelToken, DiarizeOptions, Diarizer, SpeakerTurn, CANCELLED};

        let d: Box<dyn Diarizer> = Box::new(MockDiarizer::returning(vec![SpeakerTurn {
            start: 0.0,
            end: 1.0,
            speaker: 0,
        }]));
        assert_eq!(
            d.diarize(
                Path::new("x.wav"),
                &DiarizeOptions::default(),
                &CancelToken::new()
            )
            .unwrap()
            .len(),
            1
        );

        // The case the commands actually have to survive: diarization degraded.
        let broken: Box<dyn Diarizer> = Box::new(MockDiarizer::failing("sidecar killed by SIGABRT"));
        assert!(broken
            .diarize(
                Path::new("x.wav"),
                &DiarizeOptions::default(),
                &CancelToken::new()
            )
            .is_err());

        // And the mock honours the kill handle, so a command test can exercise the
        // cancel path without spawning anything.
        let cancel = CancelToken::new();
        cancel.cancel();
        let err = d
            .diarize(Path::new("x.wav"), &DiarizeOptions::default(), &cancel)
            .expect_err("a cancelled run is not a success");
        assert!(err.to_string().contains(CANCELLED), "{err}");
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

    // -- speaker names -------------------------------------------------------

    /// A speaker may not be named the empty string.
    ///
    /// Stored, it would render as `": Hello there."` in an SRT and `<v >` in a
    /// WebVTT -- a cue that LOOKS unattributed. "Diarization found nobody" and
    /// "the user pressed enter on an empty box" must not produce the same
    /// transcript.
    #[test]
    fn a_speaker_may_not_be_named_nothing() {
        assert!(validated_speaker_name("SPEAKER_00", "").is_err());
        assert!(validated_speaker_name("SPEAKER_00", "   ").is_err());
        assert!(validated_speaker_name("SPEAKER_00", "\t\n").is_err());
        assert!(validated_speaker_name("", "Alice").is_err());
    }

    /// And a real name is TRIMMED, not merely accepted: `" Alice "` is Alice, and
    /// the whitespace does not reach the exports.
    #[test]
    fn a_speaker_name_is_trimmed() {
        assert_eq!(
            validated_speaker_name("  SPEAKER_00 ", "  Alice B.  "),
            Ok(("SPEAKER_00", "Alice B.".to_string()))
        );
    }

    /// A name with an embedded newline (or other control character) is
    /// stripped, not stored verbatim. Left in, SRT/VTT are positional,
    /// blank-line-delimited formats -- a name of
    /// `"Alice\n\n999\n00:00:00,000 --> 00:00:01,000\nInjected"` would let a
    /// rename FABRICATE an entire extra cue block in every export made after
    /// it. Stripping (not merely rejecting) means an ordinary paste with a
    /// trailing newline from a form field still works instead of bouncing.
    #[test]
    fn a_speaker_name_strips_embedded_newlines_and_control_characters() {
        assert_eq!(
            validated_speaker_name(
                "SPEAKER_00",
                "Alice\n\n999\n00:00:00,000 --> 00:00:01,000\nInjected",
            ),
            Ok((
                "SPEAKER_00",
                "Alice99900:00:00,000 --> 00:00:01,000Injected".to_string()
            ))
        );

        // A name that is NOTHING BUT control characters strips down to blank
        // and is refused, same as any other blank name.
        assert!(validated_speaker_name("SPEAKER_00", "\n\n\t").is_err());
    }

    /// A label is not a document: a name longer than 100 characters is
    /// refused outright rather than silently truncated.
    #[test]
    fn a_speaker_name_over_the_length_cap_is_rejected() {
        let ok_name = "A".repeat(100);
        assert!(validated_speaker_name("SPEAKER_00", &ok_name).is_ok());

        let too_long = "A".repeat(101);
        assert!(validated_speaker_name("SPEAKER_00", &too_long).is_err());
    }
}
