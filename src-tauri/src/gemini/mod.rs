//! The Gemini transcription engine: plan, slice, upload, interact, stitch.
//!
//! The submodules hold the pieces — credentials, chunk planning, the HTTP
//! client, transcript assembly. This file is the only place that knows the
//! order they go in, and the only place that knows a run can be cancelled.

pub mod chunking;
pub mod client;
pub mod credentials;
pub mod transcript;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::anyhow;
use serde::Serialize;
use tokio::sync::oneshot::error::TryRecvError;

use crate::gemini::client::GeminiClient;
use crate::gemini::transcript::{SpeakerTurn, Word};

pub const MODEL_ID: &str = "gemini-3.5-transcribe";
/// FLAC, not WAV: lossless (so no quality risk on a speech model) and roughly
/// half the bytes, which halves upload time on every chunk.
pub const CHUNK_MIME: &str = "audio/flac";
/// How far a chunk boundary may move to land in silence.
pub const SILENCE_SNAP_WINDOW_SECS: f64 = 30.0;
/// Retries for a retryable failure, per chunk.
pub const MAX_ATTEMPTS: u32 = 3;

/// Cancel handles for Gemini runs in flight, keyed by job.
///
/// Mirrors the shape the deleted `DiarizationRegistry` had, but simpler: there
/// is no child process to signal, only a channel the chunk loop selects on.
pub type GeminiRegistry = Arc<Mutex<HashMap<String, tokio::sync::oneshot::Sender<()>>>>;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum Speakers {
    Identified {
        turns: Vec<SpeakerTurn>,
        speaker_count: u32,
    },
    /// Carries WHY. Rendering this as an empty turn list would tell the user
    /// "one speaker" for a run that never attempted diarization.
    Unavailable { reason: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiTranscriptionResult {
    pub text: String,
    pub words: Vec<Word>,
    pub speakers: Speakers,
    pub audio_duration: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiProgress {
    pub phase: &'static str,
    pub chunk_index: usize,
    pub chunk_count: usize,
    pub fraction: f64,
}

pub fn build_request(file_uri: &str, mime: &str, diarize: bool) -> serde_json::Value {
    let mut mode = serde_json::json!({
        "type": "verbatim",
        "timestamp_granularities": ["word"],
    });
    if diarize {
        mode["diarization_mode"] = serde_json::Value::String("speaker".into());
    }

    serde_json::json!({
        "model": MODEL_ID,
        "input": [{ "type": "audio", "uri": file_uri, "mime_type": mime }],
        "generation_config": { "transcription_config": { "mode": mode } },
    })
}

/// Transcribe one already-sliced chunk, with retries for retryable failures.
///
/// The uploaded file is deleted on EVERY exit path — success, failure and
/// cancel. It would expire in 48 hours anyway, but leaving a user's audio in
/// Google's storage longer than the request needs it is not a default this app
/// should ship.
///
/// That guarantee is why `cancel` is raced in HERE rather than around the call
/// to this function: racing the whole future would drop it at whichever await
/// it was parked on and skip the delete entirely. The upload itself is not
/// interrupted for the same reason — a dropped upload can still finalize
/// server-side, and then there is no resource name left to delete it by. A
/// cancel that lands mid-upload takes effect the moment the upload returns,
/// before any interaction is started.
pub async fn transcribe_chunk(
    client: &GeminiClient,
    path: &Path,
    display_name: &str,
    diarize: bool,
    cancel: &mut tokio::sync::oneshot::Receiver<()>,
) -> anyhow::Result<Vec<Word>> {
    // Before the upload, not only in the select below: a cancel that landed
    // while ffmpeg was slicing would otherwise be honoured only after a whole
    // chunk had been sent to Google and immediately deleted again.
    //
    // ANYTHING but `Empty` ends the run, including a dropped sender. Two
    // reasons, and the first is not optional: `try_recv` clears the receiver's
    // inner state on every outcome except `Empty`, and polling a cleared
    // receiver -- which the `select!` below does -- panics with "called after
    // complete". That panic would land between the upload and `delete_file`,
    // leaving the user's audio in Google's storage for the full 48-hour expiry,
    // which is the one thing this function's shape exists to prevent. The
    // second: a dropped sender means whoever registered this run is gone, so
    // nobody is waiting for the transcript anyway.
    if !matches!(cancel.try_recv(), Err(TryRecvError::Empty)) {
        return Err(anyhow!("cancelled"));
    }

    let uploaded = client.upload(path, CHUNK_MIME, display_name).await?;

    let outcome = tokio::select! {
        // `biased` so an already-delivered cancel wins deterministically. An
        // unbiased select may poll the interaction first, which SENDS the
        // request — billing the user for a transcript nobody will read.
        biased;
        _ = &mut *cancel => Err(anyhow!("cancelled")),
        result = interact_with_retries(client, &uploaded.uri, diarize) => result,
    };

    client.delete_file(&uploaded.name).await.ok();
    transcript::words_from_response(&outcome?)
}

/// One chunk's interaction, retried only while the failure is one a later
/// identical request could survive.
async fn interact_with_retries(
    client: &GeminiClient,
    file_uri: &str,
    diarize: bool,
) -> anyhow::Result<serde_json::Value> {
    let mut attempt = 0;
    loop {
        attempt += 1;
        match client
            .interact(build_request(file_uri, CHUNK_MIME, diarize))
            .await
        {
            Ok(body) => return Ok(body),
            Err(e) => {
                let retryable = e
                    .downcast_ref::<client::GeminiError>()
                    .map(|g| g.is_retryable())
                    .unwrap_or(false);
                if !retryable || attempt >= MAX_ATTEMPTS {
                    return Err(e);
                }
                // Exponential: 1s, 2s. Short enough not to strand the user,
                // long enough to clear a burst rate limit.
                tokio::time::sleep(std::time::Duration::from_secs(1 << (attempt - 1))).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `smart` mode is INCOMPATIBLE with timestamps and diarization, and this
    /// app cannot make captions without timestamps. So the mode is always
    /// verbatim -- there is no code path that asks for `smart`.
    #[test]
    fn the_request_always_asks_for_verbatim_word_timestamps() {
        let body = build_request("files/x", "audio/flac", false);
        let mode = &body["generation_config"]["transcription_config"]["mode"];
        assert_eq!(mode["type"], "verbatim");
        assert_eq!(mode["timestamp_granularities"][0], "word");
    }

    #[test]
    fn the_model_and_audio_reference_are_pinned() {
        let body = build_request("files/x", "audio/flac", false);
        assert_eq!(body["model"], "gemini-3.5-transcribe");
        assert_eq!(body["input"][0]["type"], "audio");
        assert_eq!(body["input"][0]["uri"], "files/x");
        assert_eq!(body["input"][0]["mime_type"], "audio/flac");
    }

    /// Diarization is requested ONLY for a single-chunk run. Asking for it on
    /// chunk 2 of 5 would return speaker ids that look authoritative and are
    /// not comparable with chunk 1's.
    #[test]
    fn diarization_is_absent_from_the_request_unless_asked_for() {
        let without = build_request("files/x", "audio/flac", false);
        assert!(without["generation_config"]["transcription_config"]["mode"]
            .get("diarization_mode")
            .is_none());

        let with = build_request("files/x", "audio/flac", true);
        assert_eq!(
            with["generation_config"]["transcription_config"]["mode"]["diarization_mode"],
            "speaker"
        );
    }

    /// A client that would fail loudly if it were ever reached: nothing is
    /// listening on port 1, and the path does not exist. Both cancel tests
    /// below assert on the error, so this is what proves the run stopped
    /// BEFORE the upload rather than at it.
    fn unreachable_client() -> GeminiClient {
        GeminiClient::with_base_url("KEY".into(), "http://127.0.0.1:1".into())
    }

    /// A DROPPED cancel sender must read as a cancel, not as "no cancel yet".
    ///
    /// `try_recv` clears the receiver's inner state on `Closed` exactly as it
    /// does on a real send, and `select!` polling a cleared receiver panics
    /// with "called after complete". Under the old `.is_ok()` check that panic
    /// landed between the upload and `delete_file` -- so the regression this
    /// guards is not a wrong answer, it is a user's audio left sitting in
    /// Google's storage for the full 48-hour expiry.
    #[tokio::test]
    async fn a_dropped_cancel_sender_stops_the_chunk_instead_of_waving_it_through() {
        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
        drop(cancel_tx);

        let err = transcribe_chunk(
            &unreachable_client(),
            Path::new("/nonexistent/chunk.flac"),
            "chunk",
            false,
            &mut cancel_rx,
        )
        .await
        .unwrap_err();

        assert_eq!(err.to_string(), "cancelled");
    }

    /// The ordinary cancel, for the same reason it must be decided before the
    /// upload: a chunk already sent to Google is one that has to be deleted
    /// again, and the user is charged for the transfer either way.
    #[tokio::test]
    async fn a_cancel_delivered_before_the_upload_stops_the_chunk() {
        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
        cancel_tx.send(()).expect("the receiver is alive");

        let err = transcribe_chunk(
            &unreachable_client(),
            Path::new("/nonexistent/chunk.flac"),
            "chunk",
            false,
            &mut cancel_rx,
        )
        .await
        .unwrap_err();

        assert_eq!(err.to_string(), "cancelled");
    }
}
