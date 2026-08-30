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
///
/// SIX, not three, and the number is measured rather than chosen. Against the
/// live API on 2026-08-29, `gemini-3.5-transcribe` shed load heavily on requests
/// over ~13 minutes of audio, returning
/// `500 "currently experiencing high demand ... please try again later"`.
/// Observed: a 22-minute chunk succeeded on attempt 2, a 26-minute chunk on
/// attempt 3, and two others were still failing past attempt 6. Three attempts
/// would have failed runs that a fourth or fifth would have carried.
///
/// The cost of one more attempt is bounded and the cost of giving up is not:
/// a failed chunk fails the WHOLE run (deliberately — see `transcribe_with_gemini`),
/// so a single stubborn chunk throws away every chunk already transcribed and
/// paid for.
pub const MAX_ATTEMPTS: u32 = 6;

/// Backoff is capped at 30s and starts at 5s.
///
/// The old 1s/2s was calibrated for a burst rate limit. It is close to
/// meaningless here: a failing request takes 67-175s to come back at all, so the
/// sleep is a rounding error on the retry interval. What it must not do is
/// hammer a service that has just said it is overloaded, hence a floor of 5s and
/// a ceiling that keeps six attempts inside a few minutes of waiting.
pub const RETRY_BACKOFF_CAP_SECS: u64 = 30;

/// The delay before attempt `attempt + 1`, in seconds. 5, 10, 20, 30, 30...
pub fn retry_backoff_secs(attempt: u32) -> u64 {
    (5u64.saturating_mul(1 << attempt.saturating_sub(1).min(8))).min(RETRY_BACKOFF_CAP_SECS)
}

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

/// Dollars per hour of audio, for telling the user what a run will cost BEFORE
/// they commit to it.
///
/// DERIVED, NOT AUTHORITATIVE. It comes from the single measured data point
/// this project has -- a 6-hour file, planned as ~15 chunks, put at roughly
/// $1.80 -- and nothing here has been checked against Google's published
/// pricing. It drives a confirmation dialog and never a bill, so being wrong
/// shows the user a wrong number and does nothing worse; but check it before
/// treating the figure as real.
///
/// Per HOUR OF AUDIO, deliberately, not per chunk: chunking exists because of
/// the 30-minute request cap, and is not something the user buys.
pub const USD_PER_AUDIO_HOUR: f64 = 0.30;

/// What `duration_secs` of audio is expected to cost.
pub fn estimate_usd(duration_secs: f64) -> f64 {
    if duration_secs <= 0.0 {
        return 0.0;
    }
    (duration_secs / 3600.0) * USD_PER_AUDIO_HOUR
}

/// What a Gemini run would cost and what it would give up, answered before a
/// single request is sent.
///
/// `diarization_available` is here rather than being left for the user to infer
/// from `chunk_count`: losing speaker labels is a consequence of the 30-minute
/// cap that nothing in the UI would otherwise mention until after the money was
/// spent, when it surfaces as a `Speakers::Unavailable` reason.
#[derive(Debug, Clone, Serialize)]
pub struct GeminiCostEstimate {
    pub duration_secs: f64,
    pub chunk_count: usize,
    pub estimated_usd: f64,
    pub diarization_available: bool,
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
                tokio::time::sleep(std::time::Duration::from_secs(retry_backoff_secs(
                    attempt,
                )))
                .await;
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

    /// Measured, not chosen — see `MAX_ATTEMPTS`. Three attempts would have
    /// failed runs that the live API served on attempt 4 or 5.
    #[test]
    fn the_retry_budget_covers_what_the_live_api_actually_needed() {
        assert!(
            MAX_ATTEMPTS >= 5,
            "a 26-minute chunk needed attempt 3 and others were still failing past 6"
        );
    }

    /// The floor matters more than the curve: a failing request already takes
    /// 67-175s to return, so a 1s sleep is a rounding error, and hammering a
    /// service that just said "high demand" is the wrong response to it.
    #[test]
    fn backoff_starts_at_five_seconds_and_is_capped() {
        assert_eq!(retry_backoff_secs(1), 5);
        assert_eq!(retry_backoff_secs(2), 10);
        assert_eq!(retry_backoff_secs(3), 20);
        assert_eq!(retry_backoff_secs(4), RETRY_BACKOFF_CAP_SECS);
        // Never overflows or exceeds the cap, however many attempts are configured.
        for attempt in 1..=64 {
            assert!(retry_backoff_secs(attempt) <= RETRY_BACKOFF_CAP_SECS);
            assert!(retry_backoff_secs(attempt) >= 5);
        }
    }

    /// Total sleeping across a full budget stays bounded — the user is already
    /// waiting on requests that take minutes; the backoff must not add hours.
    #[test]
    fn a_full_retry_budget_sleeps_for_under_three_minutes() {
        let total: u64 = (1..MAX_ATTEMPTS).map(retry_backoff_secs).sum();
        assert!(total < 180, "total backoff was {total}s");
    }

    /// The one measured data point this project has: the release notes put a
    /// 6-hour file at roughly $1.80. If this drifts, the dialog is lying to the
    /// user about what they are about to spend.
    #[test]
    fn six_hours_estimates_at_about_one_eighty() {
        assert!((estimate_usd(6.0 * 3600.0) - 1.80).abs() < 0.01);
    }

    #[test]
    fn a_zero_length_file_costs_nothing() {
        assert_eq!(estimate_usd(0.0), 0.0);
    }

    /// Linear in duration, so half the audio is half the bill. Guards against
    /// someone "improving" this into a per-chunk price -- chunking is an
    /// artifact of the 30-minute request cap, not something the user buys.
    #[test]
    fn the_estimate_is_linear_in_duration() {
        assert!((estimate_usd(1800.0) * 2.0 - estimate_usd(3600.0)).abs() < 1e-9);
    }
    /// LIVE PROBE, not part of the suite — `#[ignore]`d, so `cargo test` skips
    /// it and CI never sees it. It spends real money and needs a real key.
    ///
    /// It exists because the release checklist ends on a judgement call that no
    /// mocked test can make: on 29 Aug 2026 `gemini-3.5-transcribe` was shedding
    /// load badly, with 1.8-minute requests succeeding every time while 11-to-26
    /// minute ones failed more than 60% of attempts and got worse over hours.
    /// Shipping while that holds means a user's first long transcription
    /// probably fails. The question "has the service recovered?" can only be
    /// answered by asking the service.
    ///
    ///     GEMINI_PROBE_FLAC=/path/to/chunk.flac \
    ///     cargo test --manifest-path src-tauri/Cargo.toml \
    ///         gemini_load_shedding -- --ignored --nocapture
    ///
    /// Run it against a ~20 minute chunk AND a ~2 minute one: "long fails while
    /// short succeeds" is the actual signature of shedding, and a long run that
    /// fails on its own tells you nothing about whether the service or the file
    /// is at fault.
    ///
    /// Wall time is the read on retries. `transcribe_chunk` backs off 5/10/20s
    /// internally, so a first-attempt success looks like one request's latency
    /// and a retried one is visibly longer.
    #[tokio::test]
    #[ignore = "live: spends money and needs GEMINI_API_KEY + GEMINI_PROBE_FLAC"]
    async fn gemini_load_shedding_probe() {
        let key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY");
        let path = std::env::var("GEMINI_PROBE_FLAC").expect("GEMINI_PROBE_FLAC");
        let path = std::path::PathBuf::from(path);

        let client = client::GeminiClient::new(key);
        // The SENDER stays bound. Dropping it closes the channel, and anything
        // but `Empty` from `try_recv` ends the run — a dropped sender would
        // cancel the probe before it sent a byte.
        let (_tx, mut cancel) = tokio::sync::oneshot::channel();

        let started = std::time::Instant::now();
        let outcome = transcribe_chunk(&client, &path, "probe", false, &mut cancel).await;
        let elapsed = started.elapsed();

        match outcome {
            Ok(words) => println!(
                "PROBE OK   {:>7.1}s  {} words  {}",
                elapsed.as_secs_f64(),
                words.len(),
                path.display()
            ),
            Err(e) => println!(
                "PROBE FAIL {:>7.1}s  {}  {}",
                elapsed.as_secs_f64(),
                e,
                path.display()
            ),
        }
    }

    /// The MULTI-CHUNK live probe. Also `#[ignore]`d, also spends real money.
    ///
    /// The single-chunk probe above proves the service is up; it proves nothing
    /// about the path that only long audio takes. Everything interesting about
    /// this engine lives past the 28-minute cap: `plan` splitting the file,
    /// `snap_to_silence` moving the seams off mid-word, per-chunk slicing,
    /// `shift` rebasing each chunk into whole-file time, and the stitch. None of
    /// that had ever run against the real API -- the fixtures are single
    /// responses, and a 20-minute file is one chunk.
    ///
    ///     GEMINI_PROBE_WAV=/path/to/canonical-16k-mono.wav \
    ///     cargo test --manifest-path src-tauri/Cargo.toml \
    ///         gemini_multi_chunk -- --ignored --nocapture
    ///
    /// The WAV must be the canonical 16 kHz mono the app produces, and longer
    /// than `MAX_CHUNK_SECS`, or there is nothing here to test.
    ///
    /// THE ASSERTION THAT MATTERS is monotonic time across the whole stitched
    /// transcript. Word order IS the transcript: a `shift` applied with the
    /// wrong offset, or chunks appended out of order, produces a file that
    /// still parses, still has plausible word counts, and is quietly scrambled
    /// at the seams. Only ordering catches that.
    #[tokio::test]
    #[ignore = "live: spends money and needs GEMINI_API_KEY + GEMINI_PROBE_WAV"]
    async fn gemini_multi_chunk_probe() {
        let key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY");
        let wav = std::path::PathBuf::from(
            std::env::var("GEMINI_PROBE_WAV").expect("GEMINI_PROBE_WAV"),
        );

        let probed = std::process::Command::new("ffprobe")
            .args(["-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"])
            .arg(&wav)
            .output()
            .expect("ffprobe runs");
        let duration: f64 = String::from_utf8_lossy(&probed.stdout).trim().parse().unwrap();

        let mut specs = chunking::plan(duration);
        assert!(
            specs.len() > 1,
            "this probe needs audio longer than {} s; got {duration:.0} s",
            chunking::MAX_CHUNK_SECS
        );

        // Snap the interior seams, exactly as `transcribe_with_gemini` does.
        let silenced = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-nostats", "-i"])
            .arg(&wav)
            .args(["-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"])
            .output()
            .expect("ffmpeg runs");
        let silences = chunking::parse_silencedetect(&String::from_utf8_lossy(&silenced.stderr));
        let interior: Vec<f64> = specs.iter().skip(1).map(|s| s.start_secs).collect();
        let planned = interior.clone();
        let snapped = chunking::snap_to_silence(&interior, &silences, SILENCE_SNAP_WINDOW_SECS);
        for (index, boundary) in snapped.iter().enumerate() {
            specs[index].end_secs = *boundary;
            specs[index + 1].start_secs = *boundary;
        }
        println!(
            "{duration:.0}s -> {} chunks; {} silences found; seams {planned:?} -> {snapped:?}",
            specs.len(),
            silences.len()
        );

        let client = client::GeminiClient::new(key);
        // The sender stays bound: anything but `Empty` from `try_recv` ends the
        // run, so dropping it would cancel the probe before it sent a byte.
        let (_tx, mut cancel) = tokio::sync::oneshot::channel();
        let mut all_words: Vec<Word> = Vec::new();
        let mut per_chunk: Vec<Vec<Word>> = Vec::new();

        for spec in &specs {
            let chunk = std::env::temp_dir().join(format!("probe-chunk-{}.flac", spec.index));
            let sliced = std::process::Command::new("ffmpeg")
                .args(["-hide_banner", "-loglevel", "error", "-ss"])
                .arg(format!("{}", spec.start_secs))
                .arg("-t")
                .arg(format!("{}", spec.end_secs - spec.start_secs))
                .arg("-i")
                .arg(&wav)
                .args(["-vn", "-c:a", "flac", "-y"])
                .arg(&chunk)
                .status()
                .expect("ffmpeg runs");
            assert!(sliced.success(), "slicing chunk {} failed", spec.index);

            let started = std::time::Instant::now();
            let outcome = transcribe_chunk(
                &client,
                &chunk,
                &format!("probe-{}", spec.index),
                false,
                &mut cancel,
            )
            .await;
            let elapsed = started.elapsed();
            let _ = std::fs::remove_file(&chunk);

            let mut words = outcome
                .unwrap_or_else(|e| panic!("chunk {} failed: {e}", spec.index));
            println!(
                "  chunk {} [{:>7.1}..{:>7.1}] {:>6.1}s  {:>5} words",
                spec.index,
                spec.start_secs,
                spec.end_secs,
                elapsed.as_secs_f64(),
                words.len()
            );
            assert!(!words.is_empty(), "chunk {} came back empty", spec.index);

            transcript::shift(&mut words, spec.start_secs);
            per_chunk.push(words.clone());
            all_words.extend(words);
        }

        // Monotonic across every seam, in whole-file time. This is the check.
        for pair in all_words.windows(2) {
            assert!(
                pair[1].start >= pair[0].start - 0.001,
                "time went backwards at the seam: {:?} then {:?}",
                pair[0],
                pair[1]
            );
        }

        // And the seams line up with the plan rather than restarting at zero,
        // which is what a missing or doubled `shift` would look like.
        for (index, spec) in specs.iter().enumerate().skip(1) {
            let first = &per_chunk[index][0];
            assert!(
                first.start >= spec.start_secs - 1.0,
                "chunk {index} was not shifted into whole-file time: first word at {:.1}s, chunk starts at {:.1}s",
                first.start,
                spec.start_secs
            );
        }

        let text = transcript::full_text(&all_words);
        println!(
            "STITCHED {} words, {} chars, {:.1}s..{:.1}s",
            all_words.len(),
            text.len(),
            all_words.first().unwrap().start,
            all_words.last().unwrap().end
        );

        // The resume cache, against real words and the real plan rather than
        // the synthetic bounds the unit tests use. Everything just bought is
        // banked, then a fresh `resolve_resume` must reuse ALL of it -- which is
        // the difference between a re-run costing nothing and costing again.
        let db = std::env::temp_dir().join(format!("probe-store-{}.sqlite", std::process::id()));
        let _ = std::fs::remove_file(&db);
        let store = crate::store::Store::open(&db).expect("open store");
        let source = store
            .get_or_create_source(crate::store::SourceType::File, "probe-key", None, None, None)
            .expect("create source");
        for (index, spec) in specs.iter().enumerate() {
            store
                .save_chunk_result(
                    source.id,
                    MODEL_ID,
                    spec.index,
                    spec.start_secs,
                    spec.end_secs,
                    &serde_json::to_string(&per_chunk[index]).unwrap(),
                )
                .expect("bank the chunk");
        }
        let cached = store.chunk_results(source.id, MODEL_ID).expect("read back");
        let bounds: Vec<chunking::CachedBounds> = cached
            .iter()
            .map(|row| chunking::CachedBounds {
                index: row.index,
                start_secs: row.start_secs,
                end_secs: row.end_secs,
            })
            .collect();
        let resume = chunking::resolve_resume(&specs, &bounds);
        assert!(
            resume.iter().all(Option::is_some),
            "a re-run would have paid for these chunks again: {resume:?}"
        );

        let restored: Vec<Word> = cached
            .iter()
            .flat_map(|row| serde_json::from_str::<Vec<Word>>(&row.words_json).unwrap())
            .collect();
        assert_eq!(
            restored.len(),
            all_words.len(),
            "the resume cache did not round-trip every word"
        );
        assert_eq!(
            transcript::full_text(&restored),
            text,
            "a resumed run would produce a different transcript"
        );
        println!("RESUME  all {} chunks reusable, {} words round-tripped", specs.len(), restored.len());
        let _ = std::fs::remove_file(&db);
    }
}
