//! Speaker diarization: "who spoke when".
//!
//! # The engine runs in a child process, and that is not an implementation detail
//!
//! The governing constraint is that **diarization failure must never fail
//! transcription**. ONNX Runtime makes that impossible to honour in-process: a
//! corrupt or truncated model does not return an error, it throws a C++
//! exception that nothing catches, and the C++ runtime calls `terminate` --
//!
//! ```text
//! libc++abi: terminating due to uncaught exception of type Ort::Exception
//! signal: 6, SIGABRT
//! ```
//!
//! -- taking the whole Tauri process with it. `catch_unwind` cannot help
//! (it only unwinds Rust panics, and this crate is `panic = "abort"` anyway).
//!
//! So the engine lives in the `diarize-sidecar` binary, and [`SidecarDiarizer`]
//! spawns it. A crash, a non-zero exit, a timeout, or unparseable output all
//! come back here as `Err` -- a *degradation*, to be turned into "no speaker
//! labels", never into a failed transcription. `src-tauri/diarize-sidecar/` has
//! the other half of the story. `tests` at the bottom of this file prove it: a
//! deliberately corrupt model kills the child and this process lives.
//!
//! This module does not depend on sherpa-onnx, ONNX Runtime, or any ML crate,
//! and it must stay that way -- that is what keeps ~16 MB out of the app binary:
//!
//! ```text
//! cargo tree -p remedy-transcription | grep sherpa   # must print nothing
//! ```

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};

/// One contiguous stretch of audio attributed to a single speaker.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpeakerTurn {
    /// Seconds from the start of the audio.
    pub start: f32,
    /// Seconds from the start of the audio.
    pub end: f32,
    /// An **opaque speaker id, not an index**. Callers may rely on exactly one
    /// property: *same id => same speaker, within this one run.*
    ///
    /// It is specifically **not** dense and **not** `0..n-1`. The clusterer
    /// hands back whatever cluster labels survived its merging step, and they
    /// come out sparse: a two-speaker file has been measured returning
    /// `{0, 3}`. The count is also only a hint -- asking for 4 speakers has
    /// returned 3.
    ///
    /// So this is wrong, and wrong in a way that ships:
    ///
    /// ```text
    /// colors[turn.speaker]                  // panics, or wraps to the wrong colour
    /// format!("Speaker {}", turn.speaker + 1)  // renders "Speaker 4" for a 2-speaker file
    /// ```
    ///
    /// **Build a remap first** -- collect the distinct ids, sort them, and index
    /// *that*. `speaker_ids_are_sparse_not_contiguous` in this file pins the
    /// hazard against the real model.
    pub speaker: u32,
}

/// How to cluster the speaker embeddings.
///
/// The two fields are alternatives, not a pair: when `num_speakers` is set the
/// clusterer cuts the dendrogram to exactly that many clusters and
/// `cluster_threshold` is ignored. Otherwise it cuts by distance instead.
///
/// Measured on the test fixtures (see the real-model tests below), *neither*
/// mode is universally right:
///
/// - `num_speakers` recovers a conversation exactly, where auto-detect splits
///   one voice into several. It is the better answer whenever the count is
///   actually known -- so ask the user for it.
/// - but on audio with only a few long turns it can collapse everything into a
///   single speaker, where auto-detect gets it right.
///
/// Treat the result as a strong hint, not a guarantee: the engine may return
/// more or fewer speakers than requested, and the ids it returns are sparse --
/// see [`SpeakerTurn::speaker`].
#[derive(Debug, Clone, PartialEq)]
pub struct DiarizeOptions {
    /// Set when the user knows the count. Generally the more accurate mode, but
    /// not a hard guarantee -- see the type docs.
    pub num_speakers: Option<u32>,
    /// Used only when `num_speakers` is `None`. Lower = more speakers.
    pub cluster_threshold: f32,
}

impl Default for DiarizeOptions {
    fn default() -> Self {
        Self {
            num_speakers: None,
            cluster_threshold: 0.5,
        }
    }
}

/// The largest speaker count we will accept.
///
/// The sidecar hands `num_speakers` to sherpa as an `i32`, and `u32 as i32` is
/// silently lossy in exactly the way that produces a *plausible* wrong answer
/// rather than an error. Measured against the real models:
///
/// - `4294967295` casts to `-1`, which sherpa reads as **"auto-detect"** -- so
///   the flag is silently ignored and the user gets a confident, unrequested
///   auto-detection.
/// - `1000000` collapses the whole file into **one speaker**.
///
/// Neither is a value any real caller means, so both are rejected at the
/// boundary rather than cast (or clamped -- silently answering "64" to someone
/// who asked for a million is the same species of quiet wrongness).
pub const MAX_SPEAKERS: u32 = 64;

impl DiarizeOptions {
    /// Check values that came from the UI **before** they cross a process
    /// boundary and stop being checkable.
    ///
    /// [`Diarizer::diarize`] calls this first, so no implementation has to
    /// remember to. It is `pub` so a command can also reject bad input up front,
    /// with the UI still on screen, instead of at the end of a long job.
    pub fn validate(&self) -> Result<()> {
        // `Some(0)` and `None` both mean auto-detect; that is a real choice, not
        // a bad value.
        if let Some(n) = self.num_speakers {
            if n > MAX_SPEAKERS {
                return Err(anyhow!(
                    "num_speakers is {n}, outside the supported range 0..={MAX_SPEAKERS} \
                     (0 means auto-detect). Rejected rather than cast: the engine takes an \
                     i32, and casting a u32 into it turns 4294967295 into -1, which the \
                     clusterer reads as 'auto-detect' -- a nonsense request would come back \
                     looking like a perfectly good answer."
                ));
            }
        }
        if !self.cluster_threshold.is_finite() || self.cluster_threshold <= 0.0 {
            return Err(anyhow!(
                "cluster_threshold is {}, but it must be a positive, finite number. Zero, \
                 negative and NaN thresholds make the sidecar exit 2, which would turn every \
                 single diarization into a failure -- so they are caught here, at the boundary \
                 the UI's values cross.",
                self.cluster_threshold
            ));
        }
        Ok(())
    }
}

/// The message a cancelled run comes back as. Callers that want to tell
/// "the user cancelled" apart from "diarization broke" can match on it.
pub const CANCELLED: &str = "diarization was cancelled";

/// The kill handle for an in-flight diarization.
///
/// **Why this is not optional.** [`Diarizer::diarize`] blocks for as long as the
/// engine takes, and the engine is a CPU-bound ONNX child process with a
/// 30-minute backstop timeout. Without a handle, pressing Cancel would leave
/// that child pinning a core for up to half an hour while the app sat there
/// looking idle. An abandoned ffmpeg just finishes; an abandoned diarizer does
/// not.
///
/// Clone it freely -- every clone refers to the same run. Use **one token per
/// run**: a token that has been cancelled stays cancelled, and handing it to a
/// second [`Diarizer::diarize`] makes that call fail immediately.
///
/// ```no_run
/// # use remedy_transcription_lib::diarize::*;
/// # use std::{path::PathBuf, sync::Arc};
/// # async fn example(diarizer: Arc<dyn Diarizer>, wav: PathBuf) {
/// let cancel = CancelToken::new();
/// // ... stash `cancel.clone()` where the cancel_job command can reach it ...
/// let turns = diarize_in_background(diarizer, wav, DiarizeOptions::default(), cancel).await;
/// # }
/// ```
#[derive(Clone, Default)]
pub struct CancelToken(Arc<CancelState>);

#[derive(Default)]
struct CancelState {
    cancelled: AtomicBool,
    /// The running child, if one has been adopted. `cancel()` and the poll loop
    /// take turns holding this; whoever takes the `Child` out owns reaping it.
    child: Mutex<Option<Child>>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Kill the child **now**, and make sure a child that is about to be spawned
    /// is killed the moment it exists.
    ///
    /// Safe to call from any thread, at any time, more than once, and before or
    /// after the run has started. The in-flight `diarize()` returns [`CANCELLED`].
    pub fn cancel(&self) {
        // Order matters: the flag goes up *before* we take the lock, so a
        // `diarize()` that is mid-spawn and about to adopt its child will see it.
        self.0.cancelled.store(true, Ordering::SeqCst);
        if let Some(mut child) = lock(&self.0.child).take() {
            let _ = child.kill();
            // Reap it here rather than leaving a zombie for a `wait()` that is
            // never coming: the poll loop is about to find the slot empty and
            // give up, so nobody else will.
            let _ = child.wait();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::SeqCst)
    }

    /// Hand a freshly spawned child to the token.
    ///
    /// Returns `Err(())` if [`Self::cancel`] already fired -- and kills the child
    /// before doing so. This is the whole reason adoption happens under the same
    /// lock `cancel()` takes: a cancellation that lands between the pre-spawn
    /// check and here must still kill this child rather than sail past it and
    /// leave an orphan burning a core.
    fn adopt(&self, mut child: Child) -> std::result::Result<(), ()> {
        let mut slot = lock(&self.0.child);
        if self.0.cancelled.load(Ordering::SeqCst) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(());
        }
        *slot = Some(child);
        Ok(())
    }

    /// `None` => `cancel()` took the child and killed it.
    fn poll(&self) -> Option<std::io::Result<Option<ExitStatus>>> {
        lock(&self.0.child).as_mut().map(|c| c.try_wait())
    }

    /// Kill whatever is still adopted (the timeout path) and reap it.
    fn kill_adopted(&self) {
        if let Some(mut child) = lock(&self.0.child).take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// Drop an already-exited child. `try_wait` reaped it; this just lets go.
    fn release(&self) {
        let _ = lock(&self.0.child).take();
    }
}

/// A poisoned mutex here means a thread panicked while holding a `Child`. The
/// `Child` is still perfectly usable and killing it is still the right thing to
/// do, so recovering beats propagating a panic into the cancel path.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A speaker-diarization engine.
pub trait Diarizer: Send + Sync {
    /// Diarize a 16 kHz mono WAV file.
    ///
    /// **This blocks**, for as long as the engine takes -- minutes, on a long
    /// file. Do not call it directly from an `async` Tauri command: that parks a
    /// tokio worker thread for the whole run. Use [`diarize_in_background`],
    /// which puts it on the blocking pool for you.
    ///
    /// `cancel` is not optional, on purpose: there is no signature here that
    /// lets a caller start a 30-minute ONNX child with no way to kill it. Pass
    /// `&CancelToken::new()` if the run genuinely cannot be cancelled.
    ///
    /// An `Ok(vec![])` is a legitimate success -- silence, or a zero-length
    /// file, has no speaker turns. Callers must not treat empty as failure, and
    /// must not divide by the turn count.
    ///
    /// An `Err` means "no speaker labels for this transcript", never "this
    /// transcription failed".
    fn diarize(
        &self,
        wav_path: &Path,
        opts: &DiarizeOptions,
        cancel: &CancelToken,
    ) -> Result<Vec<SpeakerTurn>>;
}

/// **Diarize from an async context. Use this one.**
///
/// [`Diarizer::diarize`] is synchronous and long-running; awaiting it inside a
/// Tauri command would block a tokio worker for the entire job. This moves it to
/// the blocking pool, so the runtime keeps serving IPC -- including the
/// `cancel` command, which is the whole point of holding a [`CancelToken`].
///
/// Keep a clone of `cancel` somewhere the cancel command can reach (the job map)
/// *before* awaiting this.
pub async fn diarize_in_background(
    diarizer: Arc<dyn Diarizer>,
    wav_path: PathBuf,
    opts: DiarizeOptions,
    cancel: CancelToken,
) -> Result<Vec<SpeakerTurn>> {
    tokio::task::spawn_blocking(move || diarizer.diarize(&wav_path, &opts, &cancel))
        .await
        .map_err(|e| anyhow!("the diarization task did not run to completion: {e}"))?
}

/// Diarization by way of the `diarize-sidecar` child process.
///
/// Holds three paths and a deadline; no models, no runtime, no memory. Every way
/// the child can go wrong -- including being killed by SIGABRT from deep inside
/// ONNX Runtime -- arrives here as an ordinary `Err`.
///
/// # Wiring this up (five things that are NOT done yet)
///
/// 1. **`diarize()` blocks, so it must not be `await`ed on a tokio worker.** It
///    runs for as long as the engine takes -- minutes on a long file. Calling it
///    from an `async` Tauri command parks that worker for the whole job and the
///    app stops answering IPC (including Cancel). Call
///    [`diarize_in_background`], which does the `spawn_blocking` for you. The
///    signature is arranged so that this is also the *easy* path.
/// 2. **Hold the [`CancelToken`] where the cancel command can reach it.** Today
///    Cancel is frontend-only (`AudioManager.tsx` says so) and an abandoned
///    ffmpeg simply finishes. Diarization is not like that: an abandoned run is a
///    CPU-bound ONNX child pinning a core for up to [`DEFAULT_TIMEOUT`] (30
///    minutes) while the app shows idle. Clone the token into the job map before
///    starting, and call `cancel()` from `cancel_job`.
/// 3. **The models are not bundled.** They are not in `tauri.conf.json`'s
///    `resources`, and the only path resolution that exists anywhere is a
///    repo-relative `../models/diarization` in this file's `#[ignore]`d tests --
///    which does not exist inside a packaged `.app`. Add them to `resources` and
///    resolve them with `app.path().resolve(.., BaseDirectory::Resource)`.
/// 4. **The sidecar executable** is declared in `externalBin`, so Tauri stages
///    it next to the main binary and signs it. Resolve it from the `AppHandle`
///    rather than hardcoding a path; do not assume the dev-tree layout.
/// 5. **`Ok(vec![])` is a success.** Silence and zero-length audio have no turns.
///    Do not treat empty as failure, and do not divide by the turn count.
pub struct SidecarDiarizer {
    exe: PathBuf,
    segmentation_model: PathBuf,
    embedding_model: PathBuf,
    timeout: Duration,
}

/// Generous on purpose: diarization is roughly real-time-ish on CPU, and a long
/// lecture is a legitimate input. The timeout is a backstop against a wedged
/// child, not a performance budget.
///
/// It is also, precisely because it is so generous, the reason [`CancelToken`]
/// has to exist: a cancelled run with no kill handle would burn a core for this
/// long.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// How often the poll loop checks on the child.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Caps on what we will buffer from the child's pipes.
///
/// `read_to_string` with no limit is a memory leak with extra steps: a wedged or
/// corrupt child that spews grows *this* process until the timeout fires. Both
/// caps are far above any legitimate output -- even a four-hour file, chopped at
/// the 0.3 s minimum turn length, is a couple of MB of JSON -- so hitting one
/// means the child is broken, which is exactly when memory must stay bounded.
const STDOUT_CAP: u64 = 8 * 1024 * 1024;
const STDERR_CAP: u64 = 64 * 1024;

/// How long the pipes get to drain **after** the child has already exited.
///
/// The invariant: *the child holds the only write end of these pipes.* It forks
/// nothing, so once it is dead the pipes see EOF and the readers finish at
/// memcpy speed -- whatever is left is at most one pipe buffer. This grace
/// period is not a performance allowance, it is the enforcement of that
/// invariant: if a grandchild ever *did* inherit a write end, a bare
/// `join()` would block here forever and "diarization hangs" would be a worse
/// bug than anything it was protecting against. Instead we time out and say so.
const DRAIN_GRACE: Duration = Duration::from_secs(10);

/// Duplicated, deliberately, from `diarize-sidecar`'s `main.rs`.
///
/// The sidecar inherits `panic = "abort"`, so **any** Rust panic in it reaches
/// us as SIGABRT -- the same signal a corrupt ONNX model produces. Without a
/// marker to tell them apart, an indexing bug in `wav.rs` would be reported to
/// the user as "your model is corrupt", confidently and wrongly. The sidecar's
/// panic hook writes this string to stderr first; `describe` looks for it.
///
/// Not shared through a crate: that would put `diarize-sidecar` in the app's
/// dependency graph, which is the exact thing this architecture prevents. Both
/// sides carry a test pinning the literal, so a rename reddens one of them.
const SIDECAR_PANIC_MARKER: &str = "diarize-sidecar panicked";

/// Exactly the document `diarize-sidecar` writes to stdout.
#[derive(serde::Deserialize)]
struct SidecarOutput {
    turns: Vec<SpeakerTurn>,
}

impl SidecarDiarizer {
    /// `exe` is the `diarize-sidecar` executable. In a bundled app it sits
    /// beside the main binary (Tauri `externalBin`); resolve it from the
    /// `AppHandle` rather than hardcoding a path.
    pub fn new(exe: PathBuf, segmentation_model: PathBuf, embedding_model: PathBuf) -> Self {
        Self {
            exe,
            segmentation_model,
            embedding_model,
            timeout: DEFAULT_TIMEOUT,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    fn args(&self, wav_path: &Path, opts: &DiarizeOptions) -> Vec<String> {
        let mut args = vec![
            "--wav".into(),
            wav_path.to_string_lossy().into_owned(),
            "--segmentation-model".into(),
            self.segmentation_model.to_string_lossy().into_owned(),
            "--embedding-model".into(),
            self.embedding_model.to_string_lossy().into_owned(),
            "--cluster-threshold".into(),
            opts.cluster_threshold.to_string(),
        ];
        // Omitted entirely when unset: the sidecar's default is auto-detect.
        if let Some(n) = opts.num_speakers.filter(|n| *n > 0) {
            args.push("--num-speakers".into());
            args.push(n.to_string());
        }
        args
    }
}

impl Diarizer for SidecarDiarizer {
    fn diarize(
        &self,
        wav_path: &Path,
        opts: &DiarizeOptions,
        cancel: &CancelToken,
    ) -> Result<Vec<SpeakerTurn>> {
        // Values from the UI stop being checkable the moment they cross the
        // process boundary as strings, so they get checked on this side of it.
        opts.validate()?;

        // Cheap short-circuit. Not the one that matters -- `adopt` below closes
        // the actual race -- but there is no point spawning ONNX for a run the
        // user has already abandoned.
        if cancel.is_cancelled() {
            return Err(anyhow!(CANCELLED));
        }

        let mut child = Command::new(&self.exe)
            .args(self.args(wav_path, opts))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("could not start the diarization sidecar at {}: {e}", self.exe.display()))?;

        // Take the pipes before the token takes the child.
        //
        // Drain both on their own threads: a child that fills a pipe buffer while
        // we are blocked waiting for it to exit is a deadlock, and "diarization
        // hangs forever" is a worse failure than any crash. `read_capped` keeps
        // draining past the cap and throws the excess away, so bounding memory
        // does not reintroduce that deadlock.
        let mut stdout = child.stdout.take().expect("stdout was piped");
        let mut stderr = child.stderr.take().expect("stderr was piped");
        let (out_tx, out_rx) = mpsc::sync_channel::<String>(1);
        let (err_tx, err_rx) = mpsc::sync_channel::<String>(1);
        std::thread::spawn(move || {
            let _ = out_tx.send(read_capped(&mut stdout, STDOUT_CAP));
        });
        std::thread::spawn(move || {
            let _ = err_tx.send(read_capped(&mut stderr, STDERR_CAP));
        });

        // From here on the child belongs to the token: it is the only thing that
        // can kill it, and it is reachable from the cancel command.
        if cancel.adopt(child).is_err() {
            return Err(anyhow!(CANCELLED));
        }

        let deadline = Instant::now() + self.timeout;
        let status = loop {
            match cancel.poll() {
                // The slot is empty: cancel() took the child and killed it.
                None => return Err(anyhow!(CANCELLED)),
                Some(Ok(Some(status))) => {
                    cancel.release();
                    break status;
                }
                Some(Ok(None)) => {
                    if Instant::now() >= deadline {
                        cancel.kill_adopted();
                        return Err(anyhow!(
                            "the diarization sidecar did not finish within {}s and was killed",
                            self.timeout.as_secs()
                        ));
                    }
                    std::thread::sleep(POLL_INTERVAL);
                }
                Some(Err(e)) => {
                    cancel.kill_adopted();
                    return Err(anyhow!("lost track of the diarization sidecar: {e}"));
                }
            }
        };

        // The child has exited, so both pipes are at EOF and the readers are
        // finishing. Bounded rather than joined outright -- see DRAIN_GRACE.
        // Missing stderr only costs us a less specific message, so it degrades
        // to empty; missing stdout is the invariant actually breaking, and says so.
        let stderr = err_rx.recv_timeout(DRAIN_GRACE).unwrap_or_default();

        if !status.success() {
            // The SIGABRT case lands here, as a signal rather than an exit code.
            return Err(anyhow!(
                "the diarization sidecar {}{}",
                describe(&status, &stderr),
                match stderr.trim() {
                    "" => String::new(),
                    msg => format!(": {}", truncate(msg, 2000)),
                }
            ));
        }

        let stdout = out_rx.recv_timeout(DRAIN_GRACE).map_err(|_| {
            anyhow!(
                "the diarization sidecar exited cleanly but its stdout did not close within \
                 {}s. The sidecar holds the only write end of that pipe -- it forks nothing -- \
                 so this means it left a grandchild behind and that invariant no longer holds.",
                DRAIN_GRACE.as_secs()
            )
        })?;

        let parsed: SidecarOutput = serde_json::from_str(stdout.trim()).map_err(|e| {
            anyhow!(
                "the diarization sidecar exited cleanly but its output was not the expected \
                 JSON ({e}); got {:?}",
                truncate(stdout.trim(), 200)
            )
        })?;

        Ok(parsed.turns)
    }
}

/// Read at most `cap` bytes, then keep draining and discarding.
///
/// The draining is not optional: stopping at the cap would leave a chatty child
/// blocked on a full pipe forever, turning a bounded-memory fix into a hang.
fn read_capped<R: Read>(r: &mut R, cap: u64) -> String {
    let mut buf = Vec::new();
    let _ = r.take(cap).read_to_end(&mut buf);
    let _ = std::io::copy(r, &mut std::io::sink());
    String::from_utf8_lossy(&buf).into_owned()
}

/// A signal is not an exit code, and conflating them is how a SIGABRT gets
/// reported as a mysterious "exit code 134" (or, on some paths, as success).
///
/// `stderr` is not decoration. The sidecar is `panic = "abort"`, so a plain Rust
/// bug in it -- an indexing slip in `wav.rs`, say -- arrives here as the very
/// same SIGABRT that a corrupt model produces. Blaming the model for that would
/// be confidently wrong and would send the user off re-downloading a file that
/// was fine. The marker is the only thing that tells the two apart.
fn describe(status: &ExitStatus, stderr: &str) -> String {
    let panicked = stderr.contains(SIDECAR_PANIC_MARKER);

    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            let name = match signal {
                6 if panicked => {
                    " (SIGABRT, but the sidecar reported a Rust panic first -- this is a BUG IN \
                     THE SIDECAR, not a bad model; it aborts because it is built with \
                     panic = \"abort\")"
                }
                6 => {
                    " (SIGABRT -- most often a corrupt or truncated ONNX model; \
                     re-download with scripts/fetch-sidecars.sh --models-only --force)"
                }
                9 => " (SIGKILL)",
                11 => " (SIGSEGV)",
                _ => "",
            };
            return format!("was killed by signal {signal}{name}");
        }
    }

    match status.code() {
        // Debug builds do not inherit panic = "abort", so a panic exits 101.
        Some(code) if panicked => format!(
            "panicked and exited with status {code} -- this is a bug in the sidecar, not a bad model"
        ),
        Some(code) => format!("exited with status {code}"),
        None => "exited abnormally".to_string(),
    }
}

fn truncate(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// A `Diarizer` that loads nothing and spawns nothing.
///
/// Not `#[cfg(test)]`-gated by accident: it is gated *and* public so that unit
/// tests in sibling modules (`commands`) can name it as
/// `crate::diarize::mock::MockDiarizer`.
///
/// **`#[cfg(test)]` items are invisible to integration tests in `src-tauri/tests/`**
/// -- those compile as a separate crate against the public library, where this
/// module does not exist. Command-level tests that need a `MockDiarizer` must
/// therefore be in-crate `#[cfg(test)] mod tests` unit tests, not files under
/// `tests/`.
#[cfg(test)]
pub mod mock {
    use super::*;

    /// A stand-in so command-level tests never spawn a process or load a model.
    pub struct MockDiarizer(pub Result<Vec<SpeakerTurn>, String>);

    impl MockDiarizer {
        pub fn returning(turns: Vec<SpeakerTurn>) -> Self {
            Self(Ok(turns))
        }

        /// The case Task 8 actually has to handle: diarization degraded.
        pub fn failing(msg: &str) -> Self {
            Self(Err(msg.to_string()))
        }
    }

    impl Diarizer for MockDiarizer {
        fn diarize(
            &self,
            _: &Path,
            opts: &DiarizeOptions,
            cancel: &CancelToken,
        ) -> Result<Vec<SpeakerTurn>> {
            // Mirror the real one's boundary behaviour, so a command test that
            // passes garbage options or a cancelled token sees what production
            // would do rather than a cheerful stub.
            opts.validate()?;
            if cancel.is_cancelled() {
                return Err(anyhow!(CANCELLED));
            }
            match &self.0 {
                Ok(turns) => Ok(turns.clone()),
                Err(msg) => Err(anyhow!("{msg}")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::mock::MockDiarizer;
    use super::*;
    use std::collections::BTreeSet;

    fn turn(start: f32, end: f32, speaker: u32) -> SpeakerTurn {
        SpeakerTurn {
            start,
            end,
            speaker,
        }
    }

    // -- the plain data types ------------------------------------------------

    #[test]
    fn default_options_auto_detect_speaker_count() {
        let opts = DiarizeOptions::default();
        assert!(opts.num_speakers.is_none());
        assert_eq!(opts.cluster_threshold, 0.5);
    }

    #[test]
    fn speaker_turn_round_trips_through_serde() {
        // The other half of the wire contract; `diarize-sidecar`'s
        // `the_json_contract_is_exactly_this` pins the emitting side.
        let t = turn(1.5, 2.25, 3);
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, r#"{"start":1.5,"end":2.25,"speaker":3}"#);
        assert_eq!(serde_json::from_str::<SpeakerTurn>(&json).unwrap(), t);
    }

    #[test]
    fn mock_diarizer_is_usable_as_a_trait_object() {
        // Task 8 stores a Diarizer behind a trait object; prove it is object safe
        // and Send + Sync without dragging in a runtime.
        let d: Box<dyn Diarizer> = Box::new(MockDiarizer::returning(vec![turn(0.0, 1.0, 0)]));
        fn assert_send_sync<T: Send + Sync + ?Sized>(_: &T) {}
        assert_send_sync(&*d);
        assert_eq!(
            d.diarize(Path::new("x.wav"), &DiarizeOptions::default(), &CancelToken::new())
                .unwrap()
                .len(),
            1
        );
        let failing: Box<dyn Diarizer> = Box::new(MockDiarizer::failing("model missing"));
        assert!(failing
            .diarize(Path::new("x.wav"), &DiarizeOptions::default(), &CancelToken::new())
            .is_err());
    }

    // -- the command line we hand the sidecar --------------------------------

    fn sidecar_at(exe: &str) -> SidecarDiarizer {
        SidecarDiarizer::new(
            PathBuf::from(exe),
            PathBuf::from("/models/seg.onnx"),
            PathBuf::from("/models/emb.onnx"),
        )
    }

    #[test]
    fn auto_detect_omits_the_speaker_count_flag() {
        let args = sidecar_at("x").args(Path::new("/a.wav"), &DiarizeOptions::default());
        assert!(!args.contains(&"--num-speakers".to_string()));
        assert!(args.contains(&"--cluster-threshold".to_string()));
    }

    #[test]
    fn a_known_speaker_count_is_passed_through() {
        let args = sidecar_at("x").args(
            Path::new("/a.wav"),
            &DiarizeOptions {
                num_speakers: Some(3),
                cluster_threshold: 0.5,
            },
        );
        let i = args.iter().position(|a| a == "--num-speakers").unwrap();
        assert_eq!(args[i + 1], "3");
    }

    #[test]
    fn zero_speakers_is_treated_as_auto_detect_not_as_zero_clusters() {
        let args = sidecar_at("x").args(
            Path::new("/a.wav"),
            &DiarizeOptions {
                num_speakers: Some(0),
                cluster_threshold: 0.5,
            },
        );
        assert!(!args.contains(&"--num-speakers".to_string()));
    }

    // -- every way the child can go wrong is a degradation, not a crash ------
    //
    // These use stub executables rather than the real sidecar: they are about
    // this side of the process boundary, and they must run in milliseconds.
    // The real thing, with a real corrupt model, is proved further down.

    fn stub(script: &str) -> (tempfile::TempDir, PathBuf) {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stub-sidecar");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "#!/bin/sh\n{script}").unwrap();
        drop(f);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        (dir, path)
    }

    fn run_stub(script: &str) -> Result<Vec<SpeakerTurn>> {
        let (_dir, exe) = stub(script);
        SidecarDiarizer::new(exe, PathBuf::from("/s.onnx"), PathBuf::from("/e.onnx"))
            .with_timeout(Duration::from_secs(10))
            .diarize(Path::new("/a.wav"), &DiarizeOptions::default(), &CancelToken::new())
    }

    #[test]
    fn a_well_behaved_sidecar_is_parsed() {
        let turns =
            run_stub(r#"echo '{"turns":[{"start":0.0,"end":7.0,"speaker":0},{"start":7.0,"end":9.0,"speaker":3}]}'"#)
                .unwrap();
        assert_eq!(turns, vec![turn(0.0, 7.0, 0), turn(7.0, 9.0, 3)]);
    }

    #[test]
    fn an_empty_turn_list_is_a_success_not_a_failure() {
        // Silence. Task 8 must render this as "no speakers found", not as an error.
        assert_eq!(run_stub(r#"echo '{"turns":[]}'"#).unwrap(), vec![]);
    }

    #[test]
    fn a_sidecar_killed_by_sigabrt_degrades() {
        // This is the shape of the ONNX abort, reproduced without ONNX.
        let err = run_stub("kill -ABRT $$").expect_err("a SIGABRT must not be a success");
        let msg = err.to_string();
        assert!(msg.contains("signal 6"), "should name the signal: {msg}");
        assert!(msg.contains("SIGABRT"), "should name the signal: {msg}");
    }

    #[test]
    fn a_sidecar_that_segfaults_degrades() {
        let err = run_stub("kill -SEGV $$").expect_err("a SIGSEGV must not be a success");
        assert!(err.to_string().contains("signal 11"), "{err}");
    }

    #[test]
    fn a_non_zero_exit_degrades_and_carries_the_reason() {
        let err = run_stub("echo 'the segmentation model is not at /s.onnx' >&2; exit 1")
            .expect_err("a non-zero exit must not be a success");
        let msg = err.to_string();
        assert!(msg.contains("exited with status 1"), "{msg}");
        assert!(
            msg.contains("segmentation model is not at"),
            "the child's stderr should survive: {msg}"
        );
    }

    #[test]
    fn unparseable_output_degrades() {
        let err = run_stub("echo 'Segmentation fault: 11'").expect_err("garbage is not a success");
        let msg = err.to_string();
        assert!(msg.contains("not the expected JSON"), "{msg}");
        assert!(msg.contains("Segmentation fault"), "should quote what it got: {msg}");
    }

    #[test]
    fn a_clean_exit_with_no_output_at_all_degrades() {
        let err = run_stub("exit 0").expect_err("silence is not valid JSON");
        assert!(err.to_string().contains("not the expected JSON"), "{err}");
    }

    #[test]
    fn a_wedged_sidecar_is_killed_and_degrades() {
        let (_dir, exe) = stub("sleep 60");
        let started = Instant::now();
        let err = SidecarDiarizer::new(exe, PathBuf::from("/s.onnx"), PathBuf::from("/e.onnx"))
            .with_timeout(Duration::from_millis(200))
            .diarize(Path::new("/a.wav"), &DiarizeOptions::default(), &CancelToken::new())
            .expect_err("a hang must not be a success");
        assert!(err.to_string().contains("did not finish"), "{err}");
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "the timeout should fire promptly, took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn a_missing_sidecar_binary_degrades() {
        let err = sidecar_at("/nonexistent/diarize-sidecar")
            .diarize(Path::new("/a.wav"), &DiarizeOptions::default(), &CancelToken::new())
            .expect_err("a missing binary must not be a success");
        assert!(
            err.to_string().contains("could not start"),
            "should say the binary is the problem: {err}"
        );
    }

    #[test]
    fn a_sidecar_that_writes_a_lot_to_stderr_does_not_deadlock() {
        // A child that fills a pipe while we wait on it would hang forever. This
        // would have been the single worst regression available here.
        let err = run_stub("i=0; while [ $i -lt 4000 ]; do echo 'ONNX Runtime is very chatty' >&2; i=$((i+1)); done; exit 1")
            .expect_err("still a failure");
        assert!(err.to_string().contains("exited with status 1"), "{err}");
    }

    // -- the pipes are bounded, and bounding them did not reintroduce the hang --

    #[test]
    fn a_torrential_child_is_capped_rather_than_buffered_without_limit() {
        // ~5 MB of stderr, comfortably past STDERR_CAP (64 KiB). The old code
        // read_to_string'd this into the parent unbounded; a genuinely wedged
        // child could do it until the 30-minute timeout fired.
        //
        // Two things must both hold: memory stays bounded, AND the child is not
        // left blocked on a full pipe (which is why read_capped keeps draining
        // past the cap). If the drain regressed, this test would hang, not fail.
        let err = run_stub(
            "i=0; while [ $i -lt 80000 ]; do \
               echo 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' >&2; \
               i=$((i+1)); done; exit 1",
        )
        .expect_err("still a failure");

        let msg = err.to_string();
        assert!(msg.contains("exited with status 1"), "{msg}");
        // The message the user could see is bounded too -- a 5 MB error string
        // is not an error message.
        assert!(
            msg.len() < 8 * 1024,
            "the error message should be truncated, got {} bytes",
            msg.len()
        );
    }

    #[test]
    fn stdout_past_the_cap_is_a_failure_not_an_out_of_memory() {
        // A child spewing JSON forever must not grow this process without limit.
        // Truncated JSON does not parse, which is the correct outcome: a sidecar
        // emitting >8 MiB of turns is broken.
        let err = run_stub("head -c 20000000 /dev/zero | tr '\\0' 'x'; exit 0")
            .expect_err("a flood of garbage on stdout is not a success");
        assert!(err.to_string().contains("not the expected JSON"), "{err}");
    }

    // -- the values that come from the UI are checked before they cross over ---

    #[test]
    fn an_absurd_speaker_count_is_rejected_rather_than_cast_to_auto_detect() {
        // u32::MAX as i32 == -1, and the clusterer reads -1 as "auto-detect". So
        // the lossy cast did not merely lose the value, it turned a nonsense
        // request into a confident, plausible, *unrequested* answer.
        for n in [u32::MAX, 1_000_000, MAX_SPEAKERS + 1] {
            let err = DiarizeOptions {
                num_speakers: Some(n),
                ..Default::default()
            }
            .validate()
            .expect_err("{n} speakers must be rejected");
            assert!(err.to_string().contains("num_speakers"), "{err}");
        }

        // And the boundary itself is fine on both sides.
        assert!(DiarizeOptions {
            num_speakers: Some(MAX_SPEAKERS),
            ..Default::default()
        }
        .validate()
        .is_ok());
        // 0 and None are auto-detect, which is a real choice.
        assert!(DiarizeOptions {
            num_speakers: Some(0),
            ..Default::default()
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn a_broken_cluster_threshold_is_rejected_at_the_boundary() {
        // Each of these makes the sidecar exit 2, i.e. turns *every* diarization
        // into a failure. They come from the UI, so they are caught here.
        for bad in [0.0, -0.5, f32::NAN, f32::INFINITY] {
            let err = DiarizeOptions {
                num_speakers: None,
                cluster_threshold: bad,
            }
            .validate()
            .expect_err("{bad} must be rejected");
            assert!(err.to_string().contains("cluster_threshold"), "{err}");
        }
    }

    #[test]
    fn bad_options_fail_before_anything_is_even_spawned() {
        // sidecar_at points at a path that does not exist, so if validation did
        // not come first the error would be "could not start", not this one.
        let err = sidecar_at("/nonexistent/diarize-sidecar")
            .diarize(
                Path::new("/a.wav"),
                &DiarizeOptions {
                    num_speakers: Some(u32::MAX),
                    cluster_threshold: 0.5,
                },
                &CancelToken::new(),
            )
            .expect_err("must not be accepted");
        assert!(err.to_string().contains("num_speakers"), "{err}");
        assert!(
            !err.to_string().contains("could not start"),
            "validation must precede the spawn: {err}"
        );
    }

    // -- cancellation ---------------------------------------------------------

    #[test]
    fn cancelling_kills_a_running_sidecar_instead_of_waiting_out_the_timeout() {
        // The finding this exists for: with the 30-minute default timeout and no
        // kill handle, a cancelled job left an ONNX child pinning a core for half
        // an hour while the app showed idle.
        let (_dir, exe) = stub("sleep 300");
        let cancel = CancelToken::new();

        let watcher = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            watcher.cancel();
        });

        let started = Instant::now();
        let err = SidecarDiarizer::new(exe, PathBuf::from("/s.onnx"), PathBuf::from("/e.onnx"))
            // The real default. If cancel did not work, this test would take 30
            // minutes -- which is precisely the bug.
            .diarize(Path::new("/a.wav"), &DiarizeOptions::default(), &cancel)
            .expect_err("a cancelled run is not a success");

        assert!(err.to_string().contains(CANCELLED), "{err}");
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "cancel must kill the child, not wait for it; took {:?}",
            started.elapsed()
        );
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn a_token_cancelled_before_the_run_never_spawns_a_child() {
        // The exe does not exist, so a spawn would fail with "could not start".
        // Getting CANCELLED instead proves nothing was spawned.
        let cancel = CancelToken::new();
        cancel.cancel();

        let err = sidecar_at("/nonexistent/diarize-sidecar")
            .diarize(Path::new("/a.wav"), &DiarizeOptions::default(), &cancel)
            .expect_err("a cancelled run is not a success");
        assert!(err.to_string().contains(CANCELLED), "{err}");
        assert!(!err.to_string().contains("could not start"), "{err}");
    }

    #[test]
    fn cancel_is_idempotent_and_safe_after_the_run_is_over() {
        let (_dir, exe) = stub(r#"echo '{"turns":[]}'"#);
        let cancel = CancelToken::new();

        let out = SidecarDiarizer::new(exe, PathBuf::from("/s.onnx"), PathBuf::from("/e.onnx"))
            .with_timeout(Duration::from_secs(10))
            .diarize(Path::new("/a.wav"), &DiarizeOptions::default(), &cancel)
            .expect("a clean run");
        assert_eq!(out, vec![]);

        // The child is long gone and the slot is empty. Cancelling now must be a
        // no-op, not a panic on a missing Child and not a hang on a dead one.
        cancel.cancel();
        cancel.cancel();
        assert!(cancel.is_cancelled());
    }

    #[tokio::test]
    async fn diarize_in_background_keeps_the_async_runtime_free_to_cancel() {
        // This is the shape Task 8 must use. If `diarize()` were awaited directly
        // from an async command it would park a tokio worker for the whole run --
        // and the cancel command, which arrives over that same runtime, could
        // never be served. So: spawn_blocking, then cancel from the async side.
        let (_dir, exe) = stub("sleep 300");
        let diarizer: Arc<dyn Diarizer> = Arc::new(SidecarDiarizer::new(
            exe,
            PathBuf::from("/s.onnx"),
            PathBuf::from("/e.onnx"),
        ));
        let cancel = CancelToken::new();

        let job = tokio::spawn(diarize_in_background(
            diarizer,
            PathBuf::from("/a.wav"),
            DiarizeOptions::default(),
            cancel.clone(),
        ));

        // The runtime is still responsive -- this is the assertion that matters.
        tokio::time::sleep(Duration::from_millis(150)).await;
        cancel.cancel();

        let err = tokio::time::timeout(Duration::from_secs(10), job)
            .await
            .expect("the job should end promptly once cancelled")
            .expect("the blocking task should not panic")
            .expect_err("a cancelled run is not a success");
        assert!(err.to_string().contains(CANCELLED), "{err}");
    }

    // -- a panic in the sidecar is a BUG, not a corrupt model ------------------

    #[test]
    fn the_panic_marker_is_the_literal_the_sidecar_writes() {
        // diarize-sidecar's main.rs carries the same literal and pins it with the
        // mirror of this test. Renaming one reddens the other -- which is the
        // point, because the two crates deliberately share no code.
        assert_eq!(SIDECAR_PANIC_MARKER, "diarize-sidecar panicked");
    }

    #[test]
    fn a_sidecar_panic_is_not_blamed_on_the_model() {
        // The sidecar inherits panic = "abort", so an ordinary Rust bug in it --
        // an indexing slip in wav.rs, say -- arrives as the very same SIGABRT a
        // corrupt model produces. Telling the user their model is corrupt would
        // be confidently wrong and send them re-downloading a perfectly good file.
        let err = run_stub(&format!(
            "echo '{SIDECAR_PANIC_MARKER}: index out of bounds at wav.rs:51' >&2; kill -ABRT $$"
        ))
        .expect_err("a panic is not a success");

        let msg = err.to_string();
        assert!(msg.contains("signal 6"), "still names the signal: {msg}");
        assert!(
            msg.contains("BUG IN THE SIDECAR"),
            "a panic must be reported as a bug: {msg}"
        );
        assert!(
            !msg.contains("corrupt or truncated ONNX model"),
            "must NOT blame the model when the sidecar said it panicked: {msg}"
        );
    }

    #[test]
    fn a_sigabrt_with_no_panic_marker_still_points_at_the_model() {
        // The other side of the same coin: a real ONNX abort prints no marker,
        // and the model really is the likely cause. Keep saying so.
        let err = run_stub(
            "echo 'libc++abi: terminating due to uncaught exception of type Ort::Exception' >&2; \
             kill -ABRT $$",
        )
        .expect_err("a SIGABRT is not a success");

        let msg = err.to_string();
        assert!(msg.contains("corrupt or truncated ONNX model"), "{msg}");
        assert!(!msg.contains("BUG IN THE SIDECAR"), "{msg}");
    }

    // -- the real sidecar, the real abort ------------------------------------

    /// The real `diarize-sidecar` binary.
    ///
    /// `CARGO_BIN_EXE_*` only covers bins in *this* package, and the sidecar is
    /// deliberately a separate one, so look in the two places it can be:
    ///
    /// 1. `target/<profile>/diarize-sidecar`, next to our own test binary. This
    ///    is what `cargo test --workspace` builds, and it is the freshest.
    /// 2. the staged `binaries/diarize-sidecar-<triple>` that Tauri bundles.
    ///    Always present: `externalBin` makes tauri-build refuse to build the app
    ///    at all without it, so anyone who got far enough to run these tests has
    ///    one. This is what covers a plain `cargo test` (which does not build
    ///    other workspace members).
    fn real_sidecar() -> PathBuf {
        let mut dir = std::env::current_exe().expect("a test binary has a path");
        dir.pop(); // deps/
        if dir.ends_with("deps") {
            dir.pop();
        }
        let built = dir.join(format!("diarize-sidecar{}", std::env::consts::EXE_SUFFIX));
        if built.is_file() {
            return built;
        }

        let staged = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("binaries");
        if let Ok(entries) = std::fs::read_dir(&staged) {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("diarize-sidecar-")
                {
                    return entry.path();
                }
            }
        }

        panic!(
            "no diarize-sidecar binary found -- looked at {} and in {}.\n\
             It is a separate workspace member, so build it with:\n    \
             ./scripts/build-diarize-sidecar.sh\n\
             or run the tests with:\n    cargo test --workspace",
            built.display(),
            staged.display()
        )
    }

    /// **The reason this whole architecture exists.**
    ///
    /// A truncated ONNX model makes ONNX Runtime throw a C++ exception that
    /// nothing catches, and the C++ runtime aborts the process. Before this
    /// change that process was the app, and the abort took transcription down
    /// with it -- unstoppably, since `catch_unwind` cannot intercept a foreign
    /// `terminate` and this crate is `panic = "abort"` regardless.
    ///
    /// Now the process that dies is the child. This test asserts that the child
    /// really does die of SIGABRT, that we get an `Err` for it, and -- by the
    /// simple fact of reaching its final line -- that *this* process is alive.
    /// If the architecture regresses, this test does not fail. The test runner
    /// dies. That is the difference, and it is the whole point.
    #[test]
    fn a_corrupt_model_kills_the_sidecar_and_the_app_survives() {
        let dir = tempfile::tempdir().unwrap();

        // Not a valid ONNX protobuf. Exactly what an interrupted download leaves
        // behind -- see the `mv`-into-place fix in scripts/fetch-sidecars.sh.
        let corrupt = dir.path().join("truncated.onnx");
        std::fs::write(&corrupt, &b"\x08\x07\x12\x0connx-corrupt"[..]).unwrap();

        let wav = dir.path().join("silence.wav");
        std::fs::write(&wav, mono_16k_wav(&[0i16; 1600])).unwrap();

        let err = SidecarDiarizer::new(real_sidecar(), corrupt.clone(), corrupt)
            .with_timeout(Duration::from_secs(60))
            .diarize(&wav, &DiarizeOptions::default(), &CancelToken::new())
            .expect_err("a corrupt model must degrade, not succeed");

        let msg = err.to_string();
        println!("a corrupt model degraded to: {msg}");

        // Measured: the child dies of SIGABRT ("libc++abi: terminating due to
        // uncaught exception of type Ort::Exception", exit 134). A clean
        // non-zero exit would be just as acceptable an outcome for the app, so
        // both are allowed -- what is *not* allowed is Ok, or not getting here.
        assert!(
            msg.contains("signal 6") || msg.contains("exited with status"),
            "a corrupt model should come back as a dead or failed child, got: {msg}"
        );

        // Reaching this line at all is the real assertion: the process that was
        // supposed to die did, and this one did not.
        assert!(std::process::id() > 0, "still alive");
    }

    /// The other half: the sidecar must still be a good citizen for the boring
    /// failures, so the user gets a real message and not just a corpse.
    #[test]
    fn a_missing_model_is_a_clean_non_zero_exit_not_a_crash() {
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("silence.wav");
        std::fs::write(&wav, mono_16k_wav(&[0i16; 1600])).unwrap();

        let err = SidecarDiarizer::new(
            real_sidecar(),
            dir.path().join("nope-seg.onnx"),
            dir.path().join("nope-emb.onnx"),
        )
        .with_timeout(Duration::from_secs(60))
        .diarize(&wav, &DiarizeOptions::default(), &CancelToken::new())
        .expect_err("a missing model must fail");

        let msg = err.to_string();
        assert!(msg.contains("exited with status 1"), "{msg}");
        assert!(msg.contains("segmentation"), "should name the missing model: {msg}");
        assert!(msg.contains("fetch-sidecars"), "should say how to fix it: {msg}");
    }

    /// A stereo file used to be accepted silently and read as mono, which halves
    /// every timestamp. And it must be rejected *before* 34 MB of models load.
    #[test]
    fn a_stereo_wav_is_rejected_by_name_and_before_any_model_loads() {
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("stereo.wav");
        std::fs::write(&wav, wav_bytes(2, 16_000, &[0i16; 3200])).unwrap();

        // The models do not exist. If the rejection did not come first, the
        // error would be about the missing models instead.
        let err = SidecarDiarizer::new(
            real_sidecar(),
            dir.path().join("nope-seg.onnx"),
            dir.path().join("nope-emb.onnx"),
        )
        .with_timeout(Duration::from_secs(60))
        .diarize(&wav, &DiarizeOptions::default(), &CancelToken::new())
        .expect_err("stereo must be rejected");

        let msg = err.to_string();
        assert!(msg.contains("2 channels"), "should say what is wrong: {msg}");
        assert!(
            !msg.contains("nope-seg"),
            "the channel check must run before the models are even looked for: {msg}"
        );
    }

    #[test]
    fn a_wrong_sample_rate_is_rejected_before_any_model_loads() {
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("48k.wav");
        std::fs::write(&wav, wav_bytes(1, 48_000, &[0i16; 4800])).unwrap();

        let err = SidecarDiarizer::new(
            real_sidecar(),
            dir.path().join("nope-seg.onnx"),
            dir.path().join("nope-emb.onnx"),
        )
        .with_timeout(Duration::from_secs(60))
        .diarize(&wav, &DiarizeOptions::default(), &CancelToken::new())
        .expect_err("48 kHz must be rejected");

        let msg = err.to_string();
        assert!(msg.contains("48000 Hz"), "{msg}");
        assert!(
            !msg.contains("nope-seg"),
            "the rate check must run before the models are even looked for: {msg}"
        );
    }

    fn mono_16k_wav(samples: &[i16]) -> Vec<u8> {
        wav_bytes(1, 16_000, samples)
    }

    fn wav_bytes(channels: u16, sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        let bits = 16u16;
        let block_align = channels * bits / 8;
        let data_len = (samples.len() * 2) as u32;
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data_len).to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&channels.to_le_bytes());
        b.extend_from_slice(&sample_rate.to_le_bytes());
        b.extend_from_slice(&(sample_rate * block_align as u32).to_le_bytes());
        b.extend_from_slice(&block_align.to_le_bytes());
        b.extend_from_slice(&bits.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            b.extend_from_slice(&s.to_le_bytes());
        }
        b
    }

    // ---------------------------------------------------------------------
    // Real-model tests. The only ones that load ONNX -- in the child, where it
    // belongs. They require:
    //     ./scripts/fetch-sidecars.sh --models-only
    //     cargo build -p diarize-sidecar
    // Run with: cargo test --workspace -- --ignored
    //
    // Two fixtures, because the two clustering modes fail in opposite
    // directions and a single fixture would hide one of them:
    //
    //   two_speakers.wav        4 long turns  (Daniel/Samantha, macOS `say`)
    //       auto-detect     -> 2 speakers  (correct)
    //       num_speakers=2  -> 1 speaker   (collapses: too few embedding rows)
    //
    //   two_speakers_dense.wav  8 shorter turns, same two voices
    //       auto-detect     -> 4 speakers  (over-segments a single voice)
    //       num_speakers=2  -> 2 speakers  (correct, perfectly alternating)
    //
    // So neither mode is universally right. Telling the clusterer the speaker
    // count is the stronger option on realistic conversational audio, which is
    // why the UI should ask for it. See the task-7 report.
    // ---------------------------------------------------------------------

    fn fixture_diarizer(fixture: &str) -> (SidecarDiarizer, PathBuf) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let models = root.join("../models/diarization");
        let d = SidecarDiarizer::new(
            real_sidecar(),
            models.join("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
            models.join("wespeaker_en_voxceleb_CAM++.onnx"),
        );
        (d, root.join("tests/fixtures").join(fixture))
    }

    fn report(label: &str, turns: &[SpeakerTurn]) -> BTreeSet<u32> {
        println!("--- {label} ---");
        for t in turns {
            println!("{:6.2}s - {:6.2}s  speaker {}", t.start, t.end, t.speaker);
        }
        turns.iter().map(|t| t.speaker).collect()
    }

    /// Ground truth for `two_speakers.wav`:
    ///     Daniel    0.00 -  7.01
    ///     Samantha  7.01 - 12.67
    ///     Daniel   12.67 - 18.93
    ///     Samantha 18.93 - 23.98
    #[test]
    #[ignore = "requires the diarization models; run with: cargo test --workspace -- --ignored"]
    fn the_sidecar_finds_two_speakers_in_the_fixture() {
        let (d, wav) = fixture_diarizer("two_speakers.wav");

        let turns = d
            .diarize(&wav, &DiarizeOptions::default(), &CancelToken::new())
            .expect("diarization should succeed on the fixture");

        let speakers = report("two_speakers.wav / auto-detect", &turns);
        assert!(!turns.is_empty(), "expected some speaker turns");
        assert_eq!(
            speakers.len(),
            2,
            "expected exactly two distinct speakers, got {speakers:?}"
        );

        // Turn boundaries should land near the real ones, not be smeared.
        assert!(
            turns[0].start < 0.5,
            "the first turn should start at the top of the file, got {:?}",
            turns[0]
        );
        assert!(
            (turns[0].end - 7.01).abs() < 0.5,
            "the first speaker change should be near 7.01s, got {:?}",
            turns[0]
        );
    }

    /// **The sparse-id hazard, pinned.**
    ///
    /// The single most expensive thing to rediscover: on this two-speaker file
    /// the engine returns speaker ids `{0, 3}`. Two speakers, ids 0 and 3.
    ///
    /// A `HashSet::len() == 2` assertion passes just as happily for `{0,1}` as
    /// for `{0,3}` and therefore pins nothing, which is why this test exists
    /// separately and asserts the *gap*.
    ///
    /// If a future model returns dense ids this test goes red. That is correct
    /// and deliberate: it means the guidance on [`SpeakerTurn::speaker`] is now
    /// out of date and someone must re-read it rather than quietly inherit an
    /// assumption that no longer holds.
    #[test]
    #[ignore = "requires the diarization models; run with: cargo test --workspace -- --ignored"]
    fn speaker_ids_are_sparse_not_contiguous() {
        let (d, wav) = fixture_diarizer("two_speakers.wav");

        let turns = d
            .diarize(&wav, &DiarizeOptions::default(), &CancelToken::new())
            .expect("diarization should succeed on the fixture");

        let ids = report("two_speakers.wav / speaker id space", &turns);
        let max = *ids.iter().next_back().expect("some turns");

        // For a set of distinct ids, "dense 0..n-1" is exactly max == len - 1.
        // So sparse is max >= len. Two speakers with ids {0, 3}: max 3, len 2.
        assert!(
            max as usize >= ids.len(),
            "speaker ids came back dense ({ids:?}). They have always been sparse -- this file \
             returns {{0, 3}} -- and SpeakerTurn::speaker's docs, plus every caller that remaps \
             them, are written against that. If the engine changed, re-read those docs and \
             decide deliberately; do not just delete this test."
        );
    }

    /// The `num_speakers` path -- the one we tell users to prefer -- against a
    /// real model. On conversational audio it recovers the speakers exactly,
    /// where auto-detect over-segments the same file into four.
    #[test]
    #[ignore = "requires the diarization models; run with: cargo test --workspace -- --ignored"]
    fn a_known_speaker_count_is_honoured_against_a_real_model() {
        let (d, wav) = fixture_diarizer("two_speakers_dense.wav");

        let turns = d
            .diarize(
                &wav,
                &DiarizeOptions {
                    num_speakers: Some(2),
                    ..Default::default()
                },
                &CancelToken::new(),
            )
            .expect("diarization should succeed on the fixture");

        let speakers = report("two_speakers_dense.wav / num_speakers = 2", &turns);
        assert_eq!(
            speakers.len(),
            2,
            "asking for 2 speakers should yield 2, got {speakers:?}"
        );

        // The fixture strictly alternates, so the labels should too.
        let labels: Vec<u32> = turns.iter().map(|t| t.speaker).collect();
        for pair in labels.windows(2) {
            assert_ne!(
                pair[0], pair[1],
                "the fixture alternates speakers, so no two adjacent turns should \
                 share a label: {labels:?}"
            );
        }
    }

    /// Guards the claim in `DiarizeOptions`' docs: auto-detect is NOT reliable
    /// on this file, so the UI must not silently default to it and call the
    /// answer authoritative. If a future model makes this pass, revisit the docs.
    #[test]
    #[ignore = "requires the diarization models; run with: cargo test --workspace -- --ignored"]
    fn auto_detect_over_segments_the_dense_fixture() {
        let (d, wav) = fixture_diarizer("two_speakers_dense.wav");

        let turns = d
            .diarize(&wav, &DiarizeOptions::default(), &CancelToken::new())
            .expect("diarization should succeed on the fixture");

        let speakers = report("two_speakers_dense.wav / auto-detect", &turns);
        assert!(
            speakers.len() > 2,
            "auto-detect was expected to over-segment this 2-speaker file, but it \
             found {}. If the engine improved, update the guidance in DiarizeOptions.",
            speakers.len()
        );
    }
}
