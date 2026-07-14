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
use std::process::{Command, ExitStatus, Stdio};
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

/// A speaker-diarization engine.
pub trait Diarizer: Send + Sync {
    /// Diarize a 16 kHz mono WAV file.
    ///
    /// An `Ok(vec![])` is a legitimate success -- silence, or a zero-length
    /// file, has no speaker turns. Callers must not treat empty as failure, and
    /// must not divide by the turn count.
    ///
    /// An `Err` means "no speaker labels for this transcript", never "this
    /// transcription failed".
    fn diarize(&self, wav_path: &Path, opts: &DiarizeOptions) -> Result<Vec<SpeakerTurn>>;
}

/// Diarization by way of the `diarize-sidecar` child process.
///
/// Holds three paths and a deadline; no models, no runtime, no memory. Every way
/// the child can go wrong -- including being killed by SIGABRT from deep inside
/// ONNX Runtime -- arrives here as an ordinary `Err`.
///
/// # Wiring this up (three things that are NOT done yet)
///
/// 1. **The models are not bundled.** They are not in `tauri.conf.json`'s
///    `resources`, and the only path resolution that exists anywhere is a
///    repo-relative `../models/diarization` in this file's `#[ignore]`d tests --
///    which does not exist inside a packaged `.app`. Add them to `resources` and
///    resolve them with `app.path().resolve(.., BaseDirectory::Resource)`.
/// 2. **The sidecar executable** is declared in `externalBin`, so Tauri stages
///    it next to the main binary and signs it. Resolve it from the `AppHandle`
///    rather than hardcoding a path; do not assume the dev-tree layout.
/// 3. **`Ok(vec![])` is a success.** Silence and zero-length audio have no turns.
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
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30 * 60);

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
    fn diarize(&self, wav_path: &Path, opts: &DiarizeOptions) -> Result<Vec<SpeakerTurn>> {
        let mut child = Command::new(&self.exe)
            .args(self.args(wav_path, opts))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("could not start the diarization sidecar at {}: {e}", self.exe.display()))?;

        // Drain both pipes on their own threads. A child that fills a pipe
        // buffer while we are blocked waiting for it to exit is a deadlock, and
        // "diarization hangs forever" is a worse failure than any crash.
        let mut stdout = child.stdout.take().expect("stdout was piped");
        let mut stderr = child.stderr.take().expect("stderr was piped");
        let out_reader = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = stdout.read_to_string(&mut s);
            s
        });
        let err_reader = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = stderr.read_to_string(&mut s);
            s
        });

        let deadline = Instant::now() + self.timeout;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(anyhow!(
                            "the diarization sidecar did not finish within {}s and was killed",
                            self.timeout.as_secs()
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(anyhow!("lost track of the diarization sidecar: {e}")),
            }
        };

        let stdout = out_reader.join().unwrap_or_default();
        let stderr = err_reader.join().unwrap_or_default();

        if !status.success() {
            // The SIGABRT case lands here, as a signal rather than an exit code.
            return Err(anyhow!(
                "the diarization sidecar {}{}",
                describe(&status),
                match stderr.trim() {
                    "" => String::new(),
                    msg => format!(": {msg}"),
                }
            ));
        }

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

/// A signal is not an exit code, and conflating them is how a SIGABRT gets
/// reported as a mysterious "exit code 134" (or, on some paths, as success).
fn describe(status: &ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            let name = match signal {
                6 => " (SIGABRT -- almost always a corrupt or truncated ONNX model)",
                9 => " (SIGKILL)",
                11 => " (SIGSEGV)",
                _ => "",
            };
            return format!("was killed by signal {signal}{name}");
        }
    }
    match status.code() {
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
        fn diarize(&self, _: &Path, _: &DiarizeOptions) -> Result<Vec<SpeakerTurn>> {
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
            d.diarize(Path::new("x.wav"), &DiarizeOptions::default())
                .unwrap()
                .len(),
            1
        );
        let failing: Box<dyn Diarizer> = Box::new(MockDiarizer::failing("model missing"));
        assert!(failing
            .diarize(Path::new("x.wav"), &DiarizeOptions::default())
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
            .diarize(Path::new("/a.wav"), &DiarizeOptions::default())
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
            .diarize(Path::new("/a.wav"), &DiarizeOptions::default())
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
            .diarize(Path::new("/a.wav"), &DiarizeOptions::default())
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
            .diarize(&wav, &DiarizeOptions::default())
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
        .diarize(&wav, &DiarizeOptions::default())
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
        .diarize(&wav, &DiarizeOptions::default())
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
        .diarize(&wav, &DiarizeOptions::default())
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
            .diarize(&wav, &DiarizeOptions::default())
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
            .diarize(&wav, &DiarizeOptions::default())
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
            .diarize(&wav, &DiarizeOptions::default())
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
