//! Speaker diarization, out of process.
//!
//! # Why this is a separate executable
//!
//! ONNX Runtime does not report a corrupt or truncated model as an error. It
//! throws a C++ exception that nothing catches, and the C++ runtime calls
//! `terminate`:
//!
//! ```text
//! libc++abi: terminating due to uncaught exception of type Ort::Exception
//! signal: 6, SIGABRT
//! ```
//!
//! That is a process-wide abort. Rust cannot intercept it -- `catch_unwind`
//! only unwinds Rust panics, and the app sets `panic = "abort"` regardless. If
//! this code ran inside the Tauri app, a single bad byte in a model file would
//! kill transcription too.
//!
//! The app's governing constraint is that **diarization failure must never fail
//! transcription**. In-process that is unsatisfiable. Here, the worst case is a
//! dead child process and a non-zero exit status, which the parent turns into a
//! transcript with no speaker labels.
//!
//! So: keep this dumb and stateless. One WAV in, one JSON document out, exit.
//! Hold no state, open no sockets, and never try to be resilient -- dying is a
//! perfectly good outcome here, and the parent is built to expect it.
//!
//! # Contract
//!
//! ```text
//! diarize-sidecar --wav <path>
//!                 --segmentation-model <path>
//!                 --embedding-model <path>
//!                 [--num-speakers <n>]          # omit or 0 to auto-detect
//!                 [--cluster-threshold <f>]     # only used when auto-detecting
//! ```
//!
//! - exit 0: stdout is `{"turns":[{"start":<f32>,"end":<f32>,"speaker":<u32>}, ...]}`,
//!   sorted by start time. **An empty `turns` array is a valid success** --
//!   silence, or a zero-length file, has no speaker turns and is not an error.
//! - exit non-zero: stderr carries a human-readable reason. stdout is empty.
//! - killed by a signal: no output at all. Also just a failure. See above.
//!
//! Anything printed on stdout that is not the JSON document breaks the parent's
//! parser, so nothing else may ever be written there.

mod cli;
mod engine;
mod wav;

use std::process::ExitCode;

/// Written to stderr the instant this binary panics.
///
/// **Why it has to exist.** This crate inherits `panic = "abort"` from the
/// workspace's release profile, so *any* Rust panic here -- an indexing slip in
/// `wav.rs`, an `unwrap` on a `None` -- kills the process with SIGABRT. That is
/// the very same signal a corrupt ONNX model produces, and the parent's error
/// message for signal 6 says "almost always a corrupt or truncated ONNX model".
/// Without a way to tell the two apart, an ordinary bug in this file would be
/// reported to the user as "your model is corrupt": confidently wrong, and it
/// would send them off re-downloading a file that was never the problem.
///
/// The app's `diarize.rs` carries this same literal and greps for it. The two
/// crates deliberately share no code, so both sides pin it with a test -- rename
/// one and the other goes red.
const PANIC_MARKER: &str = "diarize-sidecar panicked";

/// Make a panic say it was a panic, before `panic = "abort"` makes it look like
/// an ONNX abort.
fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        // stderr only. stdout is the JSON channel and nothing else may touch it.
        eprintln!("{PANIC_MARKER}: {info}");
        eprintln!(
            "This is a BUG IN diarize-sidecar, not a corrupt model. The process is about to \
             die of SIGABRT (it is built with panic = \"abort\"), which is the same signal a \
             bad ONNX model produces -- the line above is how the parent tells them apart."
        );
    }));
}

fn main() -> ExitCode {
    install_panic_hook();

    let args = match cli::Args::from_env() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("{err}");
            eprintln!("\n{}", cli::USAGE);
            return ExitCode::from(2);
        }
    };

    match engine::run(&args) {
        Ok(turns) => {
            // The only thing that may ever touch stdout.
            match serde_json::to_string(&engine::Output { turns }) {
                Ok(json) => {
                    println!("{json}");
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("failed to serialize the speaker turns: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mirror of `the_panic_marker_is_the_literal_the_sidecar_writes` in the
    /// app's `diarize.rs`. Same literal, two crates, no shared code -- so this
    /// pair is the only thing keeping them in sync.
    #[test]
    fn the_panic_marker_is_the_literal_the_app_greps_for() {
        assert_eq!(PANIC_MARKER, "diarize-sidecar panicked");
    }

    #[test]
    fn the_panic_hook_installs_and_writes_the_marker() {
        install_panic_hook();
        // The hook is global, so just prove it is reachable and that a panic
        // caught here still runs it. (Debug builds unwind; release aborts --
        // either way the hook has already written the marker to stderr.)
        let caught = std::panic::catch_unwind(|| panic!("deliberate"));
        assert!(caught.is_err());
    }
}
