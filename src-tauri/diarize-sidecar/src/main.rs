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

fn main() -> ExitCode {
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
