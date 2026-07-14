use std::path::{Path, PathBuf};

/// The two ONNX models diarization loads, relative to `models/diarization`.
///
/// Kept in step with `commands.rs`'s `SEGMENTATION_MODEL` / `EMBEDDING_MODEL` by
/// `the_bundled_model_paths_are_the_ones_tauri_will_actually_produce`, which reads
/// `tauri.conf.json` and pins the mapping both files depend on.
const MODELS: [&str; 2] = [
    "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    "wespeaker_en_voxceleb_CAM++.onnx",
];

/// Make a missing `models/diarization` a DEGRADED BUILD, not a broken one.
///
/// `tauri.conf.json` bundles `../models/diarization` as a resource, and
/// `tauri_build::build()` -- which runs during `cargo check`, not just during
/// `tauri build` -- hard-errors with `ResourcePathNotFound` on any configured
/// resource path that does not exist (`tauri_utils::resources`). The models are
/// 34 MB of downloaded weights, correctly gitignored, so on a fresh clone -- and
/// in CI, which runs `fetch-sidecars.sh --skip-models` -- that path does NOT
/// exist, and a bare `cargo check` would die with an opaque resource error
/// before compiling a line. That is the same trap the README already documents
/// for `externalBin`; it applies to `resources` too.
///
/// It would also make a liar of the runtime: `diarize_job` returns
/// `DiarizationOutcome::Degraded` when a model is missing -- "a build with no
/// models is a build without speaker labels, not a broken app" -- and that path
/// is unreachable through this repo's own build if the build refuses to run
/// without them.
///
/// So the directory is created if absent. Tauri then walks an EMPTY directory,
/// bundles nothing, and the app degrades visibly at runtime, which is exactly
/// what the runtime already promises. The `cargo:warning` is the actionable
/// part: it names the models that are missing and the script that fetches them.
///
/// This is also why the resource is the DIRECTORY and not the two files: a
/// per-file map has no "absent is fine" spelling, and neither does a glob (an
/// empty glob is `GlobPathNotFound`). The directory mapping is
/// prefix-preserving, so each model still lands at exactly
/// `$RESOURCE/models/diarization/<same relative path>`.
fn ensure_diarization_models_dir() {
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("../models/diarization");

    if let Err(e) = std::fs::create_dir_all(&dir) {
        panic!(
            "could not create {} -- tauri.conf.json bundles it as a resource, and tauri-build \
             fails on a resource path that does not exist: {e}",
            dir.display()
        );
    }

    let missing: Vec<&str> = MODELS
        .iter()
        .copied()
        .filter(|model| !Path::new(&dir).join(model).is_file())
        .collect();

    if !missing.is_empty() {
        println!(
            "cargo:warning=speaker diarization models are missing ({}). This build compiles and \
             runs, but diarization will report itself DEGRADED and produce no speaker labels. \
             Fix with: ./scripts/fetch-sidecars.sh --models-only",
            missing.join(", ")
        );
    }
}

fn main() {
    ensure_diarization_models_dir();
    tauri_build::build()
}
