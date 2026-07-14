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

    // `tauri_build::copy_resources` only emits `cargo:rerun-if-changed=<path>` for
    // resources it actually finds (tauri-build's `lib.rs`, around line 93). An
    // EMPTY directory resolves to zero resources, so it emits ZERO
    // rerun-if-changed entries anywhere under `models/`. That leaves Cargo with no
    // dependency edge from this build script to the models directory: dropping the
    // real models in later (e.g. via `fetch-sidecars.sh --models-only`, which is
    // exactly what the warning below tells you to run) does not invalidate the
    // build script's cached output, so `cargo check` stays green with the stale
    // "models are missing" warning even after the models are on disk. Watching the
    // directory ourselves closes that gap: Cargo scans a directory path given to
    // `rerun-if-changed` recursively, and the directory is guaranteed to exist by
    // this point because of `create_dir_all` above.
    println!("cargo:rerun-if-changed=../models/diarization");

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

    warn_on_unexpected_entries(&dir);
}

/// The bundle manifest is a directory resource, not a file map: whatever is
/// physically present under `models/diarization` ships in the signed `.app`,
/// unpruned, with nothing to notice if it's more than the two files above. A
/// checkout populated by an older `fetch-sidecars.sh` (before it pruned the
/// segmentation tarball's extras) would silently bundle the unused 1.5 MB int8
/// model, a README, and the tarball's Python scripts. Warn once per unexpected
/// entry so that drifts from the known set are visible in the build log instead
/// of only in the shipped bundle's size.
fn warn_on_unexpected_entries(dir: &Path) {
    // Known-good relative paths: the two models themselves, plus their
    // directory prefixes (a directory is "expected" if something under it is),
    // plus the segmentation model's LICENSE file that `fetch-sidecars.sh`
    // deliberately keeps alongside it.
    let mut known: Vec<PathBuf> = vec![PathBuf::from(
        "sherpa-onnx-pyannote-segmentation-3-0/LICENSE",
    )];
    for model in MODELS {
        let path = PathBuf::from(model);
        for ancestor in path.ancestors() {
            if ancestor.as_os_str().is_empty() {
                continue;
            }
            if !known.contains(&ancestor.to_path_buf()) {
                known.push(ancestor.to_path_buf());
            }
        }
    }

    fn walk(base: &Path, current: &Path, known: &[PathBuf], unexpected: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(current) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(rel) = path.strip_prefix(base) else {
                continue;
            };
            if known.contains(&rel.to_path_buf()) {
                if path.is_dir() {
                    walk(base, &path, known, unexpected);
                }
                continue;
            }
            unexpected.push(rel.to_path_buf());
        }
    }

    let mut unexpected = Vec::new();
    walk(dir, dir, &known, &mut unexpected);

    if !unexpected.is_empty() {
        let mut names: Vec<String> = unexpected
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        names.sort();
        println!(
            "cargo:warning=models/diarization contains entries outside the known set ({}). \
             These will be bundled into the signed .app as-is (a directory resource ships \
             whatever is present). Likely an older/unpruned fetch-sidecars.sh checkout -- \
             re-run ./scripts/fetch-sidecars.sh --models-only to prune it.",
            names.join(", ")
        );
    }
}

fn main() {
    ensure_diarization_models_dir();
    tauri_build::build()
}
