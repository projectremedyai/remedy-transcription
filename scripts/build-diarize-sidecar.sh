#!/usr/bin/env bash
# Build the diarization sidecar and stage it where Tauri's `externalBin` expects
# it: src-tauri/binaries/diarize-sidecar-<target-triple>[.exe]
#
# Unlike yt-dlp/ffmpeg/ffprobe, this sidecar is OURS -- it is compiled, not
# downloaded, which is why it lives here and not in fetch-sidecars.sh.
#
# It is a separate binary on purpose. It links ONNX Runtime, and ONNX Runtime
# aborts the whole process (SIGABRT) on a corrupt model. Out of process, that is
# a dead child and an error; in process, it would take transcription down too.
#
# This must run BEFORE any cargo command that touches the app: `externalBin`
# entries are checked for existence by tauri-build's build script, so a missing
# diarize-sidecar-<triple> fails even a bare `cargo check`.
#
# Usage:
#   ./scripts/build-diarize-sidecar.sh                        # host triple, release
#   ./scripts/build-diarize-sidecar.sh aarch64-apple-darwin
#   ./scripts/build-diarize-sidecar.sh --debug                # fast; for CI/dev only

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROFILE="release"
TRIPLE=""

for arg in "$@"; do
    case "$arg" in
        --debug) PROFILE="debug" ;;
        -*) echo "Unknown option: $arg" >&2; exit 1 ;;
        *) TRIPLE="$arg" ;;
    esac
done

HOST="$(rustc -vV | awk '/host:/ {print $2}')"
TRIPLE="${TRIPLE:-$HOST}"
DEST="src-tauri/binaries"

ext=""
case "$TRIPLE" in
    *-pc-windows-*) ext=".exe" ;;
esac

mkdir -p "$DEST"

flags=()
[[ "$PROFILE" == "release" ]] && flags+=(--release)

# Only cross-compile when actually cross-compiling. Passing --target for the host
# triple would build into target/<triple>/ instead of target/<profile>/, which is
# a second full compile of ONNX Runtime AND is not where `cargo test` looks for
# the sidecar it needs to spawn.
BUILT="src-tauri/target/$PROFILE/diarize-sidecar${ext}"
if [[ "$TRIPLE" != "$HOST" ]]; then
    flags+=(--target "$TRIPLE")
    BUILT="src-tauri/target/$TRIPLE/$PROFILE/diarize-sidecar${ext}"
fi

echo "Building diarize-sidecar for $TRIPLE ($PROFILE; this links ONNX Runtime and is not quick)..."
cargo build \
    --manifest-path src-tauri/Cargo.toml \
    --package diarize-sidecar \
    "${flags[@]}"

[[ -f "$BUILT" ]] || { echo "ERROR: expected a binary at $BUILT" >&2; exit 1; }

cp "$BUILT" "$DEST/diarize-sidecar-${TRIPLE}${ext}"
chmod +x "$DEST/diarize-sidecar-${TRIPLE}${ext}"

echo
echo "Diarization sidecar in place:"
ls -la "$DEST/diarize-sidecar-${TRIPLE}${ext}"
echo
echo "Note: on macOS, Tauri signs every externalBin with the configured signing"
echo "identity during \`tauri build\`, and notarization covers them. Nothing extra"
echo "to do here -- but a sidecar copied in by hand after a build would not be"
echo "signed, and Gatekeeper would refuse to run it."
