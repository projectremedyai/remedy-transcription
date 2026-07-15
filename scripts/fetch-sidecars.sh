#!/usr/bin/env bash
# Fetch yt-dlp / ffmpeg / ffprobe binaries into src-tauri/binaries/
# with the Tauri target-triple naming convention, plus the speaker-diarization
# ONNX models into models/diarization/.
#
# Usage:
#   ./scripts/fetch-sidecars.sh                 # detects current host triple
#   ./scripts/fetch-sidecars.sh aarch64-apple-darwin
#   ./scripts/fetch-sidecars.sh x86_64-pc-windows-msvc
#   ./scripts/fetch-sidecars.sh --models-only   # just the diarization models
#   ./scripts/fetch-sidecars.sh --skip-models   # just the binaries
#   ./scripts/fetch-sidecars.sh --force         # re-download models even if present

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

WANT_BINARIES=1
WANT_MODELS=1
FORCE=0
TRIPLE=""

for arg in "$@"; do
    case "$arg" in
        --models-only) WANT_BINARIES=0 ;;
        --skip-models) WANT_MODELS=0 ;;
        --force) FORCE=1 ;;
        -*) echo "Unknown option: $arg" >&2; exit 1 ;;
        *) TRIPLE="$arg" ;;
    esac
done

TRIPLE="${TRIPLE:-$(rustc -vV | awk '/host:/ {print $2}')}"
DEST="src-tauri/binaries"
MODELS="models/diarization"

# Staging dirs, cleaned up on ANY exit -- including the `exit 1` paths below.
#
# These used to be `trap ... RETURN` inside the function, which does NOT fire on
# `exit`: a bad tarball left a `.fetch.XXXXXX` directory sitting inside models/
# forever. An EXIT trap covers both the ordinary return and the failure paths.
BIN_TMP=""
MODELS_TMP=""
cleanup() {
    [[ -n "$BIN_TMP" ]] && rm -rf "$BIN_TMP"
    [[ -n "$MODELS_TMP" ]] && rm -rf "$MODELS_TMP"
    return 0
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Speaker-diarization models.
#
# Two models, both mandatory: pyannote segmentation-3.0 finds the speech turns,
# WeSpeaker CAM++ embeds them so they can be clustered into speakers.
#
# Both are redistributed from sherpa-onnx's OWN GitHub releases, so they are
# ungated -- no Hugging Face token, no click-through licence. They are platform
# independent (the same .onnx files serve macOS and Windows).
#
# NOTE: "speaker-recongition-models" is misspelled upstream. That typo is the
# real tag; the correctly spelled URL 404s. Do not "fix" it.
# ---------------------------------------------------------------------------
SEG_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx"

# The exact size of each model, in bytes. Both come from fixed, immutable GitHub
# release assets, so these numbers are stable.
#
# They are the ONLY thing standing between a user and a permanently broken
# checkout. See `model_is_complete` below for why existence is not enough.
SEG_BYTES=5992913
EMB_BYTES=29292684

# NEVER curl -o straight to a model's final path, and NEVER treat "the file
# exists" as "the file is good".
#
# A model file is only ever "present" if it is COMPLETE. Downloading in place
# means a Ctrl-C, a dropped connection or a full disk leaves a truncated file at
# the real path -- and a bare `[[ -f ]]` guard then cheerfully reports it
# "already present" on every subsequent run, FOREVER. The user's only clue is the
# app dying: a truncated ONNX model does not error, it makes ONNX Runtime throw a
# C++ exception that aborts the process with SIGABRT.
#
# Staging + `mv` (below) stops NEW corruption. It does nothing for the people who
# already have a truncated model from the old script -- and they are precisely
# the ones seeing the SIGABRT. So the guard checks the size too, and re-downloads
# on any mismatch. A stale or half-written model now self-heals on the next run
# instead of being wrong forever.

# Is the file on disk exactly the model we expect? Used both to decide whether a
# download can be skipped AND to verify a fresh one -- a download that lands at
# the wrong size must be an error, not a silently-accepted new baseline.
model_size_ok() {
    local path="$1" want="$2" have
    [[ -f "$path" ]] || return 1
    have="$(wc -c < "$path" | tr -d '[:space:]')"
    [[ "$have" == "$want" ]]
}

# Can we skip downloading this one? `--force` says no regardless.
model_is_complete() {
    local path="$1" want="$2"
    [[ "$FORCE" -eq 1 ]] && return 1
    [[ -f "$path" ]] || return 1

    if ! model_size_ok "$path" "$want"; then
        echo "  $path is $(wc -c < "$path" | tr -d '[:space:]') bytes, expected $want" >&2
        echo "  -- truncated or stale (this is what produces the SIGABRT). Re-downloading." >&2
        return 1
    fi
    return 0
}

fetch_models() {
    local seg_model emb_model
    seg_model="$MODELS/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    emb_model="$MODELS/wespeaker_en_voxceleb_CAM++.onnx"

    mkdir -p "$MODELS"
    # Inside $MODELS, so the mv is a same-filesystem rename (i.e. atomic: the
    # final path only ever holds a complete file, or no file).
    MODELS_TMP="$(mktemp -d "$MODELS/.fetch.XXXXXX")"

    if model_is_complete "$seg_model" "$SEG_BYTES"; then
        echo "Segmentation model already present and complete -> $seg_model"
    else
        echo "Fetching pyannote segmentation-3.0 (~7 MB) -> $seg_model"
        curl -fL --progress-bar -o "$MODELS_TMP/seg.tar.bz2" "$SEG_URL"
        # Extraction is its own staging step: a half-unpacked tree is as bad as
        # a half-downloaded file.
        mkdir -p "$MODELS_TMP/seg"
        tar -xjf "$MODELS_TMP/seg.tar.bz2" -C "$MODELS_TMP/seg"
        [[ -f "$MODELS_TMP/seg/sherpa-onnx-pyannote-segmentation-3-0/model.onnx" ]] \
            || { echo "The segmentation archive did not contain model.onnx" >&2; exit 1; }
        rm -rf "$MODELS/sherpa-onnx-pyannote-segmentation-3-0"
        mv "$MODELS_TMP/seg/sherpa-onnx-pyannote-segmentation-3-0" "$MODELS/"

        model_size_ok "$seg_model" "$SEG_BYTES" || {
            echo "The segmentation model is the wrong size even after a fresh download." >&2
            echo "Either the upstream asset changed (update SEG_BYTES) or the download is bad." >&2
            exit 1
        }
    fi

    if model_is_complete "$emb_model" "$EMB_BYTES"; then
        echo "Embedding model already present and complete -> $emb_model"
    else
        echo "Fetching WeSpeaker CAM++ embeddings (~28 MB) -> $emb_model"
        curl -fL --progress-bar -o "$MODELS_TMP/emb.onnx" "$EMB_URL"
        mv "$MODELS_TMP/emb.onnx" "$emb_model"

        model_size_ok "$emb_model" "$EMB_BYTES" || {
            echo "The embedding model is the wrong size even after a fresh download." >&2
            echo "Either the upstream asset changed (update EMB_BYTES) or the download is bad." >&2
            exit 1
        }
    fi

    # The EXIT trap would do this anyway; doing it here keeps the staging
    # directory out of the listing below.
    rm -rf "$MODELS_TMP"
    MODELS_TMP=""

    prune_seg_extras

    echo
    echo "Diarization models in place:"
    ls -la "$MODELS" "$MODELS/sherpa-onnx-pyannote-segmentation-3-0"
}

# WHAT IS IN models/diarization IS WHAT SHIPS.
#
# `tauri.conf.json` bundles the DIRECTORY (`"../models/diarization"`), not a
# hand-listed file map -- that is what lets a checkout with no models still
# `cargo check` (see src-tauri/build.rs). The cost of a directory resource is
# that it is not selective: every file in here is copied into the .app.
#
# The segmentation tarball unpacks with an unused 1.5 MB int8 model, a README and
# eight Python export/demo scripts alongside the model.onnx the engine actually
# loads. None of them belong in a shipped bundle. So the directory is pruned to
# exactly what the app opens (model.onnx) plus what redistribution requires
# (LICENSE).
#
# Unconditional, not part of the download branch: it also cleans up trees that
# were populated by the older script, which left all of it in place.
prune_seg_extras() {
    local seg_dir="$MODELS/sherpa-onnx-pyannote-segmentation-3-0" entry name
    [[ -d "$seg_dir" ]] || return 0

    for entry in "$seg_dir"/* "$seg_dir"/.??*; do
        [[ -e "$entry" ]] || continue
        name="$(basename "$entry")"
        case "$name" in
            model.onnx|LICENSE) ;;
            *) rm -rf "$entry" ;;
        esac
    done
}

if [[ "$WANT_BINARIES" -eq 0 ]]; then
    fetch_models
    exit 0
fi

mkdir -p "$DEST"

ext=""
case "$TRIPLE" in
    *-pc-windows-*) ext=".exe" ;;
esac

ytdlp_url=""
ffmpeg_url=""
ffprobe_url=""

case "$TRIPLE" in
    aarch64-apple-darwin)
        ytdlp_url="https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos"
        ffmpeg_url="https://www.osxexperts.net/ffmpeg81arm.zip"
        ffprobe_url="https://www.osxexperts.net/ffprobe81arm.zip"
        ;;
    x86_64-apple-darwin)
        ytdlp_url="https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos"
        ffmpeg_url="https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip"
        ffprobe_url="https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip"
        ;;
    x86_64-pc-windows-msvc)
        ytdlp_url="https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
        ffmpeg_url=""   # see note below
        ffprobe_url=""
        ;;
    *)
        echo "Unsupported triple: $TRIPLE" >&2
        exit 1
        ;;
esac

BIN_TMP="$(mktemp -d)"

# An ABSOLUTE mode, not a `+`/`u+` adjustment. Two different bugs meet here, and
# only 755 fixes both:
#
#   - The WRITE bit, and not just execute: the macOS ffmpeg/ffprobe zips unpack as
#     mode 555, and `tauri build` runs `xattr -crs` over the finished .app to strip
#     quarantine attributes. Clearing an extended attribute needs WRITE permission
#     on the file -- even for its owner -- so a read-only sidecar makes the bundler
#     fail with a bare "failed to run xattr" that names no file and no reason.
#
#   - GROUP and OTHER execute, and not just the owner's: `curl -o` creates a fresh
#     file 644, so a `chmod u+wx` on a freshly downloaded yt-dlp yields 744 --
#     -rwxr--r--, executable by its OWNER ALONE. Tauri's `fs::copy` preserves that
#     mode into Contents/MacOS/yt-dlp, so a release .app would ship a yt-dlp that
#     nobody but the building user can run: every admin-installed, MDM-deployed or
#     otherwise multi-user install fails YouTube ingest with EACCES.
#
#     This one hid because `curl -o` over an EXISTING file keeps the old mode. Any
#     tree that already had 755 binaries kept them, so the bug was invisible to
#     every build that did not start from a genuinely empty src-tauri/binaries/.
#     If you touch these lines, verify with `rm -rf src-tauri/binaries` first.
echo "Fetching yt-dlp -> $DEST/yt-dlp-${TRIPLE}${ext}"
curl -fsSL --output "$DEST/yt-dlp-${TRIPLE}${ext}" "$ytdlp_url"
chmod 755 "$DEST/yt-dlp-${TRIPLE}${ext}"

if [[ -n "$ffmpeg_url" ]]; then
    echo "Fetching ffmpeg -> $DEST/ffmpeg-${TRIPLE}${ext}"
    if [[ "$ffmpeg_url" == *.zip ]]; then
        curl -fsSL -o "$BIN_TMP/ffmpeg.zip" "$ffmpeg_url"
        unzip -o -j "$BIN_TMP/ffmpeg.zip" -d "$BIN_TMP"
        cp "$BIN_TMP/ffmpeg" "$DEST/ffmpeg-${TRIPLE}${ext}"
    else
        curl -fsSL -o "$DEST/ffmpeg-${TRIPLE}${ext}" "$ffmpeg_url"
    fi
    chmod 755 "$DEST/ffmpeg-${TRIPLE}${ext}"
fi

if [[ -n "$ffprobe_url" ]]; then
    echo "Fetching ffprobe -> $DEST/ffprobe-${TRIPLE}${ext}"
    if [[ "$ffprobe_url" == *.zip ]]; then
        curl -fsSL -o "$BIN_TMP/ffprobe.zip" "$ffprobe_url"
        unzip -o -j "$BIN_TMP/ffprobe.zip" -d "$BIN_TMP"
        cp "$BIN_TMP/ffprobe" "$DEST/ffprobe-${TRIPLE}${ext}"
    else
        curl -fsSL -o "$DEST/ffprobe-${TRIPLE}${ext}" "$ffprobe_url"
    fi
    chmod 755 "$DEST/ffprobe-${TRIPLE}${ext}"
fi

echo
echo "Sidecars in place:"
ls -la "$DEST"

if [[ "$WANT_MODELS" -eq 1 ]]; then
    echo
    fetch_models
fi

if [[ ! -f "$DEST/diarize-sidecar-${TRIPLE}${ext}" ]]; then
    cat <<'NOTE'

Note: the diarization sidecar is COMPILED, not downloaded, so it is not fetched
here. `tauri build` and `tauri dev` will both fail until it exists. Build it with:
    ./scripts/build-diarize-sidecar.sh
NOTE
fi

if [[ "$TRIPLE" == *-pc-windows-* ]]; then
    cat <<'NOTE'

Note: Windows ffmpeg/ffprobe binaries are not auto-fetched. Download static
builds from https://www.gyan.dev/ffmpeg/builds/ (the "release essentials" zip),
extract ffmpeg.exe and ffprobe.exe, and rename them to:
    src-tauri/binaries/ffmpeg-x86_64-pc-windows-msvc.exe
    src-tauri/binaries/ffprobe-x86_64-pc-windows-msvc.exe
NOTE
fi
