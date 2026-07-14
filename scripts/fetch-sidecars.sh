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

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

WANT_BINARIES=1
WANT_MODELS=1
TRIPLE=""

for arg in "$@"; do
    case "$arg" in
        --models-only) WANT_BINARIES=0 ;;
        --skip-models) WANT_MODELS=0 ;;
        -*) echo "Unknown option: $arg" >&2; exit 1 ;;
        *) TRIPLE="$arg" ;;
    esac
done

TRIPLE="${TRIPLE:-$(rustc -vV | awk '/host:/ {print $2}')}"
DEST="src-tauri/binaries"
MODELS="models/diarization"

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

fetch_models() {
    local tmp seg_model emb_model
    seg_model="$MODELS/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    emb_model="$MODELS/wespeaker_en_voxceleb_CAM++.onnx"

    mkdir -p "$MODELS"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN

    if [[ -f "$seg_model" ]]; then
        echo "Segmentation model already present -> $seg_model"
    else
        echo "Fetching pyannote segmentation-3.0 (~7 MB) -> $seg_model"
        curl -fL --progress-bar -o "$tmp/seg.tar.bz2" "$SEG_URL"
        tar -xjf "$tmp/seg.tar.bz2" -C "$MODELS"
        [[ -f "$seg_model" ]] || { echo "Expected $seg_model after extraction" >&2; exit 1; }
    fi

    if [[ -f "$emb_model" ]]; then
        echo "Embedding model already present -> $emb_model"
    else
        echo "Fetching WeSpeaker CAM++ embeddings (~28 MB) -> $emb_model"
        curl -fL --progress-bar -o "$emb_model" "$EMB_URL"
    fi

    echo
    echo "Diarization models in place:"
    ls -la "$MODELS" "$MODELS/sherpa-onnx-pyannote-segmentation-3-0"
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

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Fetching yt-dlp -> $DEST/yt-dlp-${TRIPLE}${ext}"
curl -fsSL --output "$DEST/yt-dlp-${TRIPLE}${ext}" "$ytdlp_url"
chmod +x "$DEST/yt-dlp-${TRIPLE}${ext}"

if [[ -n "$ffmpeg_url" ]]; then
    echo "Fetching ffmpeg -> $DEST/ffmpeg-${TRIPLE}${ext}"
    if [[ "$ffmpeg_url" == *.zip ]]; then
        curl -fsSL -o "$tmp/ffmpeg.zip" "$ffmpeg_url"
        unzip -o -j "$tmp/ffmpeg.zip" -d "$tmp"
        cp "$tmp/ffmpeg" "$DEST/ffmpeg-${TRIPLE}${ext}"
    else
        curl -fsSL -o "$DEST/ffmpeg-${TRIPLE}${ext}" "$ffmpeg_url"
    fi
    chmod +x "$DEST/ffmpeg-${TRIPLE}${ext}"
fi

if [[ -n "$ffprobe_url" ]]; then
    echo "Fetching ffprobe -> $DEST/ffprobe-${TRIPLE}${ext}"
    if [[ "$ffprobe_url" == *.zip ]]; then
        curl -fsSL -o "$tmp/ffprobe.zip" "$ffprobe_url"
        unzip -o -j "$tmp/ffprobe.zip" -d "$tmp"
        cp "$tmp/ffprobe" "$DEST/ffprobe-${TRIPLE}${ext}"
    else
        curl -fsSL -o "$DEST/ffprobe-${TRIPLE}${ext}" "$ffprobe_url"
    fi
    chmod +x "$DEST/ffprobe-${TRIPLE}${ext}"
fi

echo
echo "Sidecars in place:"
ls -la "$DEST"

if [[ "$WANT_MODELS" -eq 1 ]]; then
    echo
    fetch_models
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
