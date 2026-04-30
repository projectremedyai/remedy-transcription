#!/usr/bin/env bash
# Fetch yt-dlp / ffmpeg / ffprobe binaries into src-tauri/binaries/
# with the Tauri target-triple naming convention.
#
# Usage:
#   ./scripts/fetch-sidecars.sh                 # detects current host triple
#   ./scripts/fetch-sidecars.sh aarch64-apple-darwin
#   ./scripts/fetch-sidecars.sh x86_64-pc-windows-msvc

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

TRIPLE="${1:-$(rustc -vV | awk '/host:/ {print $2}')}"
DEST="src-tauri/binaries"
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

if [[ "$TRIPLE" == *-pc-windows-* ]]; then
    cat <<'NOTE'

Note: Windows ffmpeg/ffprobe binaries are not auto-fetched. Download static
builds from https://www.gyan.dev/ffmpeg/builds/ (the "release essentials" zip),
extract ffmpeg.exe and ffprobe.exe, and rename them to:
    src-tauri/binaries/ffmpeg-x86_64-pc-windows-msvc.exe
    src-tauri/binaries/ffprobe-x86_64-pc-windows-msvc.exe
NOTE
fi
