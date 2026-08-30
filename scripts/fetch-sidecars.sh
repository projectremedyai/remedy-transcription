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

TRIPLE=""

for arg in "$@"; do
    case "$arg" in
        -*) echo "Unknown option: $arg" >&2; exit 1 ;;
        *) TRIPLE="$arg" ;;
    esac
done

TRIPLE="${TRIPLE:-$(rustc -vV | awk '/host:/ {print $2}')}"
DEST="src-tauri/binaries"

# Staging dir, cleaned up on ANY exit -- including the `exit 1` paths below.
#
# This used to be `trap ... RETURN` inside a function, which does NOT fire on
# `exit`: a bad download could leave a temp directory sitting around forever.
# An EXIT trap covers both the ordinary return and the failure paths.
BIN_TMP=""
cleanup() {
    [[ -n "$BIN_TMP" ]] && rm -rf "$BIN_TMP"
    return 0
}
trap cleanup EXIT

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
    x86_64-unknown-linux-gnu)
        ytdlp_url="https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux"
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

if [[ "$TRIPLE" == *-pc-windows-* ]]; then
    cat <<'NOTE'

Note: Windows ffmpeg/ffprobe binaries are not auto-fetched. Download static
builds from https://www.gyan.dev/ffmpeg/builds/ (the "release essentials" zip),
extract ffmpeg.exe and ffprobe.exe, and rename them to:
    src-tauri/binaries/ffmpeg-x86_64-pc-windows-msvc.exe
    src-tauri/binaries/ffprobe-x86_64-pc-windows-msvc.exe
NOTE
fi

# Same deliberate gap as Windows, for the same reason: there is no single
# ffmpeg build that is right for every distro, and the packaged one usually is.
# Not optional, though -- tauri-build fails `cargo check` on a missing
# externalBin for the target triple, before anything is compiled.
if [[ "$TRIPLE" == *-linux-* ]]; then
    cat <<'NOTE'

Note: Linux ffmpeg/ffprobe binaries are not auto-fetched. Install them from
your distribution (e.g. `apt-get install ffmpeg`) and copy them in:
    cp "$(command -v ffmpeg)"  src-tauri/binaries/ffmpeg-x86_64-unknown-linux-gnu
    cp "$(command -v ffprobe)" src-tauri/binaries/ffprobe-x86_64-unknown-linux-gnu
NOTE
fi
