#!/usr/bin/env bash
# Build, sign, and notarize the Remedy Transcription .dmg for macOS arm64.
#
# Requires env vars (set in your shell, or in a gitignored .env.local at repo root):
#   APPLE_ID         your Apple ID email
#   APPLE_PASSWORD   app-specific password from appleid.apple.com
#   APPLE_TEAM_ID    your Apple Team ID (e.g. 7XU3QW326W)

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f .env.local ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env.local
    set +a
fi

require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: $name is not set. Set it in your shell or in .env.local at the repo root." >&2
        exit 1
    fi
}

require_var APPLE_ID
require_var APPLE_PASSWORD
require_var APPLE_TEAM_ID

if [[ ! -f src-tauri/binaries/yt-dlp-aarch64-apple-darwin ]]; then
    echo "Fetching arm64 sidecars..."
    ./scripts/fetch-sidecars.sh aarch64-apple-darwin
fi

# Ours, so it is compiled rather than downloaded. Always rebuilt: a stale sidecar
# silently shipping against new app code is not worth the seconds saved.
./scripts/build-diarize-sidecar.sh aarch64-apple-darwin

echo "Building signed + notarized .dmg for aarch64-apple-darwin..."
npm run tauri -- build --target aarch64-apple-darwin

BUNDLE_DIR="src-tauri/target/aarch64-apple-darwin/release/bundle"
APP_PATH="$(find "$BUNDLE_DIR/macos" -maxdepth 1 -name '*.app' | head -1)"
DMG_PATH="$(find "$BUNDLE_DIR/dmg" -maxdepth 1 -name '*.dmg' | head -1)"

echo
echo "Build complete:"
echo "  app: $APP_PATH"
echo "  dmg: $DMG_PATH"
echo
echo "Notarizing the .dmg (Tauri only notarizes the .app inside)..."
xcrun notarytool submit "$DMG_PATH" \
    --apple-id "$APPLE_ID" \
    --password "$APPLE_PASSWORD" \
    --team-id "$APPLE_TEAM_ID" \
    --wait
echo
echo "Stapling the .dmg..."
xcrun stapler staple "$DMG_PATH"
echo
echo "Verifying app signature..."
codesign --verify --deep --strict --verbose=2 "$APP_PATH"
echo
echo "Verifying Gatekeeper acceptance on .app..."
spctl --assess --type execute --verbose=2 "$APP_PATH"
echo
echo "Verifying Gatekeeper acceptance on .dmg..."
spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG_PATH"
echo
echo "Done. Ship $DMG_PATH"
