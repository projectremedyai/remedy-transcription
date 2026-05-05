# Remedy Transcription

A standalone native desktop app for macOS and Windows that transcribes video and audio locally. Paste a YouTube URL or drop a file; it transcribes on your machine and exports SRT / TXT / JSON.

No server, no cloud, no Python runtime. The whole pipeline — yt-dlp download, ffmpeg extraction, ONNX Whisper inference — runs in-process inside the installed app.

Why "local-first": YouTube blocks data-center IPs, so a self-hosted (VPS) version of this kept hitting 403s. Running on the user's own machine sidesteps that entirely — and with ONNX Whisper running client-side via Hugging Face Transformers.js (WebGPU when available, WASM otherwise), there's no backend to host.

## Stack

- **Tauri 2** (Rust shell, single-binary installer)
- **React + Vite** (frontend, loaded into Tauri's webview)
- **@huggingface/transformers** (ONNX Whisper inference in a Web Worker)
- **yt-dlp / ffmpeg / ffprobe** (bundled as Tauri sidecar binaries)
- **rusqlite** (transcript cache, in OS app-data dir)

## Requirements

- macOS 11+ (Apple Silicon) or Windows 10+
- Internet on first run per Whisper model (models download from Hugging Face and cache in IndexedDB; subsequent runs work offline)
- Rust toolchain + Node 18+ to build from source

## Run from source

```bash
# 1. Fetch yt-dlp / ffmpeg / ffprobe sidecars into src-tauri/binaries/.
./scripts/fetch-sidecars.sh

# 2. Install root and frontend JS deps from lockfiles.
npm ci
npm --prefix frontend ci

# 3. Run the dev build (Vite + Tauri, hot reload).
npm run dev
```

## Build an installer

```bash
./scripts/fetch-sidecars.sh
npm ci
npm --prefix frontend ci
npm run build
```

Output: `src-tauri/target/release/bundle/` — `.dmg` on macOS, `.msi` / `.exe` on Windows. The installer is fully standalone; end users don't need Node, Rust, or Python on their machine.

## CI and build checks

GitHub Actions runs the checked build path on macOS:

```bash
npm ci
npm --prefix frontend ci
npm run frontend:build
npm --prefix frontend run lint
./scripts/fetch-sidecars.sh
cargo check --manifest-path src-tauri/Cargo.toml
cargo test --manifest-path src-tauri/Cargo.toml
```

`scripts/fetch-sidecars.sh` runs in CI, not only during release packaging, because Tauri validates the configured `externalBin` sidecars during `cargo check` and `cargo test`. Release jobs should still run it for the target being packaged before `npm run build`; Windows ffmpeg/ffprobe setup remains manual as noted by the script.

## Where your data lives

Everything stays on the local machine:

- Transcripts and job history → SQLite at `~/Library/Application Support/com.remedy.transcription/` (macOS) or `%APPDATA%\com.remedy.transcription\` (Windows)
- Cached YouTube audio → `audio/` next to the DB (7-day TTL)
- Whisper models → the webview's IndexedDB

Nothing is uploaded. The only outbound traffic is to YouTube (via yt-dlp) and to Hugging Face (model downloads, first run only).

## YouTube Usage Disclaimer

Remedy Transcription can use third-party tooling to access and download media from YouTube for local transcription. YouTube's Terms of Service restrict downloading, reproducing, or otherwise using YouTube content except where expressly authorized by YouTube or with permission from YouTube and the applicable rights holders.

You are solely responsible for how you use this feature. Only use it with videos you own, videos you have permission to download and transcribe, or content you are otherwise legally authorized to use. The maintainers of this project are not responsible for violations of YouTube's Terms of Service, copyright law, account restrictions, takedowns, suspensions, or any other consequences resulting from your use of this software.

This software is provided for lawful, authorized transcription workflows only.

## Architecture

See [CLAUDE.md](./CLAUDE.md) for the full breakdown.

```
React webview ←─Tauri IPC─→ Rust core ──spawn──> yt-dlp + ffmpeg
      │
      └─ Web Worker → Transformers.js → ONNX Whisper
```

## What's where

| Path | What |
|------|------|
| `frontend/src/` | React app, services, worker, caption formatter, SRT generator |
| `src-tauri/src/` | Rust commands, SQLite store, sidecar wrappers, event emitter |
| `src-tauri/binaries/` | Bundled `yt-dlp` / `ffmpeg` / `ffprobe` per target triple |
| `src-tauri/icons/` | Generated app icons |
| `src-tauri/tauri.conf.json` | Bundle config, sidecar registration, permissions |

## License

MIT. See `LICENSE`.
