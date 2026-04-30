use std::path::{Path, PathBuf};

use tauri::{AppHandle, Manager};

pub const PREPARED_AUDIO_TTL_HOURS: i64 = 24 * 7;
pub const MAX_CONCURRENT_DOWNLOADS: usize = 2;
pub const MAX_DURATION_HOURS: u64 = 2;

pub fn resolve_app_data_dir(app: &AppHandle) -> anyhow::Result<PathBuf> {
    let dir = app.path().app_data_dir()?;
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn ensure_runtime_dirs(app_data_dir: &Path) -> anyhow::Result<()> {
    for sub in ["audio", "downloads", "models"] {
        std::fs::create_dir_all(app_data_dir.join(sub))?;
    }
    Ok(())
}

pub fn database_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("remedy-transcription.db")
}

pub fn audio_dir(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("audio")
}

pub fn downloads_dir(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("downloads")
}

pub fn models_dir(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("models")
}
