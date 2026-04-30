use std::sync::Arc;

use tauri::Manager;

mod commands;
mod events;
mod paths;
mod sidecar;
mod store;

use events::JobEvents;
use store::Store;

pub struct AppState {
    pub store: Arc<Store>,
    pub events: Arc<JobEvents>,
    pub download_semaphore: Arc<tokio::sync::Semaphore>,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            let app_data_dir = paths::resolve_app_data_dir(&app.handle())?;
            paths::ensure_runtime_dirs(&app_data_dir)?;

            let store = Arc::new(Store::open(&paths::database_path(&app_data_dir))?);
            let events = Arc::new(JobEvents::new(app.handle().clone()));
            let download_semaphore =
                Arc::new(tokio::sync::Semaphore::new(paths::MAX_CONCURRENT_DOWNLOADS));

            store.cleanup_expired_audio(paths::PREPARED_AUDIO_TTL_HOURS, &paths::audio_dir(&app_data_dir));

            app.manage(AppState {
                store,
                events,
                download_semaphore,
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::create_youtube_job,
            commands::create_file_job,
            commands::get_job,
            commands::get_prepared_audio_path,
            commands::persist_transcript,
            commands::queue_status,
            commands::health,
            commands::list_models,
            commands::resolve_models_dir,
            commands::export_transcript,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
