use std::sync::atomic::{AtomicUsize, Ordering};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::store::Job;

pub struct JobEvents {
    app: AppHandle,
    active_downloads: AtomicUsize,
}

#[derive(Debug, Serialize, Clone)]
pub struct HealthStatus {
    pub queue_length: usize,
    pub active_transcriptions: usize,
    pub active_downloads: usize,
}

impl JobEvents {
    pub fn new(app: AppHandle) -> Self {
        Self {
            app,
            active_downloads: AtomicUsize::new(0),
        }
    }

    pub fn publish(&self, job: &Job) {
        let _ = self.app.emit(&format!("job-progress::{}", job.id), job);
        let _ = self.app.emit("job-progress", job);
    }

    pub fn track_download_start(&self) {
        self.active_downloads.fetch_add(1, Ordering::SeqCst);
    }

    pub fn track_download_end(&self) {
        let prev = self.active_downloads.fetch_sub(1, Ordering::SeqCst);
        if prev == 0 {
            self.active_downloads.store(0, Ordering::SeqCst);
        }
    }

    pub fn health(&self) -> HealthStatus {
        HealthStatus {
            queue_length: 0,
            active_transcriptions: 0,
            active_downloads: self.active_downloads.load(Ordering::SeqCst),
        }
    }
}
