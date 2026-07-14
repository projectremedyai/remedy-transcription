import { TaskMode } from "../config/transcription";

export interface TranscriptionSegment {
    start: number;
    end: number;
    text: string;
}

export interface Job {
    id: string;
    source_type: "file" | "youtube";
    source_key: string;
    status: "downloading" | "extracting" | "ready" | "completed" | "failed";
    progress: number;
    cache_hit: boolean;
    error: string | null;
    filename: string | null;
    audio_url: string | null;
    audio_mime_type: string | null;
    model_id: string;
    task: TaskMode;
    language: string;
    segments: TranscriptionSegment[];
    full_text: string | null;
}

export interface FileJobRequest {
    /**
     * A real filesystem path — from the Tauri dialog or a file drop, never a
     * browser `File` (which has no path).
     *
     * Rust derives the filename, the size and the content hash from it, and runs
     * the same ffmpeg pass over it that YouTube downloads get. The webview used
     * to send a hash it computed itself over an `ArrayBuffer` of the file, which
     * meant Rust never saw the audio at all.
     */
    path: string;
    model_id: string;
    task: TaskMode;
    language: string;
}

export interface YouTubeJobRequest {
    url: string;
    model_id: string;
    task: TaskMode;
    language: string;
}

export interface PersistTranscriptRequest {
    model_id: string;
    task: TaskMode;
    language: string;
    full_text: string;
    segments: TranscriptionSegment[];
}

export interface QueueStatus {
    position: number;
    total_in_queue: number;
    estimated_wait_minutes: number | null;
}

export interface HealthResponse {
    status: string;
    queue_length: number;
    active_transcriptions: number;
    active_downloads: number;
}

export interface ModelStatusItem {
    model_id: string;
    ready: boolean;
    path: string;
}

export interface ModelStatusResponse {
    models_ready: boolean;
    missing_models: string[];
    items: ModelStatusItem[];
}
