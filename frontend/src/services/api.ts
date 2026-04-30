import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { save } from "@tauri-apps/plugin-dialog";
import { convertFileSrc } from "@tauri-apps/api/core";

import { generateJson, generateSrt, generateTxt } from "../lib/srtGenerator";
import {
    FileJobRequest,
    HealthResponse,
    Job,
    ModelStatusResponse,
    PersistTranscriptRequest,
    QueueStatus,
    YouTubeJobRequest,
} from "./types";

export type {
    FileJobRequest,
    HealthResponse,
    Job,
    ModelStatusItem,
    ModelStatusResponse,
    PersistTranscriptRequest,
    QueueStatus,
    TranscriptionSegment,
    YouTubeJobRequest,
} from "./types";

class TauriApiClient {
    async createFileJob(payload: FileJobRequest): Promise<Job> {
        return invoke<Job>("create_file_job", { request: payload });
    }

    async createYouTubeJob(payload: YouTubeJobRequest): Promise<Job> {
        return invoke<Job>("create_youtube_job", { request: payload });
    }

    async getJob(jobId: string): Promise<Job> {
        return invoke<Job>("get_job", { jobId });
    }

    async persistTranscript(
        jobId: string,
        payload: PersistTranscriptRequest,
    ): Promise<Job> {
        return invoke<Job>("persist_transcript", {
            jobId,
            request: payload,
        });
    }

    async getQueueStatus(): Promise<QueueStatus> {
        return invoke<QueueStatus>("queue_status");
    }

    async getHealth(): Promise<HealthResponse> {
        return invoke<HealthResponse>("health");
    }

    async getModelStatus(): Promise<ModelStatusResponse> {
        return invoke<ModelStatusResponse>("list_models");
    }

    subscribeToProgress(
        jobId: string,
        onMessage: (data: Job) => void,
        onError?: (error: Error) => void,
    ): () => void {
        let unlisten: UnlistenFn | null = null;
        let cancelled = false;

        listen<Job>(`job-progress::${jobId}`, (event) => {
            onMessage(event.payload);
            if (
                event.payload.status === "completed" ||
                event.payload.status === "failed"
            ) {
                cleanup();
            }
        })
            .then((fn) => {
                if (cancelled) {
                    fn();
                } else {
                    unlisten = fn;
                }
            })
            .catch((err) => {
                if (onError) {
                    onError(
                        err instanceof Error
                            ? err
                            : new Error("Failed to subscribe to job events"),
                    );
                }
            });

        const cleanup = () => {
            cancelled = true;
            if (unlisten) {
                unlisten();
                unlisten = null;
            }
        };

        return cleanup;
    }

    async getAudioUrl(jobId: string): Promise<string> {
        const path = await invoke<string>("get_prepared_audio_path", {
            jobId,
        });
        return convertFileSrc(path);
    }

    async exportTranscript(
        jobId: string,
        format: "srt" | "txt" | "json",
        segments: import("./types").TranscriptionSegment[],
        suggestedName: string,
    ): Promise<string | null> {
        const filters = {
            srt: { name: "SubRip Subtitle", extensions: ["srt"] },
            txt: { name: "Plain Text", extensions: ["txt"] },
            json: { name: "JSON", extensions: ["json"] },
        }[format];

        const destination = await save({
            defaultPath: `${suggestedName}.${format}`,
            filters: [filters],
        });
        if (!destination) {
            return null;
        }

        const content =
            format === "srt"
                ? generateSrt(segments)
                : format === "txt"
                    ? generateTxt(segments)
                    : generateJson(segments);

        await invoke("export_transcript", {
            request: {
                job_id: jobId,
                format,
                destination,
                content,
            },
        });

        return destination;
    }
}

export const api = new TauriApiClient();
