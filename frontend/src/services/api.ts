import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { save } from "@tauri-apps/plugin-dialog";

import { ConsolidatedSegment } from "../lib/captionFormatter";
import {
    generateJson,
    generateSrt,
    generateTxt,
    generateVtt,
} from "../lib/srtGenerator";
import {
    FileJobRequest,
    HealthResponse,
    Job,
    ModelStatusResponse,
    PersistTranscriptRequest,
    QueueStatus,
    SpeakerNames,
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
    SpeakerNames,
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

    /**
     * Name a speaker: `SPEAKER_00` -> `"Alice"`.
     *
     * **A METADATA WRITE. It does not re-transcribe and does not re-diarize.**
     * The segments keep their opaque `SPEAKER_00` keys; the name is a row in
     * `transcript_speakers` that the renderer joins on. So renaming is instant,
     * reversible, and works long after the audio has been cleaned up.
     *
     * Keyed by JOB id, but stored against the TRANSCRIPT the job resolves to —
     * so the name survives to every future job for the same recording, including
     * the cache hit the user gets when they drop the same file in next week.
     *
     * Renaming the same speaker twice overwrites; it does not accumulate.
     *
     * Rejects a blank name, and a job that has no transcript yet.
     */
    async setSpeakerName(
        jobId: string,
        speakerKey: string,
        displayName: string,
    ): Promise<void> {
        await invoke("set_speaker_name", { jobId, speakerKey, displayName });
    }

    /**
     * Every name given to this job's speakers, keyed by the label the segments
     * carry. `{}` when nobody has renamed anyone — the normal case, and not an
     * error. A speaker with no entry renders as its own key.
     */
    async getSpeakerNames(jobId: string): Promise<SpeakerNames> {
        return invoke<SpeakerNames>("get_speaker_names", { jobId });
    }

    /**
     * `captions` must be the output of `consolidateSegments` — the very cues on
     * screen. The `ConsolidatedSegment` brand makes that a compile-time
     * requirement: raw model segments will not typecheck here.
     *
     * `names` is optional and threaded straight through to the generators
     * (`srtGenerator.ts`) unchanged — see their doc comments for what each
     * format does with it. Omitting it must reproduce today's undiarized
     * output exactly; passing it is what makes a renamed speaker actually show
     * up in an exported file (see final-review.md, B1).
     */
    async exportTranscript(
        jobId: string,
        format: "srt" | "vtt" | "txt" | "json",
        captions: ConsolidatedSegment[],
        suggestedName: string,
        names?: SpeakerNames,
    ): Promise<string | null> {
        const filters = {
            srt: { name: "SubRip Subtitle", extensions: ["srt"] },
            vtt: { name: "WebVTT", extensions: ["vtt"] },
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

        const generators = {
            srt: generateSrt,
            vtt: generateVtt,
            txt: generateTxt,
            json: generateJson,
        };
        const content = generators[format](captions, names);

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
