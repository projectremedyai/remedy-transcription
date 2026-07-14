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
    DiarizationOutcome,
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
    DiarizationOutcome,
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
     * Speaker turns for a job's already-prepared audio.
     *
     * **Returns an outcome, not a list.** Read the `status` — see
     * {@link DiarizationOutcome}. There is no `turns` to reach for until you have
     * narrowed to `"succeeded"`, and that is deliberate: a `"degraded"` result is
     * a broken engine, and rendering it as an empty turn list would show the user
     * "0 speakers" for a crash.
     *
     * This method rejects only when the REQUEST was wrong (unknown job, no
     * prepared audio, an impossible `numSpeakers`). Every engine failure —
     * missing model, dead sidecar, timeout — resolves as `"degraded"` instead,
     * because no failure of diarization is worth failing a transcript over.
     *
     * `numSpeakers` is a HINT: asking for 4 can still return 3, and omitting it
     * auto-detects.
     */
    async diarizeJob(
        jobId: string,
        numSpeakers?: number,
    ): Promise<DiarizationOutcome> {
        return invoke<DiarizationOutcome>("diarize_job", {
            jobId,
            numSpeakers: numSpeakers ?? null,
        });
    }

    /**
     * Kill the diarization sidecar for a job, if one is running.
     *
     * Abandoning a backend child is normally harmless — an orphaned ffmpeg just
     * finishes. Diarization is not like that: it is a CPU-bound ONNX child with
     * a 30-minute backstop timeout, so a cancelled run with nobody killing it
     * would pin a core for up to half an hour while the app sat there idle.
     *
     * Resolves `false` when the job was not diarizing, which is the common case
     * and is not an error.
     */
    async cancelDiarization(jobId: string): Promise<boolean> {
        return invoke<boolean>("cancel_diarization", { jobId });
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
     */
    async exportTranscript(
        jobId: string,
        format: "srt" | "vtt" | "txt" | "json",
        captions: ConsolidatedSegment[],
        suggestedName: string,
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
        const content = generators[format](captions);

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
