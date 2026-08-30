import { api } from "../../services/api";
import type { WorkerTranscript } from "../../lib/workerTranscript";
import {
    LOCAL_ENGINE_NO_SPEAKERS,
    type EngineResult,
    type EngineRunArgs,
    type TranscriptionEngine,
} from "./types";
import type { ResolvedModelConfig } from "../../config/transcription";

/**
 * Fetch the prepared WAV and decode it.
 *
 * The WAV is 16 kHz mono and the `AudioContext` is opened at exactly 16 kHz, so
 * nothing is resampled and `duration` is the true duration of the source.
 */
export async function loadPreparedAudio(jobId: string): Promise<AudioBuffer> {
    const audioUrl = await api.getAudioUrl(jobId);
    const audioResponse = await fetch(audioUrl);
    if (!audioResponse.ok) {
        throw new Error("Failed to load prepared audio");
    }
    const arrayBuffer = await audioResponse.arrayBuffer();
    const context = new AudioContext({ sampleRate: 16000 });
    try {
        return await context.decodeAudioData(arrayBuffer);
    } finally {
        void context.close();
    }
}

export interface LocalEngineDeps {
    /**
     * The hook's existing worker round-trip, passed in rather than rebuilt.
     * The `pendingWorkerRef` / `useWorker` / `claimRun` machinery it depends on
     * stays in the hook: it is not engine-specific and it is the code whose
     * comments record three rounds of getting the run-token guard right.
     */
    runWorkerTranscription: (
        audioBuffer: AudioBuffer,
        config: ResolvedModelConfig,
        runId: number,
        filename?: string,
    ) => Promise<WorkerTranscript>;
    loadAudio?: (jobId: string) => Promise<AudioBuffer>;
}

export function createLocalEngine(deps: LocalEngineDeps): TranscriptionEngine {
    const loadAudio = deps.loadAudio ?? loadPreparedAudio;

    return {
        id: "local",

        async run(args: EngineRunArgs): Promise<EngineResult> {
            if (args.config.device === null) {
                throw new Error(
                    "the local engine needs a device (webgpu or wasm); " +
                        "resolveModelConfig returned null, which only happens " +
                        "for a cloud engine",
                );
            }

            args.onProgress({ fraction: null, status: "loading-audio" });
            const audioBuffer = await loadAudio(args.job.id);

            const transcript = await deps.runWorkerTranscription(
                audioBuffer,
                args.config,
                args.runId,
                args.job.filename || undefined,
            );

            return {
                transcript,
                // Not "we found one speaker" -- there is no diarizer here at
                // all. The distinction is what lets the UI stay silent rather
                // than claiming a result it never computed.
                speakers: {
                    status: "unavailable",
                    reason: LOCAL_ENGINE_NO_SPEAKERS,
                },
                audioDuration: audioBuffer.duration,
            };
        },

        abandon() {
            // No-op by design: `claimRun` terminates the worker, and only a
            // terminate stops a transformers.js inference.
        },
    };
}
