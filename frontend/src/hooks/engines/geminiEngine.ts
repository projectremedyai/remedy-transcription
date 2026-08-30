import { api } from "../../services/api";
import type { GeminiProgressEvent } from "../../services/types";
import type { EngineResult, EngineRunArgs, TranscriptionEngine } from "./types";

const PHASE_LABEL: Record<GeminiProgressEvent["phase"], string> = {
    slicing: "preparing-audio",
    uploading: "uploading",
    transcribing: "transcribing",
    stitching: "assembling",
};

export function createGeminiEngine(): TranscriptionEngine {
    return {
        id: "gemini",

        async run(args: EngineRunArgs): Promise<EngineResult> {
            const unsubscribe = api.subscribeToGeminiProgress(
                args.job.id,
                (event) => {
                    args.onProgress({
                        fraction: event.fraction,
                        status:
                            event.chunk_count > 1
                                ? `${PHASE_LABEL[event.phase]} (part ${
                                      event.chunk_index + 1
                                  } of ${event.chunk_count})`
                                : PHASE_LABEL[event.phase],
                    });
                },
            );

            try {
                const result = await api.transcribeWithGemini(args.job.id);
                return {
                    transcript: {
                        text: result.text,
                        // Always empty, never undefined: `chunks` is the
                        // streaming-preview fallback and there is no stream
                        // here. `words` carries the real times.
                        chunks: [],
                        words: result.words.map((word) => ({
                            text: word.text,
                            start: word.start,
                            end: word.end,
                        })),
                    },
                    speakers: result.speakers,
                    audioDuration: result.audio_duration,
                };
            } finally {
                // In `finally`, not after the await: a rejected run must not
                // leave a listener attached to a job nobody is watching.
                unsubscribe();
            }
        },

        abandon(jobId: string) {
            void api.cancelGeminiTranscription(jobId).catch(() => {
                // Best effort. The UI is being torn down either way.
            });
        },
    };
}
