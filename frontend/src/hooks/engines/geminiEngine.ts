import { api } from "../../services/api";
import type { GeminiProgressEvent } from "../../services/types";
import type { EngineResult, EngineRunArgs, TranscriptionEngine } from "./types";

/**
 * PROSE, not kebab-case machine strings.
 *
 * `EngineProgress.status` is declared "user-facing and already specific --
 * shown verbatim", and `AudioManager` now honours that: anything it does not
 * recognise as one of its own job statuses goes straight to the screen. So
 * these strings ARE the UI. Two consequences worth keeping:
 *
 *   - Sentence-cased and ellipsed, matching `AudioManager`'s own job-status
 *     copy ("Extracting audio...").
 *   - Never "in your browser". This is the cloud engine; the one axis the
 *     spec insists is stated plainly is where the user's audio actually is.
 *     Emitting a bare "transcribing" here (as this did) also COLLIDED with
 *     the local worker's own `transcribing` status, which renders
 *     "Transcribing in your browser..." -- the exact wrong sentence.
 *
 * No `uploading` arm: Rust emits `slicing`, `transcribing` and `stitching`
 * only, so an `uploading` label had no producer and could never appear. The
 * ~24 MB-per-chunk upload runs inside `gemini::transcribe_chunk` under the
 * `transcribing` label, which is why that label names both halves. Splitting
 * them would mean giving `transcribe_chunk` -- today entirely free of Tauri --
 * either an `AppHandle` or a progress callback, which is not worth the seam.
 */
const PHASE_LABEL: Record<GeminiProgressEvent["phase"], string> = {
    slicing: "Preparing the audio",
    transcribing: "Uploading to Google and transcribing",
    stitching: "Assembling the transcript",
};

export function createGeminiEngine(): TranscriptionEngine {
    return {
        id: "gemini",

        async run(args: EngineRunArgs): Promise<EngineResult> {
            const unsubscribe = api.subscribeToGeminiProgress(
                args.job.id,
                (event) => {
                    // `stitching` is emitted with `chunk_index == chunk_count`
                    // (it is a whole-run step, not a per-chunk one), so a
                    // naive `chunk_index + 1` would announce "part 4 of 3".
                    const perChunk =
                        event.chunk_count > 1 &&
                        event.chunk_index < event.chunk_count;
                    args.onProgress({
                        fraction: event.fraction,
                        status: perChunk
                            ? `${PHASE_LABEL[event.phase]} (part ${
                                  event.chunk_index + 1
                              } of ${event.chunk_count})...`
                            : `${PHASE_LABEL[event.phase]}...`,
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
