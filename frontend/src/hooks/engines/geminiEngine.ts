import { api } from "../../services/api";
import type { GeminiProgressEvent } from "../../services/types";
import {
    RUN_DECLINED,
    type EngineResult,
    type EngineRunArgs,
    type TranscriptionEngine,
} from "./types";

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
            // THE MONEY GATE, and it comes first — before the progress
            // listener, before any request, before anything that costs.
            //
            // Only for a chunked run. `chunk_count > 1` is the one honest
            // trigger: it is exactly when a run stops being pocket change AND
            // exactly when it loses its speaker labels, so one question covers
            // both. Interrupting every short file would train the user to click
            // through the dialog that matters.
            //
            // The estimate is cheap and keyless (an ffprobe of audio already on
            // disk), so asking for it costs nothing even when the answer turns
            // out to be "no need to ask".
            const estimate = await api.estimateGeminiCost(args.job.id);
            if (estimate.chunk_count > 1) {
                const proceed = await args.confirmCost({
                    durationSecs: estimate.duration_secs,
                    chunkCount: estimate.chunk_count,
                    estimatedUsd: estimate.estimated_usd,
                    diarizationAvailable: estimate.diarization_available,
                });
                if (!proceed) {
                    throw new Error(RUN_DECLINED);
                }
            }

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
                        // LEADING SPACE, deliberately. `WordToken.text` carries
                        // transformers.js's convention, where
                        // `splitTokensOnSpaces` marks every word boundary with
                        // one and `normalizeWordTokens` reads its ABSENCE as
                        // "this token continues the previous word" — the fold
                        // that glues the local engine's sub-word pieces back
                        // together. Gemini emits whole, bare words instead
                        // ("Hello"), so passing them through unchanged made
                        // every word continue the last one and folded the
                        // transcript into a single token, which
                        // `cleanCaptionText` then re-split only at `,.;:!?`:
                        // "Helloeveryone. JohnnyFunghere". This engine is the
                        // adapter between the two conventions, so restoring the
                        // boundary belongs HERE — the normalizer's fold cannot
                        // be relaxed without breaking the local engine.
                        //
                        // Safe for CJK: the space stops the fold (which is what
                        // no-space scripts want too), `normalizeWordTokens`
                        // trims it away, and `joinCaptionTexts` then re-joins
                        // CJK words with no space of its own.
                        words: result.words.map((word) => ({
                            text: ` ${word.text.trim()}`,
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
