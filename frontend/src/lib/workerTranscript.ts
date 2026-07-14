import { TranscriptionSegment } from "../services/types";
import {
    ConsolidatedSegment,
    WordToken,
    consolidateSegments,
    normalizeWordTokens,
} from "./captionFormatter";

/** A raw chunk as the transformers.js worker emits it. */
export interface WorkerChunk {
    text: string;
    timestamp: [number, number | null];
}

export interface WorkerTranscript {
    text: string;
    chunks: WorkerChunk[];
    /**
     * The model's REAL per-word times (`return_timestamps: 'word'`). Present on
     * the worker's final "complete" message; absent mid-stream, where the words
     * of the window being decoded have no times yet — DTW over the
     * cross-attentions only runs once the window is fully generated.
     */
    words?: WordToken[];
}

/** Last-resort end for a chunk with no end, no successor and no known duration. */
export const FALLBACK_CHUNK_DURATION = 2;

/**
 * Whisper leaves the final chunk's end timestamp null. Persisting or rendering
 * it as `end := start` gives the segment zero duration, which the formatter then
 * treats as a runt and glues onto the previous cue — silently dropping the last
 * seconds of every transcript.
 *
 * The fallback chain for a null end, in order:
 *   1. the next chunk's start (a null end mid-stream just means "not closed yet")
 *   2. the true audio duration, when the caller knows it (the chunk is the last)
 *   3. a nominal `start + FALLBACK_CHUNK_DURATION`
 *
 * `audioDuration` is null on mid-stream `update` messages: the trailing chunk is
 * still open there, and stretching it to the full audio length would be wrong.
 */
export function segmentsFromWorkerChunks(
    chunks: WorkerChunk[],
    audioDuration: number | null,
): TranscriptionSegment[] {
    return chunks.map((chunk, index) => ({
        start: chunk.timestamp[0],
        end:
            chunk.timestamp[1] ??
            chunks[index + 1]?.timestamp[0] ??
            audioDuration ??
            chunk.timestamp[0] + FALLBACK_CHUNK_DURATION,
        text: chunk.text,
    }));
}

function wordCount(text: string): number {
    return text.replace(/\n/g, " ").trim().split(/\s+/).filter(Boolean).length;
}

/**
 * What goes in the database as the transcript's source of truth.
 *
 * When the model gave us real word times, persist THOSE — one segment per word.
 * The alternative, persisting the worker's `chunks`, is now the same data anyway
 * (`return_timestamps: 'word'` makes every entry of `output.chunks` a word), but
 * going through `normalizeWordTokens` first is what keeps a re-read equal to the
 * live render: it folds "-level" back onto "Word" so the reloaded text does not
 * come back as "Word -level".
 *
 * A word-granular source of truth also means a transcript read back from disk is
 * NOT re-fabricated: `tokensFromSegments` over one-word segments returns those
 * words' own times unchanged. Rows written before this existed still hold
 * sentence chunks and are still fabricated on read — that is the only path left
 * that invents a word time, and it is the reason `tokensFromSegments` survives.
 */
export function segmentsForPersistence(
    workerTranscript: WorkerTranscript,
    audioDuration: number | null,
): TranscriptionSegment[] {
    const words = workerTranscript.words;
    if (words && words.length > 0) {
        return normalizeWordTokens(words).map((word) => ({
            start: word.start,
            end: word.end,
            text: word.text,
        }));
    }

    return segmentsFromWorkerChunks(workerTranscript.chunks, audioDuration);
}

/**
 * Worker output -> the cues the UI renders and the exporters serialize. This is
 * the display path's single call to `consolidateSegments`, on RAW chunks.
 */
export function consolidateWorkerTranscript(
    workerTranscript: WorkerTranscript,
    audioDuration: number | null,
    options: { hideTrailingShortCaption?: boolean } = {},
): { text: string; chunks: ConsolidatedSegment[] } {
    const chunks = consolidateSegments(
        segmentsFromWorkerChunks(workerTranscript.chunks, audioDuration),
        workerTranscript.words,
    );
    const displayChunks =
        options.hideTrailingShortCaption &&
        chunks.length > 0 &&
        wordCount(chunks[chunks.length - 1].text) < 3
            ? chunks.slice(0, -1)
            : chunks;
    const text =
        displayChunks
            .map((chunk) => chunk.text.replace(/\n/g, " "))
            .join(" ")
            .trim() || workerTranscript.text;

    return { text, chunks: displayChunks };
}
