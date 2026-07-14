import { TranscriptionSegment } from "../services/types";
import { ConsolidatedSegment, consolidateSegments } from "./captionFormatter";

/** A raw chunk as the transformers.js worker emits it. */
export interface WorkerChunk {
    text: string;
    timestamp: [number, number | null];
}

export interface WorkerTranscript {
    text: string;
    chunks: WorkerChunk[];
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
