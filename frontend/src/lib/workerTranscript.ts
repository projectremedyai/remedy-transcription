import { TranscriptionSegment } from "../services/types";
import {
    ConsolidatedSegment,
    WordToken,
    assignSpeakersToSegments,
    assignSpeakersToTokens,
    consolidateSegments,
    countCaptionWords,
    joinCaptionTexts,
    normalizeWordTokens,
} from "./captionFormatter";
import type { SpeakerTurn } from "./speakerAlignment";

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

/**
 * What goes in the database as the transcript's source of truth.
 *
 * When the model gave us real word times, persist THOSE — one segment per word.
 * The alternative, persisting the worker's `chunks`, is now the same data anyway
 * (`return_timestamps: 'word'` makes every entry of `output.chunks` a word), but
 * going through `normalizeWordTokens` first is what keeps a re-read equal to the
 * live render, and it is the ONLY thing that does. It is the single funnel both
 * paths share, so all three of its guarantees hold on the way in as well as on
 * screen:
 *
 *   - continuation fragments are folded ("-level" onto "Word"), so a reload
 *     cannot resurrect one and print "Word -level";
 *   - words are monotonic;
 *   - every word has `end > start`. DTW does emit zero-duration words. Persisted
 *     raw, `effectiveEnd` would inflate one on reload to the NEXT word's start —
 *     swallowing a pause — or, for the final word, to `start + 1.2s`, possibly
 *     past the end of the audio.
 *
 * A word-granular source of truth also means a transcript read back from disk is
 * NOT re-fabricated: `tokensFromSegments` over one-word segments returns those
 * words' own times unchanged. Rows written before this existed still hold
 * sentence chunks and are still fabricated on read — that is the only path left
 * that invents a word time, and it is the reason `tokensFromSegments` survives.
 *
 * ## Speakers
 *
 * `turns` is diarization's answer, and passing it here is what makes speaker
 * labels SURVIVE. They did not, and it is worth saying exactly how they died,
 * because "the row is an opaque JSON blob, so the field rides along for free" is
 * the intuition that killed them and it is false in both directions:
 *
 *   - here, the row was built by hand as `{start, end, text}` — so the label was
 *     dropped on the way OUT of the formatter's tokens, before it ever reached
 *     the wire;
 *   - and in Rust, `store::TranscriptionSegment` had no `speaker` field, so serde
 *     (which ignores unknown fields by default) dropped it AGAIN on the way in.
 *
 * Two independent drops, neither of which failed anything. Both are now closed,
 * and the round trip is pinned end to end — see "a DIARIZED transcript read back
 * from the database renders EXACTLY as it did live" in `workerTranscript.test.ts`
 * and `a_diarized_transcript_survives_the_round_trip_through_the_database` in
 * `store.rs`.
 *
 * The labelling itself is not done here: it is delegated to the very functions
 * the display path uses (`assignSpeakersToTokens`, `assignSpeakersToSegments`),
 * at the granularity that path would choose — WORDS when the model gave real word
 * times, SEGMENTS when it did not. A second labelling implementation would be a
 * second thing to drift, and a drifted one would put different speakers in the
 * database than on the screen.
 *
 * Omitting `turns` (or passing an empty array — a real, successful answer for
 * silence) writes rows with NO `speaker` key at all: byte-for-byte what an
 * undiarized transcript has always written.
 */
export function segmentsForPersistence(
    workerTranscript: WorkerTranscript,
    audioDuration: number | null,
    turns?: readonly SpeakerTurn[],
): TranscriptionSegment[] {
    const speakerTurns = turns && turns.length > 0 ? turns : null;
    const words = workerTranscript.words;

    if (words && words.length > 0) {
        const tokens = normalizeWordTokens(words);
        const labelled = speakerTurns
            ? assignSpeakersToTokens(tokens, speakerTurns)
            : tokens;
        return labelled.map(rowFromToken);
    }

    const chunks = segmentsFromWorkerChunks(
        workerTranscript.chunks,
        audioDuration,
    );
    return speakerTurns
        ? assignSpeakersToSegments(chunks, speakerTurns)
        : chunks;
}

/**
 * One word token -> one database row, keeping `speaker` ABSENT rather than
 * `undefined` when there is none.
 *
 * `{start, end, text, speaker: undefined}` is not the same row:
 * `JSON.stringify` of it is identical, but `"speaker" in row` is true, `toEqual`
 * tells them apart, and Rust — which now runs `deny_unknown_fields` — is one
 * `JSON.stringify` policy change away from seeing a `"speaker": null` it would
 * reject. An undiarized transcript must persist exactly as it always has.
 */
function rowFromToken(word: WordToken): TranscriptionSegment {
    return word.speaker === undefined
        ? { start: word.start, end: word.end, text: word.text }
        : {
              start: word.start,
              end: word.end,
              text: word.text,
              speaker: word.speaker,
          };
}

/**
 * Worker output -> the cues the UI renders and the exporters serialize. This is
 * the display path's single call to `consolidateSegments`, on RAW chunks.
 *
 * `turns` is diarization's answer, threaded straight through to
 * `consolidateSegments` — this is the ONLY thing that decides whether a live
 * transcript shows speakers. Before this parameter existed, every production
 * caller passed nothing, so `consolidateSegments`'s speaker-aware cue splitting
 * was reachable only from a direct test call: a live diarized run rendered
 * exactly like an undiarized one, regardless of what `diarize_job` returned.
 * Omit it (or pass `[]`, a real success for silence) and this behaves exactly
 * as it always has — that is what keeps an undiarized transcript byte-for-byte
 * unchanged.
 */
export function consolidateWorkerTranscript(
    workerTranscript: WorkerTranscript,
    audioDuration: number | null,
    options: { hideTrailingShortCaption?: boolean } = {},
    turns?: readonly SpeakerTurn[],
): { text: string; chunks: ConsolidatedSegment[] } {
    const chunks = consolidateSegments(
        segmentsFromWorkerChunks(workerTranscript.chunks, audioDuration),
        workerTranscript.words,
        turns,
    );
    const displayChunks =
        options.hideTrailingShortCaption &&
        chunks.length > 0 &&
        countCaptionWords(chunks[chunks.length - 1].text) < 3
            ? chunks.slice(0, -1)
            : chunks;
    const text =
        displayChunks
            .map((chunk) => chunk.text.replace(/\n/g, " "))
            .reduce(joinCaptionTexts, "")
            .trim() || workerTranscript.text;

    return { text, chunks: displayChunks };
}
