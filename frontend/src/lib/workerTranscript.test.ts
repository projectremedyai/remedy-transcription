import { describe, it, expect } from "vitest";
import {
    FALLBACK_CHUNK_DURATION,
    WorkerChunk,
    consolidateWorkerTranscript,
    segmentsForPersistence,
    segmentsFromWorkerChunks,
} from "./workerTranscript";
import { consolidateSegments } from "./captionFormatter";
import type { SpeakerTurn } from "./speakerAlignment";
import type { TranscriptionSegment } from "../services/types";
import PERSISTED_ROWS_FIXTURE from "./persistedSegments.fixture.json";
import PERSISTED_CHUNK_ROWS_FIXTURE from "./persistedChunkSegments.fixture.json";

/**
 * Whisper leaves the final chunk's end timestamp NULL. Rendering or persisting
 * that as `end := start` gives the segment zero duration, the formatter treats it
 * as a runt and glues it onto the previous cue, and the last seconds of every
 * transcript vanish from the captions. The fallback chain below is the fix, and
 * it is the whole reason `audioDuration` is threaded through the worker path.
 */
describe("segmentsFromWorkerChunks: the null-end fallback chain", () => {
    it("passes a real end timestamp straight through", () => {
        const chunks: WorkerChunk[] = [
            { text: " Hello there.", timestamp: [0, 2.5] },
        ];
        expect(segmentsFromWorkerChunks(chunks, 10)).toEqual([
            { start: 0, end: 2.5, text: " Hello there." },
        ]);
    });

    it("resolves a null end from the NEXT chunk's start", () => {
        // A null end mid-stream just means "this chunk is not closed yet". The
        // next chunk's start is the best evidence available, and it beats the
        // audio duration even when one is known.
        const chunks: WorkerChunk[] = [
            { text: " Hello there.", timestamp: [0, null] },
            { text: " Second chunk.", timestamp: [4.2, 8.0] },
        ];
        expect(segmentsFromWorkerChunks(chunks, 99)).toEqual([
            { start: 0, end: 4.2, text: " Hello there." },
            { start: 4.2, end: 8.0, text: " Second chunk." },
        ]);
    });

    it("resolves a null end on the FINAL chunk from the known audio duration", () => {
        // This is the production case: the last chunk really is the last, so the
        // decoded AudioBuffer's duration is its true end.
        const chunks: WorkerChunk[] = [
            { text: " Hello there.", timestamp: [0, 4.2] },
            { text: " Any questions?", timestamp: [4.2, null] },
        ];
        expect(segmentsFromWorkerChunks(chunks, 9.75)).toEqual([
            { start: 0, end: 4.2, text: " Hello there." },
            { start: 4.2, end: 9.75, text: " Any questions?" },
        ]);
    });

    it("falls back to a nominal duration when the end is null and no duration is known", () => {
        // The mid-stream `update` path passes null: the trailing chunk is still
        // open there, and stretching it to the full audio length would be wrong.
        const chunks: WorkerChunk[] = [
            { text: " Any questions?", timestamp: [4.2, null] },
        ];
        expect(segmentsFromWorkerChunks(chunks, null)).toEqual([
            {
                start: 4.2,
                end: 4.2 + FALLBACK_CHUNK_DURATION,
                text: " Any questions?",
            },
        ]);
    });

    it("never produces a zero-duration segment from a null end", () => {
        // The regression that started all this: `end := start`.
        const chunks: WorkerChunk[] = [
            {
                text: " Opening statement that runs a while.",
                timestamp: [0, 15.4],
            },
            {
                text: " Any questions before we move on?",
                timestamp: [19.74, null],
            },
        ];

        for (const audioDuration of [24.0, null]) {
            const segments = segmentsFromWorkerChunks(chunks, audioDuration);
            for (const segment of segments) {
                expect(segment.end).toBeGreaterThan(segment.start);
            }
        }
    });
});

/**
 * The mid-stream preview. `return_timestamps: 'word'` turns timestamp TOKENS off,
 * so the streamer has no per-sentence times to give and the worker opens one
 * chunk per decoding WINDOW. What that chunk claims as its END is the whole
 * contract between the worker and this module, and it is shown to the user
 * (Transcript.tsx renders these times).
 */
describe("the open decoding window must carry a provisional end", () => {
    const CHUNK_LENGTH = 29;

    it("keeps preview times inside the window instead of crushing them into 2 seconds", () => {
        // With `timestamp[1] = null` and no audioDuration (the `update` path
        // passes null by design — the window is still open), the fallback chain
        // bottoms out at `start + FALLBACK_CHUNK_DURATION`. A whole 29s window's
        // worth of words then renders inside a 2s band: speech 25 seconds in
        // displays as 0:00, and snaps when the window finalises.
        const openWindow =
            " a long decoding window of speech spoken over many seconds of audio.";

        const broken = segmentsFromWorkerChunks(
            [{ text: openWindow, timestamp: [0, null] }],
            null,
        );
        expect(broken[0].end).toBe(FALLBACK_CHUNK_DURATION);

        // What the worker now sends: the window runs to offset + chunk_length_s.
        const fixed = segmentsFromWorkerChunks(
            [{ text: openWindow, timestamp: [0, CHUNK_LENGTH] }],
            null,
        );
        expect(fixed[0].end).toBe(CHUNK_LENGTH);

        const cues = consolidateWorkerTranscript(
            {
                text: "",
                chunks: [{ text: openWindow, timestamp: [0, CHUNK_LENGTH] }],
            },
            null,
        ).chunks;
        // The preview spreads the window's words across the window, not across 2s.
        expect(cues.at(-1)!.end).toBeGreaterThan(5);
        expect(cues.at(-1)!.end).toBeLessThanOrEqual(CHUNK_LENGTH);
    });

    it("times the SECOND window from its own stride offset, not from zero", () => {
        // transformers.js overlaps each decoding window with its neighbour on BOTH
        // sides: `const jump = window - 2 * stride` (pipelines.js), so window k
        // begins at `k * (chunk_length_s - 2 * stride_length_s)` — 19s for (29, 5),
        // NOT 24s. This test used to pin 24/53, which is the worker's old
        // one-stride formula, and the error grows by 5s per window: on a
        // five-minute file the preview showed cue times a minute past the end of
        // the audio. Verified against the library source, not against arithmetic.
        const strideLength = 5;
        const secondOffset = CHUNK_LENGTH - 2 * strideLength;
        expect(secondOffset).toBe(19);

        const segments = segmentsFromWorkerChunks(
            [
                { text: " first window", timestamp: [0, CHUNK_LENGTH] },
                {
                    text: " second window",
                    timestamp: [secondOffset, secondOffset + CHUNK_LENGTH],
                },
            ],
            null,
        );
        expect(segments[1].start).toBe(19);
        expect(segments[1].end).toBe(48);
    });
});

describe("consolidateWorkerTranscript", () => {
    it("consolidates raw worker chunks into cues and joins their text", () => {
        const { text, chunks } = consolidateWorkerTranscript(
            {
                text: " raw worker text",
                chunks: [
                    {
                        text: " chlorophyll absorbs photons and converts them into sugar.",
                        timestamp: [0, 6.2],
                    },
                ],
            },
            6.2,
        );

        expect(chunks.length).toBeGreaterThan(0);
        // Formatted, not raw: sentence-cased and cleaned.
        expect(text).toBe(
            "Chlorophyll absorbs photons and converts them into sugar.",
        );
        // The joined display text never carries the two-line wrap.
        expect(text).not.toContain("\n");
    });

    it("does not swallow the final chunk when its end timestamp is null", () => {
        // End to end: null end -> audioDuration -> a real cue, rather than a
        // zero-duration runt merged into the previous cue.
        const { chunks } = consolidateWorkerTranscript(
            {
                text: "",
                chunks: [
                    {
                        text: " a long opening statement that fills a cue nicely",
                        timestamp: [0, 15.4],
                    },
                    {
                        text: " any questions before we move on?",
                        timestamp: [19.74, null],
                    },
                ],
            },
            23.5,
        );

        const last = chunks[chunks.length - 1];
        expect(last.text.replace(/\n/g, " ")).toMatch(/questions/i);
        expect(last.text.replace(/\n/g, " ")).not.toMatch(/opening statement/i);
        expect(last.start).toBeGreaterThanOrEqual(19.24);
        expect(last.end).toBeGreaterThan(last.start);
    });

    it("hides a trailing short caption only when asked (the mid-stream display path)", () => {
        // Mid-stream, the last chunk is a half-heard fragment. The display path
        // hides it so it does not flicker; the `complete` and persist paths keep
        // everything.
        const transcript = {
            text: "",
            chunks: [
                {
                    text: " chlorophyll absorbs photons and converts them into sugar.",
                    timestamp: [0, 6.2] as [number, number | null],
                },
                // Far enough away that mergeShortCaptions cannot absorb it, so it
                // really does survive as its own trailing cue.
                {
                    text: " So",
                    timestamp: [10.0, null] as [number, number | null],
                },
            ],
        };

        const shown = consolidateWorkerTranscript(transcript, null);
        const hidden = consolidateWorkerTranscript(transcript, null, {
            hideTrailingShortCaption: true,
        });

        expect(shown.chunks).toHaveLength(2);
        expect(shown.text).toMatch(/\bSo\.?$/);

        expect(hidden.chunks).toHaveLength(1);
        expect(hidden.text).not.toMatch(/\bSo\.?$/);
    });

    it("falls back to the worker's own text when there are no chunks", () => {
        expect(
            consolidateWorkerTranscript(
                { text: " raw worker text", chunks: [] },
                null,
            ),
        ).toEqual({ text: " raw worker text", chunks: [] });
    });

    /**
     * `turns` used to have no production caller at all: `consolidateWorkerTranscript`
     * had no such parameter, so a live diarized run rendered identically to an
     * undiarized one however `diarize_job` answered. This pins that the parameter
     * exists, is optional (an undiarized call is unaffected), and actually reaches
     * `consolidateSegments` — the same join `assignSpeakers` proves at the unit
     * level, exercised here through the real display entry point.
     */
    it("threads `turns` through to the cues, and leaves an undiarized call unchanged", () => {
        const transcript = {
            text: "",
            chunks: [
                {
                    text: " Hello there.",
                    timestamp: [0, 2] as [number, number],
                },
                {
                    text: " Goodbye now.",
                    timestamp: [3, 5] as [number, number],
                },
            ],
        };
        const turns: SpeakerTurn[] = [
            { start: 0, end: 2, speaker: 0 },
            { start: 3, end: 5, speaker: 1 },
        ];

        const undiarized = consolidateWorkerTranscript(transcript, 5, {});
        const diarized = consolidateWorkerTranscript(transcript, 5, {}, turns);

        expect(
            undiarized.chunks.every((chunk) => chunk.speaker === undefined),
        ).toBe(true);
        expect(diarized.chunks.map((chunk) => chunk.speaker)).toEqual([
            "SPEAKER_00",
            "SPEAKER_01",
        ]);
    });
});

/**
 * With `return_timestamps: 'word'` the worker's `output.chunks` ARE the words,
 * so a "complete" message carries both. These are verbatim model outputs for a
 * 12.82s clip (see captionFormatter.test.ts).
 */
const WORD_CHUNKS: WorkerChunk[] = [
    { text: " Hello", timestamp: [0.0, 0.42] },
    { text: " there", timestamp: [0.42, 0.9] },
    { text: " and", timestamp: [0.9, 1.1] },
    { text: " welcome.", timestamp: [1.1, 1.94] },
    { text: " Word", timestamp: [3.2, 3.4] },
    { text: "-level", timestamp: [3.4, 3.86] },
    { text: " timing", timestamp: [3.86, 4.4] },
    { text: " is", timestamp: [4.4, 4.6] },
    { text: " real", timestamp: [4.6, 5.28] },
    { text: " now.", timestamp: [5.28, 5.9] },
];

const WORD_TRANSCRIPT = {
    text: " Hello there and welcome. Word-level timing is real now.",
    chunks: WORD_CHUNKS,
    words: WORD_CHUNKS.map((c) => ({
        text: c.text,
        start: c.timestamp[0],
        end: c.timestamp[1] as number,
    })),
};

describe("segmentsForPersistence: the database keeps the REAL word times", () => {
    it("persists one segment per word, with the model's own times", () => {
        const segments = segmentsForPersistence(WORD_TRANSCRIPT, 6.0);
        expect(segments).toHaveLength(9); // 10 words, "-level" folded into "Word"
        expect(segments[0]).toEqual({ start: 0.0, end: 0.42, text: "Hello" });
        expect(segments[3]).toEqual({
            start: 1.1,
            end: 1.94,
            text: "welcome.",
        });
        // The continuation fragment is folded in BEFORE it hits the database, so
        // a reload cannot resurrect it as a separate word and print "Word -level".
        expect(segments[4]).toEqual({
            start: 3.2,
            end: 3.86,
            text: "Word-level",
        });
    });

    it("a transcript read back from the database renders EXACTLY as it did live", () => {
        // This is the point of persisting words rather than chunks. The reload
        // path has no `words` — it re-derives them from the stored segments — so
        // if the stored segments were sentence chunks, the reload would fabricate
        // word times and the cues would move. One word per segment means
        // `tokensFromSegments` hands back each word's own measured time unchanged.
        const live = consolidateWorkerTranscript(WORD_TRANSCRIPT, 6.0);
        const persisted = segmentsForPersistence(WORD_TRANSCRIPT, 6.0);
        const reloaded = consolidateSegments(persisted);

        expect(reloaded).toEqual(live.chunks);
    });

    it("does not ship the words twice: `chunks` is empty when `words` is populated", () => {
        // The worker's "complete" message used to carry `output.chunks` AND
        // `words` — with `return_timestamps: 'word'` those are the same array.
        // Persistence and display must both work off `words` alone.
        const wordsOnly = { ...WORD_TRANSCRIPT, chunks: [] };
        expect(segmentsForPersistence(wordsOnly, 6.0)).toEqual(
            segmentsForPersistence(WORD_TRANSCRIPT, 6.0),
        );
        expect(consolidateWorkerTranscript(wordsOnly, 6.0)).toEqual(
            consolidateWorkerTranscript(WORD_TRANSCRIPT, 6.0),
        );
    });

    it("falls back to chunk segments when the model gave no word times", () => {
        // Legacy rows, and any future model without cross_attentions outputs.
        const chunks: WorkerChunk[] = [
            { text: " Hello there.", timestamp: [0, 2.5] },
        ];
        expect(
            segmentsForPersistence({ text: "Hello there.", chunks }, 10),
        ).toEqual(segmentsFromWorkerChunks(chunks, 10));
    });
});

/**
 * **The speaker label used to die on the way to the database — TWICE.**
 *
 * `segmentsForPersistence` rebuilt each row by hand as `{start, end, text}`, so
 * the label never reached the wire; and Rust's `store::TranscriptionSegment` had
 * no `speaker` field, so serde (which ignores unknown fields by default) dropped
 * it again if it ever had. Neither drop failed anything: diarize a file, reopen
 * it, and the speakers were simply gone, with no error anywhere. "It's an opaque
 * JSON blob, the field rides along free" is the reasoning that produced that, and
 * it is false at both ends.
 *
 * These tests are the frontend half of the proof. The Rust half —
 * `a_diarized_transcript_survives_the_round_trip_through_the_database` in
 * `store.rs` — reads the SAME fixture this file pins, so the two halves cannot
 * drift into disagreeing about the row shape without one of them going red.
 */
describe("segmentsForPersistence: a DIARIZED transcript keeps its speakers", () => {
    // Two speakers, with the turn boundary in the 1.26s pause between the two
    // sentences — where a real one usually is.
    const TURNS: SpeakerTurn[] = [
        { start: 0.0, end: 2.0, speaker: 0 },
        { start: 3.0, end: 6.0, speaker: 1 },
    ];

    /** Exactly what crosses the IPC wire on its way into `segments_json`. */
    const persist = () => segmentsForPersistence(WORD_TRANSCRIPT, 6.0, TURNS);

    it("writes the speaker onto the rows — and writes the shape Rust reads", () => {
        const rows = persist();

        expect(rows[0]).toEqual({
            start: 0.0,
            end: 0.42,
            text: "Hello",
            speaker: "SPEAKER_00",
        });
        expect(rows[8]).toEqual({
            start: 5.28,
            end: 5.9,
            text: "now.",
            speaker: "SPEAKER_01",
        });

        // THE CROSS-LANGUAGE PIN. `store.rs` deserializes this very file into
        // `Vec<TranscriptionSegment>` under `deny_unknown_fields`. Add a field
        // here and forget to teach Rust about it and `cargo test` goes red —
        // which is the alarm that did not exist when `speaker` was dropped.
        expect(rows).toEqual(PERSISTED_ROWS_FIXTURE);
    });

    it("a DIARIZED transcript read back from the database renders EXACTLY as it did live", () => {
        // Live: raw chunks + the model's real word times + diarization's turns.
        const live = consolidateSegments(
            segmentsFromWorkerChunks(WORD_CHUNKS, 6.0),
            WORD_TRANSCRIPT.words,
            TURNS,
        );

        // Reloaded: what `find_transcript` hands back — the persisted rows, with
        // no `words` and no `turns` anywhere, because neither is stored. Through
        // `JSON.parse(JSON.stringify(...))` because that is what the row does on
        // its way to SQLite and back, and it is what would silently delete an
        // `undefined`-valued key.
        const reloaded = consolidateSegments(
            JSON.parse(JSON.stringify(persist())),
        );

        expect(reloaded).toEqual(live);

        // Toothy: the assertion above would also pass if BOTH sides had lost
        // their speakers. They did not.
        expect(reloaded.map((cue) => cue.speaker)).toEqual([
            "SPEAKER_00",
            "SPEAKER_01",
        ]);
    });

    it("an UNDIARIZED transcript persists exactly as it always has: no speaker key at all", () => {
        const rows = segmentsForPersistence(WORD_TRANSCRIPT, 6.0);

        // ABSENT, not `undefined` — `"speaker" in row` must be false. A row with
        // the key present would serialize a `null` into `segments_json` that
        // Rust's `deny_unknown_fields`-strict struct is one policy change away
        // from rejecting, and that the formatter would compare against.
        expect(rows.every((row) => !("speaker" in row))).toBe(true);

        // An EMPTY turn list is diarization SUCCEEDING on silence (or on one
        // person), not failing. It must persist the same way.
        expect(segmentsForPersistence(WORD_TRANSCRIPT, 6.0, [])).toEqual(rows);
    });

    it("labels a LEGACY sentence-granular transcript at segment granularity", () => {
        // No word times, so the rows persisted are the sentence chunks — and the
        // speaker is assigned from each chunk's own MEASURED span, never from the
        // word times `tokensFromSegments` would fabricate inside it (up to 1.3s
        // adrift, which is longer than many turns).
        const chunks: WorkerChunk[] = [
            { text: " Hello there and welcome.", timestamp: [0, 1.94] },
            { text: " Word-level timing is real now.", timestamp: [3.2, 5.9] },
        ];

        // THE OTHER CROSS-LANGUAGE PIN. This chunk-granular shape is a SECOND thing
        // that crosses the wire into `segments_json`, distinct from the
        // word-granular fixture. `store.rs` deserializes this same file under
        // `deny_unknown_fields`, so a field added on the chunk path
        // (`assignSpeakersToSegments`) and forgotten in Rust turns `cargo test` red
        // here — instead of a legacy transcript silently failing to save at runtime.
        expect(
            segmentsForPersistence({ text: "", chunks }, 6.0, TURNS),
        ).toEqual(PERSISTED_CHUNK_ROWS_FIXTURE);
    });
});

describe("a LEGACY transcript, cached under the pre-rename model id, still renders", () => {
    // The transcript cache now maps `onnx-community/whisper-base` ->
    // `onnx-community/whisper-base_timestamped` on lookup (store.rs::find_transcript),
    // so renaming the four models to their `_timestamped` exports no longer orphans
    // every transcript a user had already made. That is only SAFE because the old
    // rows — sentence-granular, no word timings — still render through the
    // formatter's no-words path. If this test ever fails, the mapping in
    // find_transcript is serving rows the frontend can no longer draw, and it must
    // go back to being a cache miss.
    const legacyRows: TranscriptionSegment[] = [
        {
            start: 0,
            end: 5.52,
            text: " The quick brown fox jumps over the lazy dog.",
        },
        { start: 5.52, end: 7.52, text: " This sentence is used for testing." },
        {
            start: 7.52,
            end: 12.24,
            text: " Word-level timestamps are extracted using DTW.",
        },
    ];

    it("consolidates sentence-granular rows into cues, with no `words` anywhere", () => {
        const { text, chunks } = consolidateWorkerTranscript(
            {
                text: "",
                chunks: legacyRows.map((row) => ({
                    text: row.text,
                    timestamp: [row.start, row.end] as [number, number],
                })),
                // NO `words`: that is exactly what a pre-rename row has.
            },
            12.24,
        );

        expect(chunks.length).toBeGreaterThan(0);
        expect(text).toContain("quick brown fox");
        expect(text).toContain("Word-level timestamps");

        // Cues are ordered, non-overlapping and inside the audio — i.e. usable.
        expect(chunks[0].start).toBeGreaterThanOrEqual(0);
        for (let i = 0; i < chunks.length; i++) {
            expect(chunks[i].end).toBeGreaterThan(chunks[i].start);
            if (i > 0) {
                expect(chunks[i].start).toBeGreaterThanOrEqual(
                    chunks[i - 1].end,
                );
            }
        }
        expect(chunks.at(-1)!.end).toBeLessThanOrEqual(12.24);
    });

    it("agrees with consolidateSegments' no-words path (the same rows read straight from SQLite)", () => {
        expect(consolidateSegments(legacyRows)).toEqual(
            consolidateWorkerTranscript(
                {
                    text: "",
                    chunks: legacyRows.map((row) => ({
                        text: row.text,
                        timestamp: [row.start, row.end] as [number, number],
                    })),
                },
                12.24,
            ).chunks,
        );
    });
});
