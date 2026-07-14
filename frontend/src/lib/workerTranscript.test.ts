import { describe, it, expect } from "vitest";
import {
    FALLBACK_CHUNK_DURATION,
    WorkerChunk,
    consolidateWorkerTranscript,
    segmentsFromWorkerChunks,
} from "./workerTranscript";

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
});
