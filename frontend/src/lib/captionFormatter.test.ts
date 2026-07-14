import { describe, it, expect } from "vitest";
import { consolidateSegments } from "./captionFormatter";
import type { TranscriptionSegment } from "../services/types";

const flat = (segs: TranscriptionSegment[]) =>
    segs.map((s) => s.text.replace(/\n/g, " ")).join(" ");

describe("BUG 1: chunk-boundary overlap dedup", () => {
    // lib/captionFormatter.ts:51 loops `for (let size = maxOverlap; size > 2; size--)`,
    // so 1- and 2-word overlaps are NEVER tested and survive into the output.

    it("removes a 1-word overlap", () => {
        const segs: TranscriptionSegment[] = [
            { start: 0, end: 4, text: "one two three four five six seven hotel" },
            { start: 3.5, end: 8, text: "hotel nine ten eleven twelve thirteen fourteen" },
        ];
        expect(flat(consolidateSegments(segs)).toLowerCase().match(/hotel/g)?.length).toBe(1);
    });

    it("removes a 2-word overlap", () => {
        const segs: TranscriptionSegment[] = [
            { start: 0, end: 4, text: "one two three four five six seven golf hotel" },
            { start: 3.5, end: 8, text: "golf hotel nine ten eleven twelve thirteen fourteen" },
        ];
        expect(flat(consolidateSegments(segs)).toLowerCase().match(/golf hotel/g)?.length).toBe(1);
    });

    it("still removes a 3-word overlap (regression — this already worked)", () => {
        const segs: TranscriptionSegment[] = [
            { start: 0, end: 4, text: "one two three four five six foxtrot golf hotel" },
            { start: 3.5, end: 8, text: "foxtrot golf hotel nine ten eleven twelve thirteen" },
        ];
        expect(flat(consolidateSegments(segs)).toLowerCase().match(/foxtrot golf hotel/g)?.length).toBe(1);
    });

    it("does NOT strip a legitimate repeated word that is not a boundary artifact", () => {
        // Guards the Task 3 fix against over-deduping. "no no no" is real speech, and the
        // chunks do NOT overlap in time, so nothing should be removed.
        const segs: TranscriptionSegment[] = [
            { start: 0, end: 4, text: "and I said no" },
            { start: 4, end: 8, text: "no no I really meant it" },
        ];
        expect(flat(consolidateSegments(segs)).toLowerCase().match(/\bno\b/g)?.length).toBe(3);
    });

    it("never emits cues that overlap in time", () => {
        const segs: TranscriptionSegment[] = [
            { start: 0, end: 4, text: "one two three four five six seven golf hotel" },
            { start: 3.5, end: 8, text: "golf hotel nine ten eleven twelve thirteen fourteen" },
        ];
        const cues = consolidateSegments(segs);
        for (let i = 1; i < cues.length; i++) {
            expect(cues[i].start).toBeGreaterThanOrEqual(cues[i - 1].end);
        }
    });
});

describe("BUG 3: the final chunk must not be swallowed", () => {
    // useTranscriber.ts:472 persists a null end as `end := start` (zero duration), which
    // becomes 0.01s, which makes mergeShortCaptions glue the final sentence onto the
    // previous cue. The last seconds of every transcript end up uncaptioned.

    it("emits a cue that covers a zero-duration final segment", () => {
        const segs: TranscriptionSegment[] = [
            { start: 0.0, end: 15.4, text: "a long opening statement that fills a cue nicely" },
            { start: 19.74, end: 19.74, text: "any questions before we move on?" },
        ];
        const cues = consolidateSegments(segs);
        const covering = cues.filter((c) => c.start <= 19.74 && c.end >= 19.74);
        expect(covering.length).toBeGreaterThan(0);
    });

    it("gives the final cue a non-zero duration", () => {
        const segs: TranscriptionSegment[] = [
            { start: 0.0, end: 15.4, text: "a long opening statement that fills a cue nicely" },
            { start: 19.74, end: 19.74, text: "any questions before we move on?" },
        ];
        const last = consolidateSegments(segs).at(-1)!;
        expect(last.end - last.start).toBeGreaterThan(0.5);
    });
});

describe("BUG 4: post-dedup cues must not start early", () => {
    // lib/captionFormatter.ts:79 offsets the start by (duration * overlap) / words.length,
    // assuming the stripped overlap words consumed time proportional to their COUNT.

    it("a cue never starts before its segment's own start time", () => {
        const segs: TranscriptionSegment[] = [
            { start: 0, end: 6, text: "one two three four five six foxtrot golf hotel" },
            { start: 6, end: 12, text: "foxtrot golf hotel nine ten eleven twelve thirteen" },
        ];
        const cues = consolidateSegments(segs);
        // No cue may begin before the first segment starts.
        for (const cue of cues) {
            expect(cue.start).toBeGreaterThanOrEqual(0);
        }
        // The cue carrying post-overlap words must not start before that segment did.
        const post = cues.find((c) => /nine|ten|eleven/.test(c.text));
        expect(post).toBeDefined();
        expect(post!.start).toBeGreaterThanOrEqual(6 - 0.25); // small tolerance only
    });
});
