import { describe, it, expect } from "vitest";
import { ConsolidatedSegment, consolidateSegments } from "./captionFormatter";
import type { TranscriptionSegment } from "../services/types";

const flat = (segs: TranscriptionSegment[]) =>
    segs.map((s) => s.text.replace(/\n/g, " ")).join(" ");

const findCue = (cues: TranscriptionSegment[], re: RegExp) =>
    cues.find((c) => re.test(c.text.replace(/\n/g, " ")));

/**
 * Realistic Whisper sliding-window output (chunk_length_s=30 / stride_length_s=5).
 * Timestamps are the TRUE word times of a synthetic lecture; the chunk boundaries
 * re-emit 4, 2, 1 and 8 words respectively, exactly as Whisper does.
 *
 * Key facts used by the assertions below:
 *   - seg1 ends at 6.11. Its last four words ("the light dependent reactions.")
 *     are re-emitted as the head of seg2, so seg2's FIRST SURVIVING word ("I")
 *     cannot legitimately begin before 6.11.
 *   - seg3's head ("about energy.") is a 2-word overlap and seg4's head
 *     ("photons.") is a 1-word overlap — neither is stripped by the current
 *     `size > 2` dedup loop.
 */
const WHISPER_WINDOW_SEGMENTS: TranscriptionSegment[] = [
    {
        start: 0.0,
        end: 2.46,
        text: "Welcome to the lecture on photosynthesis.",
    },
    {
        start: 2.81,
        end: 6.11,
        text: "Today we will cover the light dependent reactions.",
    },
    // 4-word overlap with seg1
    {
        start: 4.16,
        end: 8.6,
        text: "the light dependent reactions. I want you to think about energy.",
    },
    // 2-word overlap with seg2 — NOT stripped (dedup loop stops at size > 2)
    {
        start: 7.45,
        end: 11.55,
        text: "about energy. Chlorophyll absorbs photons.",
    },
    // 1-word overlap with seg3 — NOT stripped
    {
        start: 10.6,
        end: 15.54,
        text: "photons. That energy is then converted into chemical bonds.",
    },
    // 8-word overlap with seg4
    {
        start: 11.9,
        end: 19.39,
        text: "That energy is then converted into chemical bonds. So a plant is really a solar powered factory.",
    },
];

/** Whisper's final chunk carries a null end timestamp, which useTranscriber persists as `end := start`. */
const ZERO_DURATION_FINAL: TranscriptionSegment = {
    start: 19.74,
    end: 19.74,
    text: "Any questions before we move on?",
};

describe("BUG 1: chunk-boundary overlap dedup", () => {
    // lib/captionFormatter.ts:51 loops `for (let size = maxOverlap; size > 2; size--)`,
    // so 1- and 2-word overlaps are NEVER tested and survive into the output.

    it("removes a 1-word overlap", () => {
        const segs: TranscriptionSegment[] = [
            {
                start: 0,
                end: 4,
                text: "one two three four five six seven hotel",
            },
            {
                start: 3.5,
                end: 8,
                text: "hotel nine ten eleven twelve thirteen fourteen",
            },
        ];
        expect(
            flat(consolidateSegments(segs)).toLowerCase().match(/hotel/g)
                ?.length,
        ).toBe(1);
    });

    it("removes a 2-word overlap", () => {
        const segs: TranscriptionSegment[] = [
            {
                start: 0,
                end: 4,
                text: "one two three four five six seven golf hotel",
            },
            {
                start: 3.5,
                end: 8,
                text: "golf hotel nine ten eleven twelve thirteen fourteen",
            },
        ];
        expect(
            flat(consolidateSegments(segs))
                .toLowerCase()
                .match(/golf hotel/g)?.length,
        ).toBe(1);
    });

    it("still removes a 3-word overlap (regression — this already worked)", () => {
        const segs: TranscriptionSegment[] = [
            {
                start: 0,
                end: 4,
                text: "one two three four five six foxtrot golf hotel",
            },
            {
                start: 3.5,
                end: 8,
                text: "foxtrot golf hotel nine ten eleven twelve thirteen",
            },
        ];
        expect(
            flat(consolidateSegments(segs))
                .toLowerCase()
                .match(/foxtrot golf hotel/g)?.length,
        ).toBe(1);
    });

    it("does NOT strip a legitimate repeated word that is not a boundary artifact", () => {
        // Guards the Task 3 fix against over-deduping. "no no no" is real speech, and the
        // chunks do NOT overlap in time, so nothing should be removed.
        const segs: TranscriptionSegment[] = [
            { start: 0, end: 4, text: "and I said no" },
            { start: 4, end: 8, text: "no no I really meant it" },
        ];
        expect(
            flat(consolidateSegments(segs))
                .toLowerCase()
                .match(/\bno\b/g)?.length,
        ).toBe(3);
    });

    it("does NOT strip a repeat just because an EARLIER, longer segment is still open", () => {
        // The time-overlap test that unlocks 1-word dedup must be PAIRWISE
        // (this segment vs. the one immediately before it). A running max over
        // all previous ends is more permissive: seg1 runs to 30, so every later
        // segment starting before 30 would count as "overlapping" — including
        // seg3, which is strictly sequential with its actual neighbour seg2
        // (28 -> 29, then 29.5). Under a running max the honest repeat of
        // "right" is deleted; pairwise, it is kept.
        const segs: TranscriptionSegment[] = [
            {
                start: 0,
                end: 30,
                text: "a long opening segment that runs for a good while and covers plenty of ground.",
            },
            { start: 28, end: 29, text: "is that right?" },
            {
                start: 29.5,
                end: 33,
                text: "right so let us move on to the next topic",
            },
        ];
        expect(
            flat(consolidateSegments(segs))
                .toLowerCase()
                .match(/\bright\b/g)?.length,
        ).toBe(2);
    });

    it("never emits cues that overlap in time", () => {
        // A two-segment toy fixture does NOT surface this: it produces one cue per
        // segment and they happen not to collide. It takes the real sliding-window
        // pattern — where an un-stripped 1/2-word overlap makes a cue restart inside
        // the span the previous cue already covered — to produce invalid SRT.
        const cues = consolidateSegments(WHISPER_WINDOW_SEGMENTS);

        const overlaps = cues
            .slice(1)
            .map((cue, i) => ({ cue, previous: cues[i] }))
            .filter(({ cue, previous }) => cue.start < previous.end)
            .map(
                ({ cue, previous }) =>
                    `cue starting ${cue.start} runs back into the previous cue, which ends at ${previous.end}`,
            );

        // Currently 4 collisions, e.g. cue 4 starts at 7.45 while cue 3 runs to 8.6:
        // seg3's "about energy." duplicate survives dedup, so its cue re-opens at the
        // duplicate's own (earlier) timestamp.
        expect(overlaps).toEqual([]);
    });
});

describe("BUG 3: the final chunk must not be swallowed", () => {
    // useTranscriber.ts:472 persists a null end as `end := start` (zero duration), which
    // becomes 0.01s, which makes mergeShortCaptions glue the final sentence onto the
    // previous cue. The last seconds of every transcript end up uncaptioned.

    it("gives the zero-duration final segment its own cue instead of gluing it onto earlier speech", () => {
        // "A cue covers timestamp 19.74" is far too weak to detect the swallowing: the
        // glued cue spans 15.425 -> 19.75 and therefore "covers" 19.74 while actually
        // displaying the final sentence 4.3s early, on top of the previous sentence.
        // The real invariant is that the final segment's words are timed to the final
        // segment, not back-dated to whenever the previous cue happened to start.
        const cues = consolidateSegments([
            ...WHISPER_WINDOW_SEGMENTS,
            ZERO_DURATION_FINAL,
        ]);

        const cue = findCue(cues, /questions/i);
        expect(cue).toBeDefined();
        // Currently 15.425 — mergeShortCaptions treats the 0.01s cue as "short" and
        // absorbs it into "So a plant is really a solar powered factory."
        expect(cue!.start).toBeGreaterThanOrEqual(
            ZERO_DURATION_FINAL.start - 0.5,
        );
        expect(cue!.text.replace(/\n/g, " ")).not.toMatch(/factory/i);
    });

    it("gives the final cue a non-zero duration", () => {
        const segs: TranscriptionSegment[] = [
            {
                start: 0.0,
                end: 15.4,
                text: "a long opening statement that fills a cue nicely",
            },
            {
                start: 19.74,
                end: 19.74,
                text: "any questions before we move on?",
            },
        ];
        const last = consolidateSegments(segs).at(-1)!;
        expect(last.end - last.start).toBeGreaterThan(0.5);
    });
});

describe("characterization", () => {
    /**
     * Re-consolidating requires deliberately defeating the type system —
     * `consolidateSegments` takes `RawSegment[]` and returns branded
     * `ConsolidatedSegment[]`, so production code CANNOT do this by accident.
     * Only a characterization test has any business doing it.
     */
    const reconsolidate = (cues: ConsolidatedSegment[]) =>
        consolidateSegments(cues as unknown as TranscriptionSegment[]);

    it("consolidateSegments is NOT idempotent (characterization). If this test fails because consolidateSegments became idempotent, that is an IMPROVEMENT — delete this test, do not restore the old behavior.", () => {
        // This test asserts nothing about what SHOULD happen. It records what
        // DOES happen today, so the reason the export path must never re-format
        // (lib/srtGenerator.ts) cannot decay into folklore.
        //
        // Cause: word times inside a cue are interpolated from the raw segment
        // that produced them. Once cues exist that provenance is gone, and a
        // second pass re-interpolates each word evenly across ITS CUE's span.
        // Task 4 (real word timestamps) is the change that would fix this.

        // (a) Words migrate between cues, and the boundary moves by SECONDS.
        // Two clean, strictly sequential segments: no time overlap, no
        // zero-duration chunk, no dedup. The non-idempotence is intrinsic to the
        // timing math, not an artifact of degenerate input.
        const once = consolidateSegments([
            { start: 0, end: 6.66, text: "photons which the energy stores." },
            { start: 6.66, end: 10, text: "the sugar converts." },
        ]);
        const twice = reconsolidate(once);

        expect(once.map((c) => c.text)).toEqual([
            "Photons which the energy",
            "stores. the sugar converts.",
        ]);
        expect(twice.map((c) => c.text)).toEqual([
            "Photons which the energy stores.",
            "The sugar converts.",
        ]);
        // Not a rounding wobble: the cue boundary moves by ~1.46 SECONDS.
        expect(Math.abs(twice[0].end - once[0].end)).toBeGreaterThan(1.4);

        // (b) The dangerous one: a second pass DELETES words the speaker said.
        // The formatter correctly keeps both copies of a phrase a speaker
        // repeated inside one chunk, then breaks the cue between them. A second
        // pass mistakes that CUE boundary for a CHUNK boundary and strips it.
        const kept = consolidateSegments([
            {
                start: 0,
                end: 11.79,
                text: "really chlorophyll absorbs photons chlorophyll absorbs photons you.",
            },
        ]);
        const repeated = /chlorophyll absorbs photons/gi;

        expect(flat(kept).match(repeated)?.length).toBe(2);
        expect(flat(reconsolidate(kept)).match(repeated)?.length).toBe(1);
    });
});

describe("the monotonic clamp must not crush a segment into a sliver", () => {
    // The clamp keeps word times monotonic by starting a segment no earlier than
    // the last word already emitted. But nothing stopped `start` being clamped
    // PAST the segment's own end, at which point `Math.max(segmentEnd - start,
    // 0.01)` collapsed the entire segment into a 0.01s cue.

    it("gives a cue that is clamped past its own segment end a real duration", () => {
        // seg2 begins BEFORE seg1 ends but carries different words (no overlap to
        // strip), so its start is clamped forward to 30.0 — past its own end of
        // 28.4. Before the fix this emitted an 11-word cue lasting 6 MILLISECONDS
        // (30.000 -> 30.006), displacing 2.6s of speech by 4+ seconds.
        const cues = consolidateSegments([
            {
                start: 20.0,
                end: 30.0,
                text: "and that is how the calvin cycle fixes carbon into sugar.",
            },
            {
                start: 25.8,
                end: 28.4,
                text: "an entirely different long sentence that the second window transcribed here",
            },
        ]);

        const clamped = findCue(cues, /entirely different/i);
        expect(clamped).toBeDefined();

        // The words still have to be spoken: floor the SPAN at a nominal speaking
        // rate rather than flooring the duration at 0.01s.
        expect(clamped!.end - clamped!.start).toBeGreaterThan(3);

        // And no cue anywhere in the output is a sliver.
        const durations = cues.map((c) => c.end - c.start);
        expect(Math.min(...durations)).toBeGreaterThanOrEqual(1.0);

        // The clamp itself still holds — cues never run backwards.
        const overlaps = cues.slice(1).filter((c, i) => c.start < cues[i].end);
        expect(overlaps).toEqual([]);
    });
});

describe("BUG 4: post-dedup cues must not start early", () => {
    // lib/captionFormatter.ts:79 offsets the start by (duration * overlap) / words.length,
    // assuming the stripped overlap words consumed time proportional to their COUNT.

    it("a cue whose overlap was stripped does not start before the words it duplicated were finished", () => {
        // The overlap offset assumes the stripped words consumed time proportional to
        // their COUNT (overlap / words.length) rather than to how long they actually
        // took. Here seg2 = "the light dependent reactions. I want you to think about
        // energy." (11 words) with a 4-word overlap over 4.44s, so the code starts the
        // surviving text at 4.16 + (4.44 * 4) / 11 = 5.775.
        //
        // But those 4 duplicated words were ALREADY emitted by seg1, which ends at 6.11.
        // So the first surviving word ("I") is placed 0.335s BEFORE the duplicate it is
        // supposed to follow had even finished being spoken. This is a pure timing
        // invariant — it needs no ground truth, and it holds for any correct offset.
        const cues = consolidateSegments(WHISPER_WINDOW_SEGMENTS);

        const overlappedSegment = WHISPER_WINDOW_SEGMENTS[1]; // ends 6.11
        const postOverlapCue = findCue(cues, /want you to think/i);
        expect(postOverlapCue).toBeDefined();
        expect(postOverlapCue!.start).toBeGreaterThanOrEqual(
            overlappedSegment.end,
        );
    });
});
