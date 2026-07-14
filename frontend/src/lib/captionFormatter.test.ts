import { describe, it, expect } from "vitest";
import { consolidateSegments } from "./captionFormatter";
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

/**
 * The Task 3 plan claimed `consolidateSegments` was self-idempotent, which is what
 * made the export path's second consolidation "near-harmless". IT IS NOT, and the
 * cost is real: display shows `consolidate(raw)` while the export used to write
 * `consolidate(consolidate(raw))` — a different cut of the same words.
 */
describe("consolidateSegments is NOT idempotent", () => {
    // Two clean, strictly sequential segments. No time overlap, no zero-duration
    // chunk, no dedup — none of the degenerate cases. The non-idempotence is
    // intrinsic to the timing math, not an artifact of bad input.
    const CLEAN_SEQUENTIAL: TranscriptionSegment[] = [
        { start: 0, end: 6.66, text: "photons which the energy stores." },
        { start: 6.66, end: 10, text: "the sugar converts." },
    ];

    it("moves a word between cues on a second pass", () => {
        const once = consolidateSegments(CLEAN_SEQUENTIAL);
        const twice = consolidateSegments(once);

        // Pass 1 breaks on duration: adding "stores." would push the cue past
        // MAX_CAPTION_DURATION, so it starts a new cue before it.
        expect(once.map((c) => c.text)).toEqual([
            "Photons which the energy",
            "stores. the sugar converts.",
        ]);

        // Pass 2 re-interpolates each word evenly across its CUE's span, losing the
        // raw segment's word times. The duration rule no longer fires, so the
        // sentence-end rule wins instead and "stores." lands in the first cue.
        expect(twice.map((c) => c.text)).toEqual([
            "Photons which the energy stores.",
            "The sugar converts.",
        ]);

        expect(twice).not.toEqual(once);
        // Not a rounding wobble: the cue boundary moves by ~1.46 SECONDS.
        expect(Math.abs(twice[0].end - once[0].end)).toBeGreaterThan(1.4);
    });

    it("is stable on the realistic sliding-window fixture (which is why this went unnoticed)", () => {
        // Idempotence DOES hold here. That is exactly why the double consolidation
        // looked safe. It is a property of this fixture, not of the function.
        const once = consolidateSegments(WHISPER_WINDOW_SEGMENTS);
        expect(consolidateSegments(once)).toEqual(once);
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
