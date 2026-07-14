import { describe, it, expect } from "vitest";
import {
    ConsolidatedSegment,
    WordToken,
    cleanCaptionText,
    consolidateSegments,
    normalizeWordTokens,
} from "./captionFormatter";
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

    it("floors on the resulting DURATION, not merely on `segmentEnd > start`", () => {
        // The near-miss the `segmentEnd > start` guard let through. seg2's start is
        // clamped forward to 30.0 and its end is 30.001 — the guard passes ("it
        // ends after it starts") and the whole 11-word segment is crushed into a
        // ONE-MILLISECOND cue, which is the exact failure the span floor exists to
        // prevent. Only the legacy no-words path can reach this, and that path
        // serves users' pre-existing transcripts.
        const cues = consolidateSegments([
            {
                start: 20.0,
                end: 30.0,
                text: "and that is how the calvin cycle fixes carbon into sugar.",
            },
            {
                start: 25.8,
                end: 30.001,
                text: "an entirely different long sentence that the second window transcribed here",
            },
        ]);

        const crushed = cues.find((c) => /entirely different/i.test(c.text))!;
        expect(crushed.end - crushed.start).toBeGreaterThan(3);
        expect(Math.min(...cues.map((c) => c.end - c.start))).toBeGreaterThan(
            1.0,
        );
    });

    it("still believes a segment whose span is short but genuine (a one-word row)", () => {
        // The other side of the same guard: word-granular database rows are
        // routinely far shorter than a nominal word, and inflating THOSE would
        // break the reload. The floor must fire on 1ms and not on 20ms.
        const cues = consolidateSegments([
            { start: 0.0, end: 0.4, text: "photosynthesis" },
            { start: 0.4, end: 0.42, text: "is" },
            { start: 0.42, end: 1.1, text: "efficient." },
        ]);
        expect(cues).toHaveLength(1);
        expect(cues[0].start).toBe(0);
        expect(cues[0].end).toBe(1.1);
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

/**
 * The model's REAL word timings, from `onnx-community/whisper-tiny_timestamped`
 * with `return_timestamps: 'word'` on a 12.82s clip (two sentences, 1.2s pause).
 * Verbatim `output.chunks` — note the leading spaces Whisper uses to mark word
 * boundaries, and "-level" / "-attention" arriving WITHOUT one because they
 * continue the previous word.
 */
const REAL_WORDS: WordToken[] = [
    { text: " The", start: 0.0, end: 0.16 },
    { text: " quick", start: 0.16, end: 0.38 },
    { text: " brown", start: 0.38, end: 0.72 },
    { text: " fox", start: 0.72, end: 1.06 },
    { text: " jumps", start: 1.06, end: 1.38 },
    { text: " over", start: 1.38, end: 1.68 },
    { text: " the", start: 1.68, end: 1.86 },
    { text: " lazy", start: 1.86, end: 2.12 },
    { text: " dog,", start: 2.12, end: 2.5 },
    { text: " this", start: 2.72, end: 3.0 },
    { text: " sentence", start: 3.0, end: 3.3 },
    { text: " is", start: 3.3, end: 3.54 },
    { text: " used", start: 3.54, end: 3.8 },
    { text: " for", start: 3.8, end: 4.04 },
    { text: " testing", start: 4.04, end: 4.4 },
    { text: " speech", start: 4.4, end: 4.8 },
    { text: " recognition", start: 4.8, end: 5.36 },
    { text: " systems.", start: 5.36, end: 6.22 },
    { text: " Word", start: 7.62, end: 7.66 },
    { text: "-level", start: 7.66, end: 8.0 },
    { text: " timestamps", start: 8.0, end: 8.6 },
    { text: " are", start: 8.6, end: 8.86 },
    { text: " extracted", start: 8.86, end: 9.3 },
    { text: " using", start: 9.3, end: 9.74 },
    { text: " dynamic", start: 9.74, end: 10.3 },
    { text: " time", start: 10.3, end: 10.68 },
    { text: " warping", start: 10.68, end: 11.04 },
    { text: " over", start: 11.04, end: 11.34 },
    { text: " the", start: 11.34, end: 11.52 },
    { text: " cross", start: 11.52, end: 11.82 },
    { text: "-attention", start: 11.82, end: 12.28 },
    { text: " weights.", start: 12.28, end: 12.8 },
];

/** The chunk-level segments the SAME clip produced with `return_timestamps: true`. */
const REAL_SEGMENTS: TranscriptionSegment[] = [
    {
        start: 0,
        end: 5.52,
        text: " The quick brown fox jumps over the lazy dog, this sentence is used for testing speech recognition",
    },
    { start: 5.52, end: 7.52, text: " systems." },
    {
        start: 7.52,
        end: 12.24,
        text: " Word-level timestamps are extracted using dynamic time warping over the cross-attention",
    },
    { start: 12.24, end: 12.76, text: " weights." },
];

describe("real word timestamps replace the fabrication", () => {
    it("times every cue from the model's own word times, not from a chunk's span", () => {
        const cues = consolidateSegments(REAL_SEGMENTS, REAL_WORDS);

        // Every cue boundary must be a real word boundary. Nothing here is
        // interpolated, so each edge lands exactly on a measured time.
        const starts = new Set(REAL_WORDS.map((w) => w.start));
        const ends = new Set(REAL_WORDS.map((w) => w.end));
        for (const cue of cues) {
            expect(starts.has(cue.start)).toBe(true);
            expect(ends.has(cue.end)).toBe(true);
        }

        // The clip has 1.2s of silence after "systems.", which really ends at
        // 6.22. The fabricating path cannot see a pause — it only knows the chunk
        // ran to 7.52 and that "systems." is its last word, so it stretches the
        // word over the silence and holds the caption on screen 1.3s too long.
        const sentence = cues.find((c) => /systems\./i.test(c.text))!;
        expect(sentence.end).toBe(6.22);

        const fabricated = consolidateSegments(REAL_SEGMENTS);
        const fabricatedSentence = fabricated.find((c) =>
            /systems\./i.test(c.text),
        )!;
        expect(fabricatedSentence.end).toBeCloseTo(7.52, 2);
        expect(fabricatedSentence.end - sentence.end).toBeCloseTo(1.3, 2);
    });

    it("folds a word's continuation fragment back into the word", () => {
        // Whisper emits "Word-level" as [" Word", "-level"]. Tokens are re-joined
        // with a space, so a fragment left standing prints "Word -level".
        const text = consolidateSegments(REAL_SEGMENTS, REAL_WORDS)
            .map((c) => c.text.replace(/\n/g, " "))
            .join(" ");
        expect(text).toMatch(/Word-level/);
        expect(text).not.toMatch(/Word -level/);
        expect(text).toMatch(/cross-attention/);
        expect(text).not.toMatch(/cross -attention/);
    });

    it("never runs a cue backwards, and never past the audio", () => {
        const cues = consolidateSegments(REAL_SEGMENTS, REAL_WORDS);
        expect(cues.length).toBeGreaterThan(1);
        for (let i = 0; i < cues.length; i++) {
            expect(cues[i].end).toBeGreaterThan(cues[i].start);
            if (i > 0)
                expect(cues[i].start).toBeGreaterThanOrEqual(cues[i - 1].end);
        }
        expect(cues.at(-1)!.end).toBeLessThanOrEqual(12.83);
    });
});

/**
 * Chinese word tokens as transformers.js actually produces them.
 *
 * `combineTokensIntoWords` (tokenizers.js) routes chinese/japanese/thai/lao/
 * myanmar through `splitTokensOnUnicode`, which splits on codepoints and NEVER
 * emits a leading space; `mergePunctuations` then appends "。" to the word before
 * it. So EVERY word here begins with a CJK character and none begins with a
 * space — the exact input on which "no leading space means this continues the
 * previous word" folds an entire transcript into one token.
 */
const CHINESE_WORDS: WordToken[] = [
    { text: "光", start: 0.0, end: 0.32 },
    { text: "合", start: 0.32, end: 0.6 },
    { text: "作", start: 0.6, end: 0.88 },
    { text: "用", start: 0.88, end: 1.16 },
    { text: "是", start: 1.16, end: 1.44 },
    { text: "植", start: 1.44, end: 1.72 },
    { text: "物", start: 1.72, end: 2.0 },
    { text: "把", start: 2.0, end: 2.26 },
    { text: "光", start: 2.26, end: 2.54 },
    { text: "能", start: 2.54, end: 2.82 },
    { text: "转", start: 2.82, end: 3.1 },
    { text: "化", start: 3.1, end: 3.38 },
    { text: "为", start: 3.38, end: 3.66 },
    { text: "化", start: 3.66, end: 3.94 },
    { text: "学", start: 3.94, end: 4.22 },
    { text: "能。", start: 4.22, end: 4.6 },
    { text: "叶", start: 5.8, end: 6.08 },
    { text: "绿", start: 6.08, end: 6.36 },
    { text: "素", start: 6.36, end: 6.64 },
    { text: "吸", start: 6.64, end: 6.92 },
    { text: "收", start: 6.92, end: 7.2 },
    { text: "光", start: 7.2, end: 7.48 },
    { text: "子。", start: 7.48, end: 7.9 },
];

/** The chunk-level segments the same Chinese audio produced. */
const CHINESE_SEGMENTS: TranscriptionSegment[] = [
    { start: 0, end: 4.6, text: "光合作用是植物把光能转化为化学能。" },
    { start: 5.8, end: 7.9, text: "叶绿素吸收光子。" },
];

describe("CJK: a script with no spaces must not collapse into one word", () => {
    it("keeps every Chinese word as its own token", () => {
        // The bug: `continuesPreviousWord = !/^\s/.test(word.text)` is true for
        // EVERY Chinese word, so all 23 fold into tokens[0] — one token, one cue
        // spanning the whole audio, one database row holding the entire transcript.
        const tokens = normalizeWordTokens(CHINESE_WORDS);
        expect(tokens).toHaveLength(CHINESE_WORDS.length);
        expect(tokens[0]).toEqual({ text: "光", start: 0.0, end: 0.32 });
        expect(tokens.at(-1)).toEqual({ text: "子。", start: 7.48, end: 7.9 });
    });

    it("joins Chinese words WITHOUT spaces", () => {
        // Not folding is only half the fix: `joinWords` re-joined with " ", so
        // simply keeping the tokens apart would print "光 合 作 用".
        const cues = consolidateSegments(CHINESE_SEGMENTS, CHINESE_WORDS);
        const text = cues.map((c) => c.text.replace(/\n/g, " ")).join("");

        expect(text).toContain("光合作用是植物把光能转化为化学能。");
        expect(text).toContain("叶绿素吸收光子。");
        expect(text).not.toMatch(/光 合/);
        expect(text).not.toMatch(/\s/);
    });

    it("cuts more than one cue, on real word times, and honours the 1.2s pause", () => {
        const cues = consolidateSegments(CHINESE_SEGMENTS, CHINESE_WORDS);

        expect(cues.length).toBeGreaterThan(1);
        // The first sentence really ends at 4.6 and the next starts at 5.8. A
        // single folded token would have spanned 0 -> 7.9 and erased the pause.
        expect(cues[0].end).toBe(4.6);
        expect(cues[1].start).toBe(5.8);
        expect(cues.at(-1)!.end).toBe(7.9);
    });

    it("still folds an English continuation fragment (the CJK rule did not break it)", () => {
        expect(
            normalizeWordTokens([
                { text: " Word", start: 0, end: 0.2 },
                { text: "-level", start: 0.2, end: 0.6 },
            ]),
        ).toEqual([{ text: "Word-level", start: 0, end: 0.6 }]);
    });
});

describe("a word-granular transcript round-trips through the database exactly", () => {
    // The database stores ONE WORD PER ROW and the reload re-derives the word
    // list by splitting each row's CLEANED text on spaces. Anything that inserts
    // a space inside a persisted word turns one token into two on reload — and
    // "3." satisfies endsSentence(), so a cue can even break in the middle of a
    // number on reload but not live.

    it("does not split a number or an abbreviation", () => {
        expect(cleanCaptionText("3.14")).toBe("3.14");
        expect(cleanCaptionText("1,000")).toBe("1,000");
        expect(cleanCaptionText("3:30")).toBe("3:30");
        expect(cleanCaptionText("U.S.A.")).toBe("U.S.A.");
        // ...while still fixing the thing the rule exists for.
        expect(cleanCaptionText("Hello,world")).toBe("Hello, world");
        expect(cleanCaptionText("One sentence.Then another")).toBe(
            "One sentence. Then another",
        );
    });

    it("renders identically live and on reload, for words containing 3.14 and 1,000", () => {
        const words: WordToken[] = [
            { text: " Pi", start: 0.0, end: 0.3 },
            { text: " is", start: 0.3, end: 0.5 },
            { text: " roughly", start: 0.5, end: 0.98 },
            { text: " 3.14", start: 0.98, end: 1.7 },
            { text: " and", start: 1.7, end: 1.9 },
            { text: " a", start: 1.9, end: 2.0 },
            { text: " kilometre", start: 2.0, end: 2.7 },
            { text: " is", start: 2.7, end: 2.9 },
            // A ZERO-DURATION word: DTW quantises to 0.02s and does emit these.
            { text: " 1,000", start: 2.9, end: 2.9 },
            { text: " metres.", start: 3.2, end: 3.9 },
        ];

        const live = consolidateSegments([], words);
        const persisted = normalizeWordTokens(words).map((word) => ({
            start: word.start,
            end: word.end,
            text: word.text,
        }));
        const reloaded = consolidateSegments(persisted);

        expect(reloaded).toEqual(live);

        // The numbers survived as single words on both paths.
        const text = live.map((c) => c.text.replace(/\n/g, " ")).join(" ");
        expect(text).toContain("3.14");
        expect(text).toContain("1,000");
        expect(text).not.toContain("3. 14");
        expect(text).not.toContain("1, 000");

        // And the zero-duration word was floored on the way in, so the reload
        // cannot inflate it to the next word's start and swallow the 0.3s pause.
        const zeroDuration = persisted.find((s) => s.text === "1,000")!;
        expect(zeroDuration.end).toBeGreaterThan(zeroDuration.start);
        expect(zeroDuration.end).toBeLessThan(3.2);
    });
});

describe("a segment that knows its own end is not stretched past it", () => {
    // The span floor exists for ONE case: the monotonic clamp pushing `start` at
    // or past the segment's end. Applied unconditionally it also fires on any
    // speaker faster than 2.9 words/second, inventing time beyond the segment's
    // real end — and the next segment's start is then clamped forward to that
    // invented end, so the error compounds down the transcript.
    it("keeps a fast segment's words inside the segment", () => {
        // 12 words in 3.0s = 4 words/second. n * NOMINAL = 4.2s > 3.0s.
        const cues = consolidateSegments([
            {
                start: 0,
                end: 3.0,
                text: "one two three four five six seven eight nine ten more words.",
            },
            {
                start: 3.0,
                end: 6.0,
                text: "and then the second segment speaks.",
            },
        ]);
        expect(cues[0].end).toBeLessThanOrEqual(3.0);
        // The second segment therefore still starts where it actually starts.
        const second = cues.find((c) => /second segment/i.test(c.text))!;
        expect(second.start).toBeGreaterThanOrEqual(3.0);
        expect(second.end).toBeLessThanOrEqual(6.0);
    });
});

describe("CJK captions must obey the line-length limit", () => {
    // `wrapCaptionText` split on `" "` and nothing else, so a Chinese cue offered
    // NO break point: it came back as one line however long, and MAX_LINE_CHARS
    // (42) was silently violated on exactly the cues C2 made reachable.
    const MAX_LINE_CHARS = 42;

    /** One long Chinese sentence, one word (= one character) every 0.1s. */
    const sentence =
        "光合作用是植物利用阳光把二氧化碳和水转化为葡萄糖并且释放氧气的一个非常重要的生物化学过程。";
    const longChineseWords: WordToken[] = Array.from(sentence).map(
        (char, index) => ({
            text: char,
            start: index * 0.1,
            end: (index + 1) * 0.1,
        }),
    );

    it("wraps a long Chinese cue instead of emitting one over-long line", () => {
        const cues = consolidateSegments([], longChineseWords);

        const overLong = cues
            .flatMap((cue) => cue.text.split("\n"))
            .filter((line) => line.length > MAX_LINE_CHARS);
        expect(overLong).toEqual([]);

        // It really is long enough to have needed wrapping.
        const wrapped = cues.find((cue) => cue.text.includes("\n"));
        expect(wrapped).toBeDefined();
    });

    it("wraps without inventing spaces or losing a character", () => {
        const cues = consolidateSegments([], longChineseWords);
        const rendered = cues
            .map((cue) => cue.text.replace(/\n/g, ""))
            .join("");

        expect(rendered).toBe(sentence);
        expect(rendered).not.toMatch(/\s/);
    });

    it("never opens a line with a CJK closing mark", () => {
        const cues = consolidateSegments([], longChineseWords);
        for (const cue of cues) {
            for (const line of cue.text.split("\n")) {
                expect(line[0]).not.toMatch(/[、，。！？：；）」』]/);
            }
        }
    });

    it("wraps English exactly as before (the CJK break points did not disturb it)", () => {
        const cues = consolidateSegments([
            {
                start: 0,
                end: 5,
                text: "The quick brown fox jumps over the lazy dog while nobody watches it at all.",
            },
        ]);
        for (const cue of cues) {
            for (const line of cue.text.split("\n")) {
                expect(line.length).toBeLessThanOrEqual(MAX_LINE_CHARS);
                // Breaks only ever fall on a space: no word was cut in half.
                expect(line).toBe(line.trim());
            }
        }
        expect(flat(cues)).toContain("quick brown fox");
    });

    it("closes an unterminated CJK cue with 。, not an ASCII period", () => {
        // `normalizeCaptionStarts` appended "." unconditionally — "叶绿素吸收光子."
        const cues = consolidateSegments(
            [],
            Array.from("叶绿素吸收光子").map((char, index) => ({
                text: char,
                start: index * 0.3,
                end: (index + 1) * 0.3,
            })),
        );
        const last = cues.at(-1)!.text;
        expect(last.endsWith("。")).toBe(true);
        expect(last.endsWith(".")).toBe(false);
    });

    it("still closes an unterminated English cue with an ASCII period", () => {
        const cues = consolidateSegments([
            { start: 0, end: 2, text: "no full stop here" },
        ]);
        expect(cues.at(-1)!.text.endsWith(".")).toBe(true);
    });
});

describe("I3: one persisted row must re-derive as exactly ONE token", () => {
    // Whisper emits "Hello,world" as a SINGLE word chunk. `cleanCaptionText` puts
    // the missing space back — so that one persisted row re-derived as TWO tokens
    // on reload, and the reload saw a token stream the live render never had.
    // Enforced by construction now: `normalizeWordTokens` does the split itself,
    // so both paths start from the same tokens.
    const words: WordToken[] = [
        { text: " Hello,world", start: 0, end: 0.8 },
        { text: " again", start: 0.8, end: 1.2 },
    ];

    it("splits the glued token on the way in, sharing its span by character length", () => {
        const tokens = normalizeWordTokens(words);
        expect(tokens.map((t) => t.text)).toEqual(["Hello,", "world", "again"]);

        // The pieces tile the original word's span exactly: no time invented, none lost.
        expect(tokens[0].start).toBe(0);
        expect(tokens[1].end).toBe(0.8);
        expect(tokens[0].end).toBe(tokens[1].start);
    });

    it("renders identically live and on reload", () => {
        const live = consolidateSegments([], words);
        const persisted = normalizeWordTokens(words).map((word) => ({
            start: word.start,
            end: word.end,
            text: word.text,
        }));

        // The invariant itself: every persisted row is ONE token when re-split.
        for (const row of persisted) {
            expect(cleanCaptionText(row.text).split(" ")).toHaveLength(1);
        }

        expect(consolidateSegments(persisted)).toEqual(live);
        expect(flat(live)).toContain("Hello, world");
    });
});
