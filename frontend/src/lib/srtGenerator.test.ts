import { describe, it, expect, vi, beforeEach } from "vitest";
import * as captionFormatter from "./captionFormatter";
import type { ConsolidatedSegment } from "./captionFormatter";
import {
    generateJson,
    generateSrt,
    generateTxt,
    generateVtt,
} from "./srtGenerator";

/**
 * The generators are PURE SERIALIZERS. They receive the output of
 * `consolidateSegments` — the very array the UI renders — and must emit it
 * verbatim. They must never re-run the formatter: `consolidateSegments` is not
 * idempotent, so a second pass exports a different transcript than the one on
 * screen, and in the worst case exports FEWER WORDS than the speaker said.
 *
 * Two things enforce that here, and NEITHER depends on the formatter staying
 * non-idempotent (Task 4's real word timestamps may well make it idempotent —
 * that would be an improvement, and must not turn this suite red):
 *
 *   1. `no generator re-runs the formatter` spies on `consolidateSegments` and
 *      asserts the export path calls it ZERO times. That is the reachability
 *      guard, and it holds no matter what the function does.
 *   2. Every fixture is a HAND-WRITTEN cue array, not `consolidateSegments(raw)`.
 *      These are serializer tests; they must not break when the formatter's
 *      timing math changes.
 */

vi.mock("./captionFormatter", async (importOriginal) => {
    const actual = await importOriginal<typeof import("./captionFormatter")>();
    return {
        ...actual,
        consolidateSegments: vi.fn(actual.consolidateSegments),
    };
});

const consolidateSpy = vi.mocked(captionFormatter.consolidateSegments);

beforeEach(() => {
    consolidateSpy.mockClear();
});

/**
 * Hand-written cue. The cast is the only way to mint the brand outside
 * `consolidateSegments`, and it is exactly what we want in a serializer test:
 * these tests care about the cues -> text mapping, not about how the cues were
 * derived.
 */
const cue = (start: number, end: number, text: string): ConsolidatedSegment =>
    ({ start, end, text } as ConsolidatedSegment);

/** Cues as the formatter emits them: sentence-cased, wrapped, non-overlapping. */
const CUES: ConsolidatedSegment[] = [
    cue(0, 4.995, "Photons which the energy"),
    cue(4.995, 10, "stores. the sugar converts."),
];

/**
 * A speaker repeating themselves INSIDE one chunk — real speech, not a
 * sliding-window artifact. The formatter deliberately KEEPS both copies (its
 * dedup only strips a raw segment's prefix against the previous segment, so it
 * never mistakes this for a chunk boundary) and happens to break the cue between
 * them.
 *
 * Re-running the formatter over these CUES would see cue N ending with the same
 * three words cue N+1 begins with, mistake the cue boundary for a chunk boundary,
 * and DELETE the repeat. That is the double-consolidation bug at its worst: the
 * old export path wrote SRT/VTT/TXT/JSON missing words that were on screen.
 */
const REPEAT_CUES: ConsolidatedSegment[] = [
    cue(0, 5.6, "Really chlorophyll absorbs photons"),
    cue(5.6, 11.79, "chlorophyll absorbs photons you."),
];

/** Cue text blocks of an SRT/VTT body, in order (index and timing lines dropped). */
function cueTextsFrom(output: string, skipHeader: boolean): string[] {
    const body = skipHeader ? output.replace(/^WEBVTT\n\n/, "") : output;
    return body
        .split("\n\n")
        .map((block) => block.trim())
        .filter((block) => block.length > 0)
        .map((block) => block.split("\n").slice(2).join("\n"));
}

describe("the export path formats exactly once", () => {
    it("no generator re-runs the formatter", () => {
        // THE reachability guard. If anyone re-introduces a consolidation call
        // in srtGenerator.ts, this fails — whether or not the formatter is
        // idempotent, and whether or not the fixture happens to be affected.
        generateSrt(CUES);
        generateVtt(CUES);
        generateTxt(CUES);
        generateJson(CUES);

        expect(consolidateSpy).not.toHaveBeenCalled();
    });

    it("SRT serializes consolidated cues verbatim and does not re-cut them", () => {
        const srt = generateSrt(CUES);
        expect(cueTextsFrom(srt, false)).toEqual(CUES.map((c) => c.text));
        expect(cueTextsFrom(srt, false)).toEqual([
            "Photons which the energy",
            "stores. the sugar converts.",
        ]);
    });

    it("SRT emits well-formed, correctly numbered cues", () => {
        expect(generateSrt(CUES)).toBe(
            [
                "1",
                "00:00:00,000 --> 00:00:04,995",
                "Photons which the energy",
                "",
                "2",
                "00:00:04,995 --> 00:00:10,000",
                "stores. the sugar converts.",
                "",
            ].join("\n"),
        );
    });

    it("VTT serializes consolidated cues verbatim and does not re-cut them", () => {
        const vtt = generateVtt(CUES);
        expect(vtt.startsWith("WEBVTT\n\n")).toBe(true);
        expect(cueTextsFrom(vtt, true)).toEqual(CUES.map((c) => c.text));
    });

    it("JSON serializes consolidated cues verbatim and does not re-cut them", () => {
        expect(JSON.parse(generateJson(CUES))).toEqual(
            CUES.map((c) => ({ start: c.start, end: c.end, text: c.text })),
        );
    });

    it("no generator drops words that the formatter deliberately kept", () => {
        // Data-loss guard. Any re-tokenizing of the cue text — a re-added
        // consolidation call, or a generator that "helpfully" dedups — deletes
        // the second copy of the repeated phrase, and this catches it.
        const both = /chlorophyll absorbs photons/gi;

        expect(generateSrt(REPEAT_CUES).match(both)?.length).toBe(2);
        expect(generateVtt(REPEAT_CUES).match(both)?.length).toBe(2);
        expect(generateJson(REPEAT_CUES).match(both)?.length).toBe(2);
        expect(generateTxt(REPEAT_CUES).match(both)?.length).toBe(2);
    });

    it("TXT does not re-tokenize — it joins the cues it was given", () => {
        expect(generateTxt(REPEAT_CUES)).toBe(
            "Really chlorophyll absorbs photons chlorophyll absorbs photons you.",
        );
    });

    it("TXT unwraps the two-line caption layout into flowing prose", () => {
        const wrapped = [
            cue(
                0,
                6.2,
                "Chlorophyll absorbs photons and\nconverts them into sugar",
            ),
        ];
        // The cue really is wrapped, otherwise this proves nothing.
        expect(wrapped.some((c) => c.text.includes("\n"))).toBe(true);

        const txt = generateTxt(wrapped);
        expect(txt).not.toContain("\n");
        expect(txt).toBe(
            "Chlorophyll absorbs photons and converts them into sugar.",
        );
    });

    it("all four generators accept an empty cue list", () => {
        expect(generateSrt([])).toBe("");
        expect(generateVtt([])).toBe("WEBVTT\n");
        expect(generateTxt([])).toBe("");
        expect(JSON.parse(generateJson([]))).toEqual([]);
    });
});

/**
 * Speaker labels in exports (Task 11).
 *
 * Two conventions were verified against a primary source before writing any of
 * this, per the brief's explicit warning that the plan's original guess (a
 * bare "SPEAKER_00:" SRT prefix, a `<v Speaker>` VTT span) was never confirmed:
 *
 *   - VTT voice span: https://www.w3.org/TR/webvtt1/#webvtt-cue-voice-span —
 *     "v" is a cue span start tag that REQUIRES an annotation (the voice's
 *     name), written inside the opening tag after a space, e.g. the spec's own
 *     `<v.first.loud Esme>It's a blue apple tree!`. `</v>` MAY be omitted when
 *     the span is the cue's only content, but is not required to be.
 *   - SRT/TXT prefix: SRT has no standard at all. whisperX — the reference
 *     tool for this exact combination (Whisper word timestamps + diarization,
 *     SPEAKER_NN-style ids) — emits `[SPEAKER_00]: text` (brackets), not a
 *     bare `SPEAKER_00:` prefix. Confirmed by reading its writer source
 *     directly (`whisperx/utils.py`, `SubtitlesWriter.iterate_result` and
 *     `WriteTXT.write_result`, both `prefix = f"[{speaker}]: "`).
 */
describe("speaker labels in exports", () => {
    const speakerCue = (
        start: number,
        end: number,
        text: string,
        speaker: string,
    ): ConsolidatedSegment =>
        ({ start, end, text, speaker } as ConsolidatedSegment);

    const DIARIZED: ConsolidatedSegment[] = [
        speakerCue(0, 2, "Hello there.", "SPEAKER_00"),
        speakerCue(2, 4, "Hi, how are you?", "SPEAKER_01"),
    ];

    const PLAIN: ConsolidatedSegment[] = [cue(0, 2, "Hello there.")];

    describe("with speakers, no display names", () => {
        it("prefixes SRT cue text with the bracketed raw label", () => {
            const srt = generateSrt(DIARIZED);
            expect(cueTextsFrom(srt, false)).toEqual([
                "[SPEAKER_00]: Hello there.",
                "[SPEAKER_01]: Hi, how are you?",
            ]);
        });

        it("shows the exact diarized SRT output", () => {
            expect(generateSrt(DIARIZED)).toBe(
                [
                    "1",
                    "00:00:00,000 --> 00:00:02,000",
                    "[SPEAKER_00]: Hello there.",
                    "",
                    "2",
                    "00:00:02,000 --> 00:00:04,000",
                    "[SPEAKER_01]: Hi, how are you?",
                    "",
                ].join("\n"),
            );
        });

        it("wraps VTT cue text in a voice span carrying the raw label", () => {
            const vtt = generateVtt(DIARIZED);
            expect(cueTextsFrom(vtt, true)).toEqual([
                "<v SPEAKER_00>Hello there.</v>",
                "<v SPEAKER_01>Hi, how are you?</v>",
            ]);
        });

        it("shows the exact diarized VTT output", () => {
            expect(generateVtt(DIARIZED)).toBe(
                [
                    "WEBVTT",
                    "",
                    "1",
                    "00:00:00.000 --> 00:00:02.000",
                    "<v SPEAKER_00>Hello there.</v>",
                    "",
                    "2",
                    "00:00:02.000 --> 00:00:04.000",
                    "<v SPEAKER_01>Hi, how are you?</v>",
                    "",
                ].join("\n"),
            );
        });

        it("prefixes each TXT turn with the raw label", () => {
            expect(generateTxt(DIARIZED)).toBe(
                "[SPEAKER_00]: Hello there.\n\n[SPEAKER_01]: Hi, how are you?",
            );
        });

        it("merges consecutive same-speaker cues into one TXT turn", () => {
            const sameSpeakerTwice: ConsolidatedSegment[] = [
                speakerCue(0, 2, "Part one.", "SPEAKER_00"),
                speakerCue(2, 4, "Part two.", "SPEAKER_00"),
            ];
            expect(generateTxt(sameSpeakerTwice)).toBe(
                "[SPEAKER_00]: Part one. Part two.",
            );
        });

        it("emits a `speaker` field per JSON segment, keeping the raw label", () => {
            const parsed = JSON.parse(generateJson(DIARIZED));
            expect(parsed.segments[0]).toEqual({
                start: 0,
                end: 2,
                text: "Hello there.",
                speaker: "SPEAKER_00",
            });
            expect(parsed.segments[1].speaker).toBe("SPEAKER_01");
        });

        it("JSON carries an (empty) top-level speakerNames map when none were given", () => {
            const parsed = JSON.parse(generateJson(DIARIZED));
            expect(parsed.speakerNames).toEqual({});
        });
    });

    describe("with speakers AND display names", () => {
        const NAMES = { SPEAKER_00: "Alice", SPEAKER_01: "Bob" };

        it("SRT shows the display name in place of the raw label", () => {
            expect(generateSrt(DIARIZED, NAMES)).toContain(
                "[Alice]: Hello there.",
            );
            expect(generateSrt(DIARIZED, NAMES)).toContain(
                "[Bob]: Hi, how are you?",
            );
        });

        it("VTT voice span carries the display name", () => {
            expect(generateVtt(DIARIZED, NAMES)).toContain("<v Alice>");
            expect(generateVtt(DIARIZED, NAMES)).toContain("<v Bob>");
        });

        it("TXT turn is prefixed with the display name", () => {
            expect(generateTxt(DIARIZED, NAMES)).toBe(
                "[Alice]: Hello there.\n\n[Bob]: Hi, how are you?",
            );
        });

        it("an unnamed speaker in a partial map still falls back to its raw label", () => {
            expect(generateSrt(DIARIZED, { SPEAKER_00: "Alice" })).toContain(
                "[SPEAKER_01]: Hi, how are you?",
            );
        });

        it("JSON keeps the raw `speaker` field AND surfaces the names map separately", () => {
            const parsed = JSON.parse(generateJson(DIARIZED, NAMES));
            expect(parsed.segments[0].speaker).toBe("SPEAKER_00");
            expect(parsed.speakerNames).toEqual(NAMES);
        });
    });

    describe("without speakers — byte-identical to today, no markup at all", () => {
        // The assertion that matters most in this whole file: a legacy or
        // never-diarized transcript must produce EXACTLY what it always has.
        // NB: don't assert `.not.toContain(":")` on SRT — the cue timestamp
        // line ("00:00:00,000 --> ...") is full of colons — assert on the text
        // line specifically instead.
        it("SRT text line carries no speaker prefix", () => {
            const srt = generateSrt(PLAIN);
            expect(srt).not.toMatch(/^\[.*\]:/m);
            expect(cueTextsFrom(srt, false)).toEqual(["Hello there."]);
        });

        it("VTT contains no voice span", () => {
            expect(generateVtt(PLAIN)).not.toContain("<v");
        });

        it("TXT carries no speaker prefix", () => {
            expect(generateTxt(PLAIN)).toBe("Hello there.");
        });

        it("JSON stays a bare array with no `speaker` key and no wrapper object", () => {
            const parsed = JSON.parse(generateJson(PLAIN));
            expect(Array.isArray(parsed)).toBe(true);
            expect(parsed).toEqual([
                { start: 0, end: 2, text: "Hello there." },
            ]);
        });

        it("passing a names map with no speakers present changes nothing", () => {
            const names = { SPEAKER_00: "Alice" };
            expect(generateSrt(PLAIN, names)).toBe(generateSrt(PLAIN));
            expect(generateVtt(PLAIN, names)).toBe(generateVtt(PLAIN));
            expect(generateTxt(PLAIN, names)).toBe(generateTxt(PLAIN));
            expect(generateJson(PLAIN, names)).toBe(generateJson(PLAIN));
        });

        it("existing no-speaker fixtures (CUES, REPEAT_CUES) stay byte-identical with the new code path", () => {
            expect(generateSrt(CUES)).not.toMatch(/^\[.*\]:/m);
            expect(generateVtt(CUES)).not.toContain("<v");
            expect(JSON.parse(generateJson(CUES))).toEqual(
                CUES.map((c) => ({ start: c.start, end: c.end, text: c.text })),
            );
            expect(generateTxt(REPEAT_CUES)).toBe(
                "Really chlorophyll absorbs photons chlorophyll absorbs photons you.",
            );
        });
    });
});
