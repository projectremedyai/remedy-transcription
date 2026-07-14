import { describe, it, expect } from "vitest";
import { consolidateSegments } from "./captionFormatter";
import {
    generateJson,
    generateSrt,
    generateTxt,
    generateVtt,
} from "./srtGenerator";
import type { TranscriptionSegment } from "../services/types";

/**
 * The generators are PURE SERIALIZERS. They receive the output of
 * `consolidateSegments` — the very array the UI renders — and must emit it
 * verbatim. They must never re-run the formatter: `consolidateSegments` is not
 * idempotent, so a second pass exports a different transcript than the one on
 * screen, and in the worst case exports FEWER WORDS than the speaker said.
 *
 * Every fixture below is built the way production builds it (consolidate the raw
 * model segments exactly once), and every test first PROVES its fixture is
 * discriminating — that re-consolidating it would visibly change it. That is what
 * makes these tests fail the moment someone re-introduces consolidation — or any
 * other re-tokenizing of the cue text — inside `srtGenerator.ts`.
 */

/** Raw segments whose cues get RE-CUT (a word migrates) if consolidated twice. */
const RAW_RECUT: TranscriptionSegment[] = [
    { start: 0, end: 6.66, text: "photons which the energy stores." },
    { start: 6.66, end: 10, text: "the sugar converts." },
];

/**
 * A speaker repeating themselves INSIDE one chunk — real speech, not a
 * sliding-window artifact. The raw dedup only strips a segment's PREFIX against
 * what came before, so it never sees this repeat and correctly keeps both copies.
 * The formatter then happens to break the cue between them. A second pass now
 * sees cue N ending with the same three words cue N+1 begins with, mistakes the
 * cue boundary for a chunk boundary, and DELETES the repeat.
 *
 * This is the double-consolidation bug at its worst: the old export path wrote
 * SRT/VTT/TXT/JSON that was missing words which were visibly on screen.
 */
const RAW_INTERNAL_REPEAT: TranscriptionSegment[] = [
    {
        start: 0,
        end: 11.79,
        text: "really chlorophyll absorbs photons chlorophyll absorbs photons you.",
    },
];

const cuesOf = (raw: TranscriptionSegment[]) => consolidateSegments(raw);

const flat = (segs: TranscriptionSegment[]) =>
    segs.map((s) => s.text.replace(/\n/g, " ")).join(" ");

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
    it("SRT serializes consolidated cues verbatim and does not re-cut them", () => {
        const cues = cuesOf(RAW_RECUT);

        // The fixture is discriminating: a second pass moves "stores." into the
        // previous cue. A re-consolidating generator would show that.
        expect(consolidateSegments(cues)).not.toEqual(cues);

        const srt = generateSrt(cues);
        expect(cueTextsFrom(srt, false)).toEqual(cues.map((c) => c.text));
        expect(cueTextsFrom(srt, false)).toEqual([
            "Photons which the energy",
            "stores. the sugar converts.",
        ]);
        // The re-consolidated cut, which must NOT reach the file.
        expect(srt).not.toContain("Photons which the energy stores.");
    });

    it("SRT emits well-formed, correctly numbered cues", () => {
        expect(generateSrt(cuesOf(RAW_RECUT))).toBe(
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
        const cues = cuesOf(RAW_RECUT);
        expect(consolidateSegments(cues)).not.toEqual(cues);

        const vtt = generateVtt(cues);
        expect(vtt.startsWith("WEBVTT\n\n")).toBe(true);
        expect(cueTextsFrom(vtt, true)).toEqual(cues.map((c) => c.text));
        expect(vtt).not.toContain("Photons which the energy stores.");
    });

    it("JSON serializes consolidated cues verbatim and does not re-cut them", () => {
        const cues = cuesOf(RAW_RECUT);
        expect(consolidateSegments(cues)).not.toEqual(cues);

        expect(JSON.parse(generateJson(cues))).toEqual(
            cues.map((c) => ({ start: c.start, end: c.end, text: c.text })),
        );
    });

    it("no generator drops words that the formatter deliberately kept", () => {
        const cues = cuesOf(RAW_INTERNAL_REPEAT);

        // What the user sees: both copies of the repeated phrase.
        expect(flat(cues)).toBe(
            "Really chlorophyll absorbs photons chlorophyll absorbs photons you.",
        );
        expect(flat(cues).match(/chlorophyll absorbs photons/gi)?.length).toBe(
            2,
        );

        // The fixture is discriminating: re-running the formatter over the CUES
        // deletes three words.
        expect(
            flat(consolidateSegments(cues)).match(
                /chlorophyll absorbs photons/gi,
            )?.length,
        ).toBe(1);

        // Every export must carry what is on screen.
        const both = /chlorophyll absorbs photons/gi;
        expect(generateSrt(cues).match(both)?.length).toBe(2);
        expect(generateVtt(cues).match(both)?.length).toBe(2);
        expect(generateJson(cues).match(both)?.length).toBe(2);
        expect(generateTxt(cues).match(both)?.length).toBe(2);
    });

    it("TXT does not re-tokenize — it joins the cues it was given", () => {
        expect(generateTxt(cuesOf(RAW_INTERNAL_REPEAT))).toBe(
            "Really chlorophyll absorbs photons chlorophyll absorbs photons you.",
        );
    });

    it("TXT unwraps the two-line caption layout into flowing prose", () => {
        const cues = cuesOf([
            {
                start: 0,
                end: 6.2,
                text: "chlorophyll absorbs photons and converts them into sugar",
            },
        ]);
        // The cue really is wrapped, otherwise this proves nothing.
        expect(cues.some((c) => c.text.includes("\n"))).toBe(true);

        const txt = generateTxt(cues);
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
