import {
    ConsolidatedSegment,
    cleanCaptionText,
    endsSentence,
} from "./captionFormatter";

/**
 * Serializers for already-consolidated captions.
 *
 * PRECONDITION, enforced by the type system: every function here takes
 * `ConsolidatedSegment[]` — the output of `consolidateSegments`, which is the
 * only thing that can produce that brand. They are pure serializers: they format
 * cues into SRT/VTT/TXT/JSON text and MUST NOT re-run the formatter.
 *
 * `consolidateSegments` is NOT idempotent (see captionFormatter.test.ts):
 * consolidating a second time re-interpolates word times inside each cue's span,
 * which moves the caption break points and can DELETE words the speaker said.
 * Calling it here as well as on the display path would therefore export captions
 * that do not match the transcript on screen. Consolidation happens exactly once,
 * when raw model segments are read (`useTranscriber`).
 */

function formatTimestamp(seconds: number, separator: "," | "." = ","): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const millis = Math.floor((seconds % 1) * 1000);
    return `${pad2(hours)}:${pad2(minutes)}:${pad2(secs)}${separator}${pad3(
        millis,
    )}`;
}

function pad2(value: number): string {
    return value.toString().padStart(2, "0");
}

function pad3(value: number): string {
    return value.toString().padStart(3, "0");
}

function cueLines(
    captions: ConsolidatedSegment[],
    separator: "," | ".",
): string[] {
    const lines: string[] = [];

    captions.forEach((caption, index) => {
        lines.push(String(index + 1));
        lines.push(
            `${formatTimestamp(caption.start, separator)} --> ${formatTimestamp(
                caption.end,
                separator,
            )}`,
        );
        lines.push(caption.text);
        lines.push("");
    });

    return lines;
}

export function generateSrt(captions: ConsolidatedSegment[]): string {
    return cueLines(captions, ",").join("\n");
}

export function generateVtt(captions: ConsolidatedSegment[]): string {
    return ["WEBVTT", "", ...cueLines(captions, ".")].join("\n");
}

export function generateTxt(captions: ConsolidatedSegment[]): string {
    // Unwrap the two-line caption layout back into flowing prose. The cue text is
    // already cleaned, deduped and sentence-cased by the formatter, so this only
    // joins and fixes up sentence starts — a sentence can end mid-cue (the
    // formatter does not split a cue that is too short to stand alone), so those
    // starts were never capitalized.
    const text = cleanCaptionText(captions.map((c) => c.text).join(" "));
    if (!text) {
        return "";
    }

    const capitalized = text.replace(
        /(^|[.!?]\s+)([a-z])/g,
        (_, prefix: string, letter: string) =>
            `${prefix}${letter.toUpperCase()}`,
    );

    return endsSentence(capitalized) ? capitalized : `${capitalized}.`;
}

export function generateJson(captions: ConsolidatedSegment[]): string {
    return JSON.stringify(
        captions.map((caption) => ({
            start: caption.start,
            end: caption.end,
            text: caption.text,
        })),
        null,
        2,
    );
}
