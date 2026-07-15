import {
    ConsolidatedSegment,
    cleanCaptionText,
    endsSentence,
} from "./captionFormatter";
import type { SpeakerNames } from "../services/types";

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
 *
 * Every generator also takes an OPTIONAL `names` map (`SpeakerNames`, Task 10 —
 * `{ SPEAKER_00: "Alice" }`). Staying pure: the map is a parameter, never fetched
 * here, so these functions still have no DB/IPC dependency and no caller is forced
 * to have one either. Omitting it (or a cue's `speaker` being unset) must reproduce
 * today's output exactly — see the "no speaker" tests below.
 */

/**
 * What to print for a speaker: the display name the user gave it, falling back
 * to the raw opaque label when unnamed (or when no map was supplied at all).
 *
 * Speaker ids are OPAQUE labels (`SPEAKER_00`, ...) — this never parses one,
 * never indexes anything by it, and never invents a "Speaker N" from it. It only
 * ever prints the label as-is or looks it up in the map verbatim.
 */
function speakerDisplay(speaker: string, names?: SpeakerNames): string {
    return names?.[speaker] ?? speaker;
}

/**
 * The speaker prefix used by SRT and TXT, e.g. "[Alice]: " or "[SPEAKER_00]: "
 * when nobody has renamed that speaker yet. "" — no markup at all — when the
 * cue carries no speaker, which is what keeps an undiarized export identical to
 * what this has always produced.
 *
 * SRT has no official speaker convention to defer to. This bracket-and-colon
 * format is not a guess: it is what whisperX — the reference tool for exactly
 * this combination, Whisper word timestamps plus diarization — actually emits,
 * for SRT, VTT and TXT alike (`SubtitlesWriter.iterate_result` and
 * `WriteTXT.write_result` in whisperx/utils.py both do
 * `prefix = f"[{speaker}]: "`). The plan's original guess of a bare
 * "SPEAKER_00:" prefix (no brackets) does not match what that tool emits.
 */
function speakerPrefix(
    speaker: string | undefined,
    names?: SpeakerNames,
): string {
    return speaker === undefined ? "" : `[${speakerDisplay(speaker, names)}]: `;
}

/**
 * The WebVTT cue voice span, e.g. `<v Alice>text</v>`. Plain `caption.text`,
 * unwrapped, when the cue carries no speaker — an undiarized VTT export must
 * contain no `<v` at all.
 *
 * Per the spec (https://www.w3.org/TR/webvtt1/#webvtt-cue-voice-span): "v" is a
 * cue span start tag that REQUIRES an annotation, and "the annotation represents
 * the name of the voice" — written after a space inside the opening tag, e.g.
 * the spec's own `<v.first.loud Esme>It's a blue apple tree!`. The end tag
 * `</v>` MAY be omitted when the voice span is the cue's only content (which it
 * always is here), but omitting it is not required; it is kept here for an
 * unambiguous, self-closing cue.
 */
function voiceSpan(caption: ConsolidatedSegment, names?: SpeakerNames): string {
    return caption.speaker === undefined
        ? caption.text
        : `<v ${speakerDisplay(caption.speaker, names)}>${caption.text}</v>`;
}

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
    renderText: (caption: ConsolidatedSegment) => string,
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
        lines.push(renderText(caption));
        lines.push("");
    });

    return lines;
}

export function generateSrt(
    captions: ConsolidatedSegment[],
    names?: SpeakerNames,
): string {
    return cueLines(
        captions,
        ",",
        (caption) => `${speakerPrefix(caption.speaker, names)}${caption.text}`,
    ).join("\n");
}

export function generateVtt(
    captions: ConsolidatedSegment[],
    names?: SpeakerNames,
): string {
    return [
        "WEBVTT",
        "",
        ...cueLines(captions, ".", (caption) => voiceSpan(caption, names)),
    ].join("\n");
}

/**
 * Group consecutive cues that share a speaker into "turns". A cue never spans
 * two speakers — `shouldBreakBefore` in captionFormatter forces a hard cue
 * break on any speaker change, including into or out of "no speaker" — so
 * grouping adjacent cues by strict equality on `speaker` is exact, and it
 * collapses to exactly ONE turn covering every cue when nothing carries a
 * speaker at all. That collapse is what keeps an undiarized TXT export
 * identical to what this has always produced.
 */
function speakerTurns(
    captions: ConsolidatedSegment[],
): ConsolidatedSegment[][] {
    const turns: ConsolidatedSegment[][] = [];

    for (const caption of captions) {
        const current = turns[turns.length - 1];
        if (current && current[0].speaker === caption.speaker) {
            current.push(caption);
        } else {
            turns.push([caption]);
        }
    }

    return turns;
}

// Unwrap a turn's two-line caption layout back into flowing prose. The cue text
// is already cleaned, deduped and sentence-cased by the formatter, so this only
// joins and fixes up sentence starts — a sentence can end mid-cue (the
// formatter does not split a cue that is too short to stand alone), so those
// starts were never capitalized.
function proseForTurn(turn: ConsolidatedSegment[]): string {
    const text = cleanCaptionText(turn.map((c) => c.text).join(" "));
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

export function generateTxt(
    captions: ConsolidatedSegment[],
    names?: SpeakerNames,
): string {
    return speakerTurns(captions)
        .map((turn) => {
            const prose = proseForTurn(turn);
            return prose
                ? `${speakerPrefix(turn[0].speaker, names)}${prose}`
                : "";
        })
        .filter((block) => block.length > 0)
        .join("\n\n");
}

export function generateJson(
    captions: ConsolidatedSegment[],
    names?: SpeakerNames,
): string {
    const segments = captions.map((caption) =>
        caption.speaker === undefined
            ? { start: caption.start, end: caption.end, text: caption.text }
            : {
                  start: caption.start,
                  end: caption.end,
                  text: caption.text,
                  speaker: caption.speaker,
              },
    );

    const hasSpeakers = captions.some(
        (caption) => caption.speaker !== undefined,
    );
    if (!hasSpeakers) {
        // Undiarized transcript: byte-identical to what this has always
        // returned — a bare array, no wrapper object, no `speaker` key on any
        // segment.
        return JSON.stringify(segments, null, 2);
    }

    // Diarized: `speaker` on each segment stays the raw opaque label (fidelity
    // for re-import / cross-referencing — see `speakerDisplay`'s doc comment on
    // why ids are never rewritten), and the display names the user gave them
    // come along as a separate top-level map, exactly like `getSpeakerNames`
    // returns. A consumer resolves a name exactly as the other three formats do
    // inline: `speakerNames[segment.speaker] ?? segment.speaker`.
    return JSON.stringify({ segments, speakerNames: names ?? {} }, null, 2);
}
