import { describe, it, expect } from "vitest";
import {
    assignSpeakers,
    speakerLabel,
    type SpeakerTurn,
} from "./speakerAlignment";

const TURNS: SpeakerTurn[] = [
    { start: 0.0, end: 5.0, speaker: 0 },
    { start: 5.0, end: 10.0, speaker: 1 },
];

describe("assignSpeakers", () => {
    it("assigns a token fully inside a turn to that speaker", () => {
        const [t] = assignSpeakers(
            [{ text: "hello", start: 1.0, end: 2.0 }],
            TURNS,
        );
        expect(t.speaker).toBe(0);
    });

    it("assigns a straddling token to the turn it overlaps most", () => {
        // 4.5-5.5 overlaps speaker 0 for 0.5s and speaker 1 for 0.5s — tie.
        // 4.0-5.5 overlaps speaker 0 for 1.0s and speaker 1 for 0.5s — speaker 0 wins.
        const [t] = assignSpeakers(
            [{ text: "straddle", start: 4.0, end: 5.5 }],
            TURNS,
        );
        expect(t.speaker).toBe(0);
    });

    it("assigns a token overlapping no turn to the nearest turn", () => {
        const [t] = assignSpeakers(
            [{ text: "gap", start: 11.0, end: 12.0 }],
            TURNS,
        );
        expect(t.speaker).toBe(1);
    });

    it("returns speaker null when there are no turns at all", () => {
        const [t] = assignSpeakers(
            [{ text: "orphan", start: 1.0, end: 2.0 }],
            [],
        );
        expect(t.speaker).toBeNull();
    });

    it("handles overlapping speech by picking the greater overlap", () => {
        const overlapping: SpeakerTurn[] = [
            { start: 0.0, end: 3.0, speaker: 0 },
            { start: 2.0, end: 6.0, speaker: 1 },
        ];
        // 2.5-5.0: speaker 0 overlap = 0.5s, speaker 1 overlap = 2.5s
        const [t] = assignSpeakers(
            [{ text: "both", start: 2.5, end: 5.0 }],
            overlapping,
        );
        expect(t.speaker).toBe(1);
    });

    it("returns an empty array for empty tokens", () => {
        expect(assignSpeakers([], TURNS)).toEqual([]);
    });

    it("preserves the original token fields", () => {
        const [t] = assignSpeakers(
            [{ text: "hello", start: 1.0, end: 2.0 }],
            TURNS,
        );
        expect(t.text).toBe("hello");
        expect(t.start).toBe(1.0);
        expect(t.end).toBe(2.0);
    });

    it("does not assume turns are sorted by start time", () => {
        // Same turns as TURNS, but reversed in the array. A token fully inside
        // [0, 5) must still resolve to speaker 0 even though that turn is
        // second in the input array — selection must be by overlap, not by
        // scanning position or assumed chronological order.
        const unsorted: SpeakerTurn[] = [
            { start: 5.0, end: 10.0, speaker: 1 },
            { start: 0.0, end: 5.0, speaker: 0 },
        ];
        const [t] = assignSpeakers(
            [{ text: "hello", start: 1.0, end: 2.0 }],
            unsorted,
        );
        expect(t.speaker).toBe(0);
    });

    it("resolves a zero-length token exactly on a turn boundary deterministically", () => {
        // A token at [5.0, 5.0) has zero duration, so it overlaps NEITHER
        // adjacent turn (overlap = 0 for both under the half-open [start, end)
        // convention). It falls back to fill_nearest, where the gap to both
        // turns is also 0 (it touches both boundaries). Documented tie-break:
        // the first turn in *input array order* wins ties, not the
        // chronologically earlier one — this is intentionally the array's
        // first element, not a start-time comparison.
        const [t] = assignSpeakers(
            [{ text: "boundary", start: 5.0, end: 5.0 }],
            TURNS,
        );
        expect(t.speaker).toBe(0);

        // Reversing the array reverses the tie-break winner, proving the rule
        // is "first in array order" and not some hidden reliance on start time.
        const reversed: SpeakerTurn[] = [TURNS[1], TURNS[0]];
        const [t2] = assignSpeakers(
            [{ text: "boundary", start: 5.0, end: 5.0 }],
            reversed,
        );
        expect(t2.speaker).toBe(1);
    });
});

describe("speakerLabel", () => {
    it("pads the common case to two digits", () => {
        expect(speakerLabel(0)).toBe("SPEAKER_00");
        expect(speakerLabel(7)).toBe("SPEAKER_07");
        expect(speakerLabel(11)).toBe("SPEAKER_11");
    });

    it("treats the id as OPAQUE — it does not assume a dense 0..n-1 range", () => {
        // Rust densifies the engine's sparse ids at the command boundary, but
        // nothing downstream may re-derive that. This must not truncate, wrap, or
        // index anything: it only renders whatever id it is handed.
        expect(speakerLabel(120)).toBe("SPEAKER_120");
        expect(speakerLabel(3)).not.toBe(speakerLabel(30));
    });
});
