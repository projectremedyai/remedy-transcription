import { describe, expect, it } from "vitest";

import type { DiarizationOutcome } from "./types";

/**
 * The constraint under test is not a function's behaviour — it is that a
 * DEGRADATION CANNOT BE READ AS ZERO SPEAKERS. That has to hold for code nobody
 * has written yet (the Task 9 consumer), so the guard has to be the type, and
 * these tests have to bind at COMPILE time as much as at run time.
 *
 * `@ts-expect-error` is doing real work below: `tsc` FAILS if the line it marks
 * turns out to compile fine. So if someone flattens this union into
 * `{ status: string; turns?: SpeakerTurn[] }` — which is what the wire shape
 * would tempt you into — the typecheck goes red here, not in production.
 *
 * The outcomes come from functions, not from annotated `const`s: an annotated
 * literal gets narrowed to its own arm by control-flow analysis, which would
 * quietly test something easier than the thing a caller actually holds. What a
 * caller holds is the whole union, which is what `await api.diarizeJob(...)`
 * hands them, and what these return.
 */
const degraded = (reason: string): DiarizationOutcome => ({
    status: "degraded",
    reason,
});
const cancelled = (): DiarizationOutcome => ({ status: "cancelled" });
const succeeded = (
    turns: { start: number; end: number; speaker: number }[],
    speakerCount: number,
): DiarizationOutcome => ({
    status: "succeeded",
    turns,
    speaker_count: speakerCount,
});

describe("DiarizationOutcome", () => {
    it("does not let a degradation be read as an empty turn list", () => {
        const outcome = degraded(
            "the diarization sidecar was killed by signal 6 (SIGABRT)",
        );

        // @ts-expect-error `turns` is not on every arm of the union, so this is
        // exactly the mistake the type exists to stop: `outcome.turns ?? []`
        // would render a CRASHED ENGINE as "0 speakers found" — a confident,
        // silent, wrong answer, and the failure the three-outcome design exists
        // to prevent. It does not compile, and `tsc` fails here if it ever does.
        const smuggled = outcome.turns ?? [];
        // At runtime it really is `undefined ?? []` — i.e. the lie the type stops.
        expect(smuggled).toEqual([]);

        // The only way in is to narrow, which means the `degraded` arm has to be
        // handled, which means it gets shown.
        if (outcome.status === "succeeded") {
            throw new Error("narrowed to the wrong arm");
        }
        expect(outcome.status).toBe("degraded");
    });

    it("keeps a real zero-speaker success distinguishable from a degradation", () => {
        // Silence: the engine RAN and heard nobody. A success, and the UI may
        // legitimately draw no speakers for it.
        const silence = succeeded([], 0);
        const broken = degraded("the segmentation model is not installed");
        const stopped = cancelled();

        // Three different things to tell the user, and "no labels to draw" does
        // not collapse them into one.
        const statuses = new Set([
            silence.status,
            broken.status,
            stopped.status,
        ]);
        expect(statuses.size).toBe(3);

        const speakers = (outcome: DiarizationOutcome): number | null =>
            outcome.status === "succeeded" ? outcome.speaker_count : null;

        expect(speakers(silence)).toBe(0);
        expect(speakers(broken)).toBeNull();
        expect(speakers(stopped)).toBeNull();
    });

    it("narrows to the turns only on success", () => {
        const outcome = succeeded(
            [
                { start: 0, end: 7, speaker: 0 },
                { start: 7, end: 12, speaker: 1 },
            ],
            2,
        );

        if (outcome.status !== "succeeded") {
            throw new Error("expected a success");
        }

        // Dense ids: Rust remaps the engine's sparse `{0, 3}` at the boundary, so
        // `speaker` is a valid index into `speaker_count` things.
        expect(outcome.turns.map((t) => t.speaker)).toEqual([0, 1]);
        expect(
            outcome.turns.every((t) => t.speaker < outcome.speaker_count),
        ).toBe(true);
    });
});
