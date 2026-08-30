import { beforeEach, describe, expect, it, vi } from "vitest";

import { createGeminiEngine } from "./geminiEngine";
import { normalizeWordTokens } from "../../lib/captionFormatter";
import { RUN_DECLINED, type CostConfirmation } from "./types";
import type { ResolvedModelConfig } from "../../config/transcription";
import type { Job } from "../../services/types";

const transcribeWithGemini = vi.fn();
const cancelGeminiTranscription = vi.fn();
const estimateGeminiCost = vi.fn();
// Typed explicitly (rather than inferred from the zero-arg implementation
// below) so the two-argument call in the `vi.mock` factory typechecks; the
// implementation itself still ignores both arguments. Vitest 4 takes the whole
// FUNCTION type here, not the old `<[args], return>` pair.
const subscribeToGeminiProgress = vi.fn<
    (jobId: string, onProgress: unknown) => () => void
>(() => () => undefined);

vi.mock("../../services/api", () => ({
    api: {
        transcribeWithGemini: (jobId: string) => transcribeWithGemini(jobId),
        cancelGeminiTranscription: (jobId: string) =>
            cancelGeminiTranscription(jobId),
        estimateGeminiCost: (jobId: string) => estimateGeminiCost(jobId),
        subscribeToGeminiProgress: (jobId: string, onProgress: unknown) =>
            subscribeToGeminiProgress(jobId, onProgress),
    },
}));

const config: ResolvedModelConfig = {
    engine: "gemini",
    presetId: "balanced",
    presetLabel: "Gemini 3.5 Transcribe",
    modelId: "gemini-3.5-transcribe",
    device: null,
    task: "transcribe",
    language: "auto",
};

const job = { id: "job-1", filename: "talk.mp4" } as Job;

const run = (
    engine: ReturnType<typeof createGeminiEngine>,
    onProgress: (progress: {
        fraction: number | null;
        status: string;
    }) => void = () => undefined,
    confirmCost: (details: CostConfirmation) => Promise<boolean> = async () =>
        true,
) =>
    engine.run({
        job,
        config,
        runId: 1,
        onProgress,
        onPartial: () => undefined,
        confirmCost,
    });

describe("the gemini engine", () => {
    beforeEach(() => {
        transcribeWithGemini.mockReset();
        cancelGeminiTranscription.mockReset();
        subscribeToGeminiProgress.mockClear();
        estimateGeminiCost.mockReset();
        // One chunk unless a test says otherwise: the shape that costs least
        // and asks nothing.
        estimateGeminiCost.mockResolvedValue({
            duration_secs: 300,
            chunk_count: 1,
            estimated_usd: 0.025,
            diarization_available: true,
        });
    });

    it("maps identified speakers straight onto SpeakerTurn[]", async () => {
        transcribeWithGemini.mockResolvedValue({
            text: "hello there",
            words: [
                { text: "hello", start: 0, end: 1 },
                { text: "there", start: 1, end: 2 },
            ],
            speakers: {
                status: "identified",
                turns: [{ start: 0, end: 2, speaker: 0 }],
                speaker_count: 1,
            },
            audio_duration: 2,
        });

        const result = await run(createGeminiEngine());

        expect(result.speakers).toEqual({
            status: "identified",
            turns: [{ start: 0, end: 2, speaker: 0 }],
            speaker_count: 1,
        });
        expect(result.transcript.words).toHaveLength(2);
        expect(result.audioDuration).toBe(2);
    });

    /**
     * The whole point of the tagged union: chunked audio produces NO speakers,
     * and that must reach the UI as a stated reason rather than as an empty
     * list that reads like "one speaker".
     */
    it("carries the reason through when speakers are unavailable", async () => {
        transcribeWithGemini.mockResolvedValue({
            text: "hi",
            words: [{ text: "hi", start: 0, end: 1 }],
            speakers: { status: "unavailable", reason: "split into 4 parts" },
            audio_duration: 1,
        });

        const result = await run(createGeminiEngine());

        expect(result.speakers).toEqual({
            status: "unavailable",
            reason: "split into 4 parts",
        });
    });

    /**
     * Gemini's `word_info.text` is a BARE word ("Hello"). `normalizeWordTokens`
     * reads a MISSING leading space as "this token continues the previous word"
     * — the convention transformers.js sets, where `splitTokensOnSpaces` marks
     * every word boundary with one. Hand it Gemini's words unchanged and the
     * whole transcript folds into a single token; `cleanCaptionText` then
     * re-inserts a space only after `,.;:!?`, which is what shipped
     * "Helloeveryone. JohnnyFunghere" into the SRT/VTT/TXT/JSON exports.
     *
     * This engine is the adapter between the two conventions, so restoring the
     * boundary is ITS job, not the normalizer's: the folding is load-bearing for
     * the local engine, where transformers.js really does emit sub-word pieces
     * that must be glued back together.
     */
    it("hands on words the token normalizer will keep separate", async () => {
        transcribeWithGemini.mockResolvedValue({
            text: "Hello everyone. Johnny here",
            words: [
                { text: "Hello", start: 0, end: 1 },
                { text: "everyone.", start: 1, end: 2 },
                { text: "Johnny", start: 2, end: 3 },
                { text: "here", start: 3, end: 4 },
            ],
            speakers: { status: "unavailable", reason: "n/a" },
            audio_duration: 4,
        });

        const result = await run(createGeminiEngine());

        expect(
            normalizeWordTokens(result.transcript.words ?? []).map(
                (token) => token.text,
            ),
        ).toEqual(["Hello", "everyone.", "Johnny", "here"]);
    });

    /** Gemini has no token stream, so `chunks` must be empty, never undefined. */
    it("produces a WorkerTranscript with no streaming chunks", async () => {
        transcribeWithGemini.mockResolvedValue({
            text: "hi",
            words: [{ text: "hi", start: 0, end: 1 }],
            speakers: { status: "unavailable", reason: "n/a" },
            audio_duration: 1,
        });

        const result = await run(createGeminiEngine());
        expect(result.transcript.chunks).toEqual([]);
    });

    it("reports progress and always unsubscribes, even when the run fails", async () => {
        const unlisten = vi.fn();
        subscribeToGeminiProgress.mockReturnValue(unlisten);
        transcribeWithGemini.mockRejectedValue(
            new Error("Gemini rejected this API key"),
        );

        await expect(run(createGeminiEngine())).rejects.toThrow(/rejected/);
        expect(unlisten).toHaveBeenCalled();
    });

    /**
     * `EngineProgress.status` is shown VERBATIM by `AudioManager` — anything
     * it does not recognise as one of its own kebab-case job statuses goes
     * straight to the screen. So these strings are the UI, and two properties
     * have to hold: they must read as sentences, and they must never say the
     * work is happening in the browser. The labels used to be kebab-case
     * machine strings, and `transcribing` collided exactly with the local
     * worker's own status, rendering "Transcribing in your browser..." over a
     * run whose audio was being uploaded to Google.
     */
    describe("the progress labels it hands the UI", () => {
        /** Capture what the engine emits for one backend progress event. */
        async function statusFor(event: {
            phase: string;
            chunk_index: number;
            chunk_count: number;
            fraction: number;
        }) {
            const seen: { fraction: number | null; status: string }[] = [];
            subscribeToGeminiProgress.mockImplementation(
                (_jobId: string, onProgress: unknown) => {
                    (onProgress as (e: unknown) => void)(event);
                    return () => undefined;
                },
            );
            transcribeWithGemini.mockResolvedValue({
                text: "hi",
                words: [{ text: "hi", start: 0, end: 1 }],
                speakers: { status: "unavailable", reason: "n/a" },
                audio_duration: 1,
            });
            await run(createGeminiEngine(), (progress) => {
                seen.push(progress);
            });
            return seen[0];
        }

        it("emits presentable prose, never a kebab-case machine string", async () => {
            const progress = await statusFor({
                phase: "transcribing",
                chunk_index: 0,
                chunk_count: 1,
                fraction: 0,
            });
            expect(progress.status).toBe(
                "Uploading to Google and transcribing...",
            );
            expect(progress.status).not.toMatch(/browser/i);
        });

        it("numbers the parts on a chunked run", async () => {
            const progress = await statusFor({
                phase: "transcribing",
                chunk_index: 1,
                chunk_count: 3,
                fraction: 1 / 3,
            });
            expect(progress.status).toBe(
                "Uploading to Google and transcribing (part 2 of 3)...",
            );
            expect(progress.fraction).toBeCloseTo(1 / 3);
        });

        /**
         * `stitching` is emitted with `chunk_index == chunk_count` — it is a
         * whole-run step, not a per-chunk one — so a naive `chunk_index + 1`
         * announced "part 4 of 3" at the end of every multi-chunk run.
         */
        it("does not invent a part number for the whole-run stitch", async () => {
            const progress = await statusFor({
                phase: "stitching",
                chunk_index: 3,
                chunk_count: 3,
                fraction: 1,
            });
            expect(progress.status).toBe("Assembling the transcript...");
        });
    });

    /**
     * An abandoned Gemini run costs money and leaves the user's audio in
     * Google's storage. Unlike the local engine, `abandon` is not a no-op.
     */
    it("tells the backend to stop when abandoned", () => {
        cancelGeminiTranscription.mockResolvedValue(true);
        createGeminiEngine().abandon("job-1");
        expect(cancelGeminiTranscription).toHaveBeenCalledWith("job-1");
    });

    /**
     * THE MONEY GATE. `MAX_DURATION_HOURS` has only ever guarded the YouTube
     * path, so a long LOCAL file reached Gemini with nothing between it and the
     * bill. A dropped 6-hour file is ~15 chunks and roughly $1.80, spent
     * silently, on a mis-drop the user cannot take back.
     *
     * The order asserted here is the whole point: the question has to be
     * answered BEFORE `transcribeWithGemini`, because after it the money is
     * gone.
     */
    it("asks what a chunked run will cost before spending anything", async () => {
        estimateGeminiCost.mockResolvedValue({
            duration_secs: 21600,
            chunk_count: 15,
            estimated_usd: 1.8,
            diarization_available: false,
        });
        transcribeWithGemini.mockResolvedValue({
            text: "hello",
            words: [{ text: "hello", start: 0, end: 1 }],
            speakers: { status: "unavailable", reason: "split into 15 parts" },
            audio_duration: 21600,
        });

        const seen: CostConfirmation[] = [];
        await run(
            createGeminiEngine(),
            () => undefined,
            async (details) => {
                seen.push(details);
                // Nothing may have been bought at the moment we are asked.
                expect(transcribeWithGemini).not.toHaveBeenCalled();
                return true;
            },
        );

        expect(seen).toEqual([
            {
                durationSecs: 21600,
                chunkCount: 15,
                estimatedUsd: 1.8,
                diarizationAvailable: false,
            },
        ]);
        expect(transcribeWithGemini).toHaveBeenCalledWith("job-1");
    });

    /**
     * A short file is the ordinary case and must stay a single click. Chunking
     * is what makes a run expensive AND what costs it its speaker labels, so
     * `chunk_count > 1` is the one honest trigger for interrupting the user.
     */
    it("does not interrupt a single-chunk run", async () => {
        transcribeWithGemini.mockResolvedValue({
            text: "hello",
            words: [{ text: "hello", start: 0, end: 1 }],
            speakers: { status: "identified", turns: [], speaker_count: 1 },
            audio_duration: 300,
        });

        const confirmCost = vi.fn(async () => true);
        await run(createGeminiEngine(), () => undefined, confirmCost);

        expect(confirmCost).not.toHaveBeenCalled();
        expect(transcribeWithGemini).toHaveBeenCalledWith("job-1");
    });

    /** Declining must cost nothing at all — not one chunk, not one upload. */
    it("spends nothing when the user declines", async () => {
        estimateGeminiCost.mockResolvedValue({
            duration_secs: 21600,
            chunk_count: 15,
            estimated_usd: 1.8,
            diarization_available: false,
        });

        await expect(
            run(
                createGeminiEngine(),
                () => undefined,
                async () => false,
            ),
        ).rejects.toThrow(RUN_DECLINED);

        expect(transcribeWithGemini).not.toHaveBeenCalled();
    });
});
