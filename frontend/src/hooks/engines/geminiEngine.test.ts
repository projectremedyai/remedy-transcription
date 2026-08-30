import { beforeEach, describe, expect, it, vi } from "vitest";

import { createGeminiEngine } from "./geminiEngine";
import type { ResolvedModelConfig } from "../../config/transcription";
import type { Job } from "../../services/types";

const transcribeWithGemini = vi.fn();
const cancelGeminiTranscription = vi.fn();
// Typed explicitly (rather than inferred from the zero-arg implementation
// below) so the two-argument call in the `vi.mock` factory typechecks; the
// implementation itself still ignores both arguments.
const subscribeToGeminiProgress = vi.fn<[string, unknown], () => void>(
    () => () => undefined,
);

vi.mock("../../services/api", () => ({
    api: {
        transcribeWithGemini: (jobId: string) => transcribeWithGemini(jobId),
        cancelGeminiTranscription: (jobId: string) =>
            cancelGeminiTranscription(jobId),
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
    onProgress = () => undefined,
) =>
    engine.run({
        job,
        config,
        runId: 1,
        onProgress,
        onPartial: () => undefined,
    });

describe("the gemini engine", () => {
    beforeEach(() => {
        transcribeWithGemini.mockReset();
        cancelGeminiTranscription.mockReset();
        subscribeToGeminiProgress.mockClear();
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
     * An abandoned Gemini run costs money and leaves the user's audio in
     * Google's storage. Unlike the local engine, `abandon` is not a no-op.
     */
    it("tells the backend to stop when abandoned", () => {
        cancelGeminiTranscription.mockResolvedValue(true);
        createGeminiEngine().abandon("job-1");
        expect(cancelGeminiTranscription).toHaveBeenCalledWith("job-1");
    });
});
