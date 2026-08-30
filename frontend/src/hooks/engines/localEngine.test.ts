import { beforeEach, describe, expect, it, vi } from "vitest";

import { createLocalEngine } from "./localEngine";
import type { ResolvedModelConfig } from "../../config/transcription";
import type { Job } from "../../services/types";

const config: ResolvedModelConfig = {
    engine: "local",
    presetId: "balanced",
    presetLabel: "Balanced",
    modelId: "onnx-community/whisper-base_timestamped",
    device: "wasm",
    task: "transcribe",
    language: "auto",
};

const job = { id: "job-1", filename: "talk.mp4" } as Job;

describe("the local engine", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it("reports speakers as unavailable — it has no diarizer", async () => {
        const engine = createLocalEngine({
            runWorkerTranscription: async () => ({
                text: "hello",
                chunks: [],
                words: [{ text: "hello", start: 0, end: 1 }],
            }),
            loadAudio: async () => ({ duration: 12.5 } as AudioBuffer),
        });

        const result = await engine.run({
            job,
            config,
            runId: 1,
            onProgress: () => undefined,
            onPartial: () => undefined,
            confirmCost: async () => true,
        });

        expect(result.speakers.status).toBe("unavailable");
        expect(result.transcript.text).toBe("hello");
    });

    /**
     * `audioDuration` is load-bearing: `segmentsForPersistence` uses it to
     * CLOSE the final segment, and a wrong value silently swallows the end of
     * the transcript rather than throwing. It must come from a real decode.
     */
    it("returns the decoded audio's true duration", async () => {
        const engine = createLocalEngine({
            runWorkerTranscription: async () => ({ text: "", chunks: [] }),
            loadAudio: async () => ({ duration: 3600.25 } as AudioBuffer),
        });

        const result = await engine.run({
            job,
            config,
            runId: 1,
            onProgress: () => undefined,
            onPartial: () => undefined,
            confirmCost: async () => true,
        });

        expect(result.audioDuration).toBe(3600.25);
    });

    it("throws rather than guessing when the config has no device", async () => {
        const engine = createLocalEngine({
            runWorkerTranscription: async () => ({ text: "", chunks: [] }),
            loadAudio: async () => ({ duration: 1 } as AudioBuffer),
        });

        await expect(
            engine.run({
                job,
                config: { ...config, device: null },
                runId: 1,
                onProgress: () => undefined,
                onPartial: () => undefined,
                confirmCost: async () => true,
            }),
        ).rejects.toThrow(/device/i);
    });

    /**
     * REGRESSION GUARD for the decision behind the cost dialog: the duration
     * cap belongs to the ENGINE, not to the source.
     *
     * The local path was never capped, and that was right — on-device Whisper
     * is free at any length, so a six-hour lecture costs nothing but time. The
     * obvious reading of "local files have no duration cap" would have been to
     * extend `MAX_DURATION_HOURS` to them, which deletes a working feature to
     * solve a cost problem this engine does not have. If someone later routes
     * the confirmation through the hook for every engine instead of through
     * `geminiEngine.run`, this is what says no.
     */
    it("never asks the user to approve a cost, because it has none", async () => {
        const engine = createLocalEngine({
            runWorkerTranscription: async () => ({
                text: "hello",
                chunks: [],
                words: [{ text: "hello", start: 0, end: 1 }],
            }),
            loadAudio: async () => ({ duration: 12.5 } as AudioBuffer),
        });

        const confirmCost = vi.fn(async () => true);
        await engine.run({
            job,
            config,
            runId: 1,
            onProgress: () => undefined,
            onPartial: () => undefined,
            confirmCost,
        });

        expect(confirmCost).not.toHaveBeenCalled();
    });
});
