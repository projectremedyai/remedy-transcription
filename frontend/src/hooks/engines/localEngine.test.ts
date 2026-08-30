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
            }),
        ).rejects.toThrow(/device/i);
    });
});
