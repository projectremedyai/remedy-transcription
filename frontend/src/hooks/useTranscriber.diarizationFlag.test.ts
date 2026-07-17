// @vitest-environment jsdom
/**
 * 1.1.0: `DIARIZATION_UI_ENABLED` (`../config/features`) is false — see
 * `AudioManager.diarizationFlag.test.tsx` for why. `useTranscriber.test.ts`
 * mocks the flag back to `true` to keep the diarization wiring green for a
 * future re-enable; this file exercises the real, shipped default (false),
 * without mocking `../config/features` at all.
 *
 * The property under test is belt-and-braces, on purpose: the UI cannot even
 * render the toggle with the flag off (see `AudioManager.diarizationFlag.test.tsx`),
 * but this hook must not TRUST that — `diarizeEnabled` must read as false and
 * `api.diarizeJob` must never be called, even if something manages to call
 * `setDiarizeEnabled(true)` and `setNumSpeakersHint` on this hook directly.
 */
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MODEL_PRESETS } from "../config/transcription";
import type { Job } from "../services/api";

const mocks = vi.hoisted(() => ({
    postMessage: vi.fn(),
    restartWorker: vi.fn(),
    createFileJob: vi.fn(),
    createYouTubeJob: vi.fn(),
    getJob: vi.fn(),
    getAudioUrl: vi.fn(),
    persistTranscript: vi.fn(),
    getModelStatus: vi.fn(),
    subscribeToProgress: vi.fn(),
    unsubscribe: vi.fn(),
    cancelDiarization: vi.fn(),
    diarizeJob: vi.fn(),
    setSpeakerName: vi.fn(),
    getSpeakerNames: vi.fn(),
}));

vi.mock("./useWorker", () => ({
    useWorker: () => ({
        postMessage: mocks.postMessage,
        restart: mocks.restartWorker,
    }),
}));

vi.mock("../services/api", () => ({
    api: {
        createFileJob: mocks.createFileJob,
        createYouTubeJob: mocks.createYouTubeJob,
        getJob: mocks.getJob,
        getAudioUrl: mocks.getAudioUrl,
        persistTranscript: mocks.persistTranscript,
        getModelStatus: mocks.getModelStatus,
        subscribeToProgress: mocks.subscribeToProgress,
        cancelDiarization: mocks.cancelDiarization,
        diarizeJob: mocks.diarizeJob,
        setSpeakerName: mocks.setSpeakerName,
        getSpeakerNames: mocks.getSpeakerNames,
    },
}));

vi.mock("../utils/detectBrowserCaps", () => ({
    detectBrowserCaps: async () => ({
        secureContext: true,
        canUseWebGPU: false,
        shaderF16: false,
        deviceMemoryGiB: 8,
        logicalCores: 8,
    }),
}));

// Deliberately NOT mocked — this file exercises the real, shipped default
// (`DIARIZATION_UI_ENABLED === false`).
import { useTranscriber } from "./useTranscriber";

function makeJob(overrides: Partial<Job> & Pick<Job, "id" | "status">): Job {
    return {
        source_type: "file",
        source_key: "key",
        progress: 0,
        cache_hit: false,
        error: null,
        filename: "lecture.mp3",
        audio_url: null,
        audio_mime_type: null,
        model_id: "onnx-community/whisper-base_timestamped",
        task: "transcribe",
        language: "auto",
        segments: [],
        full_text: null,
        ...overrides,
    };
}

async function settle() {
    await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
    });
}

async function renderTranscriber() {
    const rendered = renderHook(() => useTranscriber());
    await settle();
    return rendered;
}

beforeEach(() => {
    vi.useFakeTimers();
    for (const mock of Object.values(mocks)) {
        mock.mockReset();
    }

    mocks.subscribeToProgress.mockReturnValue(mocks.unsubscribe);
    mocks.getModelStatus.mockResolvedValue({
        models_ready: true,
        missing_models: [],
        items: MODEL_PRESETS.map((preset) => ({
            model_id: preset.modelId,
            ready: true,
        })),
    });
    mocks.getAudioUrl.mockResolvedValue("asset://localhost/audio.wav");
    mocks.cancelDiarization.mockResolvedValue(false);
    mocks.getSpeakerNames.mockResolvedValue({});
    mocks.setSpeakerName.mockResolvedValue(undefined);
    mocks.persistTranscript.mockImplementation(async (jobId: string) =>
        makeJob({ id: jobId, status: "completed", progress: 1 }),
    );
    mocks.createFileJob.mockResolvedValue(
        makeJob({ id: "job-1", status: "ready", progress: 1 }),
    );

    const audioBuffer = {
        duration: 12,
        numberOfChannels: 1,
        length: 16000 * 12,
        getChannelData: () => new Float32Array(16000 * 12),
    };
    vi.stubGlobal(
        "AudioContext",
        class {
            async decodeAudioData() {
                return audioBuffer;
            }
            async close() {
                return undefined;
            }
        },
    );
    vi.stubGlobal(
        "fetch",
        vi.fn(async () => ({
            ok: true,
            arrayBuffer: async () => new ArrayBuffer(1024),
        })),
    );
});

afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.useRealTimers();
});

describe("useTranscriber with DIARIZATION_UI_ENABLED=false (1.1.0 shipped state)", () => {
    it("reports diarizeEnabled as false even after setDiarizeEnabled(true) — the flag wins regardless of state", async () => {
        const { result } = await renderTranscriber();

        expect(result.current.diarizeEnabled).toBe(false);

        act(() => {
            result.current.setDiarizeEnabled(true);
        });
        await settle();

        expect(result.current.diarizeEnabled).toBe(false);
    });

    it("never calls api.diarizeJob for a run started after setDiarizeEnabled(true) and a valid speaker count", async () => {
        const { result } = await renderTranscriber();

        act(() => {
            result.current.setDiarizeEnabled(true);
            result.current.setNumSpeakersHint(3);
        });
        await settle();

        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();

        expect(mocks.diarizeJob).not.toHaveBeenCalled();
        // "never ran" stays `null`, exactly as it does with the toggle off in
        // `useTranscriber.test.ts` — the flag must not manufacture a
        // "degraded" outcome either, since nothing was ever asked to run.
        expect(result.current.diarizationOutcome).toBeNull();
    });
});
