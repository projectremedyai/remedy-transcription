// @vitest-environment jsdom
/**
 * The wait for Rust to finish preparing a job's audio is the highest-risk code in
 * this hook, and none of it is reachable from a pure-function test: it is a
 * listener, an interval, and a ref that owns their teardown. So the hook is
 * rendered for real (jsdom + `renderHook`) with `services/api` and the worker
 * mocked, and the wait is driven with fake timers.
 *
 * What each test here is defending is written on the test.
 */
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MODEL_PRESETS } from "../config/transcription";
import type { Job } from "../services/api";

const mocks = vi.hoisted(() => ({
    postMessage: vi.fn(),
    createFileJob: vi.fn(),
    createYouTubeJob: vi.fn(),
    getJob: vi.fn(),
    getAudioUrl: vi.fn(),
    persistTranscript: vi.fn(),
    getModelStatus: vi.fn(),
    subscribeToProgress: vi.fn(),
    unsubscribe: vi.fn(),
}));

vi.mock("./useWorker", () => ({
    useWorker: () => ({ postMessage: mocks.postMessage }),
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

// Imported after the mocks it depends on, deliberately.
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

/** Flush pending promise chains without moving the clock. */
async function settle() {
    await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
    });
}

async function tick(ms: number) {
    await act(async () => {
        await vi.advanceTimersByTimeAsync(ms);
    });
    await settle();
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
    mocks.persistTranscript.mockImplementation(async (jobId: string) =>
        makeJob({ id: jobId, status: "completed", progress: 1 }),
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

describe("useTranscriber's wait for prepared audio", () => {
    /**
     * THE RACE. `api.subscribeToProgress` registers its Tauri listener
     * asynchronously, so a job that finishes inside that window emits `ready`
     * into the void. ffmpeg over a short local file does exactly that. Without
     * the poll, the UI waits forever for an event that already fired.
     *
     * This test therefore emits NO event at all — it can only pass because of
     * the poll.
     */
    it("resolves from the poll when the ready event never arrives", async () => {
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: "job-1", status: "extracting", progress: 0.1 }),
        );
        mocks.getJob.mockResolvedValue(
            makeJob({ id: "job-1", status: "ready", progress: 1 }),
        );

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();

        expect(mocks.postMessage).not.toHaveBeenCalled();

        await tick(300);

        expect(mocks.postMessage).toHaveBeenCalledWith(
            expect.objectContaining({ type: "transcribe" }),
        );
        expect(result.current.status).toBe("transcribing");
    });

    it("rejects with the backend's error when the job fails while polling", async () => {
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: "job-1", status: "extracting", progress: 0.1 }),
        );
        mocks.getJob.mockResolvedValue(
            makeJob({
                id: "job-1",
                status: "failed",
                error: "ffmpeg exited with code Some(1)",
            }),
        );

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await tick(300);

        expect(result.current.status).toBe("failed");
        expect(result.current.error).toBe("ffmpeg exited with code Some(1)");
        expect(result.current.isBusy).toBe(false);
        expect(mocks.postMessage).not.toHaveBeenCalled();
    });

    /**
     * IMPORTANT-1 REGRESSION TEST.
     *
     * A second run must cancel the first run's wait UNCONDITIONALLY. It did not:
     * the teardown sat below `waitForReady`'s terminal-status early return, and
     * `transcribePreparedJob` returns early for a `completed` (cache-hit) job
     * without calling `waitForReady` at all — so a cached YouTube run started
     * while a local file was still extracting tore down nothing.
     *
     * The first job's 300 ms poll then kept running underneath the cached
     * transcript: overwriting its status and progress every tick, and — once the
     * first job reached `ready` — resolving its wait, which resumed the first
     * job's `transcribePreparedJob` and ran a whole spurious transcription.
     */
    it("cancels the previous wait when a second run starts, even a cache hit", async () => {
        let firstJobStatus: Job["status"] = "extracting";
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: "job-1", status: "extracting", progress: 0.1 }),
        );
        mocks.getJob.mockImplementation(async (jobId: string) =>
            makeJob({
                id: jobId,
                status: firstJobStatus,
                progress: firstJobStatus === "ready" ? 1 : 0.1,
            }),
        );
        // The cached YouTube transcript: `completed` the moment it is created.
        mocks.createYouTubeJob.mockResolvedValue(
            makeJob({
                id: "job-2",
                source_type: "youtube",
                status: "completed",
                progress: 1,
                cache_hit: true,
                full_text: "the cached transcript",
                filename: "A cached video",
            }),
        );

        const { result } = await renderTranscriber();

        // Job 1: a local file, still extracting, its poll live.
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await tick(300);
        expect(result.current.status).toBe("extracting");

        // Job 2: a YouTube URL whose transcript is already cached.
        await act(async () => {
            result.current.startFromYouTube(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            );
        });
        await settle();

        expect(result.current.status).toBe("completed");
        expect(result.current.output?.text).toBe("the cached transcript");
        const pollsBefore = mocks.getJob.mock.calls.length;

        // Job 1's extraction finishes. Nothing about it may reach the UI, and it
        // must NOT resume into a transcription of its own.
        firstJobStatus = "ready";
        await tick(3000);

        expect(mocks.getJob.mock.calls.length).toBe(pollsBefore);
        expect(mocks.postMessage).not.toHaveBeenCalled();
        expect(mocks.getAudioUrl).not.toHaveBeenCalled();
        expect(result.current.status).toBe("completed");
        expect(result.current.jobId).toBe("job-2");
        expect(result.current.output?.text).toBe("the cached transcript");
        expect(result.current.isBusy).toBe(false);
    });

    it("clears the poll and the listener on unmount", async () => {
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: "job-1", status: "extracting", progress: 0.1 }),
        );
        mocks.getJob.mockResolvedValue(
            makeJob({ id: "job-1", status: "extracting", progress: 0.1 }),
        );

        const { result, unmount } = await renderTranscriber();
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await tick(300);

        const pollsBeforeUnmount = mocks.getJob.mock.calls.length;
        expect(pollsBeforeUnmount).toBeGreaterThan(0);

        unmount();
        cleanup();
        await tick(3000);

        expect(mocks.getJob.mock.calls.length).toBe(pollsBeforeUnmount);
        expect(mocks.unsubscribe).toHaveBeenCalled();
    });
});
