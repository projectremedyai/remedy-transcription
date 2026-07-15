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
import type { DiarizationOutcome, Job } from "../services/api";

const mocks = vi.hoisted(() => ({
    postMessage: vi.fn(),
    /** `useWorker`'s `restart` — i.e. `worker.terminate()` + a fresh worker. */
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

/**
 * The live worker message handler. Holding it is what lets a test post a message
 * FROM the worker — which is the only way to reproduce the real bug, because the
 * real worker keeps running (and keeps posting) after the app has abandoned it.
 */
const workerHandler = vi.hoisted(() => ({
    current: null as ((event: { data: unknown }) => void) | null,
}));

vi.mock("./useWorker", () => ({
    useWorker: (handler: (event: { data: unknown }) => void) => {
        workerHandler.current = handler;
        return { postMessage: mocks.postMessage, restart: mocks.restartWorker };
    },
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

/** A promise this test controls the settlement of, from the outside. */
function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((res) => {
        resolve = res;
    });
    return { promise, resolve };
}

async function renderTranscriber() {
    const rendered = renderHook(() => useTranscriber());
    await settle();
    return rendered;
}

/** Deliver a message from the worker, exactly as the real one would. */
async function emitFromWorker(data: unknown) {
    await act(async () => {
        workerHandler.current?.({ data });
        await Promise.resolve();
    });
    await settle();
}

/**
 * The `runId` the hook stamped on the Nth `transcribe` message it posted.
 *
 * Read rather than hardcoded: the token is bumped by every start AND by every
 * cancel, so "run 2" is not id 2. Reading it back is also the point — a build
 * that does not stamp one returns `undefined` here, and every message this test
 * then emits is unattributable, which is precisely the bug.
 */
function postedRunId(callIndex: number): unknown {
    return mocks.postMessage.mock.calls[callIndex]?.[0]?.runId;
}

function workerComplete(runId: unknown, text: string) {
    return {
        status: "complete",
        runId,
        data: {
            text,
            chunks: [],
            words: [{ text, start: 0, end: 5 }],
        },
    };
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
    // Not called unless a test flips `diarizeEnabled` on — see the diarization
    // describe block below, which sets its own resolved values per case.
    mocks.getSpeakerNames.mockResolvedValue({});
    mocks.setSpeakerName.mockResolvedValue(undefined);
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

    /**
     * THE WINDOW THE FIRST FIX LEFT OPEN.
     *
     * Cancelling the previous wait "first, before any early return" inside
     * `transcribePreparedJob` is NOT early enough. `transcribePreparedJob` runs
     * only after `beginRun()` and `api.create*Job()` have BOTH resolved — and
     * `create_file_job` sha256s the entire file before it answers, while
     * `create_youtube_job` shells out to yt-dlp. Neither is instant.
     *
     * Through that window the previous run's poll and listener are still live. If
     * the previous job reaches `ready` inside it, its wait resolves and its
     * `transcribePreparedJob` resumes — `getAudioUrl` → `fetch` → `decodeAudio` →
     * `postMessage`, a whole spurious transcription.
     *
     * The cache-hit test above CANNOT catch this: a `mockResolvedValue` settles in
     * a microtask, before any timer can tick, so the window never opens. This one
     * makes job creation take a second of fake time and lands job 1's `ready`
     * inside it. It fails unless the cancel happens synchronously, at the top of
     * the `start*` entry point, before the first `await`.
     */
    it("supersedes a run whose createJob is still in flight", async () => {
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
        // yt-dlp is on the other end of this. It takes a while.
        mocks.createYouTubeJob.mockImplementation(async () => {
            await new Promise((resolve) => setTimeout(resolve, 1000));
            return makeJob({
                id: "job-2",
                source_type: "youtube",
                status: "completed",
                progress: 1,
                cache_hit: true,
                full_text: "the cached transcript",
                filename: "A cached video",
            });
        });

        const { result } = await renderTranscriber();

        // Job 1: a local file, still extracting, its poll live.
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await tick(300);
        expect(result.current.status).toBe("extracting");

        // Job 2 starts — and while its creation is still in flight, job 1's ffmpeg
        // finishes. This is the collision.
        await act(async () => {
            result.current.startFromYouTube(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            );
        });
        firstJobStatus = "ready";
        await tick(1000);

        // Job 1 must be dead: no audio fetched, no transcription started.
        expect(mocks.getAudioUrl).not.toHaveBeenCalled();
        expect(mocks.postMessage).not.toHaveBeenCalled();

        expect(result.current.jobId).toBe("job-2");
        expect(result.current.status).toBe("completed");
        expect(result.current.output?.text).toBe("the cached transcript");
        expect(result.current.isBusy).toBe(false);
    });

    /**
     * The `settled` guard in `consider`.
     *
     * `clearInterval` stops the NEXT poll; it cannot recall the `getJob` already
     * in flight. That one still resolves — with what was true when it was asked —
     * after the wait has settled and the run has moved on. Unguarded, it pushes
     * that stale status and progress straight back into the UI: `transcribing`
     * flips back to `extracting`, and the progress bar back to 0.1.
     */
    it("ignores a poll that was already in flight when the wait settled", async () => {
        let emit: ((job: Job) => void) | undefined;
        mocks.subscribeToProgress.mockImplementation((_id, onJob) => {
            emit = onJob;
            return mocks.unsubscribe;
        });
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: "job-1", status: "extracting", progress: 0.1 }),
        );
        // A poll that is slow to answer, and answers stale.
        mocks.getJob.mockImplementation(async () => {
            await new Promise((resolve) => setTimeout(resolve, 500));
            return makeJob({
                id: "job-1",
                status: "extracting",
                progress: 0.1,
            });
        });

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });

        // The poll fires; its `getJob` is now in flight.
        await tick(300);
        expect(mocks.getJob).toHaveBeenCalledTimes(1);

        // The event beats it home. The wait settles and the run moves on.
        await act(async () => {
            emit?.(makeJob({ id: "job-1", status: "ready", progress: 1 }));
        });
        await tick(0);
        expect(result.current.status).toBe("transcribing");
        expect(result.current.progress).toBe(1);

        // Now the stale poll answers. Nothing of it may reach the UI.
        await tick(500);

        expect(result.current.status).toBe("transcribing");
        expect(result.current.progress).toBe(1);
    });

    /**
     * The escape hatch. Both entry points are locked while busy and drops are
     * refused, so a run that never terminates — a job stranded with no event ever
     * coming — leaves quitting the app as the only exit unless `cancel` works.
     *
     * "Works" means more than flipping `isBusy`: the poll must stop, and the run
     * must not be able to come back to life if whatever it was waiting on
     * eventually answers.
     */
    it("cancel stops the poll, clears busy, and disowns the run", async () => {
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: "job-1", status: "extracting", progress: 0.1 }),
        );
        // The stranded job: it never leaves `extracting`.
        mocks.getJob.mockResolvedValue(
            makeJob({ id: "job-1", status: "extracting", progress: 0.1 }),
        );

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await tick(900);
        expect(result.current.isBusy).toBe(true);
        const pollsBeforeCancel = mocks.getJob.mock.calls.length;
        expect(pollsBeforeCancel).toBeGreaterThan(0);

        await act(async () => {
            result.current.cancel();
        });

        expect(result.current.isBusy).toBe(false);
        expect(result.current.status).toBe("idle");
        expect(result.current.error).toBeNull();
        expect(mocks.unsubscribe).toHaveBeenCalled();

        // The poll is gone, and the job coming good later cannot resurrect the run.
        mocks.getJob.mockResolvedValue(
            makeJob({ id: "job-1", status: "ready", progress: 1 }),
        );
        await tick(3000);

        expect(mocks.getJob.mock.calls.length).toBe(pollsBeforeCancel);
        expect(mocks.getAudioUrl).not.toHaveBeenCalled();
        expect(mocks.postMessage).not.toHaveBeenCalled();
        expect(result.current.isBusy).toBe(false);
        expect(result.current.status).toBe("idle");
    });

    /**
     * Cancel has to REAP the diarization sidecar, not merely abandon it.
     *
     * Abandoning the backend is survivable for ffmpeg and yt-dlp — they finish on
     * their own in seconds. The diarizer is a CPU-bound ONNX child with a
     * 30-minute backstop timeout, so "the UI is idle" and "the machine is idle"
     * come apart badly: a cancelled run with nobody killing it pins a core for up
     * to half an hour. `cancel_diarization` is what closes that, and it needs the
     * job id of the run being cancelled — which is exactly the thing a stale
     * closure would get wrong.
     */
    it("cancel kills the diarization sidecar for the run it is cancelling", async () => {
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: "job-7", status: "extracting", progress: 0.1 }),
        );
        mocks.getJob.mockResolvedValue(
            makeJob({ id: "job-7", status: "extracting", progress: 0.1 }),
        );

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await tick(900);

        await act(async () => {
            result.current.cancel();
        });

        expect(mocks.cancelDiarization).toHaveBeenCalledWith("job-7");
        expect(result.current.isBusy).toBe(false);
    });

    it("cancel still clears the UI when the backend cannot be reached", async () => {
        // Fire-and-forget: a rejected cancel must not leave the app stuck busy,
        // and must not surface an error the user can do nothing about.
        mocks.cancelDiarization.mockRejectedValue(new Error("IPC is gone"));
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: "job-7", status: "extracting", progress: 0.1 }),
        );
        mocks.getJob.mockResolvedValue(
            makeJob({ id: "job-7", status: "extracting", progress: 0.1 }),
        );

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await tick(900);

        await act(async () => {
            result.current.cancel();
        });
        await tick(0);

        expect(result.current.isBusy).toBe(false);
        expect(result.current.status).toBe("idle");
        expect(result.current.error).toBeNull();
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

/**
 * The worker is a SINGLE object that outlives every run, its message handler is
 * `async`, and `terminate()` cannot recall a message already posted. So the app
 * can be handed output from a run it abandoned minutes ago. These tests post those
 * messages.
 *
 * Nothing in the suite did that before, which is why three rounds of guards on
 * `pendingWorkerRef` passed every test while the bug walked downstream: a boolean
 * "is anyone waiting?" cannot answer "is this the run I am waiting for?", and only
 * a test that posts a DEAD run's message can tell the two apart.
 */
describe("useTranscriber under an overlap the worker cannot stop", () => {
    const RUN_ONE_TEXT =
        "the long lecture, transcribed by the run we abandoned";
    const RUN_TWO_TEXT =
        "the short interview, which is what the user asked for";

    /** Both files ready the moment they are asked for; the worker is the slow part. */
    function twoReadyFiles() {
        mocks.createFileJob.mockImplementation(async ({ path }) =>
            makeJob({
                id: path.includes("lecture") ? "job-1" : "job-2",
                status: "ready",
                progress: 1,
                filename: path.includes("lecture")
                    ? "lecture.mp3"
                    : "interview.mp3",
            }),
        );
        mocks.getJob.mockImplementation(async (jobId: string) =>
            makeJob({ id: jobId, status: "ready", progress: 1 }),
        );
    }

    /**
     * THE CRITICAL REPRO — the wrong transcript, persisted under the wrong job,
     * and permanently cached. Reachable in the shipped app with the Cancel button.
     *
     *   1. Run 1 starts on a long file. Whisper begins grinding.
     *   2. The user hits Cancel. The app goes idle — but the worker does NOT stop;
     *      nothing can stop it except terminating it.
     *   3. The user starts run 2 on a DIFFERENT file. A second `transcribe` is
     *      posted; the old handler is still awaiting inside the same worker.
     *   4. Run 1's `complete` lands. `pendingWorkerRef` is truthy (it holds RUN 2),
     *      so the old `if (!pendingWorkerRef.current)` guard PASSED — and resolved
     *      RUN 2's promise with RUN 1's transcript.
     *   5. Run 2 persists run 1's text under job 2. The cache is content-keyed, so
     *      that wrong transcript is now the permanent cache hit for file 2: no
     *      recompute, no error, no way for the user to know.
     *
     * The fix is both halves. Cancel TERMINATES the worker (asserted below), which
     * is the only thing that actually stops transformers.js. And every message
     * carries the id of the run that asked for it, so a message that outlives its
     * run — a terminate cannot recall what is already in the queue — is dropped
     * instead of being mistaken for the live run's.
     */
    it("does not resolve a live run with an abandoned run's transcript", async () => {
        twoReadyFiles();
        const { result } = await renderTranscriber();

        // Run 1: the long file. It reaches the worker and Whisper starts.
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();
        expect(result.current.status).toBe("transcribing");
        expect(postedRunId(0)).toEqual(expect.any(Number));

        // The user cancels. The worker must be TERMINATED — ignoring its messages
        // is not stopping it, and it will otherwise pin every core for minutes.
        await act(async () => {
            result.current.cancel();
        });
        expect(mocks.restartWorker).toHaveBeenCalledTimes(1);
        expect(result.current.isBusy).toBe(false);

        // Run 2: a DIFFERENT file.
        await act(async () => {
            result.current.start("/tmp/interview.mp3");
        });
        await settle();
        expect(result.current.status).toBe("transcribing");
        expect(result.current.jobId).toBe("job-2");
        expect(mocks.postMessage).toHaveBeenCalledTimes(2);

        // Run 1's `complete` arrives anyway — it started minutes earlier and was
        // already in flight when the terminate landed.
        await emitFromWorker(workerComplete(postedRunId(0), RUN_ONE_TEXT));

        // It must be dropped whole. Run 2's promise is NOT settled by it...
        expect(result.current.status).toBe("transcribing");
        expect(result.current.isBusy).toBe(true);
        // ...it is not painted under run 2's filename...
        expect(result.current.output?.text ?? "").not.toContain("lecture");
        // ...and above all it is not written to the store.
        expect(mocks.persistTranscript).not.toHaveBeenCalled();

        // Run 2's own `complete` — the real one — still works, and is what gets
        // persisted, under run 2's job.
        await emitFromWorker(workerComplete(postedRunId(1), RUN_TWO_TEXT));

        expect(mocks.persistTranscript).toHaveBeenCalledTimes(1);
        const [persistedJobId, payload] = mocks.persistTranscript.mock.calls[0];
        expect(persistedJobId).toBe("job-2");
        expect(payload.full_text).toBe(RUN_TWO_TEXT);
        // The assertion the whole bug reduces to.
        expect(payload.full_text).not.toBe(RUN_ONE_TEXT);
        expect(result.current.status).toBe("completed");
    });

    /**
     * The variant that goes the other way. Run 2 picks a different preset/device,
     * so `PipelineFactory.getInstance` calls `dispose()` on the instance run 1 is
     * mid-inference on; run 1 throws and posts `error`. With one unkeyed slot that
     * `error` REJECTED RUN 2's promise, and blew run 1's message and `isBusy:
     * false` over a run that was transcribing perfectly well.
     */
    it("does not fail a live run with an abandoned run's error", async () => {
        twoReadyFiles();
        const { result } = await renderTranscriber();

        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();

        // Superseded WITHOUT a cancel: straight into a second run.
        await act(async () => {
            result.current.start("/tmp/interview.mp3");
        });
        await settle();
        expect(result.current.status).toBe("transcribing");
        expect(mocks.restartWorker).toHaveBeenCalledTimes(1);

        await emitFromWorker({
            status: "error",
            runId: postedRunId(0),
            data: { message: "Session already released" },
        });

        expect(result.current.error).toBeNull();
        expect(result.current.status).toBe("transcribing");
        expect(result.current.isBusy).toBe(true);

        await emitFromWorker(workerComplete(postedRunId(1), RUN_TWO_TEXT));

        expect(mocks.persistTranscript).toHaveBeenCalledTimes(1);
        expect(mocks.persistTranscript.mock.calls[0][1].full_text).toBe(
            RUN_TWO_TEXT,
        );
        expect(result.current.status).toBe("completed");
    });

    /**
     * `api.persistTranscript` is a Tauri IPC round-trip that writes one row per
     * segment — thousands of them for a lecture. It is SLOW, and the app sits
     * inside it with `status: "persisting"` and `isBusy: true`, which is to say
     * with the Cancel button on screen. So make it slow here: a persist that
     * settles in a microtask cannot reproduce anything, because no user event can
     * land inside a microtask.
     *
     * The persisted job echoes the text it was given, exactly as Rust does — a
     * mock that returned a fixed job would hide WHOSE transcript came back.
     */
    function slowPersist(ms: number) {
        mocks.persistTranscript.mockImplementation(
            async (jobId: string, payload: { full_text: string }) => {
                await new Promise((resolve) => setTimeout(resolve, ms));
                return makeJob({
                    id: jobId,
                    status: "completed",
                    progress: 1,
                    full_text: payload.full_text,
                    filename: "lecture.mp3",
                });
            },
        );
    }

    /**
     * THE LAST UNGUARDED AWAIT. Cancel is the trigger, and nothing holds the
     * persist.
     *
     *   1. Run 1 (a long lecture) finishes in the worker. `transcribePreparedJob`
     *      resumes and calls `persistWorkerTranscript`; the app is `persisting`,
     *      `isBusy`, so the Cancel button is on screen.
     *   2. It parks inside `await api.persistTranscript`.
     *   3. The user hits Cancel. `claimRun()` bumps the token — but the worker is
     *      idle so nothing is terminated, and nothing is rejected. NOTHING IS
     *      HOLDING THE PERSIST.
     *   4. The user starts run 2, a cached YouTube URL. It completes and paints.
     *   5. Run 1's persist resolves and calls `applyCompletedJob` — `setJobId`,
     *      `setTranscript`, `setStatus("completed")`, `setIsBusy(false)` — for a
     *      run that has been dead since step 3.
     *
     * The user asked for a YouTube video, watched it complete, and is left looking
     * at the transcript they cancelled, under the WRONG jobId. Every jobId-keyed
     * action downstream — export, and Task 8's diarizer — then targets job 1.
     *
     * The persist itself must still COMPLETE: writing run 1's transcript under run
     * 1's job is correct, and the content-keyed cache keeps the work. It simply
     * must not repaint a UI it no longer owns.
     */
    it("does not repaint a finished run with a dead run's persist", async () => {
        twoReadyFiles();
        slowPersist(1000);
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

        // Run 1: the long lecture. Whisper finishes; the persist begins.
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();
        await emitFromWorker(workerComplete(postedRunId(0), RUN_ONE_TEXT));

        expect(result.current.status).toBe("persisting");
        expect(result.current.isBusy).toBe(true);
        expect(mocks.persistTranscript).toHaveBeenCalledTimes(1);

        // The user cancels while the persist is still in flight.
        await act(async () => {
            result.current.cancel();
        });
        expect(result.current.isBusy).toBe(false);

        // Run 2: a cached YouTube URL. It completes and paints immediately.
        await act(async () => {
            result.current.startFromYouTube(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            );
        });
        await settle();
        expect(result.current.jobId).toBe("job-2");
        expect(result.current.output?.text).toBe("the cached transcript");

        // Run 1's persist lands.
        await tick(1000);

        // It wrote its OWN job — that is the feature, and it must not regress into
        // "the fix is to skip the persist".
        expect(mocks.persistTranscript.mock.calls[0][0]).toBe("job-1");
        expect(mocks.persistTranscript.mock.calls[0][1].full_text).toBe(
            RUN_ONE_TEXT,
        );

        // And it touched nothing the user is looking at.
        expect(result.current.jobId).toBe("job-2");
        expect(result.current.status).toBe("completed");
        expect(result.current.output?.text).toBe("the cached transcript");
        expect(result.current.output?.filename).toBe("A cached video");
        expect(result.current.isBusy).toBe(false);
    });

    /**
     * The same hole, under a LIVE run 2 rather than a finished one, and it is the
     * uglier half: `applyCompletedJob` ends with `setIsBusy(false)`, so a dead
     * run's persist RELEASES THE BUSY GATE while run 2's Whisper is still grinding.
     * The busy panel disappears, the Cancel button with it, the tiles unlock, and
     * run 1's transcript is on screen under run 1's job while run 2 runs.
     */
    it("does not release the busy gate under a live run when a dead run persists", async () => {
        twoReadyFiles();
        slowPersist(1000);

        const { result } = await renderTranscriber();

        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();
        await emitFromWorker(workerComplete(postedRunId(0), RUN_ONE_TEXT));
        expect(result.current.status).toBe("persisting");

        await act(async () => {
            result.current.cancel();
        });

        // Run 2: a different file, and its Whisper is still running when run 1's
        // persist lands.
        await act(async () => {
            result.current.start("/tmp/interview.mp3");
        });
        await settle();
        expect(result.current.status).toBe("transcribing");
        expect(result.current.isBusy).toBe(true);
        expect(result.current.jobId).toBe("job-2");

        await tick(1000);

        // The gate is what keeps the Cancel button — the only exit from a run —
        // on screen. A dead run may not open it.
        expect(result.current.isBusy).toBe(true);
        expect(result.current.status).toBe("transcribing");
        expect(result.current.jobId).toBe("job-2");
        expect(result.current.output?.text ?? "").not.toContain("lecture");

        // Run 2 then completes and persists normally, under its own job.
        await emitFromWorker(workerComplete(postedRunId(1), RUN_TWO_TEXT));
        await tick(1000);

        expect(mocks.persistTranscript).toHaveBeenCalledTimes(2);
        expect(mocks.persistTranscript.mock.calls[1][0]).toBe("job-2");
        expect(result.current.jobId).toBe("job-2");
        expect(result.current.status).toBe("completed");
        expect(result.current.output?.text).toBe(RUN_TWO_TEXT);
    });

    /**
     * `teardown()` did not set `settled`.
     *
     * `clearInterval` stops the NEXT poll; it cannot recall the `getJob` already in
     * flight — which is exactly why `settled` exists. `finish()` set it; the
     * cancel/supersede path (`cancelPendingWait` → `teardown`) did not. So a
     * superseded run's in-flight `getJob` sailed through the guard and handed a
     * DEAD job to `handleBackendJobUpdate`: its id, its progress, its status, and —
     * a failed job — its `setError` and `setIsBusy(false)`, all over a live run.
     *
     * The existing suite could not catch this: its `getJob` mock settles in a
     * microtask, so no `getJob` is ever in flight across a teardown. This one makes
     * the poll genuinely slow.
     */
    it("ignores a getJob still in flight when a supersede tears the wait down", async () => {
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: "job-1", status: "extracting", progress: 0.1 }),
        );
        // 500 ms to answer — and it answers with a job that FAILED.
        mocks.getJob.mockImplementation(async () => {
            await new Promise((resolve) => setTimeout(resolve, 500));
            return makeJob({
                id: "job-1",
                status: "failed",
                progress: 0.1,
                error: "ffmpeg exited with code Some(1)",
            });
        });
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

        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });

        // t=300: the poll fires. Its `getJob` is now in flight and will not answer
        // until t=800.
        await tick(300);
        expect(mocks.getJob).toHaveBeenCalledTimes(1);

        // t=400: superseded by a cached YouTube transcript. The wait is torn down —
        // but that in-flight `getJob` is still coming.
        await tick(100);
        await act(async () => {
            result.current.startFromYouTube(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            );
        });
        await settle();
        expect(result.current.status).toBe("completed");
        expect(result.current.jobId).toBe("job-2");

        // t=900: the dead job answers. Nothing of it may reach the UI, and it must
        // not settle anything.
        await tick(500);

        expect(result.current.status).toBe("completed");
        expect(result.current.jobId).toBe("job-2");
        expect(result.current.error).toBeNull();
        expect(result.current.output?.text).toBe("the cached transcript");
        expect(mocks.getAudioUrl).not.toHaveBeenCalled();
        expect(mocks.postMessage).not.toHaveBeenCalled();
    });
});

/**
 * Task 12: the toggle, the join, and the three-arm outcome actually reaching a
 * caller — the wiring nothing in the frontend did before this task. Every test
 * here drives the hook end to end (a real `start`, a real worker `complete`);
 * none of them call `consolidateSegments` or `assignSpeakers` directly, because
 * the property under test is that THIS hook calls them, not that they work.
 */
describe("useTranscriber's diarization wiring", () => {
    function readyFile(jobId = "job-1") {
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: jobId, status: "ready", progress: 1 }),
        );
    }

    it("does not call diarizeJob when the toggle is off, and renders exactly as before", async () => {
        readyFile();

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();
        await emitFromWorker(workerComplete(postedRunId(0), "hello there"));
        await settle();

        expect(mocks.diarizeJob).not.toHaveBeenCalled();
        // `null`, not any DiarizationOutcome -- "never ran" is its own state,
        // distinct from all three arms of the union.
        expect(result.current.diarizationOutcome).toBeNull();
        expect(
            result.current.output?.chunks.every(
                (chunk) => chunk.speaker === undefined,
            ),
        ).toBe(true);
    });

    it("kicks off diarizeJob once the canonical WAV is ready, with the numSpeakers hint", async () => {
        readyFile();
        mocks.diarizeJob.mockResolvedValue({
            status: "succeeded",
            turns: [],
            speaker_count: 0,
        });

        const { result } = await renderTranscriber();
        act(() => {
            result.current.setDiarizeEnabled(true);
        });

        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();

        // `readyJob.status === "ready"` here (see `readyFile`), so this fires
        // before the worker has produced anything -- the WAV exists, the
        // transcript does not.
        expect(mocks.diarizeJob).toHaveBeenCalledWith("job-1", undefined);
    });

    it("passes the optional speaker-count hint through, and omits it (auto-detect) when unset", async () => {
        readyFile();
        mocks.diarizeJob.mockResolvedValue({
            status: "succeeded",
            turns: [],
            speaker_count: 0,
        });

        const { result } = await renderTranscriber();
        act(() => {
            result.current.setDiarizeEnabled(true);
            result.current.setNumSpeakersHint(4);
        });

        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();

        expect(mocks.diarizeJob).toHaveBeenCalledWith("job-1", 4);
    });

    /**
     * The join. Diarization was kicked off concurrently with the worker
     * transcription (previous test), so by the time `complete` arrives the
     * outcome may or may not be in yet — `persistWorkerTranscript` awaits it.
     * This pins BOTH halves of hard constraint 3: the turns reach the
     * PERSISTED segments (what a reload will show) and the LIVE pre-persist
     * display (what the user sees before the persist round-trip, which writes
     * one row per segment, finishes).
     */
    it("threads a succeeded outcome's turns into the persisted segments and the live pre-persist display", async () => {
        readyFile();
        mocks.diarizeJob.mockResolvedValue({
            status: "succeeded",
            turns: [{ start: 0, end: 5, speaker: 0 }],
            speaker_count: 1,
        });
        // Slow enough to inspect the LIVE display before the persisted rows
        // come back.
        mocks.persistTranscript.mockImplementation(async (jobId: string) => {
            await new Promise((resolve) => setTimeout(resolve, 1000));
            return makeJob({ id: jobId, status: "completed", progress: 1 });
        });

        const { result } = await renderTranscriber();
        act(() => {
            result.current.setDiarizeEnabled(true);
        });

        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();
        await emitFromWorker(workerComplete(postedRunId(0), "hello there"));

        // The join already happened (diarizeJob resolves in a microtask, well
        // inside the ticks `emitFromWorker` flushes); the 1s persist has not.
        expect(result.current.status).toBe("persisting");
        expect(result.current.output?.chunks.length).toBeGreaterThan(0);
        expect(
            result.current.output?.chunks.every(
                (chunk) => chunk.speaker === "SPEAKER_00",
            ),
        ).toBe(true);

        await tick(1000);

        expect(mocks.persistTranscript).toHaveBeenCalledTimes(1);
        const segments = mocks.persistTranscript.mock.calls[0][1]
            .segments as Array<{ speaker?: string }>;
        expect(segments.length).toBeGreaterThan(0);
        expect(
            segments.every((segment) => segment.speaker === "SPEAKER_00"),
        ).toBe(true);

        expect(result.current.diarizationOutcome).toEqual({
            status: "succeeded",
            turns: [{ start: 0, end: 5, speaker: 0 }],
            speaker_count: 1,
        });
    });

    it("surfaces a degraded outcome, and never fails the transcript over it", async () => {
        readyFile();
        mocks.diarizeJob.mockResolvedValue({
            status: "degraded",
            reason: "the segmentation model is not installed",
        });

        const { result } = await renderTranscriber();
        act(() => {
            result.current.setDiarizeEnabled(true);
        });
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();
        await emitFromWorker(workerComplete(postedRunId(0), "hello there"));
        await settle();

        expect(result.current.diarizationOutcome).toEqual({
            status: "degraded",
            reason: "the segmentation model is not installed",
        });
        // The governing rule: diarization failure must never fail transcription.
        expect(result.current.status).toBe("completed");
        expect(result.current.error).toBeNull();
        expect(
            result.current.output?.chunks.every(
                (chunk) => chunk.speaker === undefined,
            ),
        ).toBe(true);
    });

    it("surfaces a cancelled outcome distinctly from degraded", async () => {
        readyFile();
        mocks.diarizeJob.mockResolvedValue({ status: "cancelled" });

        const { result } = await renderTranscriber();
        act(() => {
            result.current.setDiarizeEnabled(true);
        });
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();
        await emitFromWorker(workerComplete(postedRunId(0), "hello there"));
        await settle();

        expect(result.current.diarizationOutcome).toEqual({
            status: "cancelled",
        });
        expect(result.current.diarizationOutcome?.status).not.toBe("degraded");
    });

    it("keeps a real empty-turn success distinct from off, degraded and cancelled", async () => {
        readyFile();
        mocks.diarizeJob.mockResolvedValue({
            status: "succeeded",
            turns: [],
            speaker_count: 0,
        });

        const { result } = await renderTranscriber();
        act(() => {
            result.current.setDiarizeEnabled(true);
        });
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();
        await emitFromWorker(workerComplete(postedRunId(0), "hello there"));
        await settle();

        expect(result.current.diarizationOutcome).toEqual({
            status: "succeeded",
            turns: [],
            speaker_count: 0,
        });
    });

    /**
     * A malformed REQUEST (unknown job, or — the realistic case — a cache hit
     * whose prepared WAV has already aged out) is a REJECTED promise, not one
     * of the three `DiarizationOutcome` arms. Swallowed, it is exactly as
     * invisible a failure as an unhandled `"degraded"`, so it is folded into
     * one for display rather than silently producing an unlabelled transcript
     * with no explanation at all.
     */
    it("folds a diarizeJob rejection into a visible degraded outcome", async () => {
        readyFile();
        mocks.diarizeJob.mockRejectedValue(
            new Error("Prepared audio not found"),
        );

        const { result } = await renderTranscriber();
        act(() => {
            result.current.setDiarizeEnabled(true);
        });
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();
        await emitFromWorker(workerComplete(postedRunId(0), "hello there"));
        await settle();

        expect(result.current.diarizationOutcome).toEqual({
            status: "degraded",
            reason: "Prepared audio not found",
        });
        expect(result.current.status).toBe("completed");
    });

    /**
     * Constraint 6: legacy (no-word-timings) and cache-hit transcripts still
     * diarize. A cache hit's segments carry no `speaker` field yet, so this is
     * the one case where `applyCompletedJob` is handed turns explicitly rather
     * than reading a label already baked into the rows.
     */
    it("diarizes a cache-hit job too, without waiting on word timings that do not exist", async () => {
        mocks.createFileJob.mockResolvedValue(
            makeJob({
                id: "job-1",
                status: "completed",
                progress: 1,
                cache_hit: true,
                full_text: "Hello there and welcome.",
                segments: [
                    { start: 0, end: 2, text: "Hello there and welcome." },
                ],
            }),
        );
        mocks.diarizeJob.mockResolvedValue({
            status: "succeeded",
            turns: [{ start: 0, end: 2, speaker: 0 }],
            speaker_count: 1,
        });

        const { result } = await renderTranscriber();
        act(() => {
            result.current.setDiarizeEnabled(true);
        });
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();

        expect(mocks.diarizeJob).toHaveBeenCalledWith("job-1", undefined);
        expect(result.current.status).toBe("completed");
        expect(
            result.current.output?.chunks.some(
                (chunk) => chunk.speaker === "SPEAKER_00",
            ),
        ).toBe(true);
    });

    /**
     * THE CROSS-RUN LEAK (Task 12).
     *
     * `diarizeAudio` is the one resumption point in this hook that used to
     * write state from an async continuation with no `runIdRef.current ===
     * runId` check -- every other one (`persistWorkerTranscript`'s
     * `setTranscript`, the various `applyCompletedJob` call sites) guards the
     * moment its promise resolves, not just the moment it is kicked off.
     *
     * Here run 1's `diarizeJob` is still in flight when run 2 supersedes it.
     * Run 2 gets its own (different) outcome. Run 1's stale promise then
     * resolves. Unguarded, that write clobbers run 2's `diarizationOutcome`
     * with run 1's answer -- a stale "N speakers identified" banner on a
     * screen that has moved on to a different file.
     */
    it("does not let a superseded run's stale diarizeJob outcome overwrite the run that replaced it", async () => {
        readyFile("job-1");
        const staleOutcome = deferred<DiarizationOutcome>();
        mocks.diarizeJob.mockImplementation(async (jobId: string) => {
            if (jobId === "job-1") {
                return staleOutcome.promise;
            }
            return {
                status: "succeeded",
                turns: [{ start: 0, end: 3, speaker: 0 }],
                speaker_count: 1,
            };
        });

        const { result } = await renderTranscriber();
        act(() => {
            result.current.setDiarizeEnabled(true);
        });

        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();

        expect(mocks.diarizeJob).toHaveBeenCalledWith("job-1", undefined);
        // Run 1's diarizeJob has not answered yet -- nothing to show.
        expect(result.current.diarizationOutcome).toBeNull();

        // Run 2 supersedes run 1 before the stale diarizeJob answers.
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: "job-2", status: "ready", progress: 1 }),
        );
        await act(async () => {
            result.current.start("/tmp/other.mp3");
        });
        await settle();
        await settle();

        expect(mocks.diarizeJob).toHaveBeenCalledWith("job-2", undefined);
        expect(result.current.jobId).toBe("job-2");
        expect(result.current.diarizationOutcome).toEqual({
            status: "succeeded",
            turns: [{ start: 0, end: 3, speaker: 0 }],
            speaker_count: 1,
        });

        // Run 1's diarizeJob finally answers -- stale, and must be dropped.
        staleOutcome.resolve({
            status: "succeeded",
            turns: [{ start: 10, end: 20, speaker: 1 }],
            speaker_count: 1,
        });
        await settle();
        await settle();

        // Run 2's own legitimate outcome is still what is showing.
        expect(result.current.diarizationOutcome).toEqual({
            status: "succeeded",
            turns: [{ start: 0, end: 3, speaker: 0 }],
            speaker_count: 1,
        });
    });

    /**
     * The worst case from the task-12 report: run 2 has the toggle OFF, so it
     * never calls `diarizeJob` at all and starts (correctly) from a `null`
     * `diarizationOutcome`. Run 1's stale outcome resolving afterward must
     * not plant a speaker banner over a run whose transcript carries no
     * speaker labels -- that would violate the "byte-unchanged when the
     * toggle is off" guarantee on top of the cross-run leak itself.
     */
    it("does not let a stale diarizeJob outcome leak onto a superseding run with the toggle off", async () => {
        readyFile("job-1");
        const staleOutcome = deferred<DiarizationOutcome>();
        mocks.diarizeJob.mockImplementation(async (jobId: string) => {
            if (jobId === "job-1") {
                return staleOutcome.promise;
            }
            throw new Error("run 2 must never call diarizeJob");
        });

        const { result } = await renderTranscriber();
        act(() => {
            result.current.setDiarizeEnabled(true);
        });

        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();

        expect(mocks.diarizeJob).toHaveBeenCalledWith("job-1", undefined);
        expect(result.current.diarizationOutcome).toBeNull();

        // Run 2 supersedes run 1 with the toggle OFF.
        act(() => {
            result.current.setDiarizeEnabled(false);
        });
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: "job-2", status: "ready", progress: 1 }),
        );
        await act(async () => {
            result.current.start("/tmp/other.mp3");
        });
        await settle();
        await settle();

        expect(result.current.jobId).toBe("job-2");
        expect(mocks.diarizeJob).toHaveBeenCalledTimes(1);
        expect(result.current.diarizationOutcome).toBeNull();

        // Run 2 completes its (undiarized) transcription -- the byte-unchanged
        // path: no speaker labels, because the toggle was off.
        await emitFromWorker(workerComplete(postedRunId(1), "hello there"));
        await settle();

        expect(result.current.status).toBe("completed");
        expect(
            result.current.output?.chunks.every(
                (chunk) => chunk.speaker === undefined,
            ),
        ).toBe(true);

        // Run 1's diarizeJob finally answers -- stale, and must be dropped.
        staleOutcome.resolve({
            status: "succeeded",
            turns: [{ start: 10, end: 20, speaker: 1 }],
            speaker_count: 1,
        });
        await settle();
        await settle();

        expect(mocks.diarizeJob).toHaveBeenCalledTimes(1);
        expect(result.current.diarizationOutcome).toBeNull();
        expect(
            result.current.output?.chunks.every(
                (chunk) => chunk.speaker === undefined,
            ),
        ).toBe(true);
    });

    it("renameSpeaker writes through api.setSpeakerName and refreshes speakerNames", async () => {
        readyFile();
        mocks.getSpeakerNames
            .mockResolvedValueOnce({})
            .mockResolvedValueOnce({ SPEAKER_00: "Alice" });

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();
        await emitFromWorker(workerComplete(postedRunId(0), "hello there"));
        await settle();

        expect(result.current.jobId).toBe("job-1");
        // The fetch `applyCompletedJob` fires on completion, independent of the
        // toggle -- a speaker named in an earlier session must still show up.
        expect(mocks.getSpeakerNames).toHaveBeenCalledWith("job-1");

        await act(async () => {
            await result.current.renameSpeaker("SPEAKER_00", "Alice");
        });

        expect(mocks.setSpeakerName).toHaveBeenCalledWith(
            "job-1",
            "SPEAKER_00",
            "Alice",
        );
        expect(mocks.getSpeakerNames).toHaveBeenCalledTimes(2);
        expect(result.current.speakerNames).toEqual({ SPEAKER_00: "Alice" });
    });

    it("does not rename anything when no job is known yet", async () => {
        const { result } = await renderTranscriber();

        await act(async () => {
            await result.current.renameSpeaker("SPEAKER_00", "Alice");
        });

        expect(mocks.setSpeakerName).not.toHaveBeenCalled();
    });
});
