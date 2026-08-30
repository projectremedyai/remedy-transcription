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
    setSpeakerName: vi.fn(),
    getSpeakerNames: vi.fn(),
    /** The gemini engine's own api surface — see R8's "engine-aware readiness gates". */
    transcribeWithGemini: vi.fn(),
    subscribeToGeminiProgress: vi.fn(),
    cancelGeminiTranscription: vi.fn(),
    estimateGeminiCost: vi.fn(),
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
        setSpeakerName: mocks.setSpeakerName,
        getSpeakerNames: mocks.getSpeakerNames,
        transcribeWithGemini: mocks.transcribeWithGemini,
        subscribeToGeminiProgress: mocks.subscribeToGeminiProgress,
        cancelGeminiTranscription: mocks.cancelGeminiTranscription,
        estimateGeminiCost: mocks.estimateGeminiCost,
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
    mocks.subscribeToGeminiProgress.mockReturnValue(() => undefined);
    // A single chunk unless a test says otherwise: no confirmation, no cost
    // question, exactly as a short file behaves.
    mocks.estimateGeminiCost.mockResolvedValue({
        duration_secs: 300,
        chunk_count: 1,
        estimated_usd: 0.025,
        diarization_available: true,
    });
    mocks.getModelStatus.mockResolvedValue({
        models_ready: true,
        missing_models: [],
        items: MODEL_PRESETS.map((preset) => ({
            model_id: preset.modelId,
            ready: true,
        })),
    });
    mocks.getAudioUrl.mockResolvedValue("asset://localhost/audio.wav");
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
    /**
     * A cached transcript is a PERMANENT hit on the Rust side: `find_transcript`
     * serves the stored row forever and no engine runs. Until `force` existed
     * there was no way to ask for a fresh run from inside the app, so a
     * transcript written by a buggy engine could never be replaced -- which is
     * exactly what happened when Gemini persisted whole transcripts with every
     * word glued to the last one.
     *
     * `retranscribe` replays the last source with `force`, and `force` is the
     * whole mechanism: Rust skips the cache lookup, and `persist_transcript`
     * upserts over the stale row when the fresh run lands.
     */
    it("replays the last file with force when asked to re-transcribe", async () => {
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: "job-1", status: "ready", progress: 1 }),
        );

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();

        // The ordinary run must NOT force, or every reopen pays Gemini again.
        expect(mocks.createFileJob).toHaveBeenCalledTimes(1);
        expect(mocks.createFileJob.mock.calls[0][0].force).toBeFalsy();

        await act(async () => {
            result.current.retranscribe();
        });
        await settle();

        expect(mocks.createFileJob).toHaveBeenCalledTimes(2);
        const second = mocks.createFileJob.mock.calls[1][0];
        expect(second.force).toBe(true);
        // Same source, not a re-pick: the user pressed a button on a transcript
        // that is already on screen.
        expect(second.path).toBe("/tmp/lecture.mp3");
    });

    /** The YouTube entry point has the same cache and needs the same way out. */
    it("replays the last YouTube url with force when asked to re-transcribe", async () => {
        mocks.createYouTubeJob.mockResolvedValue(
            makeJob({ id: "job-1", status: "ready", progress: 1 }),
        );

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.startFromYouTube("https://youtu.be/abcdefghijk");
        });
        await settle();

        await act(async () => {
            result.current.retranscribe();
        });
        await settle();

        expect(mocks.createYouTubeJob).toHaveBeenCalledTimes(2);
        const second = mocks.createYouTubeJob.mock.calls[1][0];
        expect(second.force).toBe(true);
        expect(second.url).toBe("https://youtu.be/abcdefghijk");
    });

    /**
     * Nothing has run yet, so there is no source to replay. It must be a no-op
     * rather than a run against `undefined` -- the button is rendered from
     * transcript state, and a stale render could still fire it.
     */
    it("does nothing when asked to re-transcribe before any run", async () => {
        const { result } = await renderTranscriber();

        await act(async () => {
            result.current.retranscribe();
        });
        await settle();

        expect(mocks.createFileJob).not.toHaveBeenCalled();
        expect(mocks.createYouTubeJob).not.toHaveBeenCalled();
    });

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
     * action downstream, like export, then targets job 1.
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

    /**
     * THE WINDOW LEFT OPEN BY MOVING THE FETCH/DECODE INTO THE ENGINE.
     *
     * `transcribePreparedJob` used to re-check the run token between decoding
     * the audio and calling `runWorkerTranscription`. Extracting that step into
     * `localEngine.run()` (`getAudioUrl` → `fetch` the WAV → decode it) dropped
     * that check: nothing re-checked the token between the audio's real,
     * multi-await load finishing and the worker being posted to.
     *
     * The failure this opened: cancel a run while its audio is still loading.
     * `pendingWorkerRef` is still `null` at that moment — nothing has been
     * posted to the worker yet — so `claimRun`'s abandon branch has nothing to
     * reject and does not restart the worker (asserted below). Unguarded, the
     * load then finishes, the dead run resumes: it flips the UI back to
     * "transcribing" with `isBusy: true` and posts to a worker nobody
     * restarted — resurrecting a run the user already watched get cancelled,
     * and (though not exercised by this test, which stops at the post) leaving
     * the app wedged busy once that worker's never-terminated `complete`
     * arrives and the persist it would have needed is skipped for a dead run.
     *
     * The fix restores the pre-worker guard as the first statement inside
     * `runWorkerTranscription`'s promise executor, in the hook — not in the
     * engine, which must not gain run-token knowledge.
     *
     * `getAudioUrl` is made to take real time here — a `mockResolvedValue`
     * settles in a microtask, before any timer can tick, so the window would
     * never open (the same argument made above for the `createJob` and
     * `getJob` windows).
     */
    it("does not resume a transcription after a cancel lands while the audio is loading", async () => {
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: "job-1", status: "ready", progress: 1 }),
        );
        // The Tauri round trip for the prepared WAV's URL. In production this
        // is followed by a `fetch` of the whole file and a `decodeAudioData`;
        // this one delay is enough to open the window.
        mocks.getAudioUrl.mockImplementation(async () => {
            await new Promise((resolve) => setTimeout(resolve, 1000));
            return "asset://localhost/audio.wav";
        });

        const { result } = await renderTranscriber();

        await act(async () => {
            result.current.start("/tmp/lecture.mp3");
        });
        await settle();
        expect(result.current.status).toBe("loading-audio");
        expect(result.current.isBusy).toBe(true);
        expect(mocks.postMessage).not.toHaveBeenCalled();

        // The user cancels while the audio is still loading. Nothing has been
        // posted to the worker, so there is nothing for `claimRun` to abandon —
        // it must NOT restart the worker, which is exactly the state that
        // leaves a dead run free to resume if nothing else stops it.
        await act(async () => {
            result.current.cancel();
        });
        expect(mocks.restartWorker).not.toHaveBeenCalled();
        expect(result.current.isBusy).toBe(false);
        expect(result.current.status).toBe("idle");

        // The load now finishes. The dead run must not resume into a
        // transcription: no worker post, and the busy gate/status the cancel
        // set must still hold rather than flipping back to "transcribing".
        await tick(1000);

        expect(mocks.postMessage).not.toHaveBeenCalled();
        expect(result.current.isBusy).toBe(false);
        expect(result.current.status).toBe("idle");
        expect(result.current.error).toBeNull();
    });
});

/**
 * Speaker renaming: a metadata write that does not re-transcribe or
 * re-diarize, and the fetch that reflects it back into `speakerNames`. This
 * survives independently of whatever engine (if any) produced the speaker
 * labels in the first place.
 */
describe("useTranscriber's speaker renaming", () => {
    function readyFile(jobId = "job-1") {
        mocks.createFileJob.mockResolvedValue(
            makeJob({ id: jobId, status: "ready", progress: 1 }),
        );
    }

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
        // The fetch `applyCompletedJob` fires unconditionally on completion --
        // a speaker named in an earlier session must still show up.
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

/**
 * R8 (Task 13's controller ruling): `modelStatus` (from Rust's `list_models`)
 * reports LOCAL Whisper download status only, and `gemini-3.5-transcribe`
 * never appears in it — `beforeEach` above seeds `modelStatus.items` with
 * nothing but `MODEL_PRESETS`. Both gates below are the ones that used to read
 * that absence as "not ready" for every engine, not just the local one.
 */
describe("R8: engine-aware readiness gates", () => {
    it("selectedModelAvailable is true for the gemini engine (gate 1)", async () => {
        const { result } = await renderTranscriber();

        await act(async () => {
            result.current.setEngine("gemini");
        });
        await settle();

        // Pre-fix, this fell through to `modelStatus.items.some(...)`, which
        // can never match `gemini-3.5-transcribe`, and stayed `false` forever.
        expect(result.current.selectedModelAvailable).toBe(true);
    });

    it("a gemini run is not blocked by the local-install check, and reaches the engine (gate 2)", async () => {
        mocks.createFileJob.mockResolvedValue(
            makeJob({
                id: "job-1",
                status: "ready",
                progress: 1,
                model_id: "gemini-3.5-transcribe",
            }),
        );
        mocks.transcribeWithGemini.mockResolvedValue({
            text: "hello there",
            words: [
                { text: "hello", start: 0, end: 1 },
                { text: "there", start: 1, end: 2 },
            ],
            speakers: { status: "unavailable", reason: "chunked audio" },
            audio_duration: 2,
        });

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.setEngine("gemini");
        });
        await settle();

        await act(async () => {
            result.current.start("/tmp/talk.mp4");
        });
        await settle();

        // Pre-fix, `beginRun` threw "Model files for ... are not installed on
        // the server" BEFORE `createFileJob` was ever called. Reaching
        // `createFileJob` at all is gate 2 not firing; reaching `completed`
        // is confirmation the gemini engine actually ran.
        expect(mocks.createFileJob).toHaveBeenCalled();
        expect(mocks.transcribeWithGemini).toHaveBeenCalledWith("job-1");
        expect(result.current.status).toBe("completed");
        expect(result.current.error).toBeNull();
    });
});

/**
 * The test the spec asks for by name: "`useTranscriber`'s Gemini path with
 * `api.transcribeWithGemini` mocked, asserting specifically that the
 * run-token semantics survive: a cancelled Gemini run cannot paint."
 *
 * What existed was a happy path here and an isolated `abandon` unit test in
 * `geminiEngine.test.ts`. Neither reaches the thing that matters — nothing at
 * the HOOK level asserted that `cancel()` during a Gemini run actually calls
 * `api.cancelGeminiTranscription`, which is the only thing standing between an
 * abandoned cloud run and (a) Google continuing to bill for a transcript
 * nobody will read and (b) the user's audio sitting in Google's storage until
 * the 48-hour expiry.
 *
 * Everything here uses fake timers and REAL task boundaries, for the same
 * reason the overlap suite above does: a promise that settles in a microtask
 * cannot express a cancel landing mid-run, because no user event can be
 * delivered inside a microtask. `transcribeWithGemini` is therefore made to
 * take actual time.
 */
describe("useTranscriber's Gemini path under cancellation", () => {
    const CLOUD_TEXT = "the cloud transcript nobody is waiting for any more";
    const LOCAL_TEXT = "the run the user actually asked for";

    beforeEach(() => {
        // The real command resolves a bool; `geminiEngine.abandon` calls
        // `.catch()` on what it gets back. The shared `mockReset()` above
        // leaves it returning `undefined`, which no production call ever does.
        mocks.cancelGeminiTranscription.mockResolvedValue(true);
    });

    /** A Gemini round trip that spans real timer ticks, not a microtask. */
    function slowGemini(ms: number, text = CLOUD_TEXT) {
        mocks.transcribeWithGemini.mockImplementation(async () => {
            await new Promise((resolve) => setTimeout(resolve, ms));
            return {
                text,
                words: [{ text, start: 0, end: 5 }],
                speakers: {
                    status: "unavailable",
                    reason: "split into 3 parts",
                },
                audio_duration: 5,
            };
        });
    }

    /** Echoes what it was given, so a test can tell WHOSE transcript landed. */
    function echoingPersist() {
        mocks.persistTranscript.mockImplementation(
            async (jobId: string, payload: { full_text: string }) =>
                makeJob({
                    id: jobId,
                    status: "completed",
                    progress: 1,
                    full_text: payload.full_text,
                    filename: "talk.mp4",
                }),
        );
    }

    /** `talk.mp4` is the cloud run; anything else is the local one. */
    function twoReadyFiles() {
        mocks.createFileJob.mockImplementation(async ({ path }) =>
            makeJob({
                id: path.includes("talk") ? "job-1" : "job-2",
                status: "ready",
                progress: 1,
                filename: path.includes("talk") ? "talk.mp4" : "interview.mp3",
                model_id: path.includes("talk")
                    ? "gemini-3.5-transcribe"
                    : "onnx-community/whisper-base_timestamped",
            }),
        );
        mocks.getJob.mockImplementation(async (jobId: string) =>
            makeJob({ id: jobId, status: "ready", progress: 1 }),
        );
    }

    it("tells Rust to abort the run when a Gemini run is cancelled", async () => {
        twoReadyFiles();
        slowGemini(5000);

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.setEngine("gemini");
        });
        await settle();
        await act(async () => {
            result.current.start("/tmp/talk.mp4");
        });
        await settle();

        expect(mocks.transcribeWithGemini).toHaveBeenCalledWith("job-1");
        expect(result.current.isBusy).toBe(true);

        await act(async () => {
            result.current.cancel();
        });

        // The whole point. Unlike the local engine's no-op `abandon`, this one
        // has to reach the backend.
        expect(mocks.cancelGeminiTranscription).toHaveBeenCalledWith("job-1");
        // And it must not restart the worker: nothing was ever posted to it,
        // and a needless terminate throws away a warm, already-loaded model.
        expect(mocks.restartWorker).not.toHaveBeenCalled();
        expect(result.current.isBusy).toBe(false);
        expect(result.current.status).toBe("idle");
    });

    /**
     * `cancel_gemini_transcription` is best-effort by design — the request may
     * already be past the point Rust can stop it, and `abandon` swallows its
     * own rejection. So the token has to hold the line independently: the run
     * may still FINISH, it may not PAINT.
     */
    it("does not paint a cancelled Gemini run that resolves afterwards", async () => {
        twoReadyFiles();
        echoingPersist();
        slowGemini(5000);

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.setEngine("gemini");
        });
        await settle();
        await act(async () => {
            result.current.start("/tmp/talk.mp4");
        });
        await settle();

        await act(async () => {
            result.current.cancel();
        });
        expect(result.current.status).toBe("idle");

        // Google answers anyway, five seconds after nobody was listening.
        await tick(5000);

        expect(result.current.status).toBe("idle");
        expect(result.current.isBusy).toBe(false);
        expect(result.current.output).toBeUndefined();
        expect(result.current.error).toBeNull();
        // Above all: not written, and so not content-cached under
        // (source, "gemini-3.5-transcribe", ...) for every future reopen.
        expect(mocks.persistTranscript).not.toHaveBeenCalled();
    });

    /**
     * The engine `cancel()` abandons must be the RUNNING run's, not the
     * selector's.
     *
     * `AudioManager` renders the engine radios `disabled={transcriber.isBusy}`,
     * so today a user cannot move the selector mid-run by clicking — but the
     * hook is not entitled to assume that. `useTranscriber`'s own comment on
     * `cancel` enumerates what depends on that busy gate and contemplates
     * removing it for a queue or a second panel, and this was one of the
     * things quietly hanging off it: with the selector on "local", `cancel()`
     * called the LOCAL engine's no-op `abandon`, the live Gemini run was never
     * told to stop, and it kept billing.
     */
    it("abandons the engine the run is using, not the one the selector shows", async () => {
        twoReadyFiles();
        slowGemini(5000);

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.setEngine("gemini");
        });
        await settle();
        await act(async () => {
            result.current.start("/tmp/talk.mp4");
        });
        await settle();
        expect(mocks.transcribeWithGemini).toHaveBeenCalledWith("job-1");

        // The selector moves while the cloud run is still in flight.
        await act(async () => {
            result.current.setEngine("local");
        });
        await settle();
        expect(result.current.engine).toBe("local");

        await act(async () => {
            result.current.cancel();
        });

        expect(mocks.cancelGeminiTranscription).toHaveBeenCalledWith("job-1");
    });

    /**
     * The supersede variant, with no cancel in it at all: a second run simply
     * takes the UI while the first is parked inside its cloud round trip. The
     * abandoned Gemini result arrives afterwards and must be dropped whole —
     * not painted over the transcript the user is reading, and above all not
     * persisted under the second run's job, where the content-keyed cache
     * would serve it back forever.
     */
    it("does not let a late Gemini resolve paint over the run that superseded it", async () => {
        twoReadyFiles();
        echoingPersist();
        slowGemini(5000);

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.setEngine("gemini");
        });
        await settle();
        await act(async () => {
            result.current.start("/tmp/talk.mp4");
        });
        await settle();
        expect(result.current.jobId).toBe("job-1");

        // Run 2: the local engine, on a different file, straight over the top.
        await act(async () => {
            result.current.setEngine("local");
        });
        await settle();
        await act(async () => {
            result.current.start("/tmp/interview.mp3");
        });
        await settle();
        expect(result.current.jobId).toBe("job-2");

        await emitFromWorker(workerComplete(postedRunId(0), LOCAL_TEXT));
        expect(result.current.status).toBe("completed");
        expect(result.current.output?.text).toBe(LOCAL_TEXT);

        // Run 1's cloud call finally answers.
        await tick(5000);

        expect(mocks.persistTranscript).toHaveBeenCalledTimes(1);
        const [persistedJobId, payload] = mocks.persistTranscript.mock.calls[0];
        expect(persistedJobId).toBe("job-2");
        expect(payload.full_text).toBe(LOCAL_TEXT);
        expect(result.current.output?.text).toBe(LOCAL_TEXT);
        expect(result.current.status).toBe("completed");
    });
    /**
     * THE SUPERSEDE MONEY LEAK — the same abandonment as `cancel()`, reached by
     * the other door, and the one that was never told to stop.
     *
     * `cancel()` calls `claimRun()` and THEN `engines[...].abandon(jobId)`.
     * `claimRun()` on its own — which is all either `start*` entry point calls —
     * did the first half only. So a second run started over a live Gemini run
     * bumped the token, and the test above this one proves the late result is
     * correctly dropped: nothing paints, nothing persists, the cache stays
     * honest. The UI is entirely correct.
     *
     * The MONEY is not. Google is still transcribing run 1, still billing every
     * minute of it, and the user's audio is still sitting in Google's storage
     * until the 48-hour expiry — for a transcript the app has already promised
     * itself it will throw away. A 26-minute file is ~$1.80 of that.
     *
     * The user does not need to touch Cancel to reach this. Dropping a second
     * file on the window is enough, and that is the ordinary way to change your
     * mind about which file you meant.
     */
    it("tells Rust to abort a Gemini run that a second run supersedes", async () => {
        twoReadyFiles();
        slowGemini(5000);

        const { result } = await renderTranscriber();
        await act(async () => {
            result.current.setEngine("gemini");
        });
        await settle();
        await act(async () => {
            result.current.start("/tmp/talk.mp4");
        });
        await settle();

        // Run 1 is genuinely in flight in the cloud, not merely queued.
        expect(mocks.transcribeWithGemini).toHaveBeenCalledWith("job-1");
        expect(result.current.isBusy).toBe(true);
        expect(result.current.jobId).toBe("job-1");

        // Run 2 takes the UI. No cancel anywhere in this test.
        await act(async () => {
            result.current.setEngine("local");
        });
        await settle();
        await act(async () => {
            result.current.start("/tmp/interview.mp3");
        });
        await settle();
        expect(result.current.jobId).toBe("job-2");

        // The whole point: the superseded cloud run was told to stop, under the
        // engine that was RUNNING it, not the one the selector now shows.
        expect(mocks.cancelGeminiTranscription).toHaveBeenCalledWith("job-1");
    });
});

/**
 * `MAX_DURATION_HOURS` has only ever guarded the YouTube path. A long LOCAL
 * file went to Gemini with nothing between it and the bill — and the local path
 * was never capped for a good reason, since on-device Whisper is free at any
 * length. The guard therefore belongs to the ENGINE, not to the source, which
 * is why it lives in `geminiEngine.run` behind `confirmCost` and why the local
 * engine never asks.
 */
describe("useTranscriber's cost confirmation for a long Gemini run", () => {
    beforeEach(() => {
        // Same reason the suite above does this: the shared `mockReset()`
        // leaves this returning `undefined`, and `geminiEngine.abandon` calls
        // `.catch()` on whatever it gets back. No production call ever does.
        mocks.cancelGeminiTranscription.mockResolvedValue(true);
    });

    function longGeminiFile() {
        mocks.createFileJob.mockResolvedValue(
            makeJob({
                id: "job-1",
                status: "ready",
                progress: 1,
                filename: "lecture.mp3",
                model_id: "gemini-3.5-transcribe",
            }),
        );
        mocks.getJob.mockImplementation(async (jobId: string) =>
            makeJob({ id: jobId, status: "ready", progress: 1 }),
        );
        mocks.estimateGeminiCost.mockResolvedValue({
            duration_secs: 21600,
            chunk_count: 15,
            estimated_usd: 1.8,
            diarization_available: false,
        });
        mocks.transcribeWithGemini.mockResolvedValue({
            text: "the six hour lecture",
            words: [{ text: "the six hour lecture", start: 0, end: 5 }],
            speakers: { status: "unavailable", reason: "split into 15 parts" },
            audio_duration: 21600,
        });
    }

    async function startLongRun() {
        const rendered = await renderTranscriber();
        await act(async () => {
            rendered.result.current.setEngine("gemini");
        });
        await settle();
        await act(async () => {
            rendered.result.current.start("/tmp/lecture.mp3");
        });
        await settle();
        return rendered;
    }

    /** The user must be shown the bill, and nothing may be bought until they answer. */
    it("holds the run at a confirmation and spends nothing while it waits", async () => {
        longGeminiFile();
        const { result } = await startLongRun();

        expect(result.current.pendingCost).toEqual({
            durationSecs: 21600,
            chunkCount: 15,
            estimatedUsd: 1.8,
            diarizationAvailable: false,
        });
        expect(mocks.transcribeWithGemini).not.toHaveBeenCalled();
    });

    it("runs the transcription once the cost is approved", async () => {
        longGeminiFile();
        const { result } = await startLongRun();

        await act(async () => {
            result.current.approveCost();
        });
        await settle();

        expect(mocks.transcribeWithGemini).toHaveBeenCalledWith("job-1");
        expect(result.current.pendingCost).toBeNull();
    });

    /**
     * Declining is not a FAILURE. The user answered the question they were
     * asked; showing them a red error banner for saying "no" would read as
     * something having gone wrong.
     */
    it("returns to idle without an error when the cost is declined", async () => {
        longGeminiFile();
        const { result } = await startLongRun();

        await act(async () => {
            result.current.declineCost();
        });
        await settle();

        expect(mocks.transcribeWithGemini).not.toHaveBeenCalled();
        expect(result.current.status).toBe("idle");
        expect(result.current.error).toBeNull();
        expect(result.current.isBusy).toBe(false);
        expect(result.current.pendingCost).toBeNull();
    });

    /**
     * THE STRANDED FRAME. `confirmCost` hands the engine a promise that only a
     * click resolves — so a run abandoned while the dialog is open would park
     * `transcribePreparedJob` on a promise nobody will ever settle, holding its
     * decoded audio for the life of the app. That is the same failure mode
     * `cancelPendingWait` exists to prevent, and it gets the same answer:
     * `claimRun` settles it.
     */
    it("settles a confirmation left open when another run takes the UI", async () => {
        longGeminiFile();
        const { result } = await startLongRun();
        expect(result.current.pendingCost).not.toBeNull();

        await act(async () => {
            result.current.cancel();
        });
        await settle();

        expect(result.current.pendingCost).toBeNull();
        expect(mocks.transcribeWithGemini).not.toHaveBeenCalled();
        expect(result.current.status).toBe("idle");
        expect(result.current.error).toBeNull();
    });
});
