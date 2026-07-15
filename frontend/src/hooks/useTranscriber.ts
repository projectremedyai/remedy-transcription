import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
    BrowserCaps,
    LANGUAGE_OPTIONS,
    MODEL_PRESETS,
    ModelPresetId,
    ResolvedModelConfig,
    TaskMode,
    resolveModelConfig,
} from "../config/transcription";
import { useWorker } from "./useWorker";
import { detectBrowserCaps } from "../utils/detectBrowserCaps";
import {
    ConsolidatedSegment,
    WordToken,
    consolidateSegments,
} from "../lib/captionFormatter";
import {
    WorkerChunk,
    WorkerTranscript,
    consolidateWorkerTranscript,
    segmentsForPersistence,
} from "../lib/workerTranscript";
import type { SpeakerTurn } from "../lib/speakerAlignment";
import {
    api,
    DiarizationOutcome,
    Job,
    ModelStatusResponse,
    PersistTranscriptRequest,
    SpeakerNames,
} from "../services/api";

interface ProgressItem {
    file: string;
    loaded: number;
    progress: number;
    total: number;
    name: string;
    status: string;
}

interface WorkerUpdateData {
    data: {
        text: string;
        chunks: WorkerChunk[];
        /** Real per-word times. Only the final "complete" message carries them. */
        words?: WordToken[];
        tps?: number;
    };
}

interface WorkerProgressMessage {
    file: string;
    loaded: number;
    progress: number;
    total: number;
    name: string;
    status: string;
}

export interface TranscriberData {
    isBusy: boolean;
    text: string;
    /**
     * Display cues, already through the formatter. Branded so that neither the
     * exporters nor a second `consolidateSegments` call can be handed raw
     * segments (or these cues re-consolidated) without a compile error.
     */
    chunks: ConsolidatedSegment[];
    filename?: string;
    persisted: boolean;
    modelLabel?: string;
}

export interface Transcriber {
    onInputChange: () => void;
    isBusy: boolean;
    isModelLoading: boolean;
    progressItems: ProgressItem[];
    /**
     * Takes a filesystem PATH, not a browser `File`. A `File` has no path, so
     * Rust could never see it — which is why local files now come from the Tauri
     * dialog or a file drop.
     */
    start: (path: string) => void;
    startFromYouTube: (url: string) => void;
    /**
     * Abandon the run in flight and return the app to idle: the wait dies, the
     * worker is terminated, and the run token is bumped so that nothing still in
     * flight can PAINT the UI — every point at which a dead run can resume
     * re-checks the token before it writes, and every worker message carries the
     * id of the run that asked for it.
     *
     * It does not stop work already started. The backend's ffmpeg/yt-dlp runs to
     * completion (there is no `cancel_job` — see `cancel` below), and a
     * `persistTranscript` already in flight still lands, writing the transcript
     * under its own job: correct, and the content-keyed cache keeps it. Those
     * writes just cannot reach the screen.
     *
     * The only exit from a run that never terminates, since both entry points are
     * locked while busy.
     */
    cancel: () => void;
    output?: TranscriberData;
    jobId: string | null;
    error: string | null;
    progress: number;
    status: string;
    presetId: ModelPresetId;
    setPresetId: (presetId: ModelPresetId) => void;
    task: TaskMode;
    setTask: (task: TaskMode) => void;
    language: string;
    setLanguage: (language: string) => void;
    browserCaps: BrowserCaps | null;
    capabilityLabel: string;
    effectivePresetLabel: string | null;
    modelsReady: boolean;
    modelsStatusLoaded: boolean;
    modelsStatusError: string | null;
    missingModels: string[];
    selectedModelAvailable: boolean;
    selectedModelId: string | null;
    presetOptions: typeof MODEL_PRESETS;
    languageOptions: typeof LANGUAGE_OPTIONS;
    /**
     * Speaker detection is OPT-IN per job, default OFF: it costs real time (a
     * second CPU-bound child alongside Whisper) and most single-speaker content
     * has nothing for it to label. Read at the moment a run STARTS — flipping it
     * mid-run does not retroactively diarize or un-diarize the run in flight.
     */
    diarizeEnabled: boolean;
    setDiarizeEnabled: (enabled: boolean) => void;
    /**
     * "I know there are N speakers" — a HINT passed as `DiarizeOptions.num_speakers`,
     * not a guarantee: sherpa-onnx's own reference case shows asking for 4 can
     * still return 3. `undefined` means auto-detect, which is the default and is
     * NOT uniformly worse than a supplied count — do not coerce this to a number
     * before the user has actually set one.
     */
    numSpeakersHint: number | undefined;
    setNumSpeakersHint: (value: number | undefined) => void;
    /**
     * What the most recent `diarize_job` call answered, for THIS run —
     * `null` when the toggle was off, or no run has completed yet. The three
     * arms of {@link DiarizationOutcome} are handled, not collapsed: a
     * `"degraded"` run and a `"succeeded"` run that happened to find nobody both
     * end up with no speaker labels on screen, and this is the only thing that
     * lets a caller tell those apart and say so.
     */
    diarizationOutcome: DiarizationOutcome | null;
    /** Every speaker name set for the current job's source, keyed by its opaque label. */
    speakerNames: SpeakerNames;
    /**
     * Rename a speaker for the current job. A metadata write — it does not
     * re-transcribe or re-diarize — and `speakerNames` reflects it once the
     * write and the subsequent re-fetch both land.
     */
    renameSpeaker: (speakerKey: string, displayName: string) => Promise<void>;
}

type PendingWorker = {
    /** The run that posted the `transcribe` message this is waiting on. */
    runId: number;
    resolve: (value: WorkerTranscript) => void;
    reject: (reason?: unknown) => void;
    filename?: string;
    modelLabel?: string;
    audioDuration: number;
};

async function decodeAudio(arrayBuffer: ArrayBuffer): Promise<AudioBuffer> {
    const audioContext = new AudioContext({ sampleRate: 16000 });
    try {
        return await audioContext.decodeAudioData(arrayBuffer.slice(0));
    } finally {
        await audioContext.close();
    }
}

function mixToMono(audioBuffer: AudioBuffer): Float32Array {
    if (audioBuffer.numberOfChannels === 1) {
        return audioBuffer.getChannelData(0);
    }

    const left = audioBuffer.getChannelData(0);
    const right = audioBuffer.getChannelData(1);
    const mono = new Float32Array(audioBuffer.length);
    const scalingFactor = Math.sqrt(2);

    for (let index = 0; index < audioBuffer.length; index += 1) {
        mono[index] = (scalingFactor * (left[index] + right[index])) / 2;
    }

    return mono;
}

export function useTranscriber(): Transcriber {
    const [transcript, setTranscript] = useState<TranscriberData | undefined>(
        undefined,
    );
    const [isBusy, setIsBusy] = useState(false);
    const [isModelLoading, setIsModelLoading] = useState(false);
    const [progressItems, setProgressItems] = useState<ProgressItem[]>([]);
    const [jobId, setJobId] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState<string>("idle");
    const [browserCaps, setBrowserCaps] = useState<BrowserCaps | null>(null);
    const [presetId, setPresetId] = useState<ModelPresetId>("auto");
    const [task, setTask] = useState<TaskMode>("transcribe");
    const [language, setLanguage] = useState<string>("auto");
    const [effectivePresetLabel, setEffectivePresetLabel] = useState<
        string | null
    >(null);
    const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(
        null,
    );
    const [modelStatusError, setModelStatusError] = useState<string | null>(
        null,
    );
    const [diarizeEnabled, setDiarizeEnabled] = useState(false);
    const [numSpeakersHint, setNumSpeakersHint] = useState<number | undefined>(
        undefined,
    );
    const [diarizationOutcome, setDiarizationOutcome] =
        useState<DiarizationOutcome | null>(null);
    const [speakerNames, setSpeakerNames] = useState<SpeakerNames>({});

    const pendingWorkerRef = useRef<PendingWorker | null>(null);

    /**
     * The wait in flight, if any.
     *
     * `teardown` stops it listening and polling. `abandon` does that AND settles
     * its promise, so the `transcribePreparedJob` frame parked on it unwinds
     * instead of being stranded forever. Unmount uses `teardown` (there is no UI
     * left to unwind into); cancel and supersede use `abandon`.
     */
    const pendingWaitRef = useRef<{
        teardown: () => void;
        abandon: () => void;
    } | null>(null);

    /**
     * Which run owns the UI. Incremented by every start and by `cancel`, so it is
     * a monotonic token: a run holds a number, and the moment that number stops
     * being `runIdRef.current` the run is dead and may not touch anything.
     *
     * Two different things need this token, and only one of them was obvious.
     *
     * 1. CONTINUATIONS. `cancelPendingWait` can only kill a wait that has been
     *    ARMED. A run still inside `api.createFileJob` (a whole-file sha256) or
     *    `api.createYouTubeJob` (yt-dlp) has armed nothing, so there is nothing to
     *    tear down; when that call finally returns, the superseded run would carry
     *    on into a transcription unless it checks. Every resumption point checks.
     *
     * 2. WORKER MESSAGES. `pendingWorkerRef` used to be an unkeyed slot, and the
     *    guards on it asked "does ANYONE own this?" — never "does the run that
     *    POSTED this own it?". Those are different questions the moment two runs
     *    overlap, and the difference is a wrong transcript persisted under the
     *    wrong job. So the token is carried ON the worker message (see the
     *    worker's `WorkerRequest.runId`) and matched against the pending run here.
     */
    const runIdRef = useRef(0);

    const worker = useWorker((event) => {
        const message = event.data;

        /**
         * A message belongs to this run only if the pending run is the one that
         * posted it. Not "a run exists" — THIS run.
         *
         * The worker is terminated when a run is abandoned, so in practice a dead
         * run's messages usually never arrive. `terminate()` cannot recall a
         * message already in the queue, though, and this is what makes the
         * receiver correct when one slips through.
         */
        const ownedPending = (): PendingWorker | null => {
            const pending = pendingWorkerRef.current;
            if (!pending || pending.runId !== message.runId) {
                return null;
            }
            return pending;
        };

        // Model-loading traffic (`initiate`/`progress`/`done`/`ready`) is not tied
        // to a pending promise, so it is keyed against the run token directly. A
        // superseded run's download bar must not be painted over the live run's.
        //
        // FAIL CLOSED: this processes a message only if it is CURRENT. It used to
        // drop one only if it was STAMPED AND STALE (`runId !== undefined &&
        // runId !== current`), which is the weaker polarity — an UNSTAMPED message
        // sailed through. Nothing emits one today (the worker's `post()` stamps
        // every message from one place), but "no unstamped message exists" is a
        // property of a file this one does not own, and the cost of the strong
        // polarity is nil: an unstamped `runId` is `undefined`, which never equals
        // a number, so it is dropped.
        if (
            (message.status === "progress" ||
                message.status === "initiate" ||
                message.status === "done" ||
                message.status === "ready") &&
            message.runId !== runIdRef.current
        ) {
            return;
        }

        switch (message.status) {
            case "progress":
                setProgressItems((previous) =>
                    previous.map((item) =>
                        item.file === message.file
                            ? { ...item, progress: message.progress }
                            : item,
                    ),
                );
                break;
            case "initiate":
                setIsModelLoading(true);
                setProgressItems((previous) => [
                    ...previous,
                    message as WorkerProgressMessage,
                ]);
                break;
            case "done":
                setProgressItems((previous) =>
                    previous.filter((item) => item.file !== message.file),
                );
                break;
            case "ready":
                setIsModelLoading(false);
                break;
            case "update": {
                const pending = ownedPending();
                if (!pending) {
                    break;
                }
                const updateMessage = message as WorkerUpdateData;
                // Mid-stream the trailing chunk is still open, so its real end is
                // unknown — do NOT stretch it to the audio duration here.
                const displayTranscript = consolidateWorkerTranscript(
                    {
                        text: updateMessage.data.text,
                        chunks: updateMessage.data.chunks,
                    },
                    null,
                    { hideTrailingShortCaption: true },
                );
                setTranscript({
                    isBusy: true,
                    text: displayTranscript.text,
                    chunks: displayTranscript.chunks,
                    filename: pending.filename,
                    persisted: false,
                    modelLabel: pending.modelLabel,
                });
                break;
            }
            case "complete": {
                const pending = ownedPending();
                if (!pending) {
                    break;
                }
                const updateMessage = message as WorkerUpdateData;
                const nextTranscript: WorkerTranscript = {
                    text: updateMessage.data.text,
                    chunks: updateMessage.data.chunks,
                    words: updateMessage.data.words,
                };
                const displayTranscript = consolidateWorkerTranscript(
                    nextTranscript,
                    pending.audioDuration,
                );
                setTranscript({
                    isBusy: true,
                    text: displayTranscript.text,
                    chunks: displayTranscript.chunks,
                    filename: pending.filename,
                    persisted: false,
                    modelLabel: pending.modelLabel,
                });
                pendingWorkerRef.current = null;
                pending.resolve(nextTranscript);
                break;
            }
            case "error": {
                const pending = ownedPending();
                if (!pending) {
                    break;
                }
                const messageText =
                    message.data?.message ?? "Transcription failed";
                pendingWorkerRef.current = null;
                pending.reject(new Error(messageText));
                setIsBusy(false);
                setError(messageText);
                break;
            }
            default:
                break;
        }
    });

    useEffect(() => {
        detectBrowserCaps()
            .then(setBrowserCaps)
            .catch(() => {
                setBrowserCaps({
                    secureContext: window.isSecureContext,
                    canUseWebGPU: false,
                    shaderF16: false,
                    deviceMemoryGiB: null,
                    logicalCores: navigator.hardwareConcurrency ?? 4,
                });
            });
    }, []);

    useEffect(() => {
        api.getModelStatus()
            .then((status) => {
                setModelStatus(status);
                setModelStatusError(null);
            })
            .catch((statusError) => {
                setModelStatus(null);
                setModelStatusError(
                    statusError instanceof Error
                        ? statusError.message
                        : "Failed to get model status",
                );
            });
    }, []);

    useEffect(
        () => () => {
            pendingWaitRef.current?.teardown();
            pendingWaitRef.current = null;
        },
        [],
    );

    const ensureBrowserCaps = useCallback(async (): Promise<BrowserCaps> => {
        if (browserCaps) {
            return browserCaps;
        }
        const detected = await detectBrowserCaps();
        setBrowserCaps(detected);
        return detected;
    }, [browserCaps]);

    const selectedModelConfig = useMemo(() => {
        if (!browserCaps) {
            return null;
        }

        try {
            return resolveModelConfig(presetId, browserCaps, task, language);
        } catch {
            return null;
        }
    }, [browserCaps, language, presetId, task]);

    const selectedModelId = selectedModelConfig?.modelId ?? null;
    const selectedModelAvailable = useMemo(() => {
        if (!selectedModelId) {
            return true;
        }
        if (!modelStatus) {
            return false;
        }
        return modelStatus.items.some(
            (item) => item.model_id === selectedModelId && item.ready,
        );
    }, [modelStatus, selectedModelId]);

    /**
     * Every name given to the current job's speakers. Best-effort: a failed fetch
     * leaves `speakerNames` at whatever it already was (usually `{}`, the normal
     * "nobody has renamed anyone" case), which renders every speaker as its own
     * opaque label — a safe fallback, and not an error worth surfacing.
     */
    const refreshSpeakerNames = useCallback(async (targetJobId: string) => {
        try {
            const names = await api.getSpeakerNames(targetJobId);
            setSpeakerNames(names);
        } catch {
            // See above.
        }
    }, []);

    /**
     * `SPEAKER_00` -> `"Alice"`, for the current job. A metadata write —
     * `api.setSpeakerName` does not re-transcribe or re-diarize — so the only
     * thing this does afterward is re-fetch the map that write landed in, which
     * is what makes `speakerNames` reflect it.
     */
    const renameSpeaker = useCallback(
        async (speakerKey: string, displayName: string) => {
            if (!jobId) {
                return;
            }
            await api.setSpeakerName(jobId, speakerKey, displayName);
            await refreshSpeakerNames(jobId);
        },
        [jobId, refreshSpeakerNames],
    );

    /**
     * Kick off diarization for a job's already-prepared audio, if the toggle is
     * on, and resolve the turns to align against.
     *
     * Resolves `undefined` in three cases that a caller must NOT conflate:
     * the toggle was off (no `diarizationOutcome` write at all — nothing ran);
     * the sidecar answered `"succeeded"` with an empty turn list (a real
     * success — silence, or one speaker); or it answered `"degraded"` /
     * `"cancelled"`. `setDiarizationOutcome` is the one place that distinction
     * survives, which is what lets the UI show each of those three differently
     * rather than rendering all of them as the same unlabelled transcript.
     *
     * `api.diarizeJob` REJECTS only when the request itself was wrong — an
     * unknown job, no prepared audio (a cache hit whose WAV has aged out is the
     * realistic case), or an impossible `numSpeakers`. That is not one of the
     * three outcomes `DiarizationOutcome` models, but it is exactly as invisible
     * a failure if swallowed, so it is folded into `"degraded"` for display: the
     * user does not care whether the sidecar crashed or the request that reached
     * it never had a chance to, only that speaker labels did not happen and why.
     */
    const diarizeAudio = useCallback(
        async (targetJobId: string): Promise<SpeakerTurn[] | undefined> => {
            if (!diarizeEnabled) {
                return undefined;
            }
            try {
                const outcome = await api.diarizeJob(
                    targetJobId,
                    numSpeakersHint,
                );
                setDiarizationOutcome(outcome);
                return outcome.status === "succeeded"
                    ? outcome.turns
                    : undefined;
            } catch (diarizeError) {
                setDiarizationOutcome({
                    status: "degraded",
                    reason:
                        diarizeError instanceof Error
                            ? diarizeError.message
                            : "Speaker detection could not run",
                });
                return undefined;
            }
        },
        [diarizeEnabled, numSpeakersHint],
    );

    const applyCompletedJob = useCallback(
        (job: Job, modelLabel: string, turns?: readonly SpeakerTurn[]) => {
            setTranscript({
                isBusy: false,
                text: job.full_text || "",
                // The database stores RAW model segments as the source of truth, so a
                // formatter improvement retroactively improves old transcripts. They
                // must be consolidated on read, or the transcript at rest is one that
                // never went through the formatter.
                //
                // `turns` is only ever passed here for a job whose segments do NOT
                // already carry a `speaker` field (a fresh diarization of a cache hit
                // or a not-yet-persisted job) — a job persisted THROUGH this run
                // already has `speaker` baked into every row by
                // `segmentsForPersistence`, and passing `undefined` here is what
                // lets `consolidateSegments` fall through to its no-turns path and
                // read that embedded label straight off the rows, unchanged.
                chunks: consolidateSegments(job.segments, undefined, turns),
                filename: job.filename || undefined,
                persisted: true,
                modelLabel,
            });
            setJobId(job.id);
            setProgress(1);
            setStatus("completed");
            setError(null);
            setIsBusy(false);
            void refreshSpeakerNames(job.id);
        },
        [refreshSpeakerNames],
    );

    const handleBackendJobUpdate = useCallback((job: Job) => {
        setJobId(job.id);
        setProgress(job.progress);
        setStatus(job.status);
        if (job.error) {
            setError(job.error);
            setIsBusy(false);
        }
    }, []);

    /**
     * Abandon the wait already in flight: stop its listener, stop its poll, mark
     * it settled so nothing it has already asked for can still be believed, and
     * reject it so the run parked on it unwinds.
     *
     * Only ever called from `claimRun` (cancel and both `start*` entry points) and
     * as belt-and-braces at the top of `waitForReady`/`transcribePreparedJob`.
     */
    const cancelPendingWait = useCallback(() => {
        pendingWaitRef.current?.abandon();
        pendingWaitRef.current = null;
    }, []);

    /**
     * Claim the UI for a new run, killing whatever run held it — its wait, its
     * worker, and its promise.
     *
     * Synchronous by contract — every caller must invoke it before its first
     * `await`. Calling it "first, before any early return" inside
     * `transcribePreparedJob` is NOT early enough: that runs only after
     * `beginRun()` and `api.create*Job()` have both resolved, and `create_file_job`
     * sha256s the whole file while `create_youtube_job` shells out to yt-dlp.
     * Through that window the previous run's poll and listener would still be live.
     *
     * `worker.restart()` is the half that actually STOPS the abandoned run.
     * Ignoring its messages is not stopping it: transformers.js has no abort, so
     * without a terminate it keeps decoding — every core pinned for however many
     * minutes of audio are left, its `Float32Array` held — and a second
     * `postMessage` would merely start a concurrent handler sharing the same
     * pipeline instance (whose `dispose()` on a preset change would then throw
     * under the run still using it). Only the terminate ends it. It is done only
     * when a transcription is actually in flight, so the ordinary
     * one-file-then-the-next case keeps its warm, already-loaded model.
     *
     * Rejecting the abandoned promise is what frees its `AudioBuffer` and mono
     * `Float32Array`: `transcribePreparedJob` is parked on that promise, and a
     * promise that never settles strands the whole frame. The rejection is safe
     * BECAUSE the token has already been bumped — the dead run's `catch` in
     * `startFromFile`/`startFromYouTube` sees it no longer owns the UI and returns
     * without calling `failRun`, so no error is shown for a run the user
     * abandoned.
     *
     * THE INVARIANT THAT MAKES THE ORDERING WORK, since it is load-bearing and was
     * unwritten: this function is entirely SYNCHRONOUS. `cancelPendingWait()`
     * rejects before the bump lexically, and `worker.restart()`'s rejection also
     * runs before the `return` — but a rejection does not run its `catch` inline,
     * it QUEUES a microtask, and no microtask can run until this function (and the
     * event handler that called it) returns. So every abandoned continuation
     * observes the bumped token, whatever order the rejections appear in here. The
     * same invariant, read the other way, is why the token cannot move across an
     * `await` whose promise is resolved synchronously from a message handler —
     * which is what makes the pre-persist guard in `transcribePreparedJob` dead
     * and the post-persist one live. Introduce an `await` into this function and
     * both facts stop holding.
     */
    const claimRun = useCallback(() => {
        cancelPendingWait();
        // Bump BEFORE rejecting: the rejection runs the dead run's `catch`, and
        // that `catch` decides whether to show an error by comparing this token.
        runIdRef.current += 1;

        const abandoned = pendingWorkerRef.current;
        if (abandoned) {
            pendingWorkerRef.current = null;
            worker.restart();
            setIsModelLoading(false);
            setProgressItems([]);
            abandoned.reject(
                new Error("Transcription was cancelled or superseded"),
            );
        }

        return runIdRef.current;
    }, [cancelPendingWait, worker]);

    const waitForReady = useCallback(
        (job: Job) => {
            cancelPendingWait();

            if (
                job.status === "ready" ||
                job.status === "completed" ||
                job.status === "failed"
            ) {
                return Promise.resolve(job);
            }

            return new Promise<Job>((resolve, reject) => {
                let settled = false;
                let poll: ReturnType<typeof setInterval> | null = null;
                let unlistenEvents: (() => void) | null = null;

                /**
                 * `settled = true` is IN HERE, not only in `finish`, and that is
                 * load-bearing.
                 *
                 * `clearInterval` stops the NEXT poll; it cannot recall the
                 * `getJob` already in flight. `settled` is the only thing that can
                 * — which is precisely why it exists. When the teardown was
                 * reached via `finish` it was set; when it was reached via
                 * `cancelPendingWait` (cancel, or a supersede) it was NOT, so that
                 * in-flight `getJob`'s `.then(consider)` sailed through the guard
                 * and handed a DEAD job to `handleBackendJobUpdate` — writing its
                 * id, progress and status over the live run, and, if the dead job
                 * had failed, its `setError` + `setIsBusy(false)` too. It could
                 * even `finish(() => resolve(nextJob))` and settle a superseded
                 * run's wait.
                 *
                 * Idempotent: `finish` sets it and then calls this.
                 */
                const teardown = () => {
                    settled = true;
                    if (poll !== null) {
                        clearInterval(poll);
                        poll = null;
                    }
                    unlistenEvents?.();
                    unlistenEvents = null;
                };

                const finish = (settleWith: () => void) => {
                    if (settled) {
                        return;
                    }
                    settled = true;
                    teardown();
                    if (pendingWaitRef.current === handle) {
                        pendingWaitRef.current = null;
                    }
                    settleWith();
                };

                /**
                 * Settle a wait nobody is waiting for any more. The run parked on
                 * this promise is dead — its `catch` in `startFromFile` /
                 * `startFromYouTube` compares the run token, sees it no longer owns
                 * the UI, and returns without touching a thing. Leaving the promise
                 * pending instead would strand that frame for the life of the app.
                 */
                const abandon = () => {
                    finish(() =>
                        reject(
                            new Error(
                                "Transcription was cancelled or superseded",
                            ),
                        ),
                    );
                };

                const handle = { teardown, abandon };

                const consider = (nextJob: Job) => {
                    // A `getJob` already in flight when the wait settled still
                    // resolves. Without this guard it would push a stale status
                    // and progress back into the UI — flipping `loading-audio`
                    // back to `ready` and the progress bar back to 1.0 — after the
                    // run had already moved on.
                    if (settled) {
                        return;
                    }
                    handleBackendJobUpdate(nextJob);
                    if (
                        nextJob.status === "ready" ||
                        nextJob.status === "completed"
                    ) {
                        finish(() => resolve(nextJob));
                    } else if (nextJob.status === "failed") {
                        finish(() =>
                            reject(
                                new Error(
                                    nextJob.error || "Failed to prepare audio",
                                ),
                            ),
                        );
                    }
                };

                unlistenEvents = api.subscribeToProgress(
                    job.id,
                    consider,
                    (workerError) => finish(() => reject(workerError)),
                );

                // Tauri's `listen()` registers the listener ASYNCHRONOUSLY, so a job
                // that finishes fast can emit `ready` into the void before the
                // listener exists — and the UI would then wait forever for an event
                // that has already fired.
                //
                // The YouTube path never hit this only because downloading takes
                // seconds. ffmpeg over a short LOCAL file finishes in tens of
                // milliseconds, comfortably inside that window, so routing local
                // files through Rust is what exposes the race. Polling the job as a
                // backstop closes it for both sources: whichever of the event or the
                // poll first observes a terminal state wins, and `finish` makes that
                // idempotent.
                poll = setInterval(() => {
                    api.getJob(job.id)
                        .then(consider)
                        .catch(() => {
                            // Transient; the next tick (or the event) will catch it.
                        });
                }, 300);

                pendingWaitRef.current = handle;
            });
        },
        [cancelPendingWait, handleBackendJobUpdate],
    );

    const runWorkerTranscription = useCallback(
        (
            audioBuffer: AudioBuffer,
            config: ResolvedModelConfig,
            runId: number,
            filename?: string,
        ) =>
            new Promise<WorkerTranscript>((resolve, reject) => {
                pendingWorkerRef.current = {
                    runId,
                    resolve,
                    reject,
                    filename,
                    modelLabel: config.presetLabel,
                    audioDuration: audioBuffer.duration,
                };
                setStatus("transcribing");
                setIsBusy(true);
                setError(null);

                // `runId` travels with the audio and comes back on every message
                // the worker emits for it. The pending slot above holds the same
                // id, so the receiver can ask "is this MY run's output?" rather
                // than the question it used to ask, "is anyone waiting at all?".
                worker.postMessage({
                    type: "transcribe",
                    runId,
                    audio: mixToMono(audioBuffer),
                    modelId: config.modelId,
                    device: config.device,
                    task: config.task,
                    language:
                        config.language === "auto" ? null : config.language,
                });
            }),
        [worker],
    );

    /**
     * Write the transcript, then — and ONLY then — decide whether this run is
     * still allowed to paint it.
     *
     * `api.persistTranscript` is a Tauri IPC round-trip that writes one row per
     * segment: thousands of rows for a lecture, and slow. The app sits inside it
     * with `status: "persisting"` and `isBusy: true` — which is to say with the
     * Cancel button on screen — and NOTHING HOLDS IT. `claimRun` bumps the token,
     * abandons the wait and terminates the worker, but the worker is already idle
     * by now and this promise is not one of the ones it settles. So a cancel (or
     * a supersede, if the `isBusy` gate ever comes off) lands squarely inside this
     * await, the run dies, and the persist resolves into a UI it no longer owns.
     *
     * Unguarded, `applyCompletedJob` then wrote `setJobId`, `setTranscript`,
     * `setStatus("completed")` and `setIsBusy(false)` for that dead run: the user
     * is left looking at the transcript they cancelled, under the wrong job id
     * (which every jobId-keyed action downstream — export, the diarizer — then
     * targets), and under a LIVE second run the `setIsBusy(false)` takes the busy
     * panel and the Cancel button off the screen while Whisper is still grinding.
     *
     * THE PERSIST ITSELF STILL COMPLETES, deliberately. The transcript is written
     * under ITS OWN job, which is correct, and the cache is content-keyed, so the
     * work the user already paid for is kept and a re-run of that file is a cache
     * hit. Cancelling a run means "stop showing me this", not "throw the compute
     * away". What it may not do is repaint.
     *
     * Pinned by `useTranscriber.test.ts`: "does not repaint a finished run with a
     * dead run's persist" and "does not release the busy gate under a live run
     * when a dead run persists". Delete the guard below and both fail.
     *
     * `diarizationPromise` is where diarization's answer JOINS the transcript.
     * It was kicked off by the caller as soon as the canonical WAV existed —
     * concurrently with the Whisper transcription above, not after it, because
     * Whisper (webview) and sherpa-onnx (Rust sidecar) share no runtime and
     * diarizing a multi-minute file can take longer than transcribing it. By the
     * time control reaches here the worker's `complete` has already fired, so
     * this `await` is the one place both sides are known to have finished.
     */
    const persistWorkerTranscript = useCallback(
        async (
            job: Job,
            config: ResolvedModelConfig,
            workerTranscript: WorkerTranscript,
            audioDuration: number,
            runId: number,
            diarizationPromise: Promise<SpeakerTurn[] | undefined>,
        ) => {
            // Safe unguarded: the only caller reaches here in the microtask that
            // follows the worker's `complete`, and the token cannot move inside a
            // microtask (see `claimRun`).
            setStatus("persisting");
            setIsBusy(true);

            // The join. This DOES cross an await, so the token can move here —
            // unlike the two lines above, nothing after this point is safe
            // unguarded.
            const turns = await diarizationPromise;

            if (runIdRef.current === runId) {
                // Show the diarized transcript as soon as the turns are known,
                // rather than making the user wait for the persist round-trip
                // below (one row per segment — thousands for a lecture) just to
                // see who is speaking. This is the LIVE display path picking up
                // `turns`: same `consolidateWorkerTranscript` call the mid-stream
                // preview already used, now carrying the fourth argument that
                // used to have no production caller at all.
                const displayTranscript = consolidateWorkerTranscript(
                    workerTranscript,
                    audioDuration,
                    {},
                    turns,
                );
                setTranscript({
                    isBusy: true,
                    text: displayTranscript.text,
                    chunks: displayTranscript.chunks,
                    filename: job.filename || undefined,
                    persisted: false,
                    modelLabel: config.presetLabel,
                });
            }

            const payload: PersistTranscriptRequest = {
                model_id: config.modelId,
                task: config.task,
                language: config.language,
                full_text: workerTranscript.text,
                segments: segmentsForPersistence(
                    workerTranscript,
                    audioDuration,
                    turns,
                ),
            };

            const persistedJob = await api.persistTranscript(job.id, payload);

            // THE LAST GATE, and the last await in the run. Everything above this
            // line has already happened on disk; everything below it is UI.
            if (runIdRef.current !== runId) {
                return;
            }

            // No `turns` here: `persistedJob.segments` already carries `speaker`
            // baked into every row by `segmentsForPersistence` above, and
            // `applyCompletedJob` reads it straight off them. See the comment
            // there.
            applyCompletedJob(persistedJob, config.presetLabel);
        },
        [applyCompletedJob],
    );

    /**
     * The setup both sources share: reset the view, resolve which model this run
     * will actually use, and check it is available.
     */
    const beginRun = useCallback(
        async (runId: number): Promise<ResolvedModelConfig> => {
            setTranscript(undefined);
            setIsBusy(true);
            setError(null);
            setProgress(0);
            setStatus("checking-cache");
            // A fresh run's diarization has not happened yet, so a stale
            // outcome (or stale names) from the PREVIOUS job must not sit on
            // screen looking like an answer for this one.
            setDiarizationOutcome(null);
            setSpeakerNames({});

            const caps = await ensureBrowserCaps();
            const config = resolveModelConfig(presetId, caps, task, language);
            if (
                modelStatus &&
                !modelStatus.items.some(
                    (item) => item.model_id === config.modelId && item.ready,
                )
            ) {
                throw new Error(
                    `Model files for ${config.presetLabel} are not installed on the server`,
                );
            }
            // A continuation, like every other: `ensureBrowserCaps` can await a
            // GPU probe, and a run superseded during it must not relabel the UI
            // with the model IT would have used. (`setBrowserCaps` inside
            // `ensureBrowserCaps` is deliberately not guarded — the device's
            // capabilities are a property of the machine, identical for every run,
            // not state this run owns.)
            if (runIdRef.current !== runId) {
                return config;
            }
            setEffectivePresetLabel(config.presetLabel);
            return config;
        },
        [ensureBrowserCaps, language, modelStatus, presetId, task],
    );

    const failRun = useCallback((nextError: unknown, fallback: string) => {
        setTranscript((previous) =>
            previous ? { ...previous, isBusy: false } : previous,
        );
        setError(nextError instanceof Error ? nextError.message : fallback);
        setStatus("failed");
        setIsBusy(false);
    }, []);

    /**
     * Everything after a job exists — and it is now the SAME for every source.
     *
     * Wait for Rust to report `ready` (it has produced the canonical 16 kHz mono
     * WAV), fetch that WAV back through the asset protocol, decode it, transcribe
     * it, persist it.
     *
     * Local files used to take a different route entirely: the webview read the
     * browser `File` into an `ArrayBuffer` and decoded THAT, so no WAV ever
     * reached the disk and Rust never saw the audio — which left a Rust-side
     * diarizer with nothing to work from for exactly the sources users are most
     * likely to want diarized. Both sources now run this one already-proven path.
     */
    const transcribePreparedJob = useCallback(
        async (initialJob: Job, config: ResolvedModelConfig, runId: number) => {
            // Belt and braces. The call that actually closes the window is the one
            // in the `start*` entry point, which ran before `api.create*Job` was
            // even awaited; by here that job creation has already had all the time
            // it wanted to let a stale poll tick. See `cancelPendingWait`.
            cancelPendingWait();

            if (initialJob.status === "completed") {
                // A cache hit — the transcript already exists, but this run's
                // diarization toggle has not been honoured yet. This IS the
                // "legacy transcript" path a not-yet-word-granular cached row
                // reaches: `diarizeAudio` returns turns (or doesn't) the same
                // way regardless of granularity, and `consolidateSegments`
                // inside `applyCompletedJob` decides word- vs segment-level
                // alignment from the segments themselves.
                //
                // NOTE what this does NOT do: persist the turns. There is no
                // command to write a new diarization run's speakers onto an
                // already-persisted job's rows, so a cache hit's labels are
                // shown for this session only, from `job.segments` as they
                // already are. See the report for this task.
                const turns = await diarizeAudio(initialJob.id);
                if (runIdRef.current !== runId) {
                    return;
                }
                applyCompletedJob(initialJob, config.presetLabel, turns);
                return;
            }

            handleBackendJobUpdate(initialJob);
            const readyJob = await waitForReady(initialJob);

            // FIRST, above every branch. An abandoned wait now rejects rather than
            // hanging, so this is normally unreachable — but "normally" is what the
            // last three rounds of this bug were built on. The `completed` branch
            // below calls `applyCompletedJob`, which writes a transcript, a job id
            // and `isBusy: false` straight into the UI; a dead run must not reach
            // it, and it used to sit ABOVE the only check that could stop it.
            if (runIdRef.current !== runId) {
                return;
            }

            if (readyJob.status === "completed") {
                // Same cache-hit case as above, reached via the polling path
                // instead of the immediate one.
                const turns = await diarizeAudio(readyJob.id);
                if (runIdRef.current !== runId) {
                    return;
                }
                applyCompletedJob(readyJob, config.presetLabel, turns);
                return;
            }

            setStatus("loading-audio");

            // Kicked off HERE — as soon as the canonical WAV exists — and run
            // CONCURRENTLY with the whisper transcription below, not awaited
            // until `persistWorkerTranscript` needs the answer. Whisper (webview)
            // and sherpa-onnx (Rust sidecar) share no runtime, so there is
            // nothing to gain from serializing them: diarizing a multi-minute
            // file can take longer than transcribing it.
            const diarizationPromise = diarizeAudio(readyJob.id);

            const audioUrl = await api.getAudioUrl(readyJob.id);
            const audioResponse = await fetch(audioUrl);
            if (!audioResponse.ok) {
                throw new Error("Failed to load prepared audio");
            }

            // The WAV is 16 kHz mono and `decodeAudio` opens its `AudioContext` at
            // exactly 16 kHz, so nothing is resampled and `audioBuffer.duration` is
            // the true duration of the source — the same number the in-browser
            // decode of the original file used to yield.
            //
            // That number is load-bearing: `persistWorkerTranscript` passes it to
            // `segmentsForPersistence`, which uses it to CLOSE the final segment. A
            // missing or wrong duration silently swallows the end of the transcript
            // rather than throwing, so it must keep coming from a real decode of
            // the real audio.
            const audioBuffer = await decodeAudio(
                await audioResponse.arrayBuffer(),
            );

            if (runIdRef.current !== runId) {
                return;
            }

            const workerTranscript = await runWorkerTranscription(
                audioBuffer,
                config,
                runId,
                readyJob.filename || undefined,
            );

            // NO GUARD HERE, deliberately, and this is the fourth round of that
            // decision so it is worth writing down. One used to sit on this line,
            // commented "the last gate before the transcript becomes PERMANENT".
            // It was PROVABLY DEAD: `pending.resolve()` is called synchronously
            // inside the worker's message handler, so the continuation of the
            // `await` above is a microtask, microtasks drain before any task, and
            // no click can bump the token in that window. The condition could
            // never be false. Worse, it LOOKED like protection — and that is what
            // hid the live hole ten lines below it, inside `persistWorkerTranscript`,
            // for three rounds. The real gate is there, after the persist's own
            // await, which is the one place a cancel can actually land.
            //
            // If a future change makes the resolve asynchronous (a `setTimeout`,
            // a `queueMicrotask` chain that yields, an `await` before the resolve
            // in the handler), this line becomes reachable and a guard belongs
            // here again.
            await persistWorkerTranscript(
                readyJob,
                config,
                workerTranscript,
                audioBuffer.duration,
                runId,
                diarizationPromise,
            );
        },
        [
            applyCompletedJob,
            cancelPendingWait,
            diarizeAudio,
            handleBackendJobUpdate,
            persistWorkerTranscript,
            runWorkerTranscription,
            waitForReady,
        ],
    );

    const startFromFile = useCallback(
        async (path: string) => {
            // Synchronously, before ANY await: this run takes the UI, and the
            // previous run's listener and poll die now — not after `beginRun` and
            // a full-file sha256 have had their say. See `cancelPendingWait`.
            const runId = claimRun();
            try {
                const config = await beginRun(runId);
                const job = await api.createFileJob({
                    path,
                    model_id: config.modelId,
                    task: config.task,
                    language: config.language,
                });
                if (runIdRef.current !== runId) {
                    return;
                }
                setJobId(job.id);
                await transcribePreparedJob(job, config, runId);
            } catch (nextError) {
                if (runIdRef.current !== runId) {
                    return;
                }
                failRun(nextError, "Failed to transcribe file");
            }
        },
        [beginRun, claimRun, failRun, transcribePreparedJob],
    );

    const startFromYouTube = useCallback(
        async (url: string) => {
            const runId = claimRun();
            try {
                const config = await beginRun(runId);
                const job = await api.createYouTubeJob({
                    url,
                    model_id: config.modelId,
                    task: config.task,
                    language: config.language,
                });
                if (runIdRef.current !== runId) {
                    return;
                }
                setJobId(job.id);
                await transcribePreparedJob(job, config, runId);
            } catch (nextError) {
                if (runIdRef.current !== runId) {
                    return;
                }
                failRun(nextError, "YouTube preparation failed");
            }
        },
        [beginRun, claimRun, failRun, transcribePreparedJob],
    );

    /**
     * The escape hatch. Both entry points are gated on `isBusy` and the
     * `Transcribe` button is hidden while busy, so a run that never terminates —
     * a job stranded in `Extracting` with no event coming, say — used to leave
     * quitting the app as the only way out.
     *
     * `claimRun` does the work: it abandons the wait, TERMINATES the worker (which
     * is what actually stops a Whisper inference — see `useWorker.restart`),
     * settles the abandoned promises so their frames unwind, and bumps the run
     * token so anything still in flight lands on a run that no longer owns the UI.
     *
     * WHAT THE TOKEN BUYS, precisely — a dead run may still FINISH, but it may not
     * PAINT. Every point at which one can resume re-checks the token before it
     * writes to the UI: after `beginRun`, after `create*Job`, after `waitForReady`,
     * after `decodeAudio`, and — the one that was missing for three rounds — after
     * `api.persistTranscript`. Worker messages are keyed the same way, by the run
     * that posted them. Driven directly by the tests in "under an overlap the
     * worker cannot stop".
     *
     * Two things are deliberately NOT guarded, and both are correct: the persist
     * still writes (below), and `setBrowserCaps` inside `ensureBrowserCaps` writes
     * unconditionally, because the machine's GPU is a property of the machine and
     * is the same answer for every run.
     *
     * WHAT THIS DOES NOT DO: stop the ffmpeg/yt-dlp child. There is no
     * `cancel_job` command; Rust owns those and nothing tells them to stop. So
     * after a cancel the download or the extraction runs to completion, still
     * holding its `download_semaphore` permit and its `track_download_start()`
     * count — which means `health()` and `queue_status` keep counting it, and the
     * next YouTube run can queue behind a job the user has already abandoned. It
     * eventually clears itself when the child exits. The UI is idle; the machine
     * is not. That is tolerable ONLY because an abandoned ffmpeg finishes on its
     * own, in seconds to a minute.
     *
     * The diarization sidecar is NOT like that, which is why it is the one child
     * this does kill: it is a CPU-bound ONNX process with a 30-minute backstop
     * timeout, so leaving it to "finish on its own" means pinning a core for up to
     * half an hour behind an idle-looking app. `cancel_diarization` is a no-op for
     * a job that is not diarizing, and it is fire-and-forget: a cancel that cannot
     * reach the backend must still clear the UI.
     *
     * Nor does it stop a `persistTranscript` already in flight — and cancelling
     * inside that window is exactly how the last bug was reached, because the app
     * is `isBusy` while persisting, so the Cancel button is on screen. The persist
     * is allowed to finish (it writes under its OWN job; the compute is kept and
     * the content-keyed cache will hit on a re-run) and is stopped only from
     * repainting. See `persistWorkerTranscript`.
     */
    const cancel = useCallback(() => {
        claimRun();
        if (jobId) {
            void api.cancelDiarization(jobId).catch(() => {
                // Best effort. The UI is being torn down either way, and there is
                // nothing a user could do with "the cancel request failed".
            });
        }
        setTranscript((previous) =>
            previous ? { ...previous, isBusy: false } : previous,
        );
        setIsBusy(false);
        setIsModelLoading(false);
        setProgressItems([]);
        setProgress(0);
        setStatus("idle");
        setError(null);
    }, [claimRun, jobId]);

    const onInputChange = useCallback(() => {
        setTranscript(undefined);
        setError(null);
        setProgress(0);
        setStatus("idle");
        setDiarizationOutcome(null);
        setSpeakerNames({});
    }, []);

    const capabilityLabel = useMemo(() => {
        if (!browserCaps) {
            return "Detecting browser capabilities...";
        }
        return browserCaps.canUseWebGPU
            ? "WebGPU available"
            : "CPU / WASM fallback";
    }, [browserCaps]);

    return useMemo(
        () => ({
            onInputChange,
            isBusy,
            isModelLoading,
            progressItems,
            start: startFromFile,
            startFromYouTube,
            cancel,
            output: transcript,
            jobId,
            error,
            progress,
            status,
            presetId,
            setPresetId,
            task,
            setTask,
            language,
            setLanguage,
            browserCaps,
            capabilityLabel,
            effectivePresetLabel,
            modelsReady: modelStatus?.models_ready ?? false,
            modelsStatusLoaded:
                modelStatus !== null || modelStatusError !== null,
            modelsStatusError: modelStatusError,
            missingModels: modelStatus?.missing_models ?? [],
            selectedModelAvailable,
            selectedModelId,
            presetOptions: MODEL_PRESETS,
            languageOptions: LANGUAGE_OPTIONS,
            diarizeEnabled,
            setDiarizeEnabled,
            numSpeakersHint,
            setNumSpeakersHint,
            diarizationOutcome,
            speakerNames,
            renameSpeaker,
        }),
        [
            browserCaps,
            cancel,
            capabilityLabel,
            diarizeEnabled,
            diarizationOutcome,
            effectivePresetLabel,
            error,
            isBusy,
            isModelLoading,
            jobId,
            language,
            modelStatus,
            modelStatusError,
            numSpeakersHint,
            onInputChange,
            presetId,
            progress,
            progressItems,
            renameSpeaker,
            selectedModelAvailable,
            selectedModelId,
            speakerNames,
            startFromFile,
            startFromYouTube,
            status,
            task,
            transcript,
        ],
    );
}
