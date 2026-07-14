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
import {
    api,
    Job,
    ModelStatusResponse,
    PersistTranscriptRequest,
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
     * worker is terminated, and nothing still in flight can touch the UI. The only
     * exit from a run that never terminates, since both entry points are locked
     * while busy. It does NOT cancel the backend — see `cancel` below.
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
        if (
            message.runId !== undefined &&
            message.runId !== runIdRef.current &&
            (message.status === "progress" ||
                message.status === "initiate" ||
                message.status === "done" ||
                message.status === "ready")
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

    const applyCompletedJob = useCallback((job: Job, modelLabel: string) => {
        setTranscript({
            isBusy: false,
            text: job.full_text || "",
            // The database stores RAW model segments as the source of truth, so a
            // formatter improvement retroactively improves old transcripts. They
            // must be consolidated on read, or the transcript at rest is one that
            // never went through the formatter.
            chunks: consolidateSegments(job.segments),
            filename: job.filename || undefined,
            persisted: true,
            modelLabel,
        });
        setJobId(job.id);
        setProgress(1);
        setStatus("completed");
        setError(null);
        setIsBusy(false);
    }, []);

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

    const persistWorkerTranscript = useCallback(
        async (
            job: Job,
            config: ResolvedModelConfig,
            workerTranscript: WorkerTranscript,
            audioDuration: number,
        ) => {
            setStatus("persisting");
            setIsBusy(true);
            const payload: PersistTranscriptRequest = {
                model_id: config.modelId,
                task: config.task,
                language: config.language,
                full_text: workerTranscript.text,
                segments: segmentsForPersistence(
                    workerTranscript,
                    audioDuration,
                ),
            };

            const persistedJob = await api.persistTranscript(job.id, payload);
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
                applyCompletedJob(initialJob, config.presetLabel);
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
                applyCompletedJob(readyJob, config.presetLabel);
                return;
            }

            setStatus("loading-audio");
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

            // The last gate before the transcript becomes PERMANENT. `persist`
            // writes it under `readyJob.id`, and the cache is content-keyed, so a
            // transcript persisted under the wrong job is the wrong transcript
            // returned for that file for ever after — a cache hit, no recompute,
            // no way for the user to tell. The message that produced
            // `workerTranscript` is already run-keyed, so it cannot be another
            // run's text; this checks the other direction, that THIS run is still
            // the one the user is looking at.
            if (runIdRef.current !== runId) {
                return;
            }

            await persistWorkerTranscript(
                readyJob,
                config,
                workerTranscript,
                audioBuffer.duration,
            );
        },
        [
            applyCompletedJob,
            cancelPendingWait,
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
     * WHAT THIS DOES NOT DO: cancel the BACKEND. There is no `cancel_job` command;
     * Rust owns the ffmpeg/yt-dlp child and nothing tells it to stop. So after a
     * cancel the download or the extraction runs to completion, still holding its
     * `download_semaphore` permit and its `track_download_start()` count — which
     * means `health()` and `queue_status` keep counting it, and the next YouTube
     * run can queue behind a job the user has already abandoned. It eventually
     * clears itself when the child exits. The UI is idle; the machine is not.
     */
    const cancel = useCallback(() => {
        claimRun();
        setTranscript((previous) =>
            previous ? { ...previous, isBusy: false } : previous,
        );
        setIsBusy(false);
        setIsModelLoading(false);
        setProgressItems([]);
        setProgress(0);
        setStatus("idle");
        setError(null);
    }, [claimRun]);

    const onInputChange = useCallback(() => {
        setTranscript(undefined);
        setError(null);
        setProgress(0);
        setStatus("idle");
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
        }),
        [
            browserCaps,
            cancel,
            capabilityLabel,
            effectivePresetLabel,
            error,
            isBusy,
            isModelLoading,
            jobId,
            language,
            modelStatus,
            modelStatusError,
            onInputChange,
            presetId,
            progress,
            progressItems,
            selectedModelAvailable,
            selectedModelId,
            startFromFile,
            startFromYouTube,
            status,
            task,
            transcript,
        ],
    );
}
