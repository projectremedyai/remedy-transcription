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
    const unsubscribeRef = useRef<(() => void) | null>(null);

    const worker = useWorker((event) => {
        const message = event.data;

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
                    filename: pendingWorkerRef.current?.filename,
                    persisted: false,
                    modelLabel: pendingWorkerRef.current?.modelLabel,
                });
                break;
            }
            case "complete": {
                const updateMessage = message as WorkerUpdateData;
                const nextTranscript: WorkerTranscript = {
                    text: updateMessage.data.text,
                    chunks: updateMessage.data.chunks,
                    words: updateMessage.data.words,
                };
                const displayTranscript = consolidateWorkerTranscript(
                    nextTranscript,
                    pendingWorkerRef.current?.audioDuration ?? null,
                );
                setTranscript({
                    isBusy: true,
                    text: displayTranscript.text,
                    chunks: displayTranscript.chunks,
                    filename: pendingWorkerRef.current?.filename,
                    persisted: false,
                    modelLabel: pendingWorkerRef.current?.modelLabel,
                });
                pendingWorkerRef.current?.resolve(nextTranscript);
                pendingWorkerRef.current = null;
                break;
            }
            case "error": {
                const messageText =
                    message.data?.message ?? "Transcription failed";
                pendingWorkerRef.current?.reject(new Error(messageText));
                pendingWorkerRef.current = null;
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
            if (unsubscribeRef.current) {
                unsubscribeRef.current();
            }
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
     * Tear down the wait already in flight — BOTH its listener and its poll.
     *
     * Every new run calls this UNCONDITIONALLY, before it can take any early
     * return. It used to sit below `waitForReady`'s terminal-status check, and
     * `transcribePreparedJob` skips `waitForReady` entirely for an
     * already-`completed` (cache-hit) job — so starting a cached YouTube run
     * while a local file was still extracting cancelled nothing. The first job's
     * poll then kept overwriting the cached transcript's status every 300 ms and,
     * on reaching `ready`, resolved its own wait and ran a whole spurious second
     * transcription on top of it.
     */
    const cancelPendingWait = useCallback(() => {
        unsubscribeRef.current?.();
        unsubscribeRef.current = null;
    }, []);

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

                const teardown = () => {
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
                    if (unsubscribeRef.current === teardown) {
                        unsubscribeRef.current = null;
                    }
                    settleWith();
                };

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

                unsubscribeRef.current = teardown;
            });
        },
        [cancelPendingWait, handleBackendJobUpdate],
    );

    const runWorkerTranscription = useCallback(
        (
            audioBuffer: AudioBuffer,
            config: ResolvedModelConfig,
            filename?: string,
        ) =>
            new Promise<WorkerTranscript>((resolve, reject) => {
                pendingWorkerRef.current = {
                    resolve,
                    reject,
                    filename,
                    modelLabel: config.presetLabel,
                    audioDuration: audioBuffer.duration,
                };
                setStatus("transcribing");
                setIsBusy(true);
                setError(null);

                worker.postMessage({
                    type: "transcribe",
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
    const beginRun = useCallback(async (): Promise<ResolvedModelConfig> => {
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
        setEffectivePresetLabel(config.presetLabel);
        return config;
    }, [ensureBrowserCaps, language, modelStatus, presetId, task]);

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
        async (initialJob: Job, config: ResolvedModelConfig) => {
            // FIRST, before any early return: this run supersedes whatever run was
            // in flight, so that run's listener and poll die here. A cache hit
            // returns below without ever reaching `waitForReady`, so this cannot
            // be left to `waitForReady` to do.
            cancelPendingWait();

            if (initialJob.status === "completed") {
                applyCompletedJob(initialJob, config.presetLabel);
                return;
            }

            handleBackendJobUpdate(initialJob);
            const readyJob = await waitForReady(initialJob);

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
            const workerTranscript = await runWorkerTranscription(
                audioBuffer,
                config,
                readyJob.filename || undefined,
            );
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
            try {
                const config = await beginRun();
                const job = await api.createFileJob({
                    path,
                    model_id: config.modelId,
                    task: config.task,
                    language: config.language,
                });
                setJobId(job.id);
                await transcribePreparedJob(job, config);
            } catch (nextError) {
                failRun(nextError, "Failed to transcribe file");
            }
        },
        [beginRun, failRun, transcribePreparedJob],
    );

    const startFromYouTube = useCallback(
        async (url: string) => {
            try {
                const config = await beginRun();
                const job = await api.createYouTubeJob({
                    url,
                    model_id: config.modelId,
                    task: config.task,
                    language: config.language,
                });
                setJobId(job.id);
                await transcribePreparedJob(job, config);
            } catch (nextError) {
                failRun(nextError, "YouTube preparation failed");
            }
        },
        [beginRun, failRun, transcribePreparedJob],
    );

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
