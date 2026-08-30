import React, {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";

import Modal from "./modal/Modal";
import { UrlInput } from "./modal/UrlInput";
import { Transcriber } from "../hooks/useTranscriber";
import Progress from "./Progress";
import GeminiKeyDialog from "./GeminiKeyDialog";
import { ENGINES, EngineId, engineById } from "../config/engines";
import { api } from "../services/api";

/**
 * What ffmpeg is willing to be pointed at. Used for BOTH the file picker's filter
 * and the drop target's check, so the two entry points cannot disagree about what
 * counts as media.
 */
const MEDIA_EXTENSIONS = [
    "mp3",
    "wav",
    "m4a",
    "aac",
    "ogg",
    "opus",
    "flac",
    "wma",
    "mp4",
    "mkv",
    "avi",
    "mov",
    "webm",
    "m4v",
    "mpeg",
    "mpg",
];

/**
 * The statuses THIS APP sets for itself, and their prose.
 *
 * Two jobs, deliberately one table. It maps a machine string to a sentence,
 * and — by what it does NOT contain — it is also the test for whether a status
 * came from an ENGINE instead. `EngineProgress.status` is declared "user-facing
 * and already specific ... shown verbatim", and `useTranscriber` stores it
 * straight onto `status`, so anything absent here is already a sentence and
 * both `getStatusMessage` and `progressValue` treat it as one.
 *
 * That join is what was missing. This used to be a `switch` with
 * `default: return ""`, written when these were the only statuses that
 * existed, and never revisited when the engine seam arrived — so a multi-chunk
 * Gemini run rendered a spinner next to blank text for its entire duration,
 * and a single-chunk one matched `transcribing` and claimed the work was
 * happening "in your browser". The local engine only ever worked because it
 * happened to pick `loading-audio`, a string that was already here.
 *
 * The two members that are not backend job statuses: `loading-audio` comes
 * from `localEngine.run()` (it predates the seam and is kept because its copy
 * is specific to the local path), and `transcribing` is set by the hook's own
 * `runWorkerTranscription`. Both are local-only by construction; the Gemini
 * engine's labels are prose and cannot collide with either.
 */
const JOB_STATUS_MESSAGES: Record<string, string> = {
    // Not busy — nothing to say, and the block this feeds is not rendered.
    idle: "",
    "checking-cache": "Checking transcript cache...",
    downloading: "Downloading from YouTube...",
    extracting: "Extracting audio...",
    // The backend's own job-status string, on screen for the window between
    // prepared audio going "ready" and the engine's first progress event. The
    // local engine closes that window itself (its `run()` posts "loading-audio"
    // before its first `await`), but Gemini's first event waits on a Rust-side
    // silence-detection pass, so without this the spinner would sit next to
    // blank text for a couple of seconds.
    ready: "Preparing transcription...",
    "loading-audio": "Loading prepared audio...",
    transcribing: "Transcribing in your browser...",
    persisting: "Saving transcript...",
    completed: "Completed",
    failed: "Failed",
};

/**
 * The prose for one of this app's own statuses, or `undefined` if it is not
 * one — which is the same question as "did an engine supply this?".
 *
 * `hasOwnProperty` rather than a bare lookup, so the answer is the table's and
 * not `Object.prototype`'s: a plain `status in JOB_STATUS_MESSAGES` says yes
 * to `toString`, and a plain `JOB_STATUS_MESSAGES[status]` hands back a
 * function for it. Both callers below have to agree on this predicate exactly,
 * or the status text and the progress bar can disagree about which run they
 * are describing.
 */
function jobStatusMessage(status: string): string | undefined {
    return Object.prototype.hasOwnProperty.call(JOB_STATUS_MESSAGES, status)
        ? JOB_STATUS_MESSAGES[status]
        : undefined;
}

function basename(path: string): string {
    const parts = path.split(/[\\/]/);
    return parts[parts.length - 1] || path;
}

function hasMediaExtension(path: string): boolean {
    const extension = basename(path).split(".").pop()?.toLowerCase() ?? "";
    return MEDIA_EXTENSIONS.includes(extension);
}

export function AudioManager(props: { transcriber: Transcriber }) {
    // A PATH, not a browser `File`. A `File` has no path, so Rust could never
    // run ffmpeg over it — which is why local files now arrive from the Tauri
    // dialog or a file drop.
    const [selectedPath, setSelectedPath] = useState<string | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [dropError, setDropError] = useState<string | null>(null);
    const [dropNotice, setDropNotice] = useState<string | null>(null);
    const [geminiKeyConfigured, setGeminiKeyConfigured] = useState(false);
    const [keyDialogOpen, setKeyDialogOpen] = useState(false);

    useEffect(() => {
        let live = true;
        api.geminiKeyStatus()
            .then((configured) => {
                if (live) setGeminiKeyConfigured(configured);
            })
            .catch(() => {
                // A keychain that will not answer reads as "no key", which
                // routes the user to Add-key rather than into a doomed run.
            });
        return () => {
            live = false;
        };
    }, []);

    const { transcriber } = props;
    const { onInputChange } = transcriber;
    const isBusy = transcriber.isBusy;

    const engine = engineById(transcriber.engine);
    /** A cloud engine with no credential cannot start. Say so before the run. */
    const engineReady = !engine.requiresKey || geminiKeyConfigured;

    // The drop listener reads `isBusy`, but it must not DEPEND on it: Tauri's
    // `onDragDropEvent()` resolves asynchronously, so re-running the effect on
    // every busy flip unlistens immediately and re-listens a tick later, leaving
    // a window in which a drop lands on no listener at all. A ref gives the
    // listener the current value without re-registering it.
    const isBusyRef = useRef(isBusy);
    useEffect(() => {
        isBusyRef.current = isBusy;
    }, [isBusy]);

    const handleFileSelect = useCallback(
        (path: string) => {
            onInputChange();
            setDropError(null);
            setDropNotice(null);
            setSelectedPath(path);
        },
        [onInputChange],
    );

    // Dropping a file onto the window. Tauri intercepts the webview's native
    // drag-and-drop and re-emits it here WITH the real filesystem paths, which is
    // exactly the currency this app now runs on.
    useEffect(() => {
        let unlisten: (() => void) | undefined;
        let cancelled = false;

        getCurrentWebview()
            .onDragDropEvent((event) => {
                // The union is `enter | over | drop | leave`. Only `over` was
                // handled, so the drop zone stayed un-highlighted until the
                // pointer MOVED inside the window — a file dragged in and released
                // without moving showed no affordance at all.
                if (
                    event.payload.type === "enter" ||
                    event.payload.type === "over"
                ) {
                    setIsDragging(true);
                    return;
                }
                if (event.payload.type === "leave") {
                    setIsDragging(false);
                    return;
                }
                if (event.payload.type === "drop") {
                    setIsDragging(false);

                    // A run is already in flight and there is no queue, so taking
                    // the drop would silently supersede it.
                    if (isBusyRef.current) {
                        setDropNotice(null);
                        setDropError(
                            "A transcription is already running. Wait for it to finish, then drop the file again.",
                        );
                        return;
                    }

                    const dropped = event.payload.paths;
                    const media = dropped.filter(hasMediaExtension);
                    if (media.length === 0) {
                        setDropNotice(null);
                        setDropError(
                            "That is not an audio or video file we can read.",
                        );
                        return;
                    }

                    // One transcript at a time is all this app models, so a
                    // multi-file drop can only take the first — but it must SAY so
                    // rather than quietly discard the rest.
                    handleFileSelect(media[0]);
                    if (dropped.length > 1) {
                        setDropNotice(
                            `Only one file at a time — using ${basename(
                                media[0],
                            )} and ignoring the other ${dropped.length - 1}.`,
                        );
                    }
                }
            })
            .then((fn) => {
                if (cancelled) {
                    fn();
                } else {
                    unlisten = fn;
                }
            })
            .catch(() => {
                // Drag-and-drop is a convenience; the picker still works without it.
            });

        return () => {
            cancelled = true;
            unlisten?.();
        };
    }, [handleFileSelect]);

    const handleYouTubeSubmit = (url: string) => {
        transcriber.onInputChange();
        setSelectedPath(null);
        setDropError(null);
        setDropNotice(null);
        transcriber.startFromYouTube(url);
    };

    const handleTranscribe = () => {
        if (selectedPath) {
            // Clear the drop messages FIRST. `dropError` outranks
            // `transcriber.error` (it is normally the newer of the two), and it is
            // set by things the hook never hears about — a rejected `.txt` drop,
            // say. Left standing, it would sit on top of this run's real failure,
            // or under its transcript, indefinitely.
            setDropError(null);
            setDropNotice(null);
            transcriber.start(selectedPath);
        }
    };

    const handleCancel = () => {
        setDropError(null);
        setDropNotice(null);
        transcriber.cancel();
    };

    const getStatusMessage = () => {
        // Not one of ours, so it came from an engine's `EngineProgress`, which
        // `hooks/engines/types.ts` declares is "user-facing and already
        // specific -- shown verbatim". Shown verbatim.
        return (
            jobStatusMessage(props.transcriber.status) ??
            props.transcriber.status
        );
    };

    const progressValue = useMemo(() => {
        if (
            props.transcriber.status === "downloading" ||
            props.transcriber.status === "extracting" ||
            // Same join as `getStatusMessage`, on the other half of the
            // progress pair: an engine-supplied status means the fraction next
            // to it is the ENGINE's, and it is the only determinate number a
            // cloud run produces. Gemini has no token stream, so the spec's
            // "the transcript appears all at once, behind a determinate
            // progress bar" is this bar or nothing.
            jobStatusMessage(props.transcriber.status) === undefined
        ) {
            return props.transcriber.progress * 100;
        }
        if (
            props.transcriber.isModelLoading &&
            props.transcriber.progressItems.length > 0
        ) {
            const total = props.transcriber.progressItems.reduce(
                (sum, item) => sum + item.progress,
                0,
            );
            return total / props.transcriber.progressItems.length;
        }
        return 0;
    }, [
        props.transcriber.isModelLoading,
        props.transcriber.progress,
        props.transcriber.progressItems,
        props.transcriber.status,
    ]);

    /**
     * Why the YouTube modal's submit is dead, when it is.
     *
     * The Transcribe button's own gate has its explanation sitting right next
     * to it in the engine row. The modal's does not -- the engine row is
     * BEHIND the overlay -- so a user with Gemini selected and no key could
     * paste a valid URL and find "Prepare Audio" inert with nothing on screen
     * saying why. `null` means the control is live.
     */
    const youtubeDisabledReason = !engineReady
        ? "Gemini needs an API key before it can prepare audio. Close this and add one in the engine settings behind this dialog."
        : null;

    return (
        <div className='w-full'>
            <div
                className={`flex flex-col justify-center items-center rounded-lg bg-white shadow-xl shadow-black/5 ring-1 transition-all duration-150 ${
                    isDragging
                        ? "ring-2 ring-indigo-500 bg-indigo-50"
                        : "ring-slate-700/10"
                }`}
            >
                {/*
                 * Both entry points are gated on `isBusy`.
                 *
                 * WHAT THE GATE IS NOT: it is no longer the thing that keeps an
                 * overlap safe. It used to be — remove it before the current fix
                 * and a second run could be started over a live one, whose worker
                 * kept running, whose `complete` then resolved the NEW run's
                 * promise with the OLD run's transcript, which was persisted under
                 * the new run's job and content-cached forever. That is now closed
                 * in the hook and pinned by tests that drive the overlap directly,
                 * with this gate out of the picture (`useTranscriber.test.ts`,
                 * "under an overlap the worker cannot stop" — all three fail against
                 * the pre-fix hook): a new run terminates the previous worker, and
                 * every worker message carries the id of the run that asked for it,
                 * so a message that outlives its run is dropped rather than
                 * misattributed.
                 *
                 * WHAT STILL DEPENDS ON IT, and would need answering before this
                 * gate comes off for a queue or a second panel:
                 *   - The app models exactly ONE transcript — one `jobId`, one
                 *     `output`, one `status`. A second run cannot be SHOWN, only
                 *     substituted, so without the gate a user can silently lose the
                 *     run they were watching. This is the UX reason it exists.
                 *   - `cancel()` / supersede do NOT stop the backend's ffmpeg or
                 *     yt-dlp (there is no `cancel_job` command). The abandoned job's
                 *     child runs to completion holding a `download_semaphore` permit
                 *     and an active-download count, so without the gate a user can
                 *     stack up abandoned backend work that later runs queue behind.
                 *   - A finished run sits in `persisting` while `persistTranscript`
                 *     writes one row per segment over IPC — thousands of them for a
                 *     lecture, and slow. With the gate ON, only Cancel can start a
                 *     second run inside that window; with it OFF, a plain supersede
                 *     does, with no Cancel needed. The UI half of that race is
                 *     closed in the hook — the persist re-checks the run token
                 *     after its own await and will not repaint a UI it no longer
                 *     owns (`useTranscriber.test.ts`: "does not repaint a finished
                 *     run with a dead run's persist" and "does not release the busy
                 *     gate under a live run when a dead run persists", both of which
                 *     drive the overlap with this gate out of the picture) — but the
                 *     WRITE still happens by design, so without the gate a supersede
                 *     stacks DB writes the same way it stacks abandoned ffmpeg work.
                 */}
                <div className='flex flex-row space-x-2 py-2 w-full px-2'>
                    <YouTubeTile
                        icon={<YouTubeIcon />}
                        text='YouTube'
                        onUrlSubmit={handleYouTubeSubmit}
                        enabled={
                            props.transcriber.selectedModelAvailable &&
                            engineReady &&
                            !isBusy
                        }
                        disabledReason={youtubeDisabledReason}
                        disabled={isBusy}
                    />
                    <VerticalBar />
                    <FileTile
                        icon={<FolderIcon />}
                        text='From file'
                        onFileSelect={handleFileSelect}
                        disabled={isBusy}
                    />
                </div>
                {isDragging && (
                    <div className='w-full px-4 pb-2 text-center text-sm font-medium text-indigo-600'>
                        Drop an audio or video file to transcribe it
                    </div>
                )}
                <div className='w-full px-4 pb-4 flex flex-col gap-3'>
                    <fieldset className='flex flex-col gap-2 border-b border-slate-100 pb-3'>
                        <legend className='sr-only'>
                            Transcription engine
                        </legend>
                        {ENGINES.map((candidate) => (
                            <label
                                key={candidate.id}
                                className='flex items-start gap-2'
                            >
                                <input
                                    type='radio'
                                    name='engine'
                                    value={candidate.id}
                                    checked={
                                        transcriber.engine === candidate.id
                                    }
                                    disabled={transcriber.isBusy}
                                    onChange={() =>
                                        transcriber.setEngine(
                                            candidate.id as EngineId,
                                        )
                                    }
                                    className='mt-1 border-slate-300 text-indigo-600 focus:ring-indigo-500'
                                />
                                <span className='flex flex-col'>
                                    <span className='text-slate-700'>
                                        {candidate.label}
                                    </span>
                                    <span className='text-xs text-slate-500'>
                                        {candidate.description}
                                    </span>
                                </span>
                            </label>
                        ))}

                        {engine.requiresKey && (
                            <div className='flex items-center gap-3 pl-6 text-xs'>
                                {geminiKeyConfigured ? (
                                    <>
                                        <span className='text-emerald-700'>
                                            Key saved
                                        </span>
                                        <button
                                            type='button'
                                            onClick={() =>
                                                setKeyDialogOpen(true)
                                            }
                                            className='text-slate-500 underline underline-offset-2 hover:text-indigo-600'
                                        >
                                            Change
                                        </button>
                                    </>
                                ) : (
                                    <>
                                        <span className='text-amber-700'>
                                            Gemini needs an API key
                                        </span>
                                        <button
                                            type='button'
                                            onClick={() =>
                                                setKeyDialogOpen(true)
                                            }
                                            className='text-indigo-600 underline underline-offset-2'
                                        >
                                            Add key
                                        </button>
                                    </>
                                )}
                            </div>
                        )}
                    </fieldset>

                    <GeminiKeyDialog
                        show={keyDialogOpen}
                        configured={geminiKeyConfigured}
                        onClose={() => setKeyDialogOpen(false)}
                        onSaved={setGeminiKeyConfigured}
                    />

                    <SettingsPanel transcriber={props.transcriber} />
                </div>
                {!props.transcriber.modelsStatusLoaded && (
                    <div className='w-full px-4 pb-4 text-sm text-slate-500'>
                        Initializing...
                    </div>
                )}
                {props.transcriber.modelsStatusError && (
                    <div className='w-full px-4 pb-4 text-sm text-red-600'>
                        Could not initialize the transcription engine:{" "}
                        {props.transcriber.modelsStatusError}
                    </div>
                )}
                <div className='w-full px-4 pb-4 text-xs text-slate-500 flex justify-between'>
                    <span>{props.transcriber.capabilityLabel}</span>
                    {props.transcriber.effectivePresetLabel && (
                        <span>
                            Effective model:{" "}
                            {props.transcriber.effectivePresetLabel}
                        </span>
                    )}
                </div>
                <ProgressBar progress={progressValue} />
            </div>

            {props.transcriber.isBusy && (
                <div className='w-full mt-4 p-4 bg-white rounded-lg shadow-xl shadow-black/5 ring-1 ring-slate-700/10'>
                    <div className='flex items-center justify-center'>
                        <div className='animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-600 mr-3'></div>
                        <span className='text-slate-600'>
                            {getStatusMessage()}
                        </span>
                    </div>
                    {(props.transcriber.status === "downloading" ||
                        props.transcriber.status === "extracting") && (
                        <Progress
                            text='Source preparation'
                            percentage={progressValue}
                        />
                    )}
                    {props.transcriber.isModelLoading &&
                        props.transcriber.progressItems.length > 0 && (
                            <div className='mt-3'>
                                {props.transcriber.progressItems.map(
                                    (item, index) => (
                                        <Progress
                                            key={`${item.file}-${index}`}
                                            text={item.name || item.file}
                                            percentage={item.progress}
                                        />
                                    ),
                                )}
                            </div>
                        )}
                    {/*
                     * The escape hatch, and the reason it has to exist: while a run
                     * is in flight, BOTH tiles are disabled, the `Transcribe` button
                     * is hidden and drops are refused. A run that never terminates
                     * therefore had exactly one exit — quitting the app. This is the
                     * other one.
                     */}
                    <div className='mt-3 flex justify-center'>
                        <button
                            onClick={handleCancel}
                            className='text-sm text-slate-500 hover:text-red-600 underline underline-offset-2 transition-colors duration-200'
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            )}

            {/*
             * `dropError` WINS. It used to be the fallback, so a stale error from
             * an earlier transcription — which a rejected drop does not and should
             * not clear, since it never reaches the hook — kept the screen and hid
             * the reason the drop was refused. The drop error is always the newer
             * event of the two, and it is cleared the moment a file is accepted.
             */}
            {(dropError || props.transcriber.error) && (
                <div className='w-full mt-4 p-4 bg-red-50 rounded-lg border border-red-200'>
                    <div className='text-red-600 text-center'>
                        {dropError ?? props.transcriber.error}
                    </div>
                </div>
            )}

            {dropNotice && (
                <div className='w-full mt-4 p-4 bg-amber-50 rounded-lg border border-amber-200'>
                    <div className='text-amber-800 text-center text-sm'>
                        {dropNotice}
                    </div>
                </div>
            )}

            {selectedPath && !props.transcriber.isBusy && (
                <div className='w-full mt-4 flex justify-center items-center'>
                    <div className='text-sm text-slate-500 mr-4 flex items-center'>
                        {basename(selectedPath)}
                    </div>
                    <button
                        onClick={handleTranscribe}
                        disabled={
                            !props.transcriber.selectedModelAvailable ||
                            !engineReady
                        }
                        className='bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white font-medium py-2 px-6 rounded-lg transition-colors duration-200'
                    >
                        Transcribe
                    </button>
                </div>
            )}
        </div>
    );
}

function SettingsPanel(props: { transcriber: Transcriber }) {
    const engine = engineById(props.transcriber.engine);

    return (
        <div className='flex flex-col gap-3 text-sm'>
            <div className='grid grid-cols-1 md:grid-cols-3 gap-3'>
                {engine.modelId === null ? (
                    <label className='flex flex-col text-slate-600'>
                        Model
                        <select
                            value={props.transcriber.presetId}
                            onChange={(event) =>
                                props.transcriber.setPresetId(
                                    event.target
                                        .value as typeof props.transcriber.presetId,
                                )
                            }
                            className='mt-1 rounded-lg border border-slate-300 px-3 py-2'
                        >
                            {props.transcriber.presetOptions.map((preset) => (
                                <option
                                    key={preset.id}
                                    value={preset.id}
                                    disabled={
                                        preset.webgpuOnly &&
                                        !props.transcriber.browserCaps
                                            ?.canUseWebGPU
                                    }
                                >
                                    {preset.label}
                                </option>
                            ))}
                        </select>
                    </label>
                ) : (
                    <div className='flex flex-col text-slate-600'>
                        Model
                        <span className='mt-1 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-slate-700'>
                            Gemini 3.5 Transcribe
                        </span>
                    </div>
                )}

                <label className='flex flex-col text-slate-600'>
                    Task
                    <select
                        value={props.transcriber.task}
                        onChange={(event) =>
                            props.transcriber.setTask(
                                event.target.value as
                                    | "transcribe"
                                    | "translate",
                            )
                        }
                        disabled={!engine.supportsTranslate}
                        className='mt-1 rounded-lg border border-slate-300 px-3 py-2'
                    >
                        <option value='transcribe'>Transcribe</option>
                        <option value='translate'>Translate to English</option>
                    </select>
                </label>

                <label className='flex flex-col text-slate-600'>
                    Language
                    <select
                        value={props.transcriber.language}
                        onChange={(event) =>
                            props.transcriber.setLanguage(event.target.value)
                        }
                        disabled={!engine.supportsLanguageChoice}
                        className='mt-1 rounded-lg border border-slate-300 px-3 py-2'
                    >
                        {props.transcriber.languageOptions.map((option) => (
                            <option key={option.value} value={option.value}>
                                {option.label}
                            </option>
                        ))}
                    </select>
                </label>
            </div>

            {/*
             * Both controls, not just Language. The Task select above is
             * disabled by `!supportsTranslate` and the Language select by
             * `!supportsLanguageChoice`; gating the explanation on only one of
             * them leaves an engine that supports language choice but not
             * translation with a dead Task control and nothing saying why.
             * (The copy is Gemini-specific prose in a capability-driven block —
             * a third engine will need it rewritten. The polarity is the bug;
             * the wording is a known debt.)
             */}
            {(!engine.supportsLanguageChoice || !engine.supportsTranslate) && (
                <p className='text-xs text-slate-500'>
                    Gemini detects the language automatically across 85+ locales
                    and has no translation mode, so Task and Language do not
                    apply.
                </p>
            )}
        </div>
    );
}

function VerticalBar() {
    return <div className='w-[1px] bg-slate-200'></div>;
}

function ProgressBar(props: { progress: number }) {
    return (
        <div className='w-full bg-gray-200 rounded-full h-1 dark:bg-gray-700'>
            <div
                data-testid='overall-progress'
                className='bg-blue-600 h-1 rounded-full transition-all duration-100'
                style={{
                    width: `${Math.max(0, Math.min(100, props.progress))}%`,
                }}
            ></div>
        </div>
    );
}

function YouTubeTile(props: {
    icon: JSX.Element;
    text: string;
    onUrlSubmit: (url: string) => void;
    enabled: boolean;
    /** Why `enabled` is false, when it is. Rendered beside the dead submit. */
    disabledReason?: string | null;
    disabled?: boolean;
}) {
    const [showModal, setShowModal] = useState(false);

    return (
        <>
            <Tile
                icon={props.icon}
                text={props.text}
                disabled={props.disabled}
                onClick={() => setShowModal(true)}
            />
            <YouTubeModal
                show={showModal}
                enabled={props.enabled}
                disabledReason={props.disabledReason}
                onSubmit={(url) => {
                    props.onUrlSubmit(url);
                    setShowModal(false);
                }}
                onClose={() => setShowModal(false)}
            />
        </>
    );
}

function YouTubeModal(props: {
    show: boolean;
    enabled: boolean;
    disabledReason?: string | null;
    onSubmit: (url: string) => void;
    onClose: () => void;
}) {
    const [url, setUrl] = useState("");

    const isValidYouTubeUrl = (nextUrl: string) =>
        /^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/shorts\/)[\w-]+/.test(
            nextUrl,
        );

    return (
        <Modal
            show={props.show}
            title='From YouTube'
            content={
                <>
                    Enter the YouTube URL you want to transcribe.
                    <UrlInput
                        onChange={(event) => setUrl(event.target.value)}
                        value={url}
                    />
                    <div className='mt-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-800'>
                        For accessibility and education use only. Transcribe
                        YouTube videos only when you own the content, have
                        permission, or have determined your use is legally
                        authorized, including by fair use or accommodation
                        requirements. You are responsible for your use.
                    </div>
                    {url && !isValidYouTubeUrl(url) && (
                        <div className='text-red-500 text-sm mt-1'>
                            Please enter a valid YouTube URL
                        </div>
                    )}
                    {/*
                     * Last in `content`, so it lands immediately above the
                     * button row `Modal` renders — the closest this can sit to
                     * the disabled submit without `Modal` learning about it.
                     * Amber, mirroring the engine row it is pointing at.
                     */}
                    {props.disabledReason && (
                        <div className='mt-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-800'>
                            {props.disabledReason}
                        </div>
                    )}
                </>
            }
            onClose={props.onClose}
            submitText='Prepare Audio'
            submitEnabled={props.enabled && isValidYouTubeUrl(url)}
            onSubmit={() => {
                props.onSubmit(url.trim());
                setUrl("");
            }}
        />
    );
}

/**
 * The native file picker, via the Tauri dialog plugin.
 *
 * This used to build an `<input type="file">` and hand the resulting browser
 * `File` straight to the transcriber. A `File` exposes only its bytes, never its
 * path — so the audio could only ever be decoded in the webview, and Rust had no
 * file to point ffmpeg at. That matters more now, not less: ffmpeg's prepared WAV
 * is the front of the pipeline for BOTH engines, and it is also what Gemini's
 * chunker slices. The dialog returns a real path instead.
 */
function FileTile(props: {
    icon: JSX.Element;
    text: string;
    onFileSelect: (path: string) => void;
    disabled?: boolean;
}) {
    return (
        <Tile
            icon={props.icon}
            text={props.text}
            disabled={props.disabled}
            onClick={() => {
                void open({
                    multiple: false,
                    directory: false,
                    filters: [{ name: "Media", extensions: MEDIA_EXTENSIONS }],
                }).then((selected) => {
                    // `null` when the user cancels.
                    if (typeof selected === "string") {
                        props.onFileSelect(selected);
                    }
                });
            }}
        />
    );
}

function Tile(props: {
    icon: JSX.Element;
    text?: string;
    onClick?: () => void;
    disabled?: boolean;
}) {
    return (
        <button
            onClick={props.onClick}
            disabled={props.disabled}
            className='flex items-center justify-center rounded-lg p-2 bg-blue text-slate-500 hover:text-indigo-600 hover:bg-indigo-50 disabled:text-slate-300 disabled:hover:bg-transparent disabled:cursor-not-allowed transition-all duration-200'
        >
            <div className='w-7 h-7'>{props.icon}</div>
            {props.text && (
                <div className='ml-2 break-text text-center text-md w-30'>
                    {props.text}
                </div>
            )}
        </button>
    );
}

function FolderIcon() {
    return (
        <svg viewBox='0 0 20 20' fill='currentColor'>
            <path d='M2.5 4A1.5 1.5 0 0 1 4 2.5h4.379a1.5 1.5 0 0 1 1.06.44l1.121 1.12a1.5 1.5 0 0 0 1.06.44H16A1.5 1.5 0 0 1 17.5 6v8A2.5 2.5 0 0 1 15 16.5H5A2.5 2.5 0 0 1 2.5 14V4Z' />
        </svg>
    );
}

function YouTubeIcon() {
    return (
        <svg viewBox='0 0 24 24' fill='currentColor'>
            <path d='M23.5 6.2a3 3 0 0 0-2.11-2.12C19.45 3.5 12 3.5 12 3.5s-7.45 0-9.39.58A3 3 0 0 0 .5 6.2 31.3 31.3 0 0 0 0 12a31.3 31.3 0 0 0 .5 5.8 3 3 0 0 0 2.11 2.12c1.94.58 9.39.58 9.39.58s7.45 0 9.39-.58a3 3 0 0 0 2.11-2.12A31.3 31.3 0 0 0 24 12a31.3 31.3 0 0 0-.5-5.8ZM9.75 15.5v-7l6 3.5-6 3.5Z' />
        </svg>
    );
}
