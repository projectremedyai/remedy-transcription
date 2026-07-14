import React, { useCallback, useEffect, useMemo, useState } from "react";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";

import Modal from "./modal/Modal";
import { UrlInput } from "./modal/UrlInput";
import { Transcriber } from "../hooks/useTranscriber";
import Progress from "./Progress";

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

    const { transcriber } = props;
    const { onInputChange } = transcriber;
    const isBusy = transcriber.isBusy;

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
                    if (isBusy) {
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
    }, [handleFileSelect, isBusy]);

    const handleYouTubeSubmit = (url: string) => {
        transcriber.onInputChange();
        setSelectedPath(null);
        setDropError(null);
        setDropNotice(null);
        transcriber.startFromYouTube(url);
    };

    const handleTranscribe = () => {
        if (selectedPath) {
            transcriber.start(selectedPath);
        }
    };

    const getStatusMessage = () => {
        switch (props.transcriber.status) {
            case "checking-cache":
                return "Checking transcript cache...";
            case "downloading":
                return "Downloading from YouTube...";
            case "extracting":
                return "Extracting audio...";
            case "loading-audio":
                return "Loading prepared audio...";
            case "transcribing":
                return "Transcribing in your browser...";
            case "persisting":
                return "Saving transcript...";
            case "completed":
                return "Completed";
            case "failed":
                return "Failed";
            default:
                return "";
        }
    };

    const progressValue = useMemo(() => {
        if (
            props.transcriber.status === "downloading" ||
            props.transcriber.status === "extracting"
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
                 * Both entry points are gated on `isBusy`. The app models exactly
                 * ONE transcript at a time — one `jobId`, one output — so a second
                 * run started mid-flight cannot be shown, only substituted. The
                 * YouTube tile in particular was startable during a local file's
                 * extraction, which is the overlap that let a cache-hit YouTube run
                 * land on top of a still-running file job. The hook now cancels the
                 * superseded wait unconditionally, so this is defence in depth
                 * rather than the fix — but there is no legitimate use for
                 * overlapping runs here, and the `Transcribe` button was already
                 * hidden while busy, so the two entry points now agree.
                 */}
                <div className='flex flex-row space-x-2 py-2 w-full px-2'>
                    <YouTubeTile
                        icon={<YouTubeIcon />}
                        text='YouTube'
                        onUrlSubmit={handleYouTubeSubmit}
                        enabled={
                            props.transcriber.selectedModelAvailable && !isBusy
                        }
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
                <div className='w-full px-4 pb-4'>
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
                        disabled={!props.transcriber.selectedModelAvailable}
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
    return (
        <div className='grid grid-cols-1 md:grid-cols-3 gap-3 text-sm'>
            <label className='flex flex-col text-slate-600'>
                Model preset
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
                                !props.transcriber.browserCaps?.canUseWebGPU
                            }
                        >
                            {preset.label}
                        </option>
                    ))}
                </select>
            </label>

            <label className='flex flex-col text-slate-600'>
                Task
                <select
                    value={props.transcriber.task}
                    onChange={(event) =>
                        props.transcriber.setTask(
                            event.target.value as "transcribe" | "translate",
                        )
                    }
                    className='mt-1 rounded-lg border border-slate-300 px-3 py-2'
                >
                    <option value='transcribe'>Transcribe</option>
                    <option value='translate'>Translate to English</option>
                </select>
            </label>

            <label className='flex flex-col text-slate-600'>
                Source language
                <select
                    value={props.transcriber.language}
                    onChange={(event) =>
                        props.transcriber.setLanguage(event.target.value)
                    }
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
    );
}

function VerticalBar() {
    return <div className='w-[1px] bg-slate-200'></div>;
}

function ProgressBar(props: { progress: number }) {
    return (
        <div className='w-full bg-gray-200 rounded-full h-1 dark:bg-gray-700'>
            <div
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
 * path — so the audio could only ever be decoded in the webview, and Rust (where
 * diarization runs) had no file to point ffmpeg or sherpa-onnx at. The dialog
 * returns a real path instead.
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
