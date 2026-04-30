import React, { useMemo, useState } from "react";

import Modal from "./modal/Modal";
import { UrlInput } from "./modal/UrlInput";
import { Transcriber } from "../hooks/useTranscriber";
import Progress from "./Progress";

export function AudioManager(props: { transcriber: Transcriber }) {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    const handleFileSelect = (file: File) => {
        props.transcriber.onInputChange();
        setSelectedFile(file);
    };

    const handleYouTubeSubmit = (url: string) => {
        props.transcriber.onInputChange();
        setSelectedFile(null);
        props.transcriber.startFromYouTube(url);
    };

    const handleTranscribe = () => {
        if (selectedFile) {
            props.transcriber.start(selectedFile);
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
            <div className='flex flex-col justify-center items-center rounded-lg bg-white shadow-xl shadow-black/5 ring-1 ring-slate-700/10'>
                <div className='flex flex-row space-x-2 py-2 w-full px-2'>
                    <YouTubeTile
                        icon={<YouTubeIcon />}
                        text='YouTube'
                        onUrlSubmit={handleYouTubeSubmit}
                        enabled={props.transcriber.selectedModelAvailable}
                    />
                    <VerticalBar />
                    <FileTile
                        icon={<FolderIcon />}
                        text='From file'
                        onFileSelect={handleFileSelect}
                    />
                </div>
                <div className='w-full px-4 pb-4'>
                    <SettingsPanel transcriber={props.transcriber} />
                </div>
                {!props.transcriber.modelsStatusLoaded && (
                    <div className='w-full px-4 pb-4 text-sm text-slate-500'>
                        Checking browser model availability...
                    </div>
                )}
                {props.transcriber.modelsStatusLoaded &&
                    props.transcriber.modelsStatusError && (
                        <div className='w-full px-4 pb-4 text-sm text-red-600'>
                            Could not reach the backend model status endpoint.
                            Check that the backend server is running and that
                            the frontend proxy is using the correct backend
                            port.
                        </div>
                    )}
                {props.transcriber.modelsStatusLoaded &&
                    !props.transcriber.modelsStatusError &&
                    !props.transcriber.modelsReady && (
                        <div className='w-full px-4 pb-4 text-sm text-amber-700'>
                            Browser models are missing on the server. Run{" "}
                            <code>python3 scripts/bootstrap-models.py</code>{" "}
                            before transcribing.
                        </div>
                    )}
                {props.transcriber.modelsStatusLoaded &&
                    props.transcriber.modelsReady &&
                    !props.transcriber.selectedModelAvailable &&
                    props.transcriber.selectedModelId && (
                        <div className='w-full px-4 pb-4 text-sm text-amber-700'>
                            The selected model is not installed on the server:{" "}
                            <code>{props.transcriber.selectedModelId}</code>
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

            {props.transcriber.error && (
                <div className='w-full mt-4 p-4 bg-red-50 rounded-lg border border-red-200'>
                    <div className='text-red-600 text-center'>
                        {props.transcriber.error}
                    </div>
                </div>
            )}

            {selectedFile && !props.transcriber.isBusy && (
                <div className='w-full mt-4 flex justify-center items-center'>
                    <div className='text-sm text-slate-500 mr-4 flex items-center'>
                        {selectedFile.name}
                    </div>
                    <button
                        onClick={handleTranscribe}
                        disabled={!props.transcriber.selectedModelAvailable}
                        className='bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white font-medium py-2 px-6 rounded-lg transition-colors duration-200'
                    >
                        Transcribe in Browser
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
}) {
    const [showModal, setShowModal] = useState(false);

    return (
        <>
            <Tile
                icon={props.icon}
                text={props.text}
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
                    {url && !isValidYouTubeUrl(url) && (
                        <div className='text-red-500 text-sm mt-1'>
                            Please enter a valid YouTube URL
                        </div>
                    )}
                    {!props.enabled && (
                        <div className='text-amber-600 text-sm mt-1'>
                            Install the selected model on the server before
                            preparing audio.
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

function FileTile(props: {
    icon: JSX.Element;
    text: string;
    onFileSelect: (file: File) => void;
}) {
    return (
        <Tile
            icon={props.icon}
            text={props.text}
            onClick={() => {
                const input = document.createElement("input");
                input.type = "file";
                input.accept =
                    "video/*,audio/*,.mp4,.mkv,.avi,.mov,.webm,.mp3,.wav,.m4a,.aac,.ogg,.flac";
                input.onchange = (event) => {
                    const files = (event.target as HTMLInputElement).files;
                    if (files?.[0]) {
                        props.onFileSelect(files[0]);
                    }
                };
                input.click();
            }}
        />
    );
}

function Tile(props: {
    icon: JSX.Element;
    text?: string;
    onClick?: () => void;
}) {
    return (
        <button
            onClick={props.onClick}
            className='flex items-center justify-center rounded-lg p-2 bg-blue text-slate-500 hover:text-indigo-600 hover:bg-indigo-50 transition-all duration-200'
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
