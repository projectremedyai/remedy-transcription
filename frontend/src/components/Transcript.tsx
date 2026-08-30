import { useEffect, useRef, useState } from "react";

import { TranscriberData } from "../hooks/useTranscriber";
import { formatAudioTimestamp } from "../utils/AudioUtils";
import { api, SpeakerNames } from "../services/api";
import { ConsolidatedSegment } from "../lib/captionFormatter";
import {
    EngineSpeakers,
    LOCAL_ENGINE_NO_SPEAKERS,
} from "../hooks/engines/types";

interface Props {
    transcribedData: TranscriberData | undefined;
    jobId?: string | null;
    /**
     * What the engine that produced this transcript said about speakers.
     * `null` means no engine has answered yet — its own state, distinct from
     * both arms of the union.
     */
    speakerOutcome?: EngineSpeakers | null;
    /** What the user has called each speaker, keyed by the opaque label a cue carries. */
    speakerNames?: SpeakerNames;
    /**
     * Write-through rename. Omitted (rather than a no-op default) so a caller
     * that has not wired renaming yet gets a read-only label, not a button that
     * silently does nothing.
     */
    onRenameSpeaker?: (
        speakerKey: string,
        displayName: string,
    ) => Promise<void> | void;
}

/**
 * Filter only — no rebuilding. The cues are already cleaned and wrapped by the
 * formatter, and reconstructing them here would strip the `ConsolidatedSegment`
 * brand, which is what stops the exporters from being handed unformatted text.
 */
function exportableCues(
    chunks: TranscriberData["chunks"] | undefined,
): ConsolidatedSegment[] {
    if (!chunks) return [];
    return chunks.filter((chunk) => chunk.text.trim().length > 0);
}

/**
 * A cue's speaker: the display name if one has been set, otherwise the raw
 * opaque label (`SPEAKER_00`) — NEVER a parsed/renumbered "Speaker N+1". The
 * label is rendered verbatim; it is not indexed, counted, or assumed small.
 *
 * Click-to-rename, writing straight through `onRenameSpeaker`. Without it,
 * this is a read-only pill — a caller that has not wired renaming yet must not
 * get a button that looks interactive and silently does nothing.
 */
function SpeakerLabel({
    speakerKey,
    speakerNames,
    onRename,
}: {
    speakerKey: string;
    speakerNames: SpeakerNames;
    onRename?: (
        speakerKey: string,
        displayName: string,
    ) => Promise<void> | void;
}) {
    const displayName = speakerNames[speakerKey] ?? speakerKey;
    const [editing, setEditing] = useState(false);
    const [draft, setDraft] = useState(displayName);

    useEffect(() => {
        if (!editing) {
            setDraft(displayName);
        }
    }, [displayName, editing]);

    const commit = () => {
        setEditing(false);
        const trimmed = draft.trim();
        if (onRename && trimmed.length > 0 && trimmed !== displayName) {
            void onRename(speakerKey, trimmed);
        } else {
            setDraft(displayName);
        }
    };

    if (!onRename) {
        return (
            <span className='text-xs font-semibold text-indigo-700'>
                {displayName}
            </span>
        );
    }

    if (editing) {
        return (
            <input
                autoFocus
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onBlur={commit}
                onKeyDown={(event) => {
                    if (event.key === "Enter") {
                        (event.target as HTMLInputElement).blur();
                    } else if (event.key === "Escape") {
                        setDraft(displayName);
                        setEditing(false);
                    }
                }}
                aria-label={`Rename speaker ${speakerKey}`}
                data-testid={`speaker-rename-input-${speakerKey}`}
                className='text-xs font-semibold text-indigo-700 bg-transparent border-b border-indigo-300 focus:outline-none w-24'
            />
        );
    }

    return (
        <button
            type='button'
            onClick={() => setEditing(true)}
            title='Click to rename this speaker'
            data-testid={`speaker-label-${speakerKey}`}
            className='text-xs font-semibold text-indigo-600 hover:text-indigo-800 hover:underline'
        >
            {displayName}
        </button>
    );
}

/**
 * The one place an `EngineSpeakers` becomes words on screen.
 *
 * There is no feature flag here. With a single diarization producer, "does
 * this transcript have speakers?" is answered by the data, and a flag would
 * only be a second, staler answer to the same question.
 */
function SpeakerOutcomeBanner({
    outcome,
}: {
    outcome?: EngineSpeakers | null;
}) {
    if (!outcome) return null;

    if (outcome.status === "identified") {
        return (
            <div className='mt-2 rounded-md border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-700'>
                {outcome.speaker_count} speaker
                {outcome.speaker_count === 1 ? "" : "s"} identified. Click a
                name to rename them.
            </div>
        );
    }

    // The LOCAL engine reports "unavailable" on every single run — it has no
    // diarizer at all. Saying so on every on-device transcript would be noise
    // about a feature the user never asked for, so only a Gemini run's reason
    // (a real, specific "we could not do this, here is why") is shown.
    if (outcome.reason === LOCAL_ENGINE_NO_SPEAKERS) return null;

    return (
        <div className='mt-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-800'>
            Speaker labels are not available for this transcript:{" "}
            {outcome.reason}
        </div>
    );
}

export default function Transcript({
    transcribedData,
    jobId,
    speakerOutcome,
    speakerNames,
    onRenameSpeaker,
}: Props) {
    const divRef = useRef<HTMLDivElement>(null);
    const names = speakerNames ?? {};

    const getBaseName = () => {
        const name = transcribedData?.filename || "transcript";
        return name.replace(/\.[^/.]+$/, "");
    };

    const runExport = async (format: "srt" | "vtt" | "txt" | "json") => {
        const cues = exportableCues(transcribedData?.chunks);
        if (cues.length === 0) return;
        try {
            await api.exportTranscript(
                jobId ?? "local",
                format,
                cues,
                getBaseName(),
                names,
            );
        } catch (err) {
            console.error("Export failed:", err);
        }
    };

    useEffect(() => {
        if (divRef.current) {
            const diff = Math.abs(
                divRef.current.offsetHeight +
                    divRef.current.scrollTop -
                    divRef.current.scrollHeight,
            );
            if (diff <= 64) {
                divRef.current.scrollTop = divRef.current.scrollHeight;
            }
        }
    });

    return (
        <div className='w-full flex flex-col my-2'>
            <SpeakerOutcomeBanner outcome={speakerOutcome} />
            <div
                ref={divRef}
                className='w-full flex flex-col p-4 max-h-[20rem] overflow-y-auto'
            >
                {transcribedData?.chunks &&
                    transcribedData.chunks.map((chunk, i) => (
                        <div
                            key={`${i}-${chunk.text}`}
                            className='w-full flex flex-row mb-2 bg-white rounded-lg p-4 shadow-xl shadow-black/5 ring-1 ring-slate-700/10'
                        >
                            <div className='mr-5 text-slate-500 font-mono text-sm'>
                                {formatAudioTimestamp(chunk.start)}
                            </div>
                            <div className='flex-1'>
                                {chunk.speaker && (
                                    <div className='mb-1'>
                                        <SpeakerLabel
                                            speakerKey={chunk.speaker}
                                            speakerNames={names}
                                            onRename={onRenameSpeaker}
                                        />
                                    </div>
                                )}
                                {chunk.text}
                            </div>
                        </div>
                    ))}
            </div>
            {transcribedData && !transcribedData.isBusy && (
                <div className='w-full text-center mt-4'>
                    <button
                        onClick={() => runExport("srt")}
                        className='text-white bg-indigo-500 hover:bg-indigo-600 focus:ring-4 focus:ring-indigo-300 font-medium rounded-lg text-sm px-4 py-2 text-center mr-2 inline-flex items-center'
                    >
                        Export SRT
                    </button>
                    <button
                        onClick={() => runExport("vtt")}
                        className='text-white bg-indigo-500 hover:bg-indigo-600 focus:ring-4 focus:ring-indigo-300 font-medium rounded-lg text-sm px-4 py-2 text-center mr-2 inline-flex items-center'
                    >
                        Export VTT
                    </button>
                    <button
                        onClick={() => runExport("txt")}
                        className='text-white bg-green-500 hover:bg-green-600 focus:ring-4 focus:ring-green-300 font-medium rounded-lg text-sm px-4 py-2 text-center mr-2 inline-flex items-center'
                    >
                        Export TXT
                    </button>
                    <button
                        onClick={() => runExport("json")}
                        className='text-white bg-green-500 hover:bg-green-600 focus:ring-4 focus:ring-green-300 font-medium rounded-lg text-sm px-4 py-2 text-center mr-2 inline-flex items-center'
                    >
                        Export JSON
                    </button>
                </div>
            )}
        </div>
    );
}
