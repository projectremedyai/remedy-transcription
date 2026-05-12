import { useEffect, useRef } from "react";

import { TranscriberData } from "../hooks/useTranscriber";
import { formatAudioTimestamp } from "../utils/AudioUtils";
import { api } from "../services/api";
import { TranscriptionSegment } from "../services/types";

interface Props {
    transcribedData: TranscriberData | undefined;
    jobId?: string | null;
}

function chunksToSegments(
    chunks: TranscriberData["chunks"] | undefined,
): TranscriptionSegment[] {
    if (!chunks) return [];
    return chunks
        .map((chunk) => ({
            start: chunk.timestamp[0],
            end: chunk.timestamp[1] ?? chunk.timestamp[0] + 5,
            text: chunk.text.trim(),
        }))
        .filter((s) => s.text.length > 0);
}

export default function Transcript({ transcribedData, jobId }: Props) {
    const divRef = useRef<HTMLDivElement>(null);

    const getBaseName = () => {
        const name = transcribedData?.filename || "transcript";
        return name.replace(/\.[^/.]+$/, "");
    };

    const runExport = async (format: "srt" | "vtt" | "txt" | "json") => {
        const segments = chunksToSegments(transcribedData?.chunks);
        if (segments.length === 0) return;
        try {
            await api.exportTranscript(
                jobId ?? "local",
                format,
                segments,
                getBaseName(),
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
                                {formatAudioTimestamp(chunk.timestamp[0])}
                            </div>
                            <div className='flex-1'>{chunk.text}</div>
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
