import { TranscriptionSegment } from "../services/types";
import { consolidateSegments, formatPlainText } from "./captionFormatter";

function formatTimestamp(seconds: number, separator: "," | "." = ","): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const millis = Math.floor((seconds % 1) * 1000);
    return `${pad2(hours)}:${pad2(minutes)}:${pad2(secs)}${separator}${pad3(millis)}`;
}

function pad2(value: number): string {
    return value.toString().padStart(2, "0");
}

function pad3(value: number): string {
    return value.toString().padStart(3, "0");
}

export function generateSrt(segments: TranscriptionSegment[]): string {
    const consolidated = consolidateSegments(segments);
    const lines: string[] = [];

    consolidated.forEach((segment, index) => {
        lines.push(String(index + 1));
        lines.push(
            `${formatTimestamp(segment.start)} --> ${formatTimestamp(
                segment.end,
            )}`,
        );
        lines.push(segment.text);
        lines.push("");
    });

    return lines.join("\n");
}

export function generateVtt(segments: TranscriptionSegment[]): string {
    const consolidated = consolidateSegments(segments);
    const lines: string[] = ["WEBVTT", ""];

    consolidated.forEach((segment, index) => {
        lines.push(String(index + 1));
        lines.push(
            `${formatTimestamp(segment.start, ".")} --> ${formatTimestamp(
                segment.end,
                ".",
            )}`,
        );
        lines.push(segment.text);
        lines.push("");
    });

    return lines.join("\n");
}

export function generateTxt(segments: TranscriptionSegment[]): string {
    return formatPlainText(segments);
}

export function generateJson(segments: TranscriptionSegment[]): string {
    const consolidated = consolidateSegments(segments).map((s) => ({
        start: s.start,
        end: s.end,
        text: s.text,
    }));
    return JSON.stringify(consolidated, null, 2);
}
