// @vitest-environment jsdom
/**
 * Task 12: rendering the three `DiarizationOutcome` arms (plus "never ran") as
 * genuinely different UI, and the click-to-rename flow writing through to
 * `onRenameSpeaker`. These are the properties a headless `tsc`/lint pass cannot
 * see — a previous task on this branch shipped a dead button behind a passing
 * headless check, which is exactly why this renders the real component (jsdom
 * + @testing-library/react) instead of asserting on hand-built props.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ConsolidatedSegment } from "../lib/captionFormatter";
import { api, DiarizationOutcome, SpeakerNames } from "../services/api";
import type { TranscriberData } from "../hooks/useTranscriber";

vi.mock("../services/api", () => ({
    api: { exportTranscript: vi.fn() },
}));

import Transcript from "./Transcript";

afterEach(() => {
    cleanup();
});

const cue = (
    start: number,
    end: number,
    text: string,
    speaker?: string,
): ConsolidatedSegment =>
    ({ start, end, text, speaker } as unknown as ConsolidatedSegment);

function transcriptWith(chunks: ConsolidatedSegment[]): TranscriberData {
    return {
        isBusy: false,
        text: chunks.map((c) => c.text).join(" "),
        chunks,
        filename: "lecture.mp3",
        persisted: true,
        modelLabel: "test model",
    };
}

describe("Transcript: an undiarized transcript is unchanged", () => {
    it("renders no speaker pill and no status banner when nothing carries a speaker", () => {
        const data = transcriptWith([cue(0, 2, "Hello there.")]);
        render(<Transcript transcribedData={data} jobId='job-1' />);

        expect(screen.getByText("Hello there.")).toBeTruthy();
        expect(
            document.querySelector('[data-testid="diarization-status"]'),
        ).toBeNull();
        // No speaker pill of any kind — only the (unrelated) export buttons.
        expect(
            document.querySelector('[data-testid^="speaker-label-"]'),
        ).toBeNull();
    });

    it("shows no status banner when `diarizationOutcome` is null — the toggle-off state", () => {
        const data = transcriptWith([cue(0, 2, "Hello there.")]);
        render(
            <Transcript
                transcribedData={data}
                jobId='job-1'
                diarizationOutcome={null}
            />,
        );

        expect(
            document.querySelector('[data-testid="diarization-status"]'),
        ).toBeNull();
    });
});

describe("Transcript: speaker labels are opaque, never synthesized", () => {
    it("renders the raw label when no display name has been set", () => {
        const data = transcriptWith([cue(0, 2, "Hello there.", "SPEAKER_07")]);
        render(<Transcript transcribedData={data} jobId='job-1' />);

        // Verbatim — never "Speaker 8" or any renumbering of the opaque id.
        expect(screen.getByText("SPEAKER_07")).toBeTruthy();
        expect(screen.queryByText(/Speaker 8/i)).toBeNull();
        expect(screen.queryByText(/Speaker 7/i)).toBeNull();
    });

    it("renders the mapped display name when speakerNames has an entry", () => {
        const data = transcriptWith([cue(0, 2, "Hello there.", "SPEAKER_00")]);
        const speakerNames: SpeakerNames = { SPEAKER_00: "Alice" };
        render(
            <Transcript
                transcribedData={data}
                jobId='job-1'
                speakerNames={speakerNames}
            />,
        );

        expect(screen.getByText("Alice")).toBeTruthy();
        expect(screen.queryByText("SPEAKER_00")).toBeNull();
    });

    it("a speaker with no entry renders its own opaque key, unaffected by OTHER speakers' names", () => {
        const data = transcriptWith([
            cue(0, 2, "Hello.", "SPEAKER_00"),
            cue(3, 5, "Hi.", "SPEAKER_01"),
        ]);
        const speakerNames: SpeakerNames = { SPEAKER_00: "Alice" };
        render(
            <Transcript
                transcribedData={data}
                jobId='job-1'
                speakerNames={speakerNames}
            />,
        );

        expect(screen.getByText("Alice")).toBeTruthy();
        expect(screen.getByText("SPEAKER_01")).toBeTruthy();
    });
});

describe("Transcript: click-to-rename", () => {
    it("is read-only (a plain label, not a button) when no onRenameSpeaker is supplied", () => {
        const data = transcriptWith([cue(0, 2, "Hello there.", "SPEAKER_00")]);
        render(<Transcript transcribedData={data} jobId='job-1' />);

        expect(screen.getByText("SPEAKER_00").tagName).not.toBe("BUTTON");
        expect(
            document.querySelector('[data-testid^="speaker-label-"]'),
        ).toBeNull();
    });

    it("clicking the label, editing it, and blurring writes through onRenameSpeaker with the trimmed name", async () => {
        const data = transcriptWith([cue(0, 2, "Hello there.", "SPEAKER_00")]);
        const onRenameSpeaker = vi.fn().mockResolvedValue(undefined);
        render(
            <Transcript
                transcribedData={data}
                jobId='job-1'
                onRenameSpeaker={onRenameSpeaker}
            />,
        );

        fireEvent.click(screen.getByTestId("speaker-label-SPEAKER_00"));

        const input = screen.getByTestId(
            "speaker-rename-input-SPEAKER_00",
        ) as HTMLInputElement;
        fireEvent.change(input, { target: { value: "  Alice  " } });
        fireEvent.blur(input);

        expect(onRenameSpeaker).toHaveBeenCalledWith("SPEAKER_00", "Alice");
    });

    it("Enter commits the same way blur does", () => {
        const data = transcriptWith([cue(0, 2, "Hello there.", "SPEAKER_00")]);
        const onRenameSpeaker = vi.fn().mockResolvedValue(undefined);
        render(
            <Transcript
                transcribedData={data}
                jobId='job-1'
                onRenameSpeaker={onRenameSpeaker}
            />,
        );

        fireEvent.click(screen.getByTestId("speaker-label-SPEAKER_00"));
        const input = screen.getByTestId(
            "speaker-rename-input-SPEAKER_00",
        ) as HTMLInputElement;
        fireEvent.change(input, { target: { value: "Bob" } });
        fireEvent.keyDown(input, { key: "Enter" });

        expect(onRenameSpeaker).toHaveBeenCalledWith("SPEAKER_00", "Bob");
    });

    it("Escape cancels the edit without calling onRenameSpeaker", () => {
        const data = transcriptWith([cue(0, 2, "Hello there.", "SPEAKER_00")]);
        const onRenameSpeaker = vi.fn();
        render(
            <Transcript
                transcribedData={data}
                jobId='job-1'
                onRenameSpeaker={onRenameSpeaker}
            />,
        );

        fireEvent.click(screen.getByTestId("speaker-label-SPEAKER_00"));
        const input = screen.getByTestId(
            "speaker-rename-input-SPEAKER_00",
        ) as HTMLInputElement;
        fireEvent.change(input, { target: { value: "Bob" } });
        fireEvent.keyDown(input, { key: "Escape" });

        expect(onRenameSpeaker).not.toHaveBeenCalled();
        expect(screen.getByText("SPEAKER_00")).toBeTruthy();
    });

    it("blurring on the unchanged (or blank) name does not call onRenameSpeaker", () => {
        const data = transcriptWith([cue(0, 2, "Hello there.", "SPEAKER_00")]);
        const onRenameSpeaker = vi.fn();
        render(
            <Transcript
                transcribedData={data}
                jobId='job-1'
                onRenameSpeaker={onRenameSpeaker}
            />,
        );

        fireEvent.click(screen.getByTestId("speaker-label-SPEAKER_00"));
        const input = screen.getByTestId(
            "speaker-rename-input-SPEAKER_00",
        ) as HTMLInputElement;
        fireEvent.change(input, { target: { value: "   " } });
        fireEvent.blur(input);

        expect(onRenameSpeaker).not.toHaveBeenCalled();
    });
});

describe("Transcript: each DiarizationOutcome arm renders distinctly", () => {
    const data = transcriptWith([cue(0, 2, "Hello there.")]);

    it("degraded shows the reason, and is never mistaken for silence", () => {
        const outcome: DiarizationOutcome = {
            status: "degraded",
            reason: "the segmentation model is not installed",
        };
        render(
            <Transcript
                transcribedData={data}
                jobId='job-1'
                diarizationOutcome={outcome}
            />,
        );

        const banner = screen.getByTestId("diarization-status");
        expect(banner.getAttribute("data-status")).toBe("degraded");
        expect(banner.textContent).toContain(
            "the segmentation model is not installed",
        );
    });

    it("cancelled renders different text from degraded", () => {
        const outcome: DiarizationOutcome = { status: "cancelled" };
        render(
            <Transcript
                transcribedData={data}
                jobId='job-1'
                diarizationOutcome={outcome}
            />,
        );

        const banner = screen.getByTestId("diarization-status");
        expect(banner.getAttribute("data-status")).toBe("cancelled");
        expect(banner.textContent).not.toContain("model");
        expect(banner.textContent?.toLowerCase()).toContain("cancelled");
    });

    it("succeeded with an empty turn list is distinct from both degraded and cancelled", () => {
        const outcome: DiarizationOutcome = {
            status: "succeeded",
            turns: [],
            speaker_count: 0,
        };
        render(
            <Transcript
                transcribedData={data}
                jobId='job-1'
                diarizationOutcome={outcome}
            />,
        );

        const banner = screen.getByTestId("diarization-status");
        expect(banner.getAttribute("data-status")).toBe("succeeded-empty");
        expect(banner.getAttribute("data-status")).not.toBe("degraded");
        expect(banner.getAttribute("data-status")).not.toBe("cancelled");
    });

    it("succeeded with turns reports the speaker count", () => {
        const outcome: DiarizationOutcome = {
            status: "succeeded",
            turns: [
                { start: 0, end: 2, speaker: 0 },
                { start: 3, end: 5, speaker: 1 },
            ],
            speaker_count: 2,
        };
        render(
            <Transcript
                transcribedData={data}
                jobId='job-1'
                diarizationOutcome={outcome}
            />,
        );

        const banner = screen.getByTestId("diarization-status");
        expect(banner.getAttribute("data-status")).toBe("succeeded");
        expect(banner.textContent).toContain("2 speakers");
    });
});

describe("Transcript export wiring (final-review.md B1)", () => {
    // Task 10/11 made the generators name-aware, but nothing forced the
    // export call site to actually pass the names map through. Assert on
    // the wiring at the boundary: `runExport` must hand `speakerNames` to
    // `api.exportTranscript`, not drop it.
    it("passes the speakerNames map through to api.exportTranscript on export", async () => {
        const names: SpeakerNames = { SPEAKER_00: "Alice" };
        const data = transcriptWith([cue(0, 2, "Hello there.", "SPEAKER_00")]);
        render(
            <Transcript
                transcribedData={data}
                jobId='job-1'
                speakerNames={names}
            />,
        );

        fireEvent.click(screen.getByText("Export SRT"));

        expect(api.exportTranscript).toHaveBeenCalledWith(
            "job-1",
            "srt",
            expect.any(Array),
            expect.any(String),
            names,
        );
    });
});
