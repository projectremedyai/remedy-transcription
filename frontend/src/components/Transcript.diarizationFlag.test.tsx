// @vitest-environment jsdom
/**
 * 1.1.0: `DIARIZATION_UI_ENABLED` (`../config/features`) is false — see
 * `AudioManager.diarizationFlag.test.tsx` for why. `Transcript.test.tsx`
 * mocks the flag back to `true` to keep the `DiarizationStatus` plumbing
 * green for a future re-enable; this file exercises the real, shipped
 * default (false), without mocking anything.
 *
 * The property that matters here is data-preserving: hiding the banner must
 * not touch segments that already carry a persisted `speaker` field from an
 * earlier (flag-on) run. Those pills, and renaming on them, must keep
 * working — only the (now pointless, since no outcome can occur) status
 * banner goes away.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ConsolidatedSegment } from "../lib/captionFormatter";
import { DiarizationOutcome } from "../services/api";
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

describe("Transcript with DIARIZATION_UI_ENABLED=false (1.1.0 shipped state)", () => {
    it("never renders the DiarizationStatus banner, even for an outcome that would otherwise render one", () => {
        const data = transcriptWith([cue(0, 2, "Hello there.")]);
        const outcome: DiarizationOutcome = {
            status: "succeeded",
            turns: [{ start: 0, end: 2, speaker: 0 }],
            speaker_count: 1,
        };
        render(
            <Transcript
                transcribedData={data}
                jobId='job-1'
                diarizationOutcome={outcome}
            />,
        );

        expect(
            document.querySelector('[data-testid="diarization-status"]'),
        ).toBeNull();
    });

    it("still renders the speaker pill for a segment that already carries a persisted `speaker` field", () => {
        const data = transcriptWith([cue(0, 2, "Hello there.", "SPEAKER_00")]);
        render(<Transcript transcribedData={data} jobId='job-1' />);

        expect(screen.getByText("SPEAKER_00")).toBeTruthy();
    });

    it("keeps rename functional on a persisted speaker segment", () => {
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
        fireEvent.change(input, { target: { value: "Alice" } });
        fireEvent.blur(input);

        expect(onRenameSpeaker).toHaveBeenCalledWith("SPEAKER_00", "Alice");
    });
});
