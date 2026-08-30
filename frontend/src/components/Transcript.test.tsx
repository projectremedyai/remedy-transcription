// @vitest-environment jsdom
/**
 * The click-to-rename flow writing through to `onRenameSpeaker`, and the
 * speaker label rendering itself. These are properties a headless `tsc`/lint
 * pass cannot see — a previous task on this branch shipped a dead button
 * behind a passing headless check, which is exactly why this renders the real
 * component (jsdom + @testing-library/react) instead of asserting on
 * hand-built props.
 */
import type { ComponentProps } from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ConsolidatedSegment } from "../lib/captionFormatter";
import { api, SpeakerNames } from "../services/api";
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

/**
 * Renders the real component with sensible defaults (an empty, non-busy
 * transcript) so each test below only has to spell out the prop it is
 * actually exercising — here, `speakerOutcome`.
 */
function renderTranscript(
    overrides: Partial<ComponentProps<typeof Transcript>> = {},
) {
    return render(
        <Transcript
            transcribedData={transcriptWith([])}
            jobId='job-1'
            {...overrides}
        />,
    );
}

describe("Transcript: an undiarized transcript is unchanged", () => {
    it("renders no speaker pill and no status banner when nothing carries a speaker", () => {
        const data = transcriptWith([cue(0, 2, "Hello there.")]);
        render(<Transcript transcribedData={data} jobId='job-1' />);

        expect(screen.getByText("Hello there.")).toBeTruthy();
        // No speaker pill of any kind — only the (unrelated) export buttons.
        expect(
            document.querySelector('[data-testid^="speaker-label-"]'),
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

describe("the speaker outcome banner", () => {
    it("renders nothing when no engine has reported (null)", () => {
        const { container } = renderTranscript({ speakerOutcome: null });
        expect(container.textContent).not.toMatch(/speaker/i);
    });

    it("reports how many speakers were identified", () => {
        renderTranscript({
            speakerOutcome: {
                status: "identified",
                turns: [],
                speaker_count: 3,
            },
        });
        expect(screen.getByText(/3 speakers identified/i)).toBeTruthy();
    });

    it("uses the singular for one speaker", () => {
        renderTranscript({
            speakerOutcome: {
                status: "identified",
                turns: [],
                speaker_count: 1,
            },
        });
        expect(screen.getByText(/1 speaker identified/i)).toBeTruthy();
    });

    /**
     * The whole reason `speakers` is a tagged union. "No labels because we
     * split the audio into 4 parts" must not read as "one person was talking".
     */
    it("shows the stated reason when speakers are unavailable", () => {
        renderTranscript({
            speakerOutcome: {
                status: "unavailable",
                reason: "this audio was split into 4 parts",
            },
        });
        expect(screen.getByText(/split into 4 parts/i)).toBeTruthy();
    });

    /**
     * The local engine reports "unavailable" on EVERY run. Announcing that on
     * every on-device transcript would be noise about a feature the user never
     * asked for.
     */
    it("stays silent for the on-device engine's standing unavailability", () => {
        const { container } = renderTranscript({
            speakerOutcome: {
                status: "unavailable",
                reason: "the on-device engine does not identify speakers",
            },
        });
        expect(container.textContent).not.toMatch(
            /does not identify speakers/i,
        );
    });
});
