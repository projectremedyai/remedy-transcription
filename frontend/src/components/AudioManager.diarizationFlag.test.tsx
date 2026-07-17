// @vitest-environment jsdom
/**
 * 1.1.0: `DIARIZATION_UI_ENABLED` (`../config/features`) is false — real-content
 * testing found the diarization engine mislabels speakers even in its
 * count-required mode (one narrator produced 4 speaker labels, flipping 19
 * times over a 10-minute documentary sample), an embedding/engine ceiling, not
 * a bug here. The toggle and speaker-count input are hidden for this release;
 * `AudioManager.test.tsx` covers the same plumbing with the flag mocked back
 * to `true`, proving it still works for a future re-enable.
 *
 * This file does NOT mock `../config/features` — it exercises the real,
 * shipped default (false), which is the point: these are the properties that
 * must hold with nothing overridden.
 */
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LANGUAGE_OPTIONS, MODEL_PRESETS } from "../config/transcription";
import type { Transcriber } from "../hooks/useTranscriber";

const mocks = vi.hoisted(() => ({
    open: vi.fn(),
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
    open: (...args: unknown[]) => mocks.open(...args),
}));

vi.mock("@tauri-apps/api/webview", () => ({
    getCurrentWebview: () => ({
        onDragDropEvent: () => Promise.resolve(vi.fn()),
    }),
}));

import { AudioManager } from "./AudioManager";

afterEach(() => {
    cleanup();
});

beforeEach(() => {
    mocks.open.mockReset();
    vi.stubGlobal(
        "ResizeObserver",
        class {
            observe = vi.fn();
            unobserve = vi.fn();
            disconnect = vi.fn();
        },
    );
});

function makeTranscriber(overrides: Partial<Transcriber> = {}): Transcriber {
    return {
        onInputChange: vi.fn(),
        isBusy: false,
        isModelLoading: false,
        progressItems: [],
        start: vi.fn(),
        startFromYouTube: vi.fn(),
        cancel: vi.fn(),
        output: undefined,
        jobId: null,
        error: null,
        progress: 0,
        status: "idle",
        presetId: "auto",
        setPresetId: vi.fn(),
        task: "transcribe",
        setTask: vi.fn(),
        language: "auto",
        setLanguage: vi.fn(),
        browserCaps: null,
        capabilityLabel: "",
        effectivePresetLabel: null,
        modelsReady: true,
        modelsStatusLoaded: true,
        modelsStatusError: null,
        missingModels: [],
        selectedModelAvailable: true,
        selectedModelId: "test-model",
        presetOptions: MODEL_PRESETS,
        languageOptions: LANGUAGE_OPTIONS,
        diarizeEnabled: false,
        setDiarizeEnabled: vi.fn(),
        numSpeakersHint: undefined,
        setNumSpeakersHint: vi.fn(),
        diarizationOutcome: null,
        speakerNames: {},
        renameSpeaker: vi.fn(),
        ...overrides,
    };
}

async function selectAFile() {
    mocks.open.mockResolvedValue("/tmp/lecture.mp3");
    fireEvent.click(screen.getByText("From file"));
    return screen.findByRole("button", { name: "Transcribe" });
}

describe("AudioManager: diarization UI hidden behind DIARIZATION_UI_ENABLED=false (1.1.0 shipped state)", () => {
    it("does not render the toggle, the speaker-count input, or any diarization helper text", async () => {
        render(<AudioManager transcriber={makeTranscriber()} />);
        await selectAFile();

        expect(screen.queryByLabelText(/Identify speakers/)).toBeNull();
        expect(screen.queryByPlaceholderText("e.g. 2")).toBeNull();
        expect(
            screen.queryByText(
                "Enter the number of speakers to transcribe with speaker identification.",
            ),
        ).toBeNull();
    });

    it("leaves Transcribe enabled with no count entered — diarization can never be turned on, so it must never block a run", async () => {
        // `diarizeEnabled: true` stands in for a stale/leftover state (the
        // control that would normally set this no longer renders at all).
        // Even so, Transcribe must not be gated on it.
        render(
            <AudioManager
                transcriber={makeTranscriber({
                    diarizeEnabled: true,
                    numSpeakersHint: undefined,
                })}
            />,
        );
        await selectAFile();

        expect(
            screen.getByRole("button", {
                name: "Transcribe",
            }) as HTMLButtonElement,
        ).toHaveProperty("disabled", false);
    });

    it("leaves the YouTube modal's submit enabled the same way, for the same reason", async () => {
        render(
            <AudioManager
                transcriber={makeTranscriber({
                    diarizeEnabled: true,
                    numSpeakersHint: undefined,
                })}
            />,
        );

        fireEvent.click(screen.getByText("YouTube"));
        fireEvent.change(screen.getByPlaceholderText("www.example.com"), {
            target: { value: "https://www.youtube.com/watch?v=abc123" },
        });

        expect(
            screen.getByText("Prepare Audio") as HTMLButtonElement,
        ).toHaveProperty("disabled", false);
    });
});
