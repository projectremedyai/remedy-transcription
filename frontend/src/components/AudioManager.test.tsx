// @vitest-environment jsdom
/**
 * Diarization is now EXPERIMENTAL and REQUIRES a user-supplied speaker count —
 * real-content testing found auto-detect alone produces dozens of phantom
 * speakers (52 on a 53-minute documentary), so the UI must never be able to
 * reach a run with the toggle on and no valid count. That property lives in
 * `AudioManager`'s JSX (the `disabled` wiring on the Transcribe button and the
 * YouTube tile's `enabled` prop, and the helper text in `DiarizationSettings`)
 * — none of which a headless `tsc`/lint pass can see. So this renders the real
 * component (jsdom + `@testing-library/react`) with a small stateful harness
 * standing in for `useTranscriber`, exactly the way `Transcript.test.tsx`
 * exercises real button wiring instead of asserting on hand-built props.
 */
import { useState } from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LANGUAGE_OPTIONS, MODEL_PRESETS } from "../config/transcription";
import type { Transcriber } from "../hooks/useTranscriber";

const mocks = vi.hoisted(() => ({
    open: vi.fn(),
}));

// `DIARIZATION_UI_ENABLED` is off for the 1.1.0 release (see
// `../config/features`), which unmounts `DiarizationSettings` entirely — see
// `AudioManager.diarizationFlag.test.tsx` for that (shipped) behaviour. This
// suite exercises the diarization-toggle PLUMBING (the gate wiring on
// Transcribe/YouTube once the setting is visible again), so it overrides the
// flag to `true`, exactly the way it will be flipped back for a future
// release.
vi.mock("../config/features", () => ({ DIARIZATION_UI_ENABLED: true }));

vi.mock("@tauri-apps/plugin-dialog", () => ({
    open: (...args: unknown[]) => mocks.open(...args),
}));

vi.mock("@tauri-apps/api/webview", () => ({
    getCurrentWebview: () => ({
        // `AudioManager`'s drag-drop effect awaits this and only unlistens on
        // cleanup; a promise resolving to a no-op unlisten is enough to let
        // the effect settle without ever actually receiving a drop.
        onDragDropEvent: () => Promise.resolve(vi.fn()),
    }),
}));

import { AudioManager } from "./AudioManager";

afterEach(() => {
    cleanup();
});

beforeEach(() => {
    mocks.open.mockReset();
    // jsdom has no `ResizeObserver`; Headless UI's `Dialog` (used by the
    // YouTube modal) reads it on mount.
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

/**
 * A thin, REAL React state wrapper for the two diarization fields, so a click
 * on the checkbox and a change on the speaker-count input actually flow back
 * into what `AudioManager` reads next render — the same round trip
 * `useTranscriber` provides in the app, not a hand-toggled prop.
 */
function Harness(props: {
    initialDiarizeEnabled?: boolean;
    initialNumSpeakersHint?: number;
}) {
    const [diarizeEnabled, setDiarizeEnabled] = useState(
        props.initialDiarizeEnabled ?? false,
    );
    const [numSpeakersHint, setNumSpeakersHint] = useState<number | undefined>(
        props.initialNumSpeakersHint,
    );

    const transcriber = makeTranscriber({
        diarizeEnabled,
        setDiarizeEnabled,
        numSpeakersHint,
        setNumSpeakersHint,
    });

    return <AudioManager transcriber={transcriber} />;
}

async function selectAFile() {
    mocks.open.mockResolvedValue("/tmp/lecture.mp3");
    fireEvent.click(screen.getByText("From file"));
    // `open()` resolves asynchronously; `findByText` polls until the
    // Transcribe button the resolution reveals actually appears.
    return screen.findByRole("button", { name: "Transcribe" });
}

describe("AudioManager: the experimental-diarization gate", () => {
    it("disables Transcribe and shows helper text when the toggle is on with no speaker count, and clears both once a valid count is entered", async () => {
        render(<Harness />);

        await selectAFile();

        // Toggle on: `diarizeEnabled` starts true-after-click, `numSpeakersHint`
        // stays `undefined` -- exactly the state the backend has no way to run
        // sanely (auto-detect produced 52 phantom speakers on a 53-minute
        // documentary in the real-content test that motivated this change).
        fireEvent.click(screen.getByLabelText(/Identify speakers/));

        expect(
            screen.getByRole("button", {
                name: "Transcribe",
            }) as HTMLButtonElement,
        ).toHaveProperty("disabled", true);
        expect(
            screen.getByText(
                "Enter the number of speakers to transcribe with speaker identification.",
            ),
        ).toBeTruthy();

        // A valid count clears both: the disable and the warning.
        fireEvent.change(screen.getByPlaceholderText("e.g. 2"), {
            target: { value: "2" },
        });

        expect(
            screen.getByRole("button", {
                name: "Transcribe",
            }) as HTMLButtonElement,
        ).toHaveProperty("disabled", false);
        expect(
            screen.queryByText(
                "Enter the number of speakers to transcribe with speaker identification.",
            ),
        ).toBeNull();
    });

    it("keeps Transcribe disabled for an out-of-range count (0, or above the 64-speaker cap)", async () => {
        render(<Harness />);
        await selectAFile();

        fireEvent.click(screen.getByLabelText(/Identify speakers/));
        const input = screen.getByPlaceholderText("e.g. 2");

        fireEvent.change(input, { target: { value: "0" } });
        expect(
            screen.getByRole("button", {
                name: "Transcribe",
            }) as HTMLButtonElement,
        ).toHaveProperty("disabled", true);

        fireEvent.change(input, { target: { value: "65" } });
        expect(
            screen.getByRole("button", {
                name: "Transcribe",
            }) as HTMLButtonElement,
        ).toHaveProperty("disabled", true);

        fireEvent.change(input, { target: { value: "64" } });
        expect(
            screen.getByRole("button", {
                name: "Transcribe",
            }) as HTMLButtonElement,
        ).toHaveProperty("disabled", false);
    });

    it("leaves Transcribe enabled when the toggle is off, regardless of a leftover speaker count", async () => {
        render(<Harness initialNumSpeakersHint={undefined} />);
        await selectAFile();

        expect(
            screen.getByRole("button", {
                name: "Transcribe",
            }) as HTMLButtonElement,
        ).toHaveProperty("disabled", false);
        expect(
            screen.queryByText(
                "Enter the number of speakers to transcribe with speaker identification.",
            ),
        ).toBeNull();
    });

    /**
     * "The YouTube tile gets the same gate": the tile itself stays clickable
     * (it always did — only `isBusy` disables it), but the modal it opens must
     * not let the user submit a URL while diarization is on with no valid
     * count, for exactly the same reason Transcribe is blocked.
     */
    it("disables the YouTube modal's submit the same way, for the same reason", async () => {
        render(<Harness initialDiarizeEnabled />);

        fireEvent.click(screen.getByText("YouTube"));
        fireEvent.change(screen.getByPlaceholderText("www.example.com"), {
            target: { value: "https://www.youtube.com/watch?v=abc123" },
        });

        expect(
            screen.getByText("Prepare Audio") as HTMLButtonElement,
        ).toHaveProperty("disabled", true);
    });
});
