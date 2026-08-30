// @vitest-environment jsdom
/**
 * A regression guard for the Transcribe/YouTube gating JSX in `AudioManager`
 * — properties a headless `tsc`/lint pass cannot see, since they live in
 * `enabled`/`disabled` expressions on rendered elements, not in types.
 * `Transcript.test.tsx` records that a prior task on this branch shipped a
 * dead button behind a passing headless check; this is the same class of
 * regression, on the other component.
 *
 * These two tests used to also cover a since-deleted diarization toggle,
 * which could leave a stale/leftover value gating these same buttons. That
 * toggle is gone now, along with everything that could set such a value —
 * but the underlying property these tests exist to pin (a ready file, or a
 * valid YouTube URL, is never blocked by something unrelated) still holds,
 * and is still worth a real render to prove.
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

/**
 * Renders `AudioManager` with a fully-stubbed `Transcriber`, overridden a
 * field at a time via a single partial object — not positional arguments —
 * so a later caller can add new override fields (an engine selector, a
 * key-configured flag, a selected path) without touching every existing
 * call site.
 */
function renderAudioManager(overrides: Partial<Transcriber> = {}) {
    const transcriber: Transcriber = {
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
        engine: "local",
        setEngine: vi.fn(),
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
        speakerNames: {},
        renameSpeaker: vi.fn(),
        speakerOutcome: null,
        ...overrides,
    };

    return render(<AudioManager transcriber={transcriber} />);
}

async function selectAFile() {
    mocks.open.mockResolvedValue("/tmp/lecture.mp3");
    fireEvent.click(screen.getByText("From file"));
    // `open()` resolves asynchronously; `findByText` polls until the
    // Transcribe button the resolution reveals actually appears.
    return screen.findByRole("button", { name: "Transcribe" });
}

describe("AudioManager: Transcribe / YouTube submit stay enabled", () => {
    it("leaves Transcribe enabled once a file is selected and the model is available", async () => {
        renderAudioManager();
        await selectAFile();

        expect(
            screen.getByRole("button", {
                name: "Transcribe",
            }) as HTMLButtonElement,
        ).toHaveProperty("disabled", false);
    });

    it("leaves the YouTube modal's submit enabled once a valid URL is entered", () => {
        renderAudioManager();

        fireEvent.click(screen.getByText("YouTube"));
        fireEvent.change(screen.getByPlaceholderText("www.example.com"), {
            target: { value: "https://www.youtube.com/watch?v=abc123" },
        });

        expect(
            screen.getByText("Prepare Audio") as HTMLButtonElement,
        ).toHaveProperty("disabled", false);
    });
});
