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
import {
    render,
    screen,
    cleanup,
    fireEvent,
    act,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LANGUAGE_OPTIONS, MODEL_PRESETS } from "../config/transcription";
import type { Transcriber } from "../hooks/useTranscriber";

const mocks = vi.hoisted(() => ({
    open: vi.fn(),
    geminiKeyStatus: vi.fn(),
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

// `AudioManager` reads `api.geminiKeyStatus` directly (not through the
// `Transcriber` prop), and mounts the real `GeminiKeyDialog`, which reads
// `api.setGeminiKey`/`api.clearGeminiKey`. Neither Save nor Remove is
// exercised here, so both are inert stubs — only `geminiKeyStatus` needs a
// per-test resolved value.
vi.mock("../services/api", () => ({
    api: {
        geminiKeyStatus: (...args: unknown[]) => mocks.geminiKeyStatus(...args),
        setGeminiKey: vi.fn().mockResolvedValue(undefined),
        clearGeminiKey: vi.fn().mockResolvedValue(undefined),
    },
}));

import { AudioManager } from "./AudioManager";

afterEach(() => {
    cleanup();
});

beforeEach(() => {
    mocks.open.mockReset();
    mocks.geminiKeyStatus.mockReset();
    // jsdom has no `ResizeObserver`; Headless UI's `Dialog` (used by the
    // YouTube modal and the Gemini key dialog) reads it on mount.
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
 *
 * `geminiKeyConfigured` and `selectedPath` are not `Transcriber` fields —
 * the former drives the `api.geminiKeyStatus()` mock `AudioManager` fetches
 * for itself, the latter seeds the component's own internal `selectedPath`
 * state (there is no prop for it) by driving the same file-picker path
 * `selectAFile` below already used.
 *
 * This is `async` and must be awaited by any caller whose assertions depend
 * on `geminiKeyConfigured`: the value only reaches the screen after the
 * mocked promise's `.then()` runs, which — being a real microtask — cannot
 * have happened yet by the statement right after a bare `render()`. The
 * `act(() => Promise.resolve())` below is what flushes it.
 */
async function renderAudioManager(
    overrides: Partial<Transcriber> & {
        geminiKeyConfigured?: boolean;
        selectedPath?: string;
    } = {},
) {
    const {
        geminiKeyConfigured = false,
        selectedPath,
        ...transcriberOverrides
    } = overrides;
    mocks.geminiKeyStatus.mockResolvedValue(geminiKeyConfigured);

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
        ...transcriberOverrides,
    };

    const result = render(<AudioManager transcriber={transcriber} />);
    // Flushes the `api.geminiKeyStatus()` effect's pending `.then()` — see
    // the doc comment above.
    await act(() => Promise.resolve());

    if (selectedPath) {
        mocks.open.mockResolvedValueOnce(selectedPath);
        fireEvent.click(screen.getByText("From file"));
        // `open()` resolves asynchronously; `findByRole` polls until the
        // Transcribe button the resolution reveals actually appears.
        await screen.findByRole("button", { name: /transcribe/i });
    }

    return result;
}

async function selectAFile() {
    mocks.open.mockResolvedValue("/tmp/lecture.mp3");
    fireEvent.click(screen.getByText("From file"));
    // `open()` resolves asynchronously; `findByText` polls until the
    // Transcribe button the resolution reveals actually appears.
    return screen.findByRole("button", { name: "Transcribe" });
}

// `@testing-library/jest-dom` (the usual home of `toBeDisabled()`) is not a
// dependency of this project — every other test in this file reads the DOM
// `disabled` property directly instead, and this keeps that same style.
function isDisabled(element: HTMLElement): boolean {
    return (element as HTMLButtonElement | HTMLSelectElement).disabled;
}

describe("AudioManager: Transcribe / YouTube submit stay enabled", () => {
    it("leaves Transcribe enabled once a file is selected and the model is available", async () => {
        await renderAudioManager();
        await selectAFile();

        expect(
            screen.getByRole("button", {
                name: "Transcribe",
            }) as HTMLButtonElement,
        ).toHaveProperty("disabled", false);
    });

    it("leaves the YouTube modal's submit enabled once a valid URL is entered", async () => {
        await renderAudioManager();

        fireEvent.click(screen.getByText("YouTube"));
        fireEvent.change(screen.getByPlaceholderText("www.example.com"), {
            target: { value: "https://www.youtube.com/watch?v=abc123" },
        });

        expect(
            screen.getByText("Prepare Audio") as HTMLButtonElement,
        ).toHaveProperty("disabled", false);
    });
});

describe("the engine selector", () => {
    it("offers both engines", async () => {
        await renderAudioManager();
        expect(screen.getByLabelText(/on-device/i)).toBeTruthy();
        expect(screen.getByLabelText(/google gemini/i)).toBeTruthy();
    });

    /**
     * gemini-3.5-transcribe has no translation mode and no language-forcing
     * parameter. Leaving these enabled would let the user set a value that
     * silently does nothing.
     */
    it("disables Task and Language on the gemini engine, with a reason", async () => {
        await renderAudioManager({
            engine: "gemini",
            geminiKeyConfigured: true,
        });
        expect(isDisabled(screen.getByLabelText(/^task$/i))).toBe(true);
        expect(isDisabled(screen.getByLabelText(/^language$/i))).toBe(true);
        expect(
            screen.getByText(/detects the language automatically/i),
        ).toBeTruthy();
    });

    it("leaves Task and Language enabled on the local engine", async () => {
        await renderAudioManager({ engine: "local" });
        expect(isDisabled(screen.getByLabelText(/^task$/i))).toBe(false);
        expect(isDisabled(screen.getByLabelText(/^language$/i))).toBe(false);
    });

    it("shows the fixed Gemini model instead of the local preset dropdown", async () => {
        await renderAudioManager({
            engine: "gemini",
            geminiKeyConfigured: true,
        });
        expect(screen.queryByLabelText(/^model$/i)).toBeNull();
        expect(screen.getByText(/Gemini 3\.5 Transcribe/)).toBeTruthy();
    });

    /**
     * The no-key gate. Starting a run that can only fail at the first API call
     * wastes an ffmpeg pass and tells the user nothing useful.
     */
    it("blocks transcription and prompts for a key when none is stored", async () => {
        await renderAudioManager({
            engine: "gemini",
            geminiKeyConfigured: false,
            selectedPath: "/a.mp4",
        });
        expect(
            isDisabled(screen.getByRole("button", { name: /transcribe/i })),
        ).toBe(true);
        expect(screen.getByRole("button", { name: /add key/i })).toBeTruthy();
    });

    it("allows transcription once a key is stored", async () => {
        await renderAudioManager({
            engine: "gemini",
            geminiKeyConfigured: true,
            selectedPath: "/a.mp4",
        });
        expect(
            isDisabled(screen.getByRole("button", { name: /transcribe/i })),
        ).toBe(false);
    });

    /** The privacy trade must be stated where the choice is made. */
    it("warns that audio leaves the machine on the gemini engine", async () => {
        await renderAudioManager({
            engine: "gemini",
            geminiKeyConfigured: true,
        });
        expect(screen.getByText(/uploaded to Google/i)).toBeTruthy();
    });
});

/**
 * The engine seam's other half. `hooks/engines/types.ts` declares
 * `EngineProgress.status` "user-facing and already specific -- shown
 * verbatim", `useTranscriber` stores it straight onto `status`, and this
 * component is where it either reaches the screen or does not. It did not:
 * `getStatusMessage` was a closed `switch` over the backend job statuses with
 * `default: return ""`, and `progressValue` returned the fraction only for
 * `downloading`/`extracting`. A Gemini run therefore had NEITHER text nor bar
 * — the exact opposite of the spec's "behind a determinate progress bar".
 *
 * These drive the `transcriber` stub with the strings `geminiEngine.ts`
 * actually emits, which is the only place the two halves meet.
 */
describe("a cloud engine's own progress reaches the screen", () => {
    it("shows an engine-supplied status verbatim instead of blank text", async () => {
        await renderAudioManager({
            engine: "gemini",
            geminiKeyConfigured: true,
            isBusy: true,
            status: "Uploading to Google and transcribing (part 2 of 3)...",
            progress: 1 / 3,
        });

        expect(
            screen.getByText(
                "Uploading to Google and transcribing (part 2 of 3)...",
            ),
        ).toBeTruthy();
    });

    /**
     * The single-chunk case, which was worse than blank: Gemini used to emit a
     * bare "transcribing", which MATCHED the local worker's own status and
     * rendered "Transcribing in your browser..." for a run whose audio was
     * being uploaded to Google. Wrong on precisely the axis the spec insists
     * be stated plainly.
     */
    it("never claims a cloud run is happening in the browser", async () => {
        await renderAudioManager({
            engine: "gemini",
            geminiKeyConfigured: true,
            isBusy: true,
            status: "Uploading to Google and transcribing...",
        });

        expect(screen.queryByText(/in your browser/i)).toBeNull();
        expect(
            screen.getByText("Uploading to Google and transcribing..."),
        ).toBeTruthy();
    });

    it("drives the progress bar from the engine's fraction", async () => {
        await renderAudioManager({
            engine: "gemini",
            geminiKeyConfigured: true,
            isBusy: true,
            status: "Uploading to Google and transcribing (part 3 of 5)...",
            progress: 0.4,
        });

        expect(
            (screen.getByTestId("overall-progress") as HTMLElement).style.width,
        ).toBe("40%");
    });

    /** The bar has to MOVE, not merely be non-zero once. */
    it("moves the bar as the engine's fraction advances", async () => {
        await renderAudioManager({
            engine: "gemini",
            geminiKeyConfigured: true,
            isBusy: true,
            status: "Preparing the audio (part 1 of 5)...",
            progress: 0,
        });
        expect(
            (screen.getByTestId("overall-progress") as HTMLElement).style.width,
        ).toBe("0%");

        cleanup();
        await renderAudioManager({
            engine: "gemini",
            geminiKeyConfigured: true,
            isBusy: true,
            status: "Assembling the transcript...",
            progress: 1,
        });
        expect(
            (screen.getByTestId("overall-progress") as HTMLElement).style.width,
        ).toBe("100%");
    });

    /**
     * The other side of the same join: a kebab-case status this app sets for
     * itself must still be turned into prose, never echoed raw.
     */
    it("still renders this app's own job statuses as prose", async () => {
        await renderAudioManager({ isBusy: true, status: "extracting" });
        expect(screen.getByText("Extracting audio...")).toBeTruthy();
        expect(screen.queryByText("extracting")).toBeNull();
    });

    it("still says 'in your browser' for the local worker's own status", async () => {
        await renderAudioManager({
            engine: "local",
            isBusy: true,
            status: "transcribing",
        });
        expect(
            screen.getByText("Transcribing in your browser..."),
        ).toBeTruthy();
    });
});

/**
 * The YouTube modal's submit is gated on `engineReady` — but the ONLY thing
 * explaining that gate ("Gemini needs an API key / Add key") lives in the
 * engine row, which is behind the modal's own overlay. So the user pasted a
 * valid URL, found "Prepare Audio" inert, and had nothing on screen to read.
 * Deleting the gate entirely left all 188 tests green, which is what this
 * test exists to stop happening again; `Transcript.test.tsx` records the last
 * time a permanently-dead control shipped behind a fully-green headless suite.
 */
describe("the YouTube modal says why its submit is dead", () => {
    function openTheModalWithAValidUrl() {
        fireEvent.click(screen.getByText("YouTube"));
        fireEvent.change(screen.getByPlaceholderText("www.example.com"), {
            target: { value: "https://www.youtube.com/watch?v=abc123" },
        });
    }

    it("disables Prepare Audio and states the reason when Gemini has no key", async () => {
        await renderAudioManager({
            engine: "gemini",
            geminiKeyConfigured: false,
        });
        openTheModalWithAValidUrl();

        expect(isDisabled(screen.getByText("Prepare Audio"))).toBe(true);
        // Matched on the part that is unique to the modal: the engine row's
        // own "Gemini needs an API key" is in the document too, behind the
        // overlay, which is the whole problem this copy exists to solve.
        expect(screen.getByText(/behind this dialog/i)).toBeTruthy();
    });

    it("shows no reason at all once the key is stored", async () => {
        await renderAudioManager({
            engine: "gemini",
            geminiKeyConfigured: true,
        });
        openTheModalWithAValidUrl();

        expect(isDisabled(screen.getByText("Prepare Audio"))).toBe(false);
        expect(screen.queryByText(/behind this dialog/i)).toBeNull();
    });
});
