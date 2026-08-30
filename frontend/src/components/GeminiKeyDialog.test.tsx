// @vitest-environment jsdom
import {
    cleanup,
    fireEvent,
    render,
    screen,
    waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import GeminiKeyDialog from "./GeminiKeyDialog";

const setGeminiKey = vi.fn();
const clearGeminiKey = vi.fn();

vi.mock("../services/api", () => ({
    api: {
        setGeminiKey: (key: string) => setGeminiKey(key),
        clearGeminiKey: () => clearGeminiKey(),
    },
}));

afterEach(() => {
    cleanup();
});

describe("GeminiKeyDialog", () => {
    beforeEach(() => {
        setGeminiKey.mockReset().mockResolvedValue(undefined);
        clearGeminiKey.mockReset().mockResolvedValue(undefined);
        // jsdom has no `ResizeObserver`; Headless UI's `Dialog` (which
        // `Modal` wraps) reads it on mount. Same stub as AudioManager.test.tsx.
        vi.stubGlobal(
            "ResizeObserver",
            class {
                observe = vi.fn();
                unobserve = vi.fn();
                disconnect = vi.fn();
            },
        );
    });

    it("will not submit an empty key", () => {
        render(
            <GeminiKeyDialog
                show
                configured={false}
                onClose={vi.fn()}
                onSaved={vi.fn()}
            />,
        );
        expect(
            (
                screen.getByRole("button", {
                    name: /save key/i,
                }) as HTMLButtonElement
            ).disabled,
        ).toBe(true);
    });

    it("saves the pasted key and tells the caller", async () => {
        const onSaved = vi.fn();
        render(
            <GeminiKeyDialog
                show
                configured={false}
                onClose={vi.fn()}
                onSaved={onSaved}
            />,
        );

        fireEvent.change(
            screen.getByLabelText(/api key/i, { selector: "input" }),
            {
                target: { value: "AIzaSyTEST" },
            },
        );
        fireEvent.click(screen.getByRole("button", { name: /save key/i }));

        await waitFor(() =>
            expect(setGeminiKey).toHaveBeenCalledWith("AIzaSyTEST"),
        );
        await waitFor(() => expect(onSaved).toHaveBeenCalledWith(true));
    });

    /**
     * Validation happens at paste time (Rust spends one GET /v1beta/models),
     * so a bad key fails HERE rather than twenty minutes into the user's first
     * transcription. The message must survive to the screen.
     */
    it("shows the backend's rejection instead of a generic failure", async () => {
        setGeminiKey.mockRejectedValue(
            new Error("Gemini rejected this API key"),
        );
        render(
            <GeminiKeyDialog
                show
                configured={false}
                onClose={vi.fn()}
                onSaved={vi.fn()}
            />,
        );

        fireEvent.change(
            screen.getByLabelText(/api key/i, { selector: "input" }),
            {
                target: { value: "bad" },
            },
        );
        fireEvent.click(screen.getByRole("button", { name: /save key/i }));

        expect(await screen.findByText(/rejected this API key/i)).toBeTruthy();
    });

    it("offers removal only when a key is already stored", () => {
        const { rerender } = render(
            <GeminiKeyDialog
                show
                configured={false}
                onClose={vi.fn()}
                onSaved={vi.fn()}
            />,
        );
        expect(
            screen.queryByRole("button", { name: /remove key/i }),
        ).toBeNull();

        rerender(
            <GeminiKeyDialog
                show
                configured
                onClose={vi.fn()}
                onSaved={vi.fn()}
            />,
        );
        expect(
            screen.getByRole("button", { name: /remove key/i }),
        ).toBeTruthy();
    });

    it("never renders the stored key back to the user", () => {
        render(
            <GeminiKeyDialog
                show
                configured
                onClose={vi.fn()}
                onSaved={vi.fn()}
            />,
        );
        expect(
            (
                screen.getByLabelText(/api key/i, {
                    selector: "input",
                }) as HTMLInputElement
            ).value,
        ).toBe("");
    });
});
