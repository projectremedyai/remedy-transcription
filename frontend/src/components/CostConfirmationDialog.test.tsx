// @vitest-environment jsdom
/**
 * The dialog that stands between a mis-dropped six-hour file and a real bill.
 *
 * `MAX_DURATION_HOURS` only ever guarded the YouTube path, so a long LOCAL file
 * reached Gemini with nothing in front of it. What the user needs to see before
 * committing is the money, and the OTHER cost nothing else in the UI mentions
 * until afterwards: a chunked run gets no speaker labels.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import CostConfirmationDialog from "./CostConfirmationDialog";

afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
});

const sixHours = {
    durationSecs: 21600,
    chunkCount: 15,
    estimatedUsd: 1.8,
    diarizationAvailable: false,
};

describe("the cost confirmation dialog", () => {
    beforeEach(() => {
        // jsdom has no `ResizeObserver`; Headless UI's `Dialog` (which `Modal`
        // wraps) reads it on mount. Same stub as GeminiKeyDialog.test.tsx.
        vi.stubGlobal(
            "ResizeObserver",
            class {
                observe = vi.fn();
                unobserve = vi.fn();
                disconnect = vi.fn();
            },
        );
    });

    it("states the money before anything is spent", () => {
        render(
            <CostConfirmationDialog
                pending={sixHours}
                onApprove={() => undefined}
                onDecline={() => undefined}
            />,
        );
        expect(screen.getByText(/\$1\.80/)).toBeTruthy();
    });

    /** Hours, not "21600 seconds". The user dropped a film, not a number. */
    it("states the length in a unit a human dropped on the window", () => {
        render(
            <CostConfirmationDialog
                pending={sixHours}
                onApprove={() => undefined}
                onDecline={() => undefined}
            />,
        );
        expect(screen.getByText(/6 hours/)).toBeTruthy();
    });

    /**
     * The second cost. Splitting is what makes speaker labels impossible, and
     * today that only surfaces AFTER the money is gone, as a
     * `Speakers::Unavailable` reason on a transcript already paid for.
     */
    it("warns that a split run gets no speaker labels", () => {
        render(
            <CostConfirmationDialog
                pending={sixHours}
                onApprove={() => undefined}
                onDecline={() => undefined}
            />,
        );
        expect(screen.getByText(/speaker/i)).toBeTruthy();
    });

    it("approves and declines through its own callbacks", () => {
        const onApprove = vi.fn();
        const onDecline = vi.fn();
        render(
            <CostConfirmationDialog
                pending={sixHours}
                onApprove={onApprove}
                onDecline={onDecline}
            />,
        );

        fireEvent.click(screen.getByText(/Transcribe anyway/i));
        expect(onApprove).toHaveBeenCalledTimes(1);

        fireEvent.click(screen.getByText(/^Close$/i));
        expect(onDecline).toHaveBeenCalledTimes(1);
    });

    /**
     * A null `pending` is the ordinary state — a short file never asks. The
     * dialog must be genuinely absent, not merely invisible, or its buttons stay
     * reachable to the accessibility tree.
     */
    it("renders nothing at all when there is no question to ask", () => {
        const { container } = render(
            <CostConfirmationDialog
                pending={null}
                onApprove={() => undefined}
                onDecline={() => undefined}
            />,
        );
        expect(container.textContent).toBe("");
    });
});
