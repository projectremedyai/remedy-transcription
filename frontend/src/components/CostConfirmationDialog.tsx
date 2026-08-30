import Modal from "./modal/Modal";
import type { CostConfirmation } from "../hooks/engines/types";

interface Props {
    /** The question to ask, or `null` when there is none. */
    pending: CostConfirmation | null;
    onApprove: () => void;
    onDecline: () => void;
}

/** `21600` -> `"6 hours"`. The user dropped a film, not a number of seconds. */
function humanDuration(seconds: number): string {
    if (seconds >= 3600) {
        const hours = seconds / 3600;
        // `6`, not `6.0`; `1.5`, not `1.50`.
        const rounded = Math.round(hours * 10) / 10;
        return `${rounded} ${rounded === 1 ? "hour" : "hours"}`;
    }
    const minutes = Math.max(1, Math.round(seconds / 60));
    return `${minutes} ${minutes === 1 ? "minute" : "minutes"}`;
}

/**
 * The last thing between a mis-dropped six-hour file and a real bill.
 *
 * `MAX_DURATION_HOURS` has only ever guarded the YouTube path, and the local
 * path was uncapped for a good reason: on-device Whisper is free at any length.
 * The guard therefore belongs to the ENGINE rather than the source, which is
 * why this is reached from `geminiEngine.run` and never from the local one.
 *
 * DECLINING IS THE DEFAULT. Esc and a backdrop click both land on `onDecline`
 * through `Modal`'s `onClose`, so every way of dismissing this without reading
 * it spends nothing. Only the explicit button spends money.
 */
export default function CostConfirmationDialog({
    pending,
    onApprove,
    onDecline,
}: Props) {
    // Genuinely absent rather than merely hidden: a dialog kept mounted leaves
    // its buttons in the accessibility tree for every short file that never
    // needed to ask anything.
    if (!pending) {
        return null;
    }

    return (
        <Modal
            show
            onClose={onDecline}
            onSubmit={onApprove}
            submitText='Transcribe anyway'
            title='This is a long transcription'
            content={
                <div className='flex flex-col gap-3 text-sm text-slate-600'>
                    <p>
                        This audio is {humanDuration(pending.durationSecs)}{" "}
                        long. Gemini will be sent {pending.chunkCount} separate
                        requests, at an estimated cost of{" "}
                        <strong>${pending.estimatedUsd.toFixed(2)}</strong>.
                    </p>
                    {!pending.diarizationAvailable && (
                        <p className='text-amber-700'>
                            Because it has to be split, this run will not
                            identify speakers — speaker identities cannot be
                            matched across parts.
                        </p>
                    )}
                    <p className='text-xs text-slate-500'>
                        The estimate is approximate and is not a quote from
                        Google.
                    </p>
                </div>
            }
        />
    );
}
