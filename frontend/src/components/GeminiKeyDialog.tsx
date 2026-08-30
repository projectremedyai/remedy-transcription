import { useState } from "react";

import Modal from "./modal/Modal";
import { api } from "../services/api";

interface Props {
    show: boolean;
    /** Whether a key is already in the keychain. Drives the Remove affordance. */
    configured: boolean;
    onClose: () => void;
    /** Called with the new configured-state after a save or a removal. */
    onSaved: (configured: boolean) => void;
}

/**
 * Paste, validate and clear the Gemini API key.
 *
 * The input starts EMPTY even when a key is stored, and is never prefilled:
 * there is no command that returns the key, by design. "Change" means "replace
 * with a new one", not "edit the old one".
 */
export default function GeminiKeyDialog({
    show,
    configured,
    onClose,
    onSaved,
}: Props) {
    const [key, setKey] = useState("");
    const [error, setError] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);

    const finish = (nextConfigured: boolean) => {
        setKey("");
        setError(null);
        setBusy(false);
        onSaved(nextConfigured);
        onClose();
    };

    const save = () => {
        setBusy(true);
        setError(null);
        api.setGeminiKey(key.trim())
            .then(() => finish(true))
            .catch((saveError: unknown) => {
                setBusy(false);
                setError(
                    saveError instanceof Error
                        ? saveError.message
                        : String(saveError),
                );
            });
    };

    const remove = () => {
        setBusy(true);
        setError(null);
        api.clearGeminiKey()
            .then(() => finish(false))
            .catch((clearError: unknown) => {
                setBusy(false);
                setError(
                    clearError instanceof Error
                        ? clearError.message
                        : String(clearError),
                );
            });
    };

    return (
        <Modal
            show={show}
            onClose={onClose}
            onSubmit={save}
            submitText={busy ? "Saving…" : "Save key"}
            submitEnabled={key.trim().length > 0 && !busy}
            title='Google Gemini API key'
            content={
                <div className='flex flex-col gap-3 text-sm text-slate-600'>
                    <p>
                        Get a key from Google AI Studio. It is stored in your
                        operating system&apos;s keychain, never in this
                        app&apos;s database, and never shown again once saved.
                    </p>
                    <label className='flex flex-col'>
                        <span className='text-slate-600'>API key</span>
                        <input
                            type='password'
                            value={key}
                            autoComplete='off'
                            spellCheck={false}
                            placeholder={
                                configured
                                    ? "A key is saved — paste a new one to replace it"
                                    : "AIza…"
                            }
                            onChange={(event) => setKey(event.target.value)}
                            className='mt-1 rounded-lg border border-slate-300 px-3 py-2 font-mono'
                        />
                    </label>
                    <p className='text-xs text-amber-700'>
                        Audio you transcribe with Gemini is uploaded to Google
                        and deleted again as soon as the transcript comes back.
                    </p>
                    {error && (
                        <p className='text-xs text-red-600' role='alert'>
                            {error}
                        </p>
                    )}
                    {configured && (
                        <button
                            type='button'
                            onClick={remove}
                            disabled={busy}
                            className='self-start text-xs text-slate-500 underline underline-offset-2 hover:text-red-600'
                        >
                            Remove key
                        </button>
                    )}
                </div>
            }
        />
    );
}
