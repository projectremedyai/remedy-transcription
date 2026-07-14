import { useCallback, useEffect, useMemo, useRef } from "react";

export interface MessageEventHandler {
    (event: MessageEvent): void;
}

export interface WorkerHandle {
    /** Post to the live worker, standing one up first if there is not one. */
    postMessage: (message: unknown) => void;
    /**
     * Kill the worker in flight. The next `postMessage` stands a fresh one up.
     *
     * This is the ONLY way to stop a transformers.js inference that has already
     * started. `transcriber(audio, ...)` is one long `await` inside the worker's
     * message handler: there is no abort signal to raise, no flag the decode loop
     * checks, and posting a second message does not interrupt the first — it just
     * starts a SECOND handler running concurrently against the same shared
     * `PipelineFactory.instance`. Ignoring the doomed run's messages (which the
     * `runId` on every message does) stops its output reaching the UI, but it does
     * not stop Whisper: it keeps every core pinned for however many minutes the
     * audio needs, and it keeps its `Float32Array` alive. Terminating does.
     */
    restart: () => void;
}

function spawnWorker(onMessage: (event: MessageEvent) => void): Worker {
    const worker = new Worker(
        new URL("../workers/transcription.worker.ts", import.meta.url),
        { type: "module" },
    );
    worker.addEventListener("message", onMessage);
    return worker;
}

/**
 * A worker that can be REPLACED, not a worker that lives as long as the app.
 *
 * The worker is created lazily on the first `postMessage` and re-created after
 * every `restart`, so the caller can throw away a worker that is stuck inside an
 * inference it no longer wants. It is deliberately not created at mount: the
 * worker does nothing at all until it is asked to transcribe.
 *
 * The message handler is read from a ref, so a fresh worker always dispatches to
 * the CURRENT render's handler and the caller does not have to keep one stable.
 */
export function useWorker(
    messageEventHandler: MessageEventHandler,
): WorkerHandle {
    const handlerRef = useRef<MessageEventHandler>(messageEventHandler);
    handlerRef.current = messageEventHandler;

    const workerRef = useRef<Worker | null>(null);

    const dispatch = useCallback((event: MessageEvent) => {
        handlerRef.current(event);
    }, []);

    const postMessage = useCallback(
        (message: unknown) => {
            if (workerRef.current === null) {
                workerRef.current = spawnWorker(dispatch);
            }
            workerRef.current.postMessage(message);
        },
        [dispatch],
    );

    const restart = useCallback(() => {
        workerRef.current?.terminate();
        workerRef.current = null;
    }, []);

    useEffect(
        () => () => {
            workerRef.current?.terminate();
            workerRef.current = null;
        },
        [],
    );

    return useMemo(() => ({ postMessage, restart }), [postMessage, restart]);
}
