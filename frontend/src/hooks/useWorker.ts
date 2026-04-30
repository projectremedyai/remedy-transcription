import { useState } from "react";

export interface MessageEventHandler {
    (event: MessageEvent): void;
}

export function useWorker(messageEventHandler: MessageEventHandler): Worker {
    const [worker] = useState(() => {
        const nextWorker = new Worker(
            new URL("../workers/transcription.worker.ts", import.meta.url),
            { type: "module" },
        );
        nextWorker.addEventListener("message", messageEventHandler);
        return nextWorker;
    });

    return worker;
}
