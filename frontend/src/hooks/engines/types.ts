import type { EngineId } from "../../config/engines";
import type { ResolvedModelConfig } from "../../config/transcription";
import type { WorkerTranscript } from "../../lib/workerTranscript";
import type { Job, SpeakerOutcome } from "../../services/types";

export interface EngineProgress {
    /** 0..1, or `null` when the phase has no determinate fraction. */
    fraction: number | null;
    /** User-facing and already specific. Shown verbatim. */
    status: string;
}

/**
 * Whether the engine identified speakers — and if not, WHY NOT.
 *
 * Re-exported from `services/types` rather than redeclared: Task 12's
 * `GeminiTranscriptionResult.speakers` is the SAME union, and two structurally
 * identical declarations would compile happily while drifting apart.
 * `services` is the lower layer, so it owns the type.
 */
export type EngineSpeakers = SpeakerOutcome;
export type { SpeakerOutcome };

/**
 * The local engine's standing reason for having no speakers.
 *
 * Exported so the UI can recognise it and stay quiet (Task 14) instead of
 * string-matching prose that someone will later reword.
 */
export const LOCAL_ENGINE_NO_SPEAKERS =
    "the on-device engine does not identify speakers";

export interface EngineResult {
    transcript: WorkerTranscript;
    speakers: EngineSpeakers;
    /**
     * The audio's true duration. `segmentsForPersistence` uses it to close the
     * final segment; a wrong value silently truncates the transcript.
     */
    audioDuration: number;
}

export interface EngineRunArgs {
    job: Job;
    config: ResolvedModelConfig;
    runId: number;
    onProgress: (progress: EngineProgress) => void;
    /**
     * Live preview, for an engine that streams outside the hook's own worker
     * message handler. NEITHER engine calls this today: the LOCAL engine
     * leaves partials entirely to the hook's existing `useWorker` handler,
     * which already paints them straight from the worker's `update` messages
     * (see `useTranscriber.ts`), so `localEngine.ts` never references
     * `onPartial` at all. The GEMINI engine will not call it either — the
     * Interactions request is one opaque round trip with no token stream,
     * which is what `EngineDescriptor.supportsLivePreview` tells the UI in
     * advance. So this callback has no caller yet; it exists for a future
     * engine that streams over a channel the hook does not already own.
     */
    onPartial: (transcript: WorkerTranscript) => void;
}

export interface TranscriptionEngine {
    id: EngineId;
    run(args: EngineRunArgs): Promise<EngineResult>;
    /**
     * Stop a run in flight. The local engine is a no-op: the hook's `claimRun`
     * already terminates the worker, which is the only thing that stops a
     * transformers.js inference. The Gemini engine tells Rust to abort the
     * request and reap its uploads.
     */
    abandon(jobId: string): void;
}
