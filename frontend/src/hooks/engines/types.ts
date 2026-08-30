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

/**
 * What a run is about to cost, put to the user before it is spent.
 *
 * Deliberately not named for Gemini. It describes a property of a RUN — this
 * much audio, this many requests, this much money — and any future paid engine
 * asks the same question. The local engine never asks it at all.
 */
export interface CostConfirmation {
    durationSecs: number;
    chunkCount: number;
    estimatedUsd: number;
    /**
     * Whether speaker labels are possible at this length. Carried here because
     * the 30-minute request cap costs a long run its diarization, and nothing
     * else in the UI mentions that until after the money is gone.
     */
    diarizationAvailable: boolean;
}

/**
 * Thrown by an engine whose cost the user declined.
 *
 * A constant rather than prose matched at the call site, for the same reason
 * `LOCAL_ENGINE_NO_SPEAKERS` is one: the hook has to tell "the user said no"
 * apart from "the run failed" — one goes quietly back to idle, the other shows
 * an error — and a reworded string would silently turn the first into the
 * second.
 */
export const RUN_DECLINED = "the user declined this run's cost";

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
    /**
     * Ask the user to approve what this run will cost, and wait for the answer.
     *
     * Resolves `true` to proceed, `false` to abandon before anything is spent.
     * An engine that costs nothing never calls it — the local engine does not.
     *
     * That this await lives INSIDE `run()` is deliberate rather than
     * incidental. `transcribePreparedJob` documents that its run token cannot
     * move across an await whose promise settles synchronously, and a real
     * user-interaction await would break that reasoning wherever it were
     * placed. Here it is already covered: the hook re-checks the token after
     * `engines[config.engine].run()` returns, which is one of the guarded
     * resume points it enumerates.
     */
    confirmCost: (details: CostConfirmation) => Promise<boolean>;
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
