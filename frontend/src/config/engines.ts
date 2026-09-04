/**
 * Which transcription engine runs a job.
 *
 * This is a SEPARATE axis from `MODEL_PRESETS`, deliberately. Every field on a
 * preset (`webgpuOnly`, `englishOnly`, and everything `chooseAutoPreset` reads)
 * is a statement about the local machine's hardware. A cloud provider has no
 * answer to any of them, and burying "your audio leaves this machine" in the
 * model dropdown would be the wrong place to say it.
 */
export type EngineId = "local" | "gemini";

/** The only Gemini model this app calls. Pinned; not user-selectable. */
export const GEMINI_MODEL_ID = "gemini-3.5-transcribe";

export interface EngineDescriptor {
    id: string;
    label: string;
    description: string;
    /** Blocks the run before it starts when no credential is stored. */
    requiresKey: boolean;
    supportsTranslate: boolean;
    supportsLanguageChoice: boolean;
    /** Whether partial text arrives mid-run. Gemini: no token stream. */
    supportsLivePreview: boolean;
    /** Fixed for a cloud engine; resolved from `MODEL_PRESETS` for local. */
    modelId: string | null;
}

export const ENGINES = [
    {
        id: "local",
        label: "On-device",
        description: "Private — nothing leaves this machine",
        requiresKey: false,
        supportsTranslate: true,
        supportsLanguageChoice: true,
        supportsLivePreview: true,
        modelId: null,
    },
    {
        id: "gemini",
        label: "Google Gemini",
        description:
            "Cloud — your audio is uploaded to Google. Roughly $0.005 per minute.",
        requiresKey: true,
        supportsTranslate: false,
        supportsLanguageChoice: false,
        supportsLivePreview: false,
        modelId: GEMINI_MODEL_ID,
    },
] as const satisfies readonly EngineDescriptor[];

/**
 * Never returns `undefined` — same reasoning as `modelIdForPreset`. A silent
 * `undefined` here would disable the Task control for the LOCAL engine (a
 * missing `supportsTranslate` is falsy) with nothing to say why.
 */
export function engineById(id: EngineId): EngineDescriptor {
    const engine = ENGINES.find((candidate) => candidate.id === id);
    if (!engine) {
        throw new Error(
            `no engine "${id}" in ENGINES — an engine was removed or renamed ` +
                `without updating the code that names it.`,
        );
    }
    return engine;
}

/**
 * Whether a transcript persisted under `modelId` could have got its speaker
 * labels from a diarizer this build would still stand behind.
 *
 * This is a question about PROVENANCE, not capability: it is asked of a row
 * already in the database, to decide whether the `speaker` baked into it means
 * anything. 1.1.0 shipped a local sherpa-onnx diarizer and 1.2.0 deleted it for
 * mislabelling a single narrator as dozens of speakers — but deleting the code
 * did not delete its OUTPUT, which is still sitting in every transcript it
 * touched. Those labels have to be recognised on the way out.
 *
 * Derived from `ENGINES` rather than written as `modelId === GEMINI_MODEL_ID`:
 * a cloud engine PINS its model id here, and the local engine's is `null`
 * because it is chosen per run from `MODEL_PRESETS`. So "this id names an
 * engine's pinned model" is exactly "a cloud engine wrote this", and a second
 * cloud engine added to the array has its labels honoured with no change here.
 *
 * Deliberately NOT a lookup against `MODEL_PRESETS`. The local model ids have
 * been renamed once already, so a transcript from before that rename names a
 * preset that no longer exists — and matching on the preset list would call it
 * "not local" and trust its labels, which is the exact case this exists to
 * catch. Anything that is not a known cloud model is local.
 */
export function modelDiarizes(modelId: string): boolean {
    return ENGINES.some(
        (engine) => engine.modelId !== null && engine.modelId === modelId,
    );
}
