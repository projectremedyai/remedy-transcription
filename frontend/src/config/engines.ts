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
