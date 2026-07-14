export type TaskMode = "transcribe" | "translate";
export type DeviceMode = "webgpu" | "wasm";

export interface BrowserCaps {
    secureContext: boolean;
    canUseWebGPU: boolean;
    shaderF16: boolean;
    deviceMemoryGiB: number | null;
    logicalCores: number;
}

/**
 * The SHAPE of a preset — the contract `MODEL_PRESETS` is checked against.
 *
 * `id` is a plain `string` here on purpose: `ModelPresetId` is derived FROM
 * `MODEL_PRESETS`, so typing it as `ModelPresetId` would make the array's type
 * reference itself. The strict union lives on `ModelPresetId`; this exists only to
 * catch a malformed entry.
 */
export interface ModelPreset {
    id: string;
    label: string;
    /** A HuggingFace repo id, or the `"__auto__"` sentinel. */
    modelId: string;
    description: string;
    webgpuOnly: boolean;
    englishOnly: boolean;
}

export interface ResolvedModelConfig {
    presetId: ModelPresetId;
    presetLabel: string;
    modelId: string;
    device: DeviceMode;
    task: TaskMode;
    language: string;
}

/**
 * The single source of truth for which models exist. The Rust backend parses the
 * `modelId:` lines of this very array at compile time (`commands.rs`,
 * `include_str!`), so this list and `list_models` cannot hold different models.
 *
 * `as const satisfies` is load-bearing, not style. `ModelPresetId` is DERIVED from
 * it below, so deleting or renaming a preset here is a COMPILE ERROR at every site
 * that names it — including the worker's WebGPU dtype override, which is keyed on
 * the `"quality"` preset. Written as a hand-maintained union, the id could outlive
 * the preset: the lookup would quietly return `undefined`, the dtype override
 * would stop applying, and large-v3-turbo would load an fp32 encoder on WebGPU
 * with nothing to say so — the same silent-drift class as the dead Transcribe
 * button, one level over.
 */
export const MODEL_PRESETS = [
    {
        id: "auto",
        label: "Auto (Recommended)",
        modelId: "__auto__",
        description: "Choose the best model for this browser",
        webgpuOnly: false,
        englishOnly: false,
    },
    {
        id: "fast",
        label: "Fast",
        modelId: "onnx-community/whisper-tiny_timestamped",
        description: "Lowest memory usage, lower accuracy",
        webgpuOnly: false,
        englishOnly: false,
    },
    {
        id: "balanced",
        label: "Balanced",
        modelId: "onnx-community/whisper-base_timestamped",
        description: "Default multilingual model",
        webgpuOnly: false,
        englishOnly: false,
    },
    {
        id: "quality",
        label: "High Quality",
        modelId: "onnx-community/whisper-large-v3-turbo_timestamped",
        description: "Best quality for stronger WebGPU desktops",
        webgpuOnly: true,
        englishOnly: false,
    },
    {
        id: "fast_en",
        label: "Fast English",
        modelId: "onnx-community/distil-small.en_timestamped",
        description: "Fastest option for English-only transcription",
        webgpuOnly: false,
        englishOnly: true,
    },
] as const satisfies readonly ModelPreset[];

export type ModelPresetId = (typeof MODEL_PRESETS)[number]["id"];

/** A preset that names a real model, as opposed to the `__auto__` sentinel. */
export type ConcreteModelPresetId = Exclude<ModelPresetId, "auto">;

/**
 * The model id a preset resolves to — or a throw.
 *
 * The worker keys its WebGPU dtype override on a preset id, and a bare
 * `MODEL_PRESETS.find(...)?.modelId` hands back `undefined` if the preset is ever
 * gone: the override silently stops applying. The derived `ModelPresetId` above
 * makes that a compile error, and this makes it a loud runtime one as well, for
 * anything that reaches the lookup with an id the type system did not check
 * (a persisted preference, a `select` value). Never return `undefined`.
 */
export function modelIdForPreset(presetId: ConcreteModelPresetId): string {
    const preset = MODEL_PRESETS.find((item) => item.id === presetId);
    if (!preset || preset.modelId === "__auto__") {
        throw new Error(
            `No model preset "${presetId}" in MODEL_PRESETS — a preset was removed ` +
                `or renamed without updating the code that reads it.`,
        );
    }
    return preset.modelId;
}

export const LANGUAGE_OPTIONS = [
    { value: "auto", label: "Auto detect" },
    { value: "english", label: "English" },
    { value: "spanish", label: "Spanish" },
    { value: "french", label: "French" },
    { value: "german", label: "German" },
    { value: "portuguese", label: "Portuguese" },
    { value: "chinese", label: "Chinese" },
    { value: "japanese", label: "Japanese" },
    { value: "korean", label: "Korean" },
];

function isEnglishRequest(language: string): boolean {
    return ["english", "en"].includes(language.toLowerCase());
}

export function chooseAutoPreset(
    caps: BrowserCaps,
    task: TaskMode,
    language: string,
): ModelPresetId {
    if (task === "translate") {
        return "balanced";
    }

    const englishOnly = isEnglishRequest(language);
    const lowEndCpu =
        !caps.canUseWebGPU &&
        ((caps.deviceMemoryGiB ?? 4) <= 4 || caps.logicalCores <= 4);

    if (lowEndCpu) {
        return englishOnly ? "fast_en" : "fast";
    }

    if (
        caps.canUseWebGPU &&
        (caps.deviceMemoryGiB ?? 0) >= 8 &&
        caps.logicalCores >= 8
    ) {
        return "quality";
    }

    return "balanced";
}

export function resolveModelConfig(
    requestedPresetId: ModelPresetId,
    caps: BrowserCaps,
    task: TaskMode,
    language: string,
): ResolvedModelConfig {
    const normalizedLanguage = language || "auto";
    let presetId =
        requestedPresetId === "auto"
            ? chooseAutoPreset(caps, task, normalizedLanguage)
            : requestedPresetId;

    const englishOnly = isEnglishRequest(normalizedLanguage);
    if (presetId === "quality" && !caps.canUseWebGPU) {
        presetId = "balanced";
    }
    if (presetId === "fast_en" && (task === "translate" || !englishOnly)) {
        presetId = "balanced";
    }

    const preset = MODEL_PRESETS.find((item) => item.id === presetId);
    if (!preset || preset.modelId === "__auto__") {
        throw new Error("Unable to resolve transcription model");
    }

    return {
        presetId,
        presetLabel: preset.label,
        modelId: preset.modelId,
        device: caps.canUseWebGPU ? "webgpu" : "wasm",
        task,
        language: normalizedLanguage,
    };
}
