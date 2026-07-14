export type TaskMode = "transcribe" | "translate";
export type DeviceMode = "webgpu" | "wasm";
export type ModelPresetId =
    | "auto"
    | "fast"
    | "balanced"
    | "quality"
    | "fast_en";

export interface BrowserCaps {
    secureContext: boolean;
    canUseWebGPU: boolean;
    shaderF16: boolean;
    deviceMemoryGiB: number | null;
    logicalCores: number;
}

export interface ModelPreset {
    id: ModelPresetId;
    label: string;
    modelId: string | "__auto__";
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

export const MODEL_PRESETS: ModelPreset[] = [
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
];

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
