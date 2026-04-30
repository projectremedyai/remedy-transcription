import { BrowserCaps } from "../config/transcription";

export async function detectBrowserCaps(): Promise<BrowserCaps> {
    const deviceMemoryGiB =
        typeof navigator.deviceMemory === "number"
            ? navigator.deviceMemory
            : null;

    let canUseWebGPU = false;
    let shaderF16 = false;

    if (window.isSecureContext && "gpu" in navigator) {
        try {
            const adapter = await navigator.gpu?.requestAdapter();
            canUseWebGPU = !!adapter;
            shaderF16 = !!adapter?.features?.has("shader-f16");
        } catch {
            canUseWebGPU = false;
            shaderF16 = false;
        }
    }

    return {
        secureContext: window.isSecureContext,
        canUseWebGPU,
        shaderF16,
        deviceMemoryGiB,
        logicalCores: navigator.hardwareConcurrency ?? 4,
    };
}
