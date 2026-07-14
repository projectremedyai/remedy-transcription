import { describe, it, expect } from "vitest";
import {
    MODEL_PRESETS,
    ModelPresetId,
    modelIdForPreset,
    resolveModelConfig,
} from "./transcription";

const caps = {
    secureContext: true,
    canUseWebGPU: true,
    shaderF16: true,
    deviceMemoryGiB: 16,
    logicalCores: 12,
};

/**
 * The worker's WebGPU dtype override is keyed on the `"quality"` PRESET id, not
 * on a model id — the C1 fix moved it there so it could not drift when the models
 * were renamed. But a preset id can drift too: written as a hand-maintained union,
 * `ModelPresetId` could keep `"quality"` after the preset was deleted, the lookup
 * would return `undefined`, `modelId === undefined` is never true, and the
 * override would silently stop applying. large-v3-turbo would load an fp32 encoder
 * on WebGPU with nothing anywhere to say so.
 *
 * `ModelPresetId` is now DERIVED from `MODEL_PRESETS`, so that drift is a compile
 * error (verified: deleting the `quality` preset fails `tsc` at the worker's
 * override). These tests guard the runtime half — that the lookup is total and
 * never hands back `undefined`.
 */
describe("a preset id always resolves to a model", () => {
    it("resolves every concrete preset id", () => {
        for (const preset of MODEL_PRESETS) {
            if (preset.modelId === "__auto__") continue;
            expect(modelIdForPreset(preset.id as never)).toBe(preset.modelId);
        }
    });

    it("resolves the preset the worker's WebGPU dtype override is keyed on", () => {
        const modelId = modelIdForPreset("quality");
        expect(modelId).toBeTruthy();
        expect(modelId).toContain("large-v3-turbo");
    });

    it("THROWS rather than returning undefined for a preset that is not there", () => {
        expect(() => modelIdForPreset("gone" as never)).toThrow(
            /No model preset/,
        );
        // The `__auto__` sentinel is not a model either.
        expect(() => modelIdForPreset("auto" as never)).toThrow(
            /No model preset/,
        );
    });

    it("every ModelPresetId the resolver can return names a real preset", () => {
        const resolvable: ModelPresetId[] = MODEL_PRESETS.map(
            (preset) => preset.id,
        );
        for (const presetId of resolvable) {
            const resolved = resolveModelConfig(
                presetId,
                caps,
                "transcribe",
                presetId === "fast_en" ? "english" : "auto",
            );
            expect(resolved.modelId).not.toBe("__auto__");
            expect(resolved.modelId).toContain("/");
        }
    });
});
