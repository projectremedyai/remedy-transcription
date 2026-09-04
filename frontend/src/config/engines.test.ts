import { describe, expect, it } from "vitest";

import { ENGINES, GEMINI_MODEL_ID, engineById, modelDiarizes } from "./engines";

describe("engine descriptors", () => {
    it("exposes exactly the two engines the app ships", () => {
        expect(ENGINES.map((engine) => engine.id)).toEqual(["local", "gemini"]);
    });

    it("marks gemini as needing a key and local as not", () => {
        expect(engineById("gemini").requiresKey).toBe(true);
        expect(engineById("local").requiresKey).toBe(false);
    });

    /**
     * gemini-3.5-transcribe has no language-forcing parameter and no
     * translation mode. The UI reads these to disable the Task and Language
     * controls WITH A REASON rather than leaving them silently inert.
     */
    it("records that gemini supports neither translation nor a language choice", () => {
        const gemini = engineById("gemini");
        expect(gemini.supportsTranslate).toBe(false);
        expect(gemini.supportsLanguageChoice).toBe(false);
        expect(engineById("local").supportsTranslate).toBe(true);
        expect(engineById("local").supportsLanguageChoice).toBe(true);
    });

    /**
     * The Interactions request is one opaque round trip: no token stream, so
     * no live preview. Callers must render a determinate bar instead.
     */
    it("records that gemini cannot stream a live preview", () => {
        expect(engineById("gemini").supportsLivePreview).toBe(false);
        expect(engineById("local").supportsLivePreview).toBe(true);
    });

    it("pins the gemini model id", () => {
        expect(engineById("gemini").modelId).toBe(GEMINI_MODEL_ID);
        expect(GEMINI_MODEL_ID).toBe("gemini-3.5-transcribe");
    });

    it("throws rather than returning undefined for an unknown engine", () => {
        expect(() => engineById("nope" as never)).toThrow(/no engine "nope"/);
    });
});

/**
 * The predicate that decides whether a stored transcript's speaker labels can
 * be believed. It exists because 1.1.0's local sherpa-onnx diarizer was deleted
 * in 1.2.0 but its labels were not — they sit in the database, on rows whose
 * `model_id` names an on-device Whisper model.
 */
describe("modelDiarizes: which stored labels are worth believing", () => {
    it("trusts the pinned cloud model", () => {
        expect(modelDiarizes(GEMINI_MODEL_ID)).toBe(true);
    });

    /**
     * Every on-device model, whether or not it is still a preset. The one
     * transcript the deleted diarizer left behind in the wild names
     * `whisper-base_timestamped` — a CURRENT preset — so a test that only used
     * a made-up id would pass against a build that trusted anything familiar.
     */
    it("does not trust any on-device model, current preset or not", () => {
        expect(modelDiarizes("onnx-community/whisper-base_timestamped")).toBe(
            false,
        );
        expect(
            modelDiarizes("onnx-community/whisper-large-v3-turbo_timestamped"),
        ).toBe(false);
        // Pre-rename, from before the model ids moved. No preset matches it,
        // and it must still be read as local rather than as "unknown, so
        // probably cloud".
        expect(modelDiarizes("Xenova/whisper-base")).toBe(false);
        expect(modelDiarizes("")).toBe(false);
    });

    /**
     * The point of deriving it from `ENGINES` instead of comparing against
     * `GEMINI_MODEL_ID`: a second cloud engine is trusted by being added to the
     * array, and `modelDiarizes` does not have to be found and edited.
     */
    it("is derived from ENGINES, not from a hardcoded gemini check", () => {
        const pinned = ENGINES.filter((engine) => engine.modelId !== null).map(
            (engine) => engine.modelId as string,
        );
        expect(pinned.every(modelDiarizes)).toBe(true);
        expect(
            ENGINES.filter((engine) => engine.modelId === null),
        ).toHaveLength(1);
    });
});
