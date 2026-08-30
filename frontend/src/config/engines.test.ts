import { describe, expect, it } from "vitest";

import { ENGINES, GEMINI_MODEL_ID, engineById } from "./engines";

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
