/// <reference lib="webworker" />

import { env, pipeline, WhisperTextStreamer } from "@huggingface/transformers";

import { modelIdForPreset } from "../config/transcription";

env.allowRemoteModels = true;
env.allowLocalModels = false;

/**
 * The dtype override below is keyed on a model ID. Typing the ID out here made it
 * a second copy of a fact owned by `config/transcription.ts`, and it drifted
 * silently the first time the models were renamed — the override simply stopped
 * applying and large-v3-turbo loaded an fp32 encoder on WebGPU. Read it from the
 * config instead.
 *
 * `modelIdForPreset` rather than `find(...)?.modelId`: the optional chain returns
 * `undefined` if the `"quality"` preset ever goes away, `modelId === undefined` is
 * never true, and the override silently stops applying all over again. Now the id
 * is a compile error if the preset is gone (`ModelPresetId` is derived from
 * `MODEL_PRESETS`) and a throw if it is somehow still reached.
 */
const LARGE_TURBO_MODEL_ID: string = modelIdForPreset("quality");

type WorkerRequest = {
    type: "transcribe";
    /**
     * Which run asked for this. Echoed back on EVERY message this request
     * produces — `initiate`/`progress`/`done`/`ready`, `update`, `complete`,
     * `error` — so the hook can tell whose output it is holding.
     *
     * This handler is `async`: a second `transcribe` posted while the first is
     * still awaiting `transcriber(...)` does not queue behind it, it starts a
     * SECOND handler that runs concurrently against the same shared
     * `PipelineFactory.instance`. Both then post `update`s and a `complete` at
     * the same unlabelled receiver. Without this id the receiver cannot answer
     * the only question that matters — "is this the run I am waiting for?" — and
     * will happily resolve run 2's promise with run 1's transcript, which is then
     * persisted under run 2's job and content-cached against run 2's file
     * forever.
     *
     * The app also terminates the worker when it abandons a run, which is what
     * actually stops the inference. The id is what makes the receiver correct
     * even so, because `terminate()` cannot recall a message already in flight.
     */
    runId: number;
    audio: Float32Array;
    modelId: string;
    device: "webgpu" | "wasm";
    task: "transcribe" | "translate";
    language: string | null;
};

type PipelineInstance = any;

class PipelineFactory {
    static task = "automatic-speech-recognition" as const;
    static instance: PipelineInstance | null = null;
    static modelId: string | null = null;
    static device: string | null = null;

    static async getInstance(
        modelId: string,
        device: "webgpu" | "wasm",
        progressCallback?: (event: unknown) => void,
    ) {
        if (
            this.instance !== null &&
            (this.modelId !== modelId || this.device !== device)
        ) {
            this.instance.dispose();
            this.instance = null;
        }

        if (this.instance === null) {
            const options: Record<string, unknown> = {
                progress_callback: progressCallback,
            };

            if (device === "webgpu") {
                options.device = "webgpu";
            }

            if (modelId === LARGE_TURBO_MODEL_ID && device === "webgpu") {
                options.dtype = {
                    encoder_model: "fp16",
                    decoder_model_merged: "q4",
                };
            }

            this.instance = await (pipeline as any)(
                this.task,
                modelId,
                options,
            );
            this.modelId = modelId;
            this.device = device;
        }

        return this.instance;
    }
}

self.addEventListener("message", async (event: MessageEvent<WorkerRequest>) => {
    const message = event.data;
    if (message.type !== "transcribe") {
        return;
    }

    /**
     * Every message this request emits is stamped with the run that asked for it.
     * There is no unlabelled path out of this handler — the stamp is applied here,
     * once, rather than at each of the five call sites, so a new message type
     * cannot forget it.
     */
    const post = (data: object) => {
        self.postMessage({ ...data, runId: message.runId });
    };

    try {
        const transcriber: any = await PipelineFactory.getInstance(
            message.modelId,
            message.device,
            (progress) => post(progress as object),
        );

        const timePrecision =
            transcriber.processor.feature_extractor.config.chunk_length /
            transcriber.model.config.max_source_positions;

        const isDistil = message.modelId.startsWith("onnx-community/distil");
        // 29, not 30: transformers.js #1357/#1358 — word timestamps break at
        // exactly chunk_length_s=30. The fix (#1594) shipped only in 4.x, after
        // the pinned 3.8.1.
        const chunkLength = isDistil ? 20 : 29;
        const strideLength = isDistil ? 3 : 5;

        const chunks: Array<{
            text: string;
            timestamp: [number, number];
        }> = [];

        let chunkCount = 0;
        let startedAt: number | null = null;
        let tokenCount = 0;
        let tokensPerSecond: number | undefined;

        /**
         * `return_timestamps: 'word'` makes transformers.js turn timestamp TOKENS
         * off (pipelines.js:1812-1815 — word times come from DTW over the
         * cross-attentions instead), so `WhisperTextStreamer`'s `on_chunk_start` /
         * `on_chunk_end` — which only fire on timestamp tokens — never fire at all.
         * The old code appended streamed text to the chunk that `on_chunk_start`
         * had opened, so with word timestamps it would append to nothing and the
         * live preview would stay empty for the whole run.
         *
         * Open one chunk per decoding WINDOW instead. Its span is the window,
         * which is the only timing information that exists mid-stream — good
         * enough for a preview, and the "complete" message replaces it wholesale
         * with the model's real per-word times.
         *
         * The window's end is provisional but it is NOT null. A null end reaches
         * `segmentsFromWorkerChunks` on the `update` path with no audio duration
         * to fall back on, so the whole open window — up to 29 seconds and dozens
         * of words — collapses into the 2s nominal fallback: speech 25 seconds in
         * previews as "0:00", then snaps when the window finalises. The window
         * runs to `offset + chunkLength` by construction; say so.
         */
        const openWindowChunk = () => {
            if (chunks.length > chunkCount) {
                return;
            }
            // TWO strides, not one. transformers.js advances its decoding window
            // by `jump = window - 2 * stride` (pipelines.js: `const jump = window
            // - 2 * stride; ... offset += jump;`) — the window overlaps its
            // neighbour on BOTH sides, so consecutive windows start 19s apart for
            // (29, 5) and 14s apart for distil's (20, 3). Subtracting one stride
            // put window k five seconds late per window, compounding linearly: on
            // a five-minute file the preview claimed cue times about a minute past
            // the end of the audio.
            const chunkOffset = (chunkLength - 2 * strideLength) * chunkCount;
            chunks.push({
                text: "",
                timestamp: [chunkOffset, chunkOffset + chunkLength],
            });
        };

        const streamer = new WhisperTextStreamer(transcriber.tokenizer, {
            time_precision: timePrecision,
            token_callback_function: () => {
                startedAt ??= performance.now();
                tokenCount += 1;
                if (tokenCount > 1 && startedAt !== null) {
                    tokensPerSecond =
                        (tokenCount / (performance.now() - startedAt)) * 1000;
                }
            },
            callback_function: (text: string) => {
                openWindowChunk();
                chunks[chunks.length - 1].text += text;
                post({
                    status: "update",
                    data: {
                        text: "",
                        chunks: chunks.map((chunk) => ({
                            text: chunk.text,
                            timestamp: chunk.timestamp,
                        })),
                        tps: tokensPerSecond,
                    },
                });
            },
            on_finalize: () => {
                startedAt = null;
                tokenCount = 0;
                chunkCount += 1;
            },
        });

        const output: any = await transcriber(message.audio, {
            top_k: 0,
            do_sample: false,
            chunk_length_s: chunkLength,
            stride_length_s: strideLength,
            language: message.language,
            task: message.task,
            return_timestamps: "word",
            force_full_sequences: false,
            streamer,
        });

        // With `return_timestamps: 'word'` every entry of `output.chunks` is a
        // WORD — `{ text, timestamp: [start, end] }` straight out of
        // `collateWordTimestamps` (tokenizers.js:4036) — not a sentence chunk.
        // These are the real times: no more spreading a chunk's duration across
        // its words by character length.
        const wordChunks: Array<{
            text: string;
            timestamp: [number | null, number | null];
        }> = output.chunks ?? [];

        // A word whose DTW timestamp came back null cannot be placed on a cue, so
        // it is dropped from `words`. NOTE the consequence: `output.text` — which
        // is persisted as `full_text` and is what the plain-text export writes —
        // still CONTAINS that word, so a dropped word makes the cues disagree with
        // full_text by exactly the words that had no time. That is the right
        // trade (a word with an invented time corrupts the timeline and, in Task
        // 6, the speaker attribution; a word missing from the cues does not), but
        // it is a real divergence and not an invariant. In practice the filter
        // removes nothing: transformers.js only leaves a null when the token has
        // no aligned frame at all.
        const words = wordChunks
            .filter(
                (chunk) =>
                    chunk.timestamp?.[0] != null &&
                    chunk.timestamp?.[1] != null,
            )
            .map((chunk) => ({
                text: chunk.text,
                start: chunk.timestamp[0] as number,
                end: chunk.timestamp[1] as number,
            }));

        post({
            status: "complete",
            data: {
                text: output.text,
                // `output.chunks` ARE the words, so shipping both would send the
                // same array twice across the worker boundary. `chunks` is only
                // the FALLBACK, for a model that returned no usable word times;
                // when there are words, nothing downstream reads it.
                chunks: words.length > 0 ? [] : output.chunks ?? [],
                words,
                tps: tokensPerSecond,
            },
        });
    } catch (error) {
        const messageText =
            error instanceof Error
                ? error.message
                : "Unknown transcription error";
        post({
            status: "error",
            data: { message: messageText },
        });
    }
});
