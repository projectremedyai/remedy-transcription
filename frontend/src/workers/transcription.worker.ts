/// <reference lib="webworker" />

import { env, pipeline, WhisperTextStreamer } from "@huggingface/transformers";

env.allowRemoteModels = true;
env.allowLocalModels = false;

type WorkerRequest = {
    type: "transcribe";
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

            if (
                modelId ===
                    "onnx-community/whisper-large-v3-turbo_timestamped" &&
                device === "webgpu"
            ) {
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

function postMessageSafe(data: unknown) {
    self.postMessage(data);
}

self.addEventListener("message", async (event: MessageEvent<WorkerRequest>) => {
    const message = event.data;
    if (message.type !== "transcribe") {
        return;
    }

    try {
        const transcriber: any = await PipelineFactory.getInstance(
            message.modelId,
            message.device,
            (progress) => postMessageSafe(progress),
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
            timestamp: [number, number | null];
            offset: number;
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
         */
        const openWindowChunk = () => {
            if (chunks.length > chunkCount) {
                return;
            }
            const chunkOffset = (chunkLength - strideLength) * chunkCount;
            chunks.push({
                text: "",
                timestamp: [chunkOffset, null],
                offset: chunkOffset,
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
                postMessageSafe({
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
                const current = chunks[chunks.length - 1];
                if (current && current.timestamp[1] === null) {
                    current.timestamp[1] = current.offset + chunkLength;
                }
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

        postMessageSafe({
            status: "complete",
            data: {
                text: output.text,
                chunks: output.chunks,
                words,
                tps: tokensPerSecond,
            },
        });
    } catch (error) {
        const messageText =
            error instanceof Error
                ? error.message
                : "Unknown transcription error";
        postMessageSafe({
            status: "error",
            data: { message: messageText },
        });
    }
});
