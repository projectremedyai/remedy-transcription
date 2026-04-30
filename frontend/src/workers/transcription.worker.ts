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
                modelId === "onnx-community/whisper-large-v3-turbo" &&
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
        const chunkLength = isDistil ? 20 : 30;
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

        const streamer = new WhisperTextStreamer(transcriber.tokenizer, {
            time_precision: timePrecision,
            on_chunk_start: (offset: number) => {
                const chunkOffset = (chunkLength - strideLength) * chunkCount;
                chunks.push({
                    text: "",
                    timestamp: [chunkOffset + offset, null],
                    offset: chunkOffset,
                });
            },
            token_callback_function: () => {
                startedAt ??= performance.now();
                tokenCount += 1;
                if (tokenCount > 1 && startedAt !== null) {
                    tokensPerSecond =
                        (tokenCount / (performance.now() - startedAt)) * 1000;
                }
            },
            callback_function: (text: string) => {
                if (!chunks.length) {
                    return;
                }
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
            on_chunk_end: (offset: number) => {
                const current = chunks[chunks.length - 1];
                current.timestamp[1] = offset + current.offset;
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
            return_timestamps: true,
            force_full_sequences: false,
            streamer,
        });

        postMessageSafe({
            status: "complete",
            data: {
                text: output.text,
                chunks: output.chunks,
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
