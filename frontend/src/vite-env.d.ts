// eslint-disable-next-line spaced-comment
/// <reference types="vite/client" />

interface MinimalGPUFeatureSet {
    has: (feature: string) => boolean;
}

interface MinimalGPUAdapter {
    features: MinimalGPUFeatureSet;
}

interface MinimalGPU {
    requestAdapter: () => Promise<MinimalGPUAdapter | null>;
}

interface Navigator {
    deviceMemory?: number;
    gpu?: MinimalGPU;
}
