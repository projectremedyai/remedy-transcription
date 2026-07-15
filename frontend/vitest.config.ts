import { defineConfig } from "vitest/config";

export default defineConfig({
    // No `@vitejs/plugin-react` here, deliberately: it injects a Fast Refresh
    // preamble that this project's file-per-suite Vitest run never satisfies
    // ("can't detect preamble"), which is irrelevant to tests anyway. `.tsx`
    // still transforms fine through Vite's default esbuild pipeline, which
    // reads `tsconfig.json`'s `"jsx": "react-jsx"` on its own — this config
    // exists only to widen `include` to `Transcript.test.tsx` (Task 12).
    test: {
        environment: "node",
        include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
    },
});
