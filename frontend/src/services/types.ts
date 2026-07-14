import { TaskMode } from "../config/transcription";
import type { SpeakerTurn } from "../lib/speakerAlignment";

export interface TranscriptionSegment {
    start: number;
    end: number;
    text: string;
    /**
     * Who said it, e.g. "SPEAKER_00" — an OPAQUE label, produced by
     * `speakerLabel` from a diarization turn's id.
     *
     * ABSENT, not empty, when diarization did not run, was cancelled, degraded,
     * or found no turns at all. A segment without this key must render and
     * serialize exactly as it did before diarization existed, which is why it is
     * left off rather than set to `undefined` or `""`.
     *
     * A cue (`ConsolidatedSegment`) carrying this label is guaranteed to have
     * exactly ONE speaker: `consolidateSegments` treats a speaker change as a
     * hard break. See `shouldBreakBefore` and `canMerge` in `captionFormatter`.
     */
    speaker?: string;
}

export interface Job {
    id: string;
    source_type: "file" | "youtube";
    source_key: string;
    status: "downloading" | "extracting" | "ready" | "completed" | "failed";
    progress: number;
    cache_hit: boolean;
    error: string | null;
    filename: string | null;
    audio_url: string | null;
    audio_mime_type: string | null;
    model_id: string;
    task: TaskMode;
    language: string;
    segments: TranscriptionSegment[];
    full_text: string | null;
}

export interface FileJobRequest {
    /**
     * A real filesystem path — from the Tauri dialog or a file drop, never a
     * browser `File` (which has no path).
     *
     * Rust derives the filename, the size and the content hash from it, and runs
     * the same ffmpeg pass over it that YouTube downloads get. The webview used
     * to send a hash it computed itself over an `ArrayBuffer` of the file, which
     * meant Rust never saw the audio at all.
     */
    path: string;
    model_id: string;
    task: TaskMode;
    language: string;
}

export interface YouTubeJobRequest {
    url: string;
    model_id: string;
    task: TaskMode;
    language: string;
}

export interface PersistTranscriptRequest {
    model_id: string;
    task: TaskMode;
    language: string;
    full_text: string;
    segments: TranscriptionSegment[];
}

/**
 * What the user CALLS each speaker, keyed by the opaque label the segments carry:
 * `{ SPEAKER_00: "Alice" }`.
 *
 * A map, and NOT an array indexed by speaker number. The keys are the segments'
 * own strings — do not parse them back into numbers, do not assume they are
 * dense, sorted or small. A speaker nobody has renamed has NO ENTRY, and the
 * caller renders the key itself; an entry is never blank (Rust rejects a blank
 * name, which would render as a nameless speaker and be indistinguishable from a
 * cue with no speaker at all).
 */
export type SpeakerNames = Record<string, string>;

export interface QueueStatus {
    position: number;
    total_in_queue: number;
    estimated_wait_minutes: number | null;
}

export interface HealthResponse {
    status: string;
    queue_length: number;
    active_transcriptions: number;
    active_downloads: number;
}

export interface ModelStatusItem {
    model_id: string;
    ready: boolean;
    path: string;
}

export interface ModelStatusResponse {
    models_ready: boolean;
    missing_models: string[];
    items: ModelStatusItem[];
}

/**
 * What `diarize_job` produced. **Three outcomes, not two — and the type is the
 * only thing that keeps the third one visible.**
 *
 * A transcript with no speaker labels *because the engine crashed* must never be
 * indistinguishable from a transcript with no speaker labels *because one person
 * was talking*. `SpeakerTurn[]` cannot tell those apart — both are "nothing to
 * draw" — so a failure would arrive in the UI as a confident, silent, wrong
 * answer: "0 speakers", rendered as if it were the truth.
 *
 * Rust already refuses to serialize them the same way (`#[serde(tag = "status")]`,
 * and a `degraded` payload carries **no `turns` key at all**). This union is the
 * other half of that guard rail, and the half that actually binds a caller:
 *
 * ```ts
 * outcome.turns ?? []          // ✗ does not compile — `turns` is not on every arm
 * if (outcome.status === "succeeded") outcome.turns   // ✓ the only way in
 * ```
 *
 * So a degradation cannot be read as zero speakers by accident. It has to be
 * handled — which means shown. That is the point.
 *
 * `cancelled` is separate from `degraded` for the same reason at one remove:
 * telling users "speaker detection failed" for something they themselves stopped
 * is a lie in the other direction.
 */
export type DiarizationOutcome =
    | {
          status: "succeeded";
          /**
           * MAY BE EMPTY, and that is a real success: silence has no speaker
           * turns. Do not treat `[]` as a failure, and do not divide by
           * `speaker_count`.
           *
           * Speaker ids are DENSE (`0..speaker_count`) — Rust remaps the engine's
           * sparse ids at the boundary, so `speaker` is a valid index.
           */
          turns: SpeakerTurn[];
          speaker_count: number;
      }
    | {
          status: "degraded";
          /**
           * User-facing, and already specific: it names the missing model, the
           * signal that killed the sidecar, or the timeout. Show it. The
           * transcript itself is completely unaffected.
           */
          reason: string;
      }
    | { status: "cancelled" };
