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
     * ABSENT, not empty, when diarization did not run, was unavailable, or
     * found no turns at all. A segment without this key must render and
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
 * Whether an engine identified speakers — and if not, WHY NOT.
 *
 * A tagged union, not `SpeakerTurn[] | undefined`: a transcript with no speaker
 * labels *because the engine gave us nothing* must never be indistinguishable
 * from one *because a single person was talking*.
 *
 * ```ts
 * outcome.turns                                 // ✗ does not compile
 * if (outcome.status === "identified") ...       // ✓ the only way in
 * ```
 */
export type SpeakerOutcome =
    | {
          status: "identified";
          /**
           * MAY BE EMPTY, and that is a real success: silence has no speaker
           * turns. Do not treat `[]` as a failure, and do not divide by
           * `speaker_count`.
           *
           * Ids are dense (0..n-1), but treat the value as an opaque label;
           * do not index arrays with it.
           */
          turns: SpeakerTurn[];
          speaker_count: number;
      }
    | {
          status: "unavailable";
          /**
           * User-facing, and already specific: it names what went wrong.
           * Show it. The transcript itself is completely unaffected.
           */
          reason: string;
      };

/** One word as Gemini timed it, already offset into whole-file time by Rust. */
export interface GeminiWord {
    text: string;
    start: number;
    end: number;
}

export interface GeminiProgressEvent {
    phase: "slicing" | "uploading" | "transcribing" | "stitching";
    chunk_index: number;
    chunk_count: number;
    fraction: number;
}

/**
 * What `transcribe_with_gemini` produced.
 *
 * `speakers` is a tagged union for the reason the old `DiarizationOutcome`
 * documented: a transcript with no speaker labels *because the engine gave us
 * nothing* must never be indistinguishable from one *because a single person
 * was talking*. Here that is not hypothetical — audio over 28 minutes is split
 * into chunks, and `spk_1` in chunk 2 is not the same person as `spk_1` in
 * chunk 1, so no speakers are produced at all.
 *
 * ```ts
 * result.speakers.turns                              // ✗ does not compile
 * if (result.speakers.status === "identified") ...    // ✓ the only way in
 * ```
 */
export interface GeminiTranscriptionResult {
    text: string;
    words: GeminiWord[];
    /** The union added in Task 5. Reused, not redeclared. */
    speakers: SpeakerOutcome;
    audio_duration: number;
}
