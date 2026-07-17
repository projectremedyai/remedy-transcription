/**
 * Diarization UI feature flag — OFF for the 1.1.0 release.
 *
 * WHY: real-content smoke testing (a documentary, one narrator, under music)
 * put diarization in its *count-required* mode — the one meant to be the
 * reliable path, per-speaker count supplied, no auto-detect guessing — and
 * it still produced 4 distinct speaker labels for a single narrator, flipping
 * between them 19 times over a 10-minute sample. That is an embedding/engine
 * ceiling (sherpa-onnx's speaker embeddings do not hold up against music
 * under narration), not a bug in this app's plumbing, and there is no UI
 * fix for it: the count hint is a request the engine may not honour (see
 * `isValidSpeakerCount` in `../hooks/useTranscriber`), and asking harder
 * does not change what the embeddings can tell apart.
 *
 * So the product decision is to hide the diarization UI for this release
 * rather than ship a control that reliably mislabels speakers. Nothing
 * about the backend, the `useTranscriber` hook's diarization plumbing, the
 * speaker-alignment/export code, or persisted `speaker` fields on segments
 * is removed — it is fully tested and stays in the tree so it can be
 * re-enabled the moment the engine (or a swappable one) clears this bar.
 *
 * Flip this back to `true` to restore the toggle, speaker-count input, and
 * status banner.
 */
export const DIARIZATION_UI_ENABLED = false;
