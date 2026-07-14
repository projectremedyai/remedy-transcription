/**
 * Joins timed tokens (either word-level or segment-level) to speaker turns
 * produced by diarization, by maximum temporal overlap.
 *
 * Deliberately generic over `TimedToken` rather than specialized to words:
 *
 *   - Fresh transcripts have REAL word-level timings (Task 4) — pass words
 *     for word-level speaker labels, accurate at turn boundaries.
 *   - Transcripts cached before the model rename are SENTENCE-granular with
 *     FABRICATED word times (character-length interpolation, measured off by
 *     up to 1.3s). The app preserves those cached transcripts by design and
 *     there is no re-transcribe path, so they exist indefinitely. Those must
 *     pass segments, not words — aligning turns against fabricated word
 *     times produces confidently wrong labels, which is worse than coarse
 *     (segment-level) ones.
 *
 * Both shapes satisfy `{ text, start, end }`, so the same function serves
 * both without specializing on a `words`-shaped field.
 */

export interface TimedToken {
    text: string;
    start: number;
    end: number;
}

export interface SpeakerTurn {
    start: number;
    end: number;
    speaker: number;
}

export type SpeakerTagged<T> = T & { speaker: number | null };

function overlap(a: TimedToken, b: SpeakerTurn): number {
    return Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));
}

/** Distance from a token to a turn it does not overlap. Zero if they touch. */
function gap(token: TimedToken, turn: SpeakerTurn): number {
    if (token.start >= turn.end) return token.start - turn.end;
    if (turn.start >= token.end) return turn.start - token.end;
    return 0;
}

/**
 * Assign each token the speaker whose turn it overlaps most.
 *
 * Tokens overlapping no turn fall back to the nearest turn (WhisperX calls this
 * fill_nearest). A token with no turns at all gets speaker: null — the caller
 * must render that as "unknown speaker", never as a real one (never invent
 * speaker 0 for the no-turns case).
 *
 * Turns are NOT assumed to be sorted by start time; every turn is checked.
 *
 * Tie-break rule (deterministic, not accidental): both the max-overlap scan
 * and the nearest-turn fallback keep the first candidate found and only
 * replace it on a STRICTLY greater overlap / strictly smaller gap. So on an
 * exact tie — including a zero-length token sitting exactly on the boundary
 * between two turns, where overlap and gap are both 0 for each — the turn
 * that appears FIRST in the input `turns` array wins, regardless of its
 * start time.
 */
export function assignSpeakers<T extends TimedToken>(
    tokens: T[],
    turns: SpeakerTurn[],
): SpeakerTagged<T>[] {
    if (turns.length === 0) {
        return tokens.map((t) => ({ ...t, speaker: null }));
    }

    return tokens.map((token) => {
        let best: SpeakerTurn | null = null;
        let bestOverlap = 0;

        for (const turn of turns) {
            const o = overlap(token, turn);
            if (o > bestOverlap) {
                bestOverlap = o;
                best = turn;
            }
        }

        if (best === null) {
            best = turns.reduce((nearest, turn) =>
                gap(token, turn) < gap(token, nearest) ? turn : nearest,
            );
        }

        return { ...token, speaker: best.speaker };
    });
}
