import { TranscriptionSegment } from "../services/types";
import {
    assignSpeakers,
    speakerLabel,
    type SpeakerTurn,
} from "./speakerAlignment";

const MAX_LINE_CHARS = 42;
const MAX_CAPTION_CHARS = 78;
const MAX_CAPTION_DURATION = 6.5;
const SHORT_MERGE_MAX_DURATION = 8.0;
const MIN_CAPTION_DURATION = 1.2;
const MIN_CAPTION_WORDS = 3;
const MAX_OVERLAP_WORDS = 16;
const NOMINAL_WORD_DURATION = 0.35;

/**
 * The shortest span a single word is allowed to occupy.
 *
 * DTW quantises word times to 0.02s and DOES emit words whose end equals their
 * start. Left alone, such a word is a zero-duration database row, and on reload
 * `effectiveEnd` inflates it to the next word's start (swallowing a pause) or,
 * for the final word, to `start + MIN_CAPTION_DURATION` — possibly past the end
 * of the audio. Flooring it here, on the path BOTH the live render and the
 * persisted rows go through, is what keeps a reload identical to the live cues.
 */
const MIN_WORD_DURATION = 0.01;

/** Absorbs float noise in duration comparisons (a - b where b was a + eps). */
const TIME_EPSILON = 1e-6;

const WHITESPACE_RE = /\s+/g;
const SPACE_BEFORE_PUNCT_RE = /\s+([,.;:!?])/g;
const MISSING_SPACE_AFTER_PUNCT_RE = /([,.;:!?])(?=[^\s"'])/g;
const WORD_NORMALIZE_RE = /[^a-z0-9']+/g;
const SENTENCE_END_RE = /[.!?。！？…]["'”’)\]]*$/;
const CLAUSE_END_RE = /[,;:，、；：]["'”’)\]]*$/;
const DIGIT_RE = /\d/;
const LETTER_RE = /[A-Za-z]/;

/**
 * Scripts that do not put spaces between words.
 *
 * This is not decoration — it is the same list transformers.js branches on.
 * `combineTokensIntoWords` (tokenizers.js) sends chinese/japanese/thai/lao/
 * myanmar through `splitTokensOnUnicode`, which NEVER emits a leading space, and
 * everything else through `splitTokensOnSpaces`, which marks each word boundary
 * with one. Code that reads "no leading space" as "this continues the previous
 * word" is therefore correct for English and catastrophically wrong for Chinese:
 * every word continues the previous one, and the entire transcript folds into a
 * single token. Ranges: CJK punctuation, kana, CJK ideographs (incl. ext-A and
 * compatibility), fullwidth/halfwidth forms, Thai, Lao, Myanmar. Hangul is
 * deliberately absent — Korean is space-separated and takes the normal path.
 */
const CJK_SCRIPT_CHARS = [
    "\\u3000-\\u303f", // CJK symbols and punctuation
    "\\u3040-\\u30ff", // Hiragana and Katakana
    "\\u3400-\\u4dbf", // CJK unified ideographs extension A
    "\\u4e00-\\u9fff", // CJK unified ideographs
    "\\uf900-\\ufaff", // CJK compatibility ideographs
    "\\uff00-\\uffef", // Halfwidth and fullwidth forms
].join("");
const SEA_SCRIPT_CHARS = [
    "\\u0e00-\\u0e7f", // Thai
    "\\u0e80-\\u0eff", // Lao
    "\\u1000-\\u109f", // Myanmar
].join("");
const NO_SPACE_SCRIPT_CHARS = `${SEA_SCRIPT_CHARS}${CJK_SCRIPT_CHARS}`;

const STARTS_NO_SPACE_SCRIPT_RE = new RegExp(`^[${NO_SPACE_SCRIPT_CHARS}]`);
const ENDS_NO_SPACE_SCRIPT_RE = new RegExp(`[${NO_SPACE_SCRIPT_CHARS}]$`);
const NO_SPACE_SCRIPT_GLOBAL_RE = new RegExp(`[${NO_SPACE_SCRIPT_CHARS}]`, "g");
const IS_NO_SPACE_SCRIPT_RE = new RegExp(`^[${NO_SPACE_SCRIPT_CHARS}]$`);
const ENDS_CJK_RE = new RegExp(`[${CJK_SCRIPT_CHARS}]$`);

/**
 * Characters that may not START a line. A line break before a CJK closing mark
 * strands the comma or full stop at the head of line two, which is precisely the
 * thing CJK typesetting rules (kinsoku) forbid. Cheap to honour: the break
 * candidate is simply not offered.
 */
const NO_LINE_START_RE = /[、，。．！？：；）〕】》」』〉’”…・]/;

/**
 * One word with its own start and end. When these come from the model
 * (`return_timestamps: 'word'`) they are REAL times, recovered by DTW over the
 * decoder's cross-attentions. When they don't, `tokensFromSegments` fabricates
 * them from a segment's span — see the warning there.
 */
export interface WordToken {
    text: string;
    start: number;
    end: number;
    /**
     * Who said this word — the OPAQUE label from `speakerLabel`, never a raw
     * turn id. Absent unless diarization produced turns to align against.
     *
     * This is what lets the cue splitter see a speaker change: `shouldBreakBefore`
     * compares it and breaks unconditionally when it differs.
     */
    speaker?: string;
}

/**
 * Build a token, keeping `speaker` ABSENT when there is none.
 *
 * `{...t, speaker: undefined}` would be a different object: the key exists,
 * `"speaker" in token` is true, and `JSON.stringify` of a persisted row starts
 * differing from one written before diarization existed. An undiarized transcript
 * must be byte-identical to what it was, so the key is omitted, not emptied.
 */
function token(
    text: string,
    start: number,
    end: number,
    speaker?: string,
): WordToken {
    return speaker === undefined
        ? { text, start, end }
        : { text, start, end, speaker };
}

/** As `token`, for captions. Same absent-not-empty rule, same reason. */
function caption(
    start: number,
    end: number,
    text: string,
    speaker?: string,
): TranscriptionSegment {
    return speaker === undefined
        ? { start, end, text }
        : { start, end, text, speaker };
}

/**
 * A caption cue: the output of `consolidateSegments`, and the ONLY thing the
 * serializers in `lib/srtGenerator.ts` accept.
 *
 * The brand exists because `consolidateSegments` is not idempotent and silently
 * DELETES words when run on its own output (see the characterization test in
 * captionFormatter.test.ts). A cue is structurally identical to a raw
 * `TranscriptionSegment`, so without the brand `consolidateSegments(cues)` and
 * `generateSrt(rawSegments)` both typecheck and both are wrong. With it, the
 * compiler rejects them: consolidation consumes raw segments and produces cues,
 * and there is no way back.
 */
export type ConsolidatedSegment = TranscriptionSegment & {
    readonly __consolidated: unique symbol;
};

/**
 * A segment that has NOT been through the formatter — the only thing
 * `consolidateSegments` accepts.
 *
 * The optional-`never` property is what makes double-consolidation a COMPILE
 * ERROR. A brand alone is not enough: `ConsolidatedSegment` is an intersection,
 * hence a subtype of `TranscriptionSegment`, so a plain
 * `consolidateSegments(segments: TranscriptionSegment[])` would happily accept
 * its own output. Declaring the input as "has no `__consolidated` property"
 * rejects it: a raw segment (property absent) is assignable, a cue (property
 * present, typed `unique symbol`) is not.
 */
export type RawSegment = TranscriptionSegment & {
    readonly __consolidated?: never;
};

/**
 * Whisper sometimes drops the space after a punctuation mark ("Hello,world"), so
 * the formatter puts it back. But punctuation is not always a word boundary, and
 * splitting where there is no boundary is not cosmetic: the database stores ONE
 * WORD PER ROW, and a reload re-derives the word list by splitting each row's
 * cleaned text on spaces. Insert a space inside a persisted word and the reload
 * gets two tokens where the live render had one — "3.14" comes back as "3." and
 * "14", "3." satisfies `endsSentence`, and a cue can break in the middle of a
 * number on reload but not live.
 *
 * So: never split a number (3.14, 1,000, 3:30) or an abbreviation (U.S.A., e.g.).
 */
function needsSpaceAfterPunctuation(
    text: string,
    punctuation: string,
    index: number,
): boolean {
    const before = text[index - 1] ?? "";
    const after = text[index + 1] ?? "";

    if (
        ".,:".includes(punctuation) &&
        DIGIT_RE.test(before) &&
        DIGIT_RE.test(after)
    ) {
        return false;
    }

    // A dot whose left neighbour is a SINGLE letter, followed by another letter:
    // U.S.A., e.g., i.e. A sentence end has a word before the dot, not a letter.
    if (
        punctuation === "." &&
        LETTER_RE.test(before) &&
        LETTER_RE.test(after) &&
        !LETTER_RE.test(text[index - 2] ?? "")
    ) {
        return false;
    }

    return true;
}

export function cleanCaptionText(text: string): string {
    let cleaned = text.trim().replace(WHITESPACE_RE, " ");
    cleaned = cleaned.replace(SPACE_BEFORE_PUNCT_RE, "$1");
    cleaned = cleaned.replace(
        MISSING_SPACE_AFTER_PUNCT_RE,
        (_match, punctuation: string, index: number, whole: string) =>
            needsSpaceAfterPunctuation(whole, punctuation, index)
                ? `${punctuation} `
                : punctuation,
    );
    cleaned = cleaned.replace(/\bi\b/g, "I");
    return cleaned.trim();
}

/**
 * Join two pieces of caption text, inserting a space only where the scripts
 * involved actually use one. `"光合" + "作用"` is `"光合作用"`, not `"光合 作用"`.
 */
export function joinCaptionTexts(left: string, right: string): string {
    if (left.length === 0) return right;
    if (right.length === 0) return left;
    const needsSpace =
        !ENDS_NO_SPACE_SCRIPT_RE.test(left) &&
        !STARTS_NO_SPACE_SCRIPT_RE.test(right);
    return needsSpace ? `${left} ${right}` : `${left}${right}`;
}

/**
 * How many "words" a caption holds, for the runt/length rules.
 *
 * Splitting on whitespace reports 1 for any Chinese or Japanese caption however
 * long, which makes `mergeShortCaptions` treat every one of them as a runt and
 * glue them together. Count each no-space-script character as its own word,
 * which is exactly how the model tokenised them.
 */
export function countCaptionWords(text: string): number {
    const normalized = text.replace(/\n/g, " ").trim();
    if (normalized.length === 0) return 0;

    const scriptChars =
        normalized.match(NO_SPACE_SCRIPT_GLOBAL_RE)?.length ?? 0;
    const spaced = normalized
        .replace(NO_SPACE_SCRIPT_GLOBAL_RE, " ")
        .split(/\s+/)
        .filter(Boolean).length;

    return scriptChars + spaced;
}

function normalizeWord(word: string): string {
    return word.toLowerCase().replace(WORD_NORMALIZE_RE, "");
}

function splitWords(text: string): string[] {
    return cleanCaptionText(text)
        .split(" ")
        .filter((word) => word.length > 0);
}

function overlapPrefixSize(
    previous: string[],
    current: string[],
    minOverlapSize: number,
): number {
    const prevNorm = previous.map(normalizeWord).filter((w) => w.length > 0);
    const currNorm = current.map(normalizeWord).filter((w) => w.length > 0);
    const maxOverlap = Math.min(
        MAX_OVERLAP_WORDS,
        prevNorm.length,
        currNorm.length,
    );

    for (let size = maxOverlap; size >= minOverlapSize; size--) {
        const prevTail = prevNorm.slice(prevNorm.length - size);
        const currHead = currNorm.slice(0, size);
        if (arraysEqual(prevTail, currHead)) {
            return size;
        }
    }
    return 0;
}

function arraysEqual(a: string[], b: string[]): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

function charWeight(word: string): number {
    return Math.max(word.length, 1);
}

function charWeightOf(words: string[]): number {
    return words.reduce((sum, word) => sum + charWeight(word), 0);
}

/**
 * A segment whose end is not after its start carries no usable duration. That
 * happens for Whisper's final chunk, whose end timestamp is null and gets
 * persisted as `end := start`. Left at zero, mergeShortCaptions treats the cue
 * as a runt and glues it onto the previous one, so the last sentence of the
 * transcript is displayed seconds early on top of earlier speech. Fall back to
 * the next segment's start, then to a nominal speaking rate.
 */
function effectiveEnd(
    segments: TranscriptionSegment[],
    index: number,
    wordCount: number,
): number {
    const segment = segments[index];
    if (segment.end > segment.start) {
        return segment.end;
    }

    const next = segments[index + 1];
    if (next && next.start > segment.start) {
        return next.start;
    }

    return (
        segment.start +
        Math.max(wordCount * NOMINAL_WORD_DURATION, MIN_CAPTION_DURATION)
    );
}

/**
 * Real word timings from the model, cleaned up for the cue splitter.
 *
 * Three things have to happen before Whisper's words are usable as tokens:
 *
 *   1. Fold continuation fragments. In a SPACE-SEPARATED language Whisper marks a
 *      word boundary with a LEADING SPACE (" quick") and emits intra-word
 *      fragments without one: "Word-level" arrives as [" Word", "-level"] and
 *      "cross-attention" as [" cross", "-attention"]. A fragment left standing
 *      prints "Word -level".
 *
 *      "No leading space" therefore does NOT mean "continues the previous word" —
 *      it means that only in a script that uses spaces. In Chinese, Japanese,
 *      Thai, Lao and Myanmar, transformers.js splits on unicode codepoints and
 *      NEVER emits a leading space (tokenizers.js, `combineTokensIntoWords`), so
 *      the naive rule folds EVERY word into the first one and the whole
 *      transcript collapses to a single token — one cue spanning the entire
 *      audio, one database row holding everything. Both languages are offered in
 *      the UI and auto-detect reaches them. So a token that STARTS with a
 *      no-space-script character opens a new word.
 *   2. Split a token that `cleanCaptionText` will turn into TWO. Whisper emits
 *      "Hello,world" as ONE word chunk, and the formatter — rightly — puts the
 *      missing space back. But the database stores one word per row, and a reload
 *      re-derives the tokens by splitting each row's CLEANED text on spaces: the
 *      row persisted as one token comes back as two, so the reload sees a
 *      different token stream than the live render did, and "Hello," ends a
 *      clause where it did not before. Split it HERE, in the single funnel both
 *      the live render and the persisted rows pass through, and the two streams
 *      are identical by construction. Its span is shared out by character length,
 *      the same weighting the fabricating path uses.
 *   3. Clamp monotonically. The model's words are already ordered and
 *      non-overlapping in practice, but a cue whose start ran backwards would
 *      produce invalid SRT, so do not take that on trust.
 *   4. Floor each word's duration. DTW does emit `end === start`; see
 *      MIN_WORD_DURATION.
 *
 * Steps 3 and 4 run LAST, over the already-split tokens, so a split cannot
 * introduce a backwards or zero-length token of its own.
 */
export function normalizeWordTokens(words: readonly WordToken[]): WordToken[] {
    const folded: WordToken[] = [];

    for (const word of words) {
        const continuesPreviousWord =
            !/^\s/.test(word.text) &&
            !STARTS_NO_SPACE_SCRIPT_RE.test(word.text);
        const text = word.text.trim();
        if (text.length === 0) continue;

        const previous = folded[folded.length - 1];
        if (continuesPreviousWord && previous) {
            previous.text += text;
            previous.end = Math.max(previous.end, word.end);
            continue;
        }

        folded.push(token(text, word.start, word.end, word.speaker));
    }

    const tokens: WordToken[] = [];
    for (const word of folded) {
        for (const piece of splitGluedToken(word)) {
            const previous = tokens[tokens.length - 1];
            const start = previous
                ? Math.max(piece.start, previous.end)
                : piece.start;
            tokens.push(
                token(
                    piece.text,
                    start,
                    Math.max(piece.end, start + MIN_WORD_DURATION),
                    piece.speaker,
                ),
            );
        }
    }

    return tokens;
}

/**
 * One model word -> the token(s) it renders as, sharing the word's span out by
 * character length. Almost always exactly one; more only when `cleanCaptionText`
 * inserts a space the model did not emit ("Hello,world"). Storing the CLEANED
 * text is what makes the persisted row round-trip: `tokensFromSegments` re-cleans
 * and re-splits it on reload and must get back this same single token.
 */
function splitGluedToken(word: WordToken): WordToken[] {
    const parts = splitWords(word.text);
    if (parts.length === 0) return [];
    if (parts.length === 1) {
        return [token(parts[0], word.start, word.end, word.speaker)];
    }

    const totalWeight = charWeightOf(parts);
    const duration = Math.max(word.end - word.start, 0);
    let elapsed = 0;

    return parts.map((part) => {
        const start = word.start + (elapsed / totalWeight) * duration;
        elapsed += charWeight(part);
        const end = word.start + (elapsed / totalWeight) * duration;
        // One model word cannot have been said by two people, so every piece it
        // splits into keeps its speaker.
        return token(part, start, end, word.speaker);
    });
}

/**
 * FABRICATED word times: spread each segment's duration across its words by
 * character length. Only for input that has no real word timings — a transcript
 * loaded from the database that predates `return_timestamps: 'word'`. Measured
 * against the model's real word times on a 12.8s clip, this is off by 0.26s on
 * average and up to 1.30s. Prefer `normalizeWordTokens`.
 *
 * Each token inherits its SEGMENT's `speaker`, unchanged. That is what makes the
 * legacy diarization path honest: those segment times are real (only the word
 * times inside them are invented), so a speaker assigned to the segment is a
 * measured answer, and stamping it on every word of that segment is the coarse
 * but true one. Assigning speakers to the fabricated word times instead would be
 * a confident answer built on a number nobody measured — up to 1.3s adrift, which
 * is longer than many turns.
 */
function tokensFromSegments(segments: RawSegment[]): WordToken[] {
    const tokens: WordToken[] = [];
    let emittedWords: string[] = [];
    let previousEnd: number | null = null;

    for (let index = 0; index < segments.length; index++) {
        const segment = segments[index];
        const allWords = splitWords(segment.text);
        if (allWords.length === 0) continue;

        const segmentEnd = effectiveEnd(segments, index, allWords.length);
        const duration = Math.max(segmentEnd - segment.start, 0.01);

        // A 1- or 2-word repeat is only a sliding-window artifact if the two
        // segments actually overlap in time. Real speech ("no, no, no") repeats
        // words in chunks that are sequential, not overlapping — stripping those
        // would delete words the speaker said. Whisper's chunk timestamps are
        // real (only the *word* times are synthesized), so this test is
        // trustworthy.
        //
        // The test is PAIRWISE — this segment against the one immediately before
        // it. A running max over all previous ends would be more permissive: one
        // long segment would mark every later segment that starts inside its span
        // as "overlapping", enabling 1-word dedup between segments that are
        // actually sequential, and so deleting words the speaker said. Sliding
        // windows only ever overlap their immediate neighbour, so pairwise is
        // both the intent and sufficient.
        const chunksOverlapInTime =
            previousEnd !== null && segment.start < previousEnd;
        const minOverlapSize = chunksOverlapInTime ? 1 : 3;
        const overlap = overlapPrefixSize(
            emittedWords,
            allWords,
            minOverlapSize,
        );
        previousEnd = segmentEnd;

        // Weight the stripped overlap by characters, not word count — the
        // tokenizer distributes time by character length everywhere else, and
        // mixing the two metrics makes every post-dedup cue start early.
        const totalWeight = charWeightOf(allWords);
        const strippedWeight = charWeightOf(allWords.slice(0, overlap));
        const offsetStart =
            segment.start +
            (totalWeight > 0 ? (duration * strippedWeight) / totalWeight : 0);

        const words = allWords.slice(overlap);
        if (words.length === 0) continue;

        // Overlapping chunks time the SAME duplicated words differently, so the
        // offset can still land before the duplicate finished according to the
        // chunk that actually emitted it. Word times must be monotonic: never
        // re-open inside a span already emitted, or the cues collide.
        const lastEmittedEnd = tokens[tokens.length - 1]?.end ?? offsetStart;
        const start = Math.max(offsetStart, lastEmittedEnd);

        // The monotonic clamp can push `start` at or past this segment's own end
        // (a later chunk can begin before an earlier, longer one finished). Then
        // `segmentEnd - start` is zero or negative, and flooring the DURATION at
        // 0.01s crushes the whole segment into a 6ms cue that displaces seconds
        // of speech. Floor the SPAN instead: the words still have to be spoken,
        // so give them a nominal speaking rate's worth of time.
        //
        // The test is on the resulting DURATION, not merely on `segmentEnd >
        // start`. A clamp that lands `start` a millisecond before `segmentEnd`
        // passes "ends after it starts" and still crushes the whole segment into a
        // 1ms cue — the exact failure this floor exists to prevent. Demand enough
        // span for the words to exist at all: MIN_WORD_DURATION each.
        //
        // But ONLY that. A `Math.max(segmentEnd, start + n * NOMINAL)` would also
        // fire on any segment whose speaker is simply faster than 2.9 words/second,
        // stretching it past its own real end — and since the next segment's start
        // is then clamped forward to that invented end, the error compounds down
        // the transcript. It would also break the word-granular reload: real words
        // routinely last less than 0.35s, and `MIN_WORD_DURATION * 1` is precisely
        // the floor those rows were persisted under, so a one-word row is always
        // believed and comes back with its own measured time.
        const minimumUsableSpan =
            words.length * MIN_WORD_DURATION - TIME_EPSILON;
        const usableEnd =
            segmentEnd - start >= minimumUsableSpan
                ? segmentEnd
                : start + words.length * NOMINAL_WORD_DURATION;
        const survivingWeight = totalWeight - strippedWeight;
        const usableDuration = usableEnd - start;
        let elapsed = 0;

        for (const word of words) {
            const weight = charWeight(word) / survivingWeight;
            const wordStart = start + elapsed * usableDuration;
            elapsed += weight;
            const wordEnd = start + elapsed * usableDuration;
            tokens.push(token(word, wordStart, wordEnd, segment.speaker));
        }

        emittedWords = emittedWords.concat(words);
        if (emittedWords.length > MAX_OVERLAP_WORDS) {
            emittedWords = emittedWords.slice(-MAX_OVERLAP_WORDS);
        }
    }

    return tokens;
}

/**
 * Re-join tokens into caption text. NOT `join(" ")`: that prints Chinese as
 * "光 合 作 用". `joinCaptionTexts` inserts a space only between scripts that use
 * one, so not folding CJK words (above) does not simply move the damage here.
 */
function joinWords(tokens: WordToken[]): string {
    return cleanCaptionText(
        tokens.reduce((text, token) => joinCaptionTexts(text, token.text), ""),
    );
}

export function endsSentence(text: string): boolean {
    return SENTENCE_END_RE.test(text);
}

function endsClause(text: string): boolean {
    return CLAUSE_END_RE.test(text);
}

function shouldBreakBefore(
    current: WordToken[],
    nextToken: WordToken,
): boolean {
    // A CUE MUST NEVER SPAN TWO SPEAKERS, and this is checked FIRST, above every
    // length and duration rule below, because it OUTRANKS them: a speaker change
    // breaks the cue even when the result is a single word far under
    // MAX_CAPTION_CHARS and MAX_CAPTION_DURATION — the exact cue every heuristic
    // below would rather merge away. `mergeShortCaptions` is the other half of
    // this guarantee; see `canMerge`, which refuses to glue that runt back on.
    //
    // Every token in `current` shares one speaker by induction (this rule is why),
    // so comparing the last one is comparing all of them.
    //
    // Undiarized text has no speaker on either side: `undefined !== undefined` is
    // false, no break, and the cues come out exactly as they did before speakers
    // existed. Pinned by "renders an undiarized transcript exactly as before".
    if (current[current.length - 1].speaker !== nextToken.speaker) {
        return true;
    }

    const currentText = joinWords(current);
    const nextText = joinWords([...current, nextToken]);
    const currentDuration = current[current.length - 1].end - current[0].start;
    const nextDuration = nextToken.end - current[0].start;
    const currentWords = current.length;

    if (
        nextText.length > MAX_CAPTION_CHARS &&
        currentWords >= MIN_CAPTION_WORDS
    ) {
        return true;
    }

    if (
        currentDuration >= MIN_CAPTION_DURATION &&
        currentWords >= MIN_CAPTION_WORDS &&
        endsSentence(current[current.length - 1].text)
    ) {
        return true;
    }

    if (
        nextDuration > MAX_CAPTION_DURATION &&
        currentWords >= MIN_CAPTION_WORDS &&
        currentText.length >= 24
    ) {
        return true;
    }

    if (
        currentDuration >= 4.5 &&
        currentWords >= MIN_CAPTION_WORDS &&
        currentText.length >= 56 &&
        endsClause(current[current.length - 1].text)
    ) {
        return true;
    }

    return false;
}

function captionFromTokens(tokens: WordToken[]): TranscriptionSegment {
    const first = tokens[0];
    const last = tokens[tokens.length - 1];
    // Every token here shares one speaker — `shouldBreakBefore` cut the cue the
    // moment it changed — so the first token's speaker is the cue's speaker.
    return caption(
        round3(first.start),
        round3(last.end),
        wrapCaptionText(joinWords(tokens)),
        first.speaker,
    );
}

function round3(value: number): number {
    return Math.round(value * 1000) / 1000;
}

function mergeCaptionText(
    left: TranscriptionSegment,
    right: TranscriptionSegment,
): string {
    return cleanCaptionText(joinCaptionTexts(left.text, right.text));
}

function canMerge(
    left: TranscriptionSegment,
    right: TranscriptionSegment,
): boolean {
    // NEVER across a speaker boundary — the other half of the guarantee
    // `shouldBreakBefore` opens with. Without this, the split it just made is
    // undone one function later: a one-word answer ("Yes.") is a runt by both the
    // word-count and the duration test, it fits inside MAX_CAPTION_CHARS and
    // SHORT_MERGE_MAX_DURATION with room to spare, so `mergeShortCaptions` would
    // glue it onto the previous speaker's cue and attribute it to them. Pinned by
    // "breaks on speaker change even when under the char and duration limits".
    if (left.speaker !== right.speaker) {
        return false;
    }

    const mergedText = mergeCaptionText(left, right);
    return (
        mergedText.replace(/\n/g, " ").length <= MAX_CAPTION_CHARS &&
        right.end - left.start <= SHORT_MERGE_MAX_DURATION
    );
}

function mergeShortCaptions(
    captions: TranscriptionSegment[],
): TranscriptionSegment[] {
    const merged: TranscriptionSegment[] = [];

    for (const short of captions) {
        const isShort =
            countCaptionWords(short.text) < MIN_CAPTION_WORDS ||
            short.end - short.start < MIN_CAPTION_DURATION;

        if (
            isShort &&
            merged.length > 0 &&
            canMerge(merged[merged.length - 1], short)
        ) {
            const previous = merged.pop()!;
            merged.push(
                caption(
                    previous.start,
                    short.end,
                    wrapCaptionText(mergeCaptionText(previous, short)),
                    // Identical on both sides — `canMerge` refused otherwise.
                    previous.speaker,
                ),
            );
            continue;
        }

        merged.push(short);
    }

    return merged;
}

function capitalizeFirstAlpha(text: string): string {
    const match = text.match(/[A-Za-z]/);
    if (!match || match.index === undefined) {
        return text;
    }
    const index = match.index;
    return (
        text.slice(0, index) + text[index].toUpperCase() + text.slice(index + 1)
    );
}

function normalizeCaptionStarts(
    captions: TranscriptionSegment[],
): TranscriptionSegment[] {
    const normalized: TranscriptionSegment[] = [];

    for (const cue of captions) {
        let text = cue.text;
        const previous = normalized[normalized.length - 1];
        if (!previous || endsSentence(previous.text.replace(/\n/g, " "))) {
            text = capitalizeFirstAlpha(text);
        }
        normalized.push(caption(cue.start, cue.end, text, cue.speaker));
    }

    if (
        normalized.length > 0 &&
        !endsSentence(
            normalized[normalized.length - 1].text.replace(/\n/g, " "),
        )
    ) {
        const final = normalized.pop()!;
        const terminator = sentenceTerminatorFor(final.text);
        normalized.push(
            caption(
                final.start,
                final.end,
                `${final.text}${terminator}`,
                final.speaker,
            ),
        );
    }

    return normalized;
}

/**
 * The full stop to close an unterminated final cue with.
 *
 * An ASCII "." on a Chinese or Japanese cue is simply the wrong character —
 * "光合作用." — and these cues are reachable now that C2 made CJK work. Use the
 * ideographic full stop. Thai, Lao and Myanmar get nothing: those scripts do not
 * end a sentence with a period at all (Thai marks it with a space), so any
 * terminator would be an invented mark. `endsSentence` already recognises 。, so
 * the cue this produces is terminated.
 */
function sentenceTerminatorFor(text: string): string {
    const stripped = text.replace(/\n/g, " ").trimEnd();
    if (ENDS_CJK_RE.test(stripped)) return "。";
    if (ENDS_NO_SPACE_SCRIPT_RE.test(stripped)) return "";
    return ".";
}

/**
 * Every place this text is ALLOWED to break, as the two lines it would produce.
 *
 * Splitting on `" "` — which is all this used to do — offers no break at all in a
 * script that has no spaces, so a Chinese or Japanese cue came back as one line
 * however long and `MAX_LINE_CHARS` was silently violated on the exact cues C2
 * made reachable. Two kinds of break point:
 *
 *   - at a space, which is consumed by the break (English);
 *   - between two adjacent no-space-script characters, which are both kept (CJK,
 *     Thai, Lao, Myanmar) — minus the positions kinsoku forbids, so a line never
 *     opens with 。 or 、.
 *
 * For space-separated text this yields exactly the old word-boundary candidates.
 */
function lineBreakCandidates(cleaned: string): Array<[string, string]> {
    const breaks: Array<[string, string]> = [];

    for (let i = 1; i < cleaned.length; i++) {
        const previous = cleaned[i - 1];
        const current = cleaned[i];

        if (current === " ") continue;

        if (previous === " ") {
            breaks.push([cleaned.slice(0, i - 1), cleaned.slice(i)]);
            continue;
        }

        if (
            IS_NO_SPACE_SCRIPT_RE.test(previous) &&
            IS_NO_SPACE_SCRIPT_RE.test(current) &&
            !NO_LINE_START_RE.test(current)
        ) {
            breaks.push([cleaned.slice(0, i), cleaned.slice(i)]);
        }
    }

    return breaks;
}

function wrapCaptionText(text: string): string {
    const cleaned = cleanCaptionText(text);
    if (cleaned.length <= MAX_LINE_CHARS) {
        return cleaned;
    }

    type Candidate = {
        overflow: number;
        orphanPenalty: number;
        balance: number;
        joined: string;
    };
    const candidates: Candidate[] = [];

    for (const [first, second] of lineBreakCandidates(cleaned)) {
        const overflow =
            Math.max(0, first.length - MAX_LINE_CHARS) +
            Math.max(0, second.length - MAX_LINE_CHARS);
        // `countCaptionWords`, not `split(" ").length`: a Chinese line is one
        // whitespace-token however long, so the space-counting version called
        // every CJK line an orphan and the penalty stopped discriminating.
        const orphanPenalty =
            countCaptionWords(first) === 1 || countCaptionWords(second) === 1
                ? 1
                : 0;
        const balance = Math.abs(first.length - second.length);
        candidates.push({
            overflow,
            orphanPenalty,
            balance,
            joined: `${first}\n${second}`,
        });
    }

    if (candidates.length === 0) {
        return cleaned;
    }

    candidates.sort((a, b) => {
        if (a.overflow !== b.overflow) return a.overflow - b.overflow;
        if (a.orphanPenalty !== b.orphanPenalty)
            return a.orphanPenalty - b.orphanPenalty;
        return a.balance - b.balance;
    });

    return candidates[0].joined;
}

/**
 * Do these segments carry REAL word times, or will they be FABRICATED?
 *
 * This is the question speaker alignment turns on, and it must be ASKED, never
 * assumed — a transcript's granularity is not knowable from where it came from.
 * `job.segments` off a cache hit may be either (see the doc comment on
 * `store.rs::find_transcript`): the model rename aliased old rows to the new
 * model id, and nothing supersedes a legacy row once it exists, so pre-rename
 * transcripts persist indefinitely and are served for the same sources forever.
 *
 * The two shapes, and how they are told apart:
 *
 *   - WORD-GRANULAR (`segmentsForPersistence` wrote it): one word per row, each
 *     with the model's own DTW-measured start and end. Every row is one word, so
 *     `countCaptionWords` of it is 1.
 *   - SENTENCE-GRANULAR (legacy): one Whisper chunk per row. The chunk's OWN times
 *     are real, but the word times `tokensFromSegments` derives inside it are
 *     interpolated by character length — off by 0.26s on average and up to 1.30s.
 *     A row holds a sentence, so `countCaptionWords` of it is >1.
 *
 * COUNT THE CLEANED TEXT, because that is the text the tokenizer will split.
 * `countCaptionWords(raw)` and `splitWords` are two different notions of "word",
 * and this predicate's only job is to predict what `tokensFromSegments` will do —
 * so it must use the one the tokenizer uses. `splitWords` runs `cleanCaptionText`
 * FIRST, and cleaning puts back the space Whisper drops after a punctuation mark:
 * a row of "Hello,world" is ONE word raw and TWO words as the tokenizer sees it.
 * Count it raw and the row is called word-granular, `tokensFromSegments`
 * fabricates two word times by character-length interpolation, and the speaker
 * turns are aligned against numbers nobody measured — precisely the failure this
 * predicate exists to prevent. Pinned by "counts the CLEANED text, so a glued
 * 'Hello,world' row is not word-granular".
 *
 * `countCaptionWords`, not `split(" ")`, and the difference is worth being precise
 * about because a whitespace count is WRONG here in a way that currently hides.
 *
 * A legacy Chinese row ("光合作用是植物") contains no spaces at all, so a whitespace
 * count reports ONE word and declares an entire legacy CJK transcript
 * word-granular. `countCaptionWords` counts each no-space-script character as its
 * own word — the way the model tokenised them, and the way they are persisted — so
 * it sees the sentence for what it is.
 *
 * TODAY, both answers happen to produce the same CUES for that row, and the honest
 * reason is that `tokensFromSegments` is blind to CJK in exactly the same way:
 * `splitWords` splits on spaces, so a legacy Chinese segment becomes ONE token
 * spanning the whole segment, and aligning that token is aligning the segment.
 * Two blindnesses cancelling is not a safety property. Teach `tokensFromSegments`
 * to split CJK — which it arguably should, since a legacy Chinese transcript
 * currently comes back as one un-wrappable cue — and a whitespace-based detector
 * becomes a live bug that hangs speaker labels off per-character times nobody
 * measured. So the predicate is correct ON ITS OWN TERMS, and is tested directly
 * ("tells a legacy Chinese sentence from a word-granular Chinese row") rather than
 * through cues it does not currently change.
 *
 * Degenerate case, stated because it is real and harmless: a transcript whose every
 * legacy chunk happens to be one word is called word-granular. Its chunk times ARE
 * its word times then, so both alignments agree and the answer is the same.
 *
 * Exported for that direct test, and for callers that must decide the same thing.
 */
export function hasRealWordTimings(
    segments: readonly TranscriptionSegment[],
): boolean {
    return (
        segments.length > 0 &&
        segments.every(
            (segment) =>
                countCaptionWords(cleanCaptionText(segment.text)) === 1,
        )
    );
}

/**
 * The tokens the cue splitter runs on, each already knowing who said it.
 *
 * WHICH GRANULARITY the speakers are aligned at is the whole point of this
 * function, and it is decided per-transcript:
 *
 *   - REAL word times (live from the model, or read back from word-granular rows)
 *     -> align the WORDS. A turn boundary lands between two words, which is where
 *     it actually is.
 *   - FABRICATED word times (a legacy sentence-granular row) -> align the
 *     SEGMENTS, and stamp each segment's speaker on every token derived from it.
 *     Coarse: a cue cannot then change speaker mid-segment. But the segment times
 *     are MEASURED, so the label is true — where aligning turns against word times
 *     that are up to 1.3s adrift (longer than many turns) would hand back a
 *     confidently wrong speaker, which is strictly worse than a coarse right one.
 *
 * `assignSpeakers` is generic over `{text,start,end}` for exactly this reason, so
 * both paths are the same function over different tokens.
 */
function tokensWithSpeakers(
    segments: RawSegment[],
    words: readonly WordToken[] | undefined,
    turns: readonly SpeakerTurn[] | null,
): WordToken[] {
    if (words && words.length > 0) {
        const wordTokens = normalizeWordTokens(words);
        return turns ? assignSpeakersToTokens(wordTokens, turns) : wordTokens;
    }

    if (turns && !hasRealWordTimings(segments)) {
        return tokensFromSegments(assignSpeakersToSegments(segments, turns));
    }

    // Either word-granular rows (whose times are the model's own, so aligning the
    // tokens IS aligning the words) or no diarization at all, in which case any
    // speaker already persisted on a row rides through `tokensFromSegments`
    // untouched.
    const tokens = tokensFromSegments(segments);
    return turns ? assignSpeakersToTokens(tokens, turns) : tokens;
}

/**
 * `assignSpeakers` answers with the turn's own id (a number, or null when there
 * are no turns). The transcript carries an opaque LABEL. These two adapters are
 * the only place the one becomes the other.
 *
 * They pass `assignSpeakers` a bare `{text,start,end}` rather than the token or
 * segment itself, deliberately: `SpeakerTagged<T>` sets `speaker: number | null`,
 * which would collide with the `speaker?: string` these types already carry.
 *
 * EXPORTED so that `segmentsForPersistence` labels the rows it writes with the
 * same two functions the screen labels its cues with. That sharing is the reason
 * a transcript reloaded from the database renders identically to the one that was
 * on screen when it was saved: two independent labelling paths would be two
 * things to drift, and the drift would be invisible until a user reopened a file.
 */
export function assignSpeakersToTokens(
    tokens: WordToken[],
    turns: readonly SpeakerTurn[],
): WordToken[] {
    const tagged = assignSpeakers(timedOnly(tokens), [...turns]);
    return tokens.map((t, index) =>
        token(t.text, t.start, t.end, labelOf(tagged[index].speaker)),
    );
}

export function assignSpeakersToSegments(
    segments: RawSegment[],
    turns: readonly SpeakerTurn[],
): RawSegment[] {
    const tagged = assignSpeakers(timedOnly(segments), [...turns]);
    return segments.map((segment, index) => {
        const speaker = labelOf(tagged[index].speaker);
        return speaker === undefined ? segment : { ...segment, speaker };
    });
}

function timedOnly(
    items: readonly { text: string; start: number; end: number }[],
): { text: string; start: number; end: number }[] {
    return items.map(({ text, start, end }) => ({ text, start, end }));
}

/**
 * Render an assigned turn id as a label.
 *
 * WHAT THIS DOES NOT DO, stated plainly because an earlier version of this comment
 * claimed the opposite: it does NOT leave a non-overlapping token unlabelled. A
 * token that overlaps no turn is assigned the NEAREST turn's speaker by
 * `assignSpeakers` (`fill_nearest`, Task 5's deliberate choice, documented at
 * `speakerAlignment.ts`), and it arrives here as a real number. So a word far from
 * every turn IS given a speaker — see "labels a token that overlaps no turn with
 * the nearest turn's speaker", which pins exactly that.
 *
 * `assignSpeakers` answers `null` in ONE case only: `turns.length === 0`. That case
 * cannot reach this function. `consolidateSegments` collapses empty turns to `null`
 * and `tokensWithSpeakers` then skips speaker assignment entirely, so both adapters
 * above are only ever called with a NON-EMPTY turns array.
 *
 * The `null` branch below is therefore unreachable at runtime today. It is kept
 * because `assignSpeakers` is generic and its return type (`number | null`) admits
 * null for the zero-turns case; this is a total function over that signature, not a
 * safety net. Same for the `undefined` check in `assignSpeakersToSegments`. Neither
 * guards against fill_nearest — nothing here does.
 */
function labelOf(speaker: number | null): string | undefined {
    return speaker === null ? undefined : speakerLabel(speaker);
}

/**
 * Turn RAW model segments into display/export captions. This is the single
 * formatting boundary: it must be called exactly ONCE, on raw segments, and its
 * output fed unchanged to both the screen and the exporters.
 *
 * `words` is the model's real per-word timing (`return_timestamps: 'word'`).
 * When it is supplied, EVERY decision below — where to cut a cue, how long it
 * lasts — is made from measured times. When it is not (a transcript read back
 * from the database that was made before those existed), the word times are
 * fabricated from each segment's span by character length, and every one of
 * those decisions is made from a number nobody measured.
 *
 * It is deliberately NOT idempotent, and cannot cheaply be made so on the
 * fabricating path. Word times inside a cue are interpolated from the raw
 * segment that produced them; once cues exist, that provenance is gone, and a
 * second pass re-interpolates each word evenly across its cue's span instead.
 * Different word times mean `shouldBreakBefore` cuts in different places, so
 * words migrate between cues. `captionFormatter.test.ts` pins a two-segment
 * counterexample where a second pass moves a word to the previous cue and shifts
 * the boundary by 1.46s.
 *
 * Consequence: never call this on its own output. That is enforced by the type
 * system, not by convention: it consumes raw `TranscriptionSegment[]` and
 * produces branded `ConsolidatedSegment[]`, which it will not accept back.
 * `lib/srtGenerator.ts` is a set of pure serializers over `ConsolidatedSegment[]`
 * precisely so that the export path cannot re-format.
 *
 * `turns` is diarization's answer, and it is deliberately typed `SpeakerTurn[]`
 * and NOT `DiarizationOutcome`. A caller cannot reach turns without narrowing
 * that union to its `succeeded` arm first, so the two things that both LOOK like
 * "no speakers" cannot arrive here as the same value:
 *
 *   - `succeeded { turns: [] }` — a real, measured answer (silence, or one
 *     speaker). Passing `[]`, or nothing at all, is correct: no cue gets a
 *     speaker, and the transcript renders exactly as it did before diarization
 *     existed.
 *   - `degraded { reason }` — the sidecar crashed, or its model is missing. It
 *     has NO `turns` key, cannot be coerced into one (`outcome.turns ?? []` does
 *     not compile, and a test pins that), and so cannot reach this function at
 *     all. It is not this function's answer to give; the caller must SHOW the
 *     reason. Silently consolidating with no turns would render a crash as
 *     silence, which is the one failure the union exists to prevent.
 *
 * When turns ARE supplied, the alignment granularity is chosen by
 * `hasRealWordTimings`, not assumed. See it.
 */
export function consolidateSegments(
    segments: RawSegment[],
    words?: readonly WordToken[],
    turns?: readonly SpeakerTurn[],
): ConsolidatedSegment[] {
    const tokens = tokensWithSpeakers(
        segments,
        words,
        turns && turns.length > 0 ? turns : null,
    );
    if (tokens.length === 0) {
        return [];
    }

    const captions: TranscriptionSegment[] = [];
    let current: WordToken[] = [];

    for (const next of tokens) {
        if (current.length > 0 && shouldBreakBefore(current, next)) {
            captions.push(captionFromTokens(current));
            current = [];
        }
        current.push(next);
    }

    if (current.length > 0) {
        captions.push(captionFromTokens(current));
    }

    // The single site that mints the brand. Everything above this line works in
    // plain TranscriptionSegments; everything downstream of it is a cue.
    return normalizeCaptionStarts(
        mergeShortCaptions(captions),
    ) as ConsolidatedSegment[];
}
