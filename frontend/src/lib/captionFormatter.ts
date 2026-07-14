import { TranscriptionSegment } from "../services/types";

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
const NO_SPACE_SCRIPT_CHARS = [
    "\\u0e00-\\u0e7f", // Thai
    "\\u0e80-\\u0eff", // Lao
    "\\u1000-\\u109f", // Myanmar
    "\\u3000-\\u303f", // CJK symbols and punctuation
    "\\u3040-\\u30ff", // Hiragana and Katakana
    "\\u3400-\\u4dbf", // CJK unified ideographs extension A
    "\\u4e00-\\u9fff", // CJK unified ideographs
    "\\uf900-\\ufaff", // CJK compatibility ideographs
    "\\uff00-\\uffef", // Halfwidth and fullwidth forms
].join("");
const STARTS_NO_SPACE_SCRIPT_RE = new RegExp(`^[${NO_SPACE_SCRIPT_CHARS}]`);
const ENDS_NO_SPACE_SCRIPT_RE = new RegExp(`[${NO_SPACE_SCRIPT_CHARS}]$`);
const NO_SPACE_SCRIPT_GLOBAL_RE = new RegExp(`[${NO_SPACE_SCRIPT_CHARS}]`, "g");

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
 *   2. Clamp monotonically. The model's words are already ordered and
 *      non-overlapping in practice, but a cue whose start ran backwards would
 *      produce invalid SRT, so do not take that on trust.
 *   3. Floor each word's duration. DTW does emit `end === start`; see
 *      MIN_WORD_DURATION.
 */
export function normalizeWordTokens(words: readonly WordToken[]): WordToken[] {
    const tokens: WordToken[] = [];

    for (const word of words) {
        const continuesPreviousWord =
            !/^\s/.test(word.text) &&
            !STARTS_NO_SPACE_SCRIPT_RE.test(word.text);
        const text = word.text.trim();
        if (text.length === 0) continue;

        const previous = tokens[tokens.length - 1];
        if (continuesPreviousWord && previous) {
            previous.text += text;
            previous.end = Math.max(
                previous.end,
                word.end,
                previous.start + MIN_WORD_DURATION,
            );
            continue;
        }

        const start = previous
            ? Math.max(word.start, previous.end)
            : word.start;
        tokens.push({
            text,
            start,
            end: Math.max(word.end, start + MIN_WORD_DURATION),
        });
    }

    return tokens;
}

/**
 * FABRICATED word times: spread each segment's duration across its words by
 * character length. Only for input that has no real word timings — a transcript
 * loaded from the database that predates `return_timestamps: 'word'`. Measured
 * against the model's real word times on a 12.8s clip, this is off by 0.26s on
 * average and up to 1.30s. Prefer `normalizeWordTokens`.
 */
function tokensFromSegments(segments: TranscriptionSegment[]): WordToken[] {
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
            tokens.push({ text: word, start: wordStart, end: wordEnd });
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
    return {
        start: round3(first.start),
        end: round3(last.end),
        text: wrapCaptionText(joinWords(tokens)),
    };
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

    for (const caption of captions) {
        const isShort =
            countCaptionWords(caption.text) < MIN_CAPTION_WORDS ||
            caption.end - caption.start < MIN_CAPTION_DURATION;

        if (
            isShort &&
            merged.length > 0 &&
            canMerge(merged[merged.length - 1], caption)
        ) {
            const previous = merged.pop()!;
            merged.push({
                start: previous.start,
                end: caption.end,
                text: wrapCaptionText(mergeCaptionText(previous, caption)),
            });
            continue;
        }

        merged.push(caption);
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

    for (const caption of captions) {
        let text = caption.text;
        const previous = normalized[normalized.length - 1];
        if (!previous || endsSentence(previous.text.replace(/\n/g, " "))) {
            text = capitalizeFirstAlpha(text);
        }
        normalized.push({
            start: caption.start,
            end: caption.end,
            text,
        });
    }

    if (
        normalized.length > 0 &&
        !endsSentence(
            normalized[normalized.length - 1].text.replace(/\n/g, " "),
        )
    ) {
        const final = normalized.pop()!;
        normalized.push({
            start: final.start,
            end: final.end,
            text: `${final.text}.`,
        });
    }

    return normalized;
}

function wrapCaptionText(text: string): string {
    const cleaned = cleanCaptionText(text);
    if (cleaned.length <= MAX_LINE_CHARS) {
        return cleaned;
    }

    const words = cleaned.split(" ");
    type Candidate = {
        overflow: number;
        orphanPenalty: number;
        balance: number;
        joined: string;
    };
    const candidates: Candidate[] = [];

    for (let i = 1; i < words.length; i++) {
        const first = words.slice(0, i).join(" ");
        const second = words.slice(i).join(" ");
        const overflow =
            Math.max(0, first.length - MAX_LINE_CHARS) +
            Math.max(0, second.length - MAX_LINE_CHARS);
        const orphanPenalty =
            first.split(" ").length === 1 || second.split(" ").length === 1
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
 */
export function consolidateSegments(
    segments: RawSegment[],
    words?: readonly WordToken[],
): ConsolidatedSegment[] {
    const tokens =
        words && words.length > 0
            ? normalizeWordTokens(words)
            : tokensFromSegments(segments);
    if (tokens.length === 0) {
        return [];
    }

    const captions: TranscriptionSegment[] = [];
    let current: WordToken[] = [];

    for (const token of tokens) {
        if (current.length > 0 && shouldBreakBefore(current, token)) {
            captions.push(captionFromTokens(current));
            current = [];
        }
        current.push(token);
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
