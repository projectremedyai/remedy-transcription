export interface CaptionChunk {
    text: string;
    timestamp: [number, number | null];
}

interface WordToken {
    text: string;
    start: number;
    end: number;
}

const MAX_LINE_CHARS = 42;
const MAX_CAPTION_CHARS = 78;
const MAX_CAPTION_DURATION = 6.5;
const SHORT_MERGE_MAX_DURATION = 8.0;
const MIN_CAPTION_DURATION = 1.2;
const MIN_CAPTION_WORDS = 3;
const MAX_OVERLAP_WORDS = 16;

const WHITESPACE_RE = /\s+/g;
const SPACE_BEFORE_PUNCT_RE = /\s+([,.;:!?])/g;
const MISSING_SPACE_AFTER_PUNCT_RE = /([,.;:!?])(?=[^\s"'])/g;
const WORD_NORMALIZE_RE = /[^a-z0-9']+/g;
const SENTENCE_END_RE = /[.!?]["')\]]*$/;
const CLAUSE_END_RE = /[,;:]["')\]]*$/;

function cleanCaptionText(text: string): string {
    return text
        .trim()
        .replace(WHITESPACE_RE, " ")
        .replace(SPACE_BEFORE_PUNCT_RE, "$1")
        .replace(MISSING_SPACE_AFTER_PUNCT_RE, "$1 ")
        .replace(/\bi\b/g, "I")
        .trim();
}

function normalizeWord(word: string): string {
    return word.toLowerCase().replace(WORD_NORMALIZE_RE, "");
}

function splitWords(text: string): string[] {
    return cleanCaptionText(text).split(" ").filter(Boolean);
}

function overlapPrefixSize(
    previousWords: string[],
    currentWords: string[],
): number {
    const previous = previousWords.map(normalizeWord).filter(Boolean);
    const current = currentWords.map(normalizeWord).filter(Boolean);
    const maxOverlap = Math.min(
        MAX_OVERLAP_WORDS,
        previous.length,
        current.length,
    );

    for (let size = maxOverlap; size > 2; size -= 1) {
        if (
            previous.slice(-size).join("\u0000") ===
            current.slice(0, size).join("\u0000")
        ) {
            return size;
        }
    }

    return 0;
}

function chunkEnd(chunks: CaptionChunk[], index: number): number {
    const [start, explicitEnd] = chunks[index].timestamp;
    if (explicitEnd !== null) {
        return explicitEnd;
    }
    const nextStart = chunks[index + 1]?.timestamp[0];
    return nextStart ?? start + 4;
}

function tokensFromChunks(chunks: CaptionChunk[]): WordToken[] {
    const tokens: WordToken[] = [];
    let emittedWords: string[] = [];

    chunks.forEach((chunk, index) => {
        let words = splitWords(chunk.text);
        if (words.length === 0) {
            return;
        }

        const overlap = overlapPrefixSize(emittedWords, words);
        const segmentStart = chunk.timestamp[0];
        const segmentEnd = chunkEnd(chunks, index);
        const duration = Math.max(segmentEnd - segmentStart, 0.01);
        const start = segmentStart + (duration * overlap) / words.length;
        words = words.slice(overlap);
        if (words.length === 0) {
            return;
        }

        const usableDuration =
            duration * (words.length / (words.length + overlap));
        const totalWeight = words.reduce(
            (sum, word) => sum + Math.max(word.length, 1),
            0,
        );
        let elapsed = 0;

        words.forEach((word) => {
            const weight = Math.max(word.length, 1) / totalWeight;
            const wordStart = start + elapsed * usableDuration;
            elapsed += weight;
            const wordEnd = start + elapsed * usableDuration;
            tokens.push({ text: word, start: wordStart, end: wordEnd });
        });

        emittedWords = [...emittedWords, ...words].slice(-MAX_OVERLAP_WORDS);
    });

    return tokens;
}

function joinWords(tokens: WordToken[]): string {
    return cleanCaptionText(tokens.map((token) => token.text).join(" "));
}

function endsSentence(text: string): boolean {
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

    return (
        currentDuration >= 4.5 &&
        currentWords >= MIN_CAPTION_WORDS &&
        currentText.length >= 56 &&
        endsClause(current[current.length - 1].text)
    );
}

function wrapCaptionText(text: string): string {
    const cleanText = cleanCaptionText(text);
    if (cleanText.length <= MAX_LINE_CHARS) {
        return cleanText;
    }

    const words = cleanText.split(" ");
    const candidates = words.slice(1).map((_, offset) => {
        const index = offset + 1;
        const first = words.slice(0, index).join(" ");
        const second = words.slice(index).join(" ");
        const overflow =
            Math.max(0, first.length - MAX_LINE_CHARS) +
            Math.max(0, second.length - MAX_LINE_CHARS);
        const orphanPenalty =
            first.split(" ").length === 1 || second.split(" ").length === 1
                ? 1
                : 0;
        const balance = Math.abs(first.length - second.length);
        return {
            score: [overflow, orphanPenalty, balance],
            text: `${first}\n${second}`,
        };
    });

    candidates.sort((left, right) => {
        for (let index = 0; index < left.score.length; index += 1) {
            const diff = left.score[index] - right.score[index];
            if (diff !== 0) {
                return diff;
            }
        }
        return 0;
    });

    return candidates[0]?.text ?? cleanText;
}

function chunkFromTokens(tokens: WordToken[]): CaptionChunk {
    return {
        text: wrapCaptionText(joinWords(tokens)),
        timestamp: [
            Number(tokens[0].start.toFixed(3)),
            Number(tokens[tokens.length - 1].end.toFixed(3)),
        ],
    };
}

function canMerge(left: CaptionChunk, right: CaptionChunk): boolean {
    const mergedText = cleanCaptionText(`${left.text} ${right.text}`);
    const start = left.timestamp[0];
    const end = right.timestamp[1] ?? right.timestamp[0];
    return (
        mergedText.length <= MAX_CAPTION_CHARS &&
        end - start <= SHORT_MERGE_MAX_DURATION
    );
}

function mergeShortCaptions(captions: CaptionChunk[]): CaptionChunk[] {
    const merged: CaptionChunk[] = [];

    captions.forEach((caption) => {
        const wordCount = caption.text.replace(/\n/g, " ").split(/\s+/).length;
        const duration =
            (caption.timestamp[1] ?? caption.timestamp[0]) -
            caption.timestamp[0];
        const isShort =
            wordCount < MIN_CAPTION_WORDS || duration < MIN_CAPTION_DURATION;

        if (
            isShort &&
            merged.length > 0 &&
            canMerge(merged[merged.length - 1], caption)
        ) {
            const previous = merged.pop();
            if (!previous) {
                merged.push(caption);
                return;
            }
            merged.push({
                text: wrapCaptionText(
                    cleanCaptionText(`${previous.text} ${caption.text}`),
                ),
                timestamp: [previous.timestamp[0], caption.timestamp[1]],
            });
            return;
        }

        merged.push(caption);
    });

    return merged;
}

function capitalizeFirstAlpha(text: string): string {
    const index = text.search(/[A-Za-z]/);
    if (index < 0) {
        return text;
    }
    return `${text.slice(0, index)}${text[index].toUpperCase()}${text.slice(
        index + 1,
    )}`;
}

function normalizeCaptionStarts(captions: CaptionChunk[]): CaptionChunk[] {
    const normalized = captions.map((caption, index) => {
        const previous = captions[index - 1];
        const shouldCapitalize =
            index === 0 || endsSentence(previous.text.replace(/\n/g, " "));

        return {
            ...caption,
            text: shouldCapitalize
                ? capitalizeFirstAlpha(caption.text)
                : caption.text,
        };
    });

    const finalCaption = normalized[normalized.length - 1];
    if (finalCaption && !endsSentence(finalCaption.text.replace(/\n/g, " "))) {
        normalized[normalized.length - 1] = {
            ...finalCaption,
            text: `${finalCaption.text}.`,
        };
    }

    return normalized;
}

export function consolidateChunks(chunks: CaptionChunk[]): CaptionChunk[] {
    const tokens = tokensFromChunks(chunks);
    if (tokens.length === 0) {
        return [];
    }

    const captions: CaptionChunk[] = [];
    let current: WordToken[] = [];

    tokens.forEach((token) => {
        if (current.length > 0 && shouldBreakBefore(current, token)) {
            captions.push(chunkFromTokens(current));
            current = [];
        }
        current.push(token);
    });

    if (current.length > 0) {
        captions.push(chunkFromTokens(current));
    }

    return normalizeCaptionStarts(mergeShortCaptions(captions));
}
