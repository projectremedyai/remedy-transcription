//! Turning an Interactions response into this app's transcript shape.
//!
//! Pure: no HTTP, no filesystem, no Tauri. Everything here is exercised by
//! `cargo test` with hand-written JSON.

// Not called yet: Task 11's orchestration is what stitches per-chunk client
// responses through this module to build the final transcript. Allowed dead
// here rather than adding a premature caller (out of scope for this task), so
// `cargo check` stays warning-clean at this checkpoint; the tests are the
// only caller until then. Task 11 must remove this attribute.
#![allow(dead_code)]

use anyhow::{anyhow, bail};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
    /// Gemini's own opaque id (`spk_1`). Densified by `turns_from_words`; never
    /// sent to the frontend as-is.
    #[serde(skip)]
    pub speaker: Option<String>,
}

/// Matches the frontend's `SpeakerTurn` field-for-field.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpeakerTurn {
    pub start: f64,
    pub end: f64,
    pub speaker: u32,
}

/// `"0.100s"` -> `0.1`.
///
/// Rejects non-finite results: `f64::from_str` happily accepts `"NaN"` and
/// `"inf"`/`"-inf"` as valid floats, and a NaN or infinite offset would flow
/// straight into `Word.start`/`end` and corrupt every downstream comparison
/// (`shift`, the time sort in `words_from_response`, turn stitching) without
/// ever tripping an error. Also rejects negatives: a word cannot start before
/// t=0.
pub fn parse_offset(raw: &str) -> anyhow::Result<f64> {
    let trimmed = raw.trim().strip_suffix('s').unwrap_or(raw.trim());
    if trimmed.is_empty() {
        bail!("empty duration offset");
    }
    let value = trimmed
        .parse::<f64>()
        .map_err(|_| anyhow!("could not read {raw:?} as a duration offset"))?;
    if !value.is_finite() {
        bail!("duration offset {raw:?} is not a finite number");
    }
    if value < 0.0 {
        bail!("duration offset {raw:?} is negative");
    }
    Ok(value)
}

/// Every object tagged `"type": "word_info"`, wherever it sits.
///
/// A structural search rather than a fixed path: the published docs pin the
/// SHAPE of a word annotation but describe its envelope only loosely, and a
/// wrong path would fail silently by finding zero words rather than loudly.
pub fn collect_word_infos(value: &serde_json::Value) -> Vec<&serde_json::Value> {
    let mut found = Vec::new();
    walk(value, &mut found);
    found
}

fn walk<'a>(value: &'a serde_json::Value, found: &mut Vec<&'a serde_json::Value>) {
    match value {
        serde_json::Value::Object(map) => {
            if map.get("type").and_then(|t| t.as_str()) == Some("word_info") {
                found.push(value);
            }
            for nested in map.values() {
                walk(nested, found);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                walk(item, found);
            }
        }
        _ => {}
    }
}

pub fn words_from_response(value: &serde_json::Value) -> anyhow::Result<Vec<Word>> {
    let infos = collect_word_infos(value);
    if infos.is_empty() {
        bail!(
            "Gemini returned no word timestamps. The transcript cannot be turned \
             into captions without them."
        );
    }

    let mut words = infos
        .into_iter()
        .map(|info| {
            Ok(Word {
                text: info["text"].as_str().unwrap_or_default().to_string(),
                start: parse_offset(info["start_offset"].as_str().unwrap_or(""))?,
                end: parse_offset(info["end_offset"].as_str().unwrap_or(""))?,
                speaker: info["speaker"].as_str().map(|s| s.to_string()),
            })
        })
        .collect::<anyhow::Result<Vec<Word>>>()?;

    // `collect_word_infos`'s traversal order is NOT speech order: this crate
    // builds `serde_json::Value::Object` on a `BTreeMap` (no `preserve_order`
    // feature), so a walk visits sibling keys alphabetically. If the real
    // envelope ever spreads `word_info` nodes across multiple keys (the docs
    // describe them living in "steps AND content sections"), alphabetical
    // order would scramble the transcript. Sorting by `start` derives order
    // from the API's own timestamps -- the one thing the docs do pin -- so
    // this is correct under any envelope shape, not just the ones tested
    // here. `total_cmp` is safe only because `parse_offset` above already
    // rejected non-finite values; a stable sort keeps equal-start words in
    // traversal order.
    words.sort_by(|a, b| a.start.total_cmp(&b.start));

    Ok(words)
}

pub fn shift(words: &mut [Word], offset_secs: f64) {
    for word in words {
        word.start += offset_secs;
        word.end += offset_secs;
    }
}

/// Collapse contiguous same-speaker runs into turns, densifying the ids.
///
/// Returns `(turns, speaker_count)`. `speaker_count` is the number of DISTINCT
/// speakers, not the number of turns.
///
/// An unattributed word (`speaker: None`) ENDS the current turn rather than
/// being silently spanned by it: Gemini declined to say who was speaking
/// then, so letting the previous speaker's turn stretch across that gap
/// would claim something the response never said. The next attributed word
/// starts a new turn even if it is the same speaker as before the gap. The
/// resulting fragmentation is harmless downstream: `speaker_count` counts
/// distinct speakers, not turns, and the frontend's cue splitter only breaks
/// on a speaker-label change, so two adjacent turns by the same speaker
/// produce no visible cue break.
pub fn turns_from_words(words: &[Word]) -> (Vec<SpeakerTurn>, u32) {
    let mut order: Vec<&str> = Vec::new();
    let mut turns: Vec<SpeakerTurn> = Vec::new();
    let mut turn_open = false;

    for word in words {
        let Some(raw) = word.speaker.as_deref() else {
            turn_open = false;
            continue;
        };
        let dense = match order.iter().position(|seen| *seen == raw) {
            Some(index) => index as u32,
            None => {
                order.push(raw);
                (order.len() - 1) as u32
            }
        };

        match turns.last_mut() {
            Some(last) if turn_open && last.speaker == dense => last.end = word.end,
            _ => turns.push(SpeakerTurn {
                start: word.start,
                end: word.end,
                speaker: dense,
            }),
        }
        turn_open = true;
    }

    (turns, order.len() as u32)
}

/// Joined from the words, NOT from `output_text`: in verbatim mode the words
/// ARE the transcript, and joining them guarantees the cues and `full_text`
/// agree.
pub fn full_text(words: &[Word]) -> String {
    words
        .iter()
        .map(|word| word.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn offsets_are_parsed_from_the_trailing_s_form() {
        assert_eq!(parse_offset("0.100s").unwrap(), 0.1);
        assert_eq!(parse_offset("12s").unwrap(), 12.0);
        assert_eq!(parse_offset("1234.567s").unwrap(), 1234.567);
    }

    #[test]
    fn a_malformed_offset_is_an_error_not_a_zero() {
        assert!(parse_offset("").is_err());
        assert!(parse_offset("abc").is_err());
        assert!(parse_offset("s").is_err());
    }

    /// `f64::from_str` happily parses `"NaN"`/`"inf"`/`"-inf"` as valid floats.
    /// A NaN or infinite offset would flow straight into `Word.start`/`end`
    /// and corrupt every downstream comparison (`shift`, turn stitching,
    /// sorting) without ever tripping an error. Negative offsets are rejected
    /// too: a word cannot start before t=0.
    #[test]
    fn non_finite_and_negative_offsets_are_errors() {
        assert!(parse_offset("NaNs").is_err());
        assert!(parse_offset("infs").is_err());
        assert!(parse_offset("-infs").is_err());
        assert!(parse_offset("-3.5s").is_err());
    }

    /// The envelope around `word_info` is not pinned by the published docs, so
    /// the parser searches structurally. This test IS the contract: nesting
    /// depth and key names along the way must not matter.
    #[test]
    fn word_infos_are_found_at_any_depth_in_the_response() {
        let response = json!({
            "steps": [{ "content": [
                { "type": "word_info", "text": "Hello", "start_offset": "0.1s", "end_offset": "0.4s" },
                { "type": "something_else", "text": "ignore me" }
            ]}],
            "output_text": "Hello"
        });
        assert_eq!(collect_word_infos(&response).len(), 1);
    }

    #[test]
    fn words_carry_their_times_and_speaker() {
        let response = json!({"c": [
            { "type": "word_info", "text": "Hi", "speaker": "spk_1",
              "start_offset": "0.1s", "end_offset": "0.4s" }
        ]});
        let words = words_from_response(&response).unwrap();
        assert_eq!(words[0].text, "Hi");
        assert_eq!(words[0].start, 0.1);
        assert_eq!(words[0].end, 0.4);
        assert_eq!(words[0].speaker.as_deref(), Some("spk_1"));
    }

    /// A response with no word annotations at all means timestamps did not
    /// come back. Returning an empty transcript would persist a blank
    /// transcript under the content cache and serve it back forever.
    #[test]
    fn a_response_with_no_words_is_an_error() {
        assert!(words_from_response(&json!({"output_text": "hi"})).is_err());
    }

    /// `serde_json::Value::Object` here is a `BTreeMap` (no `preserve_order`
    /// feature), so plain tree traversal visits keys alphabetically, not in
    /// speech order. If the real envelope ever spreads `word_info` nodes
    /// across sibling keys, alphabetical traversal would silently scramble
    /// the transcript. Sorting by `start` after parsing derives order from
    /// the API's own timestamps -- the ground truth -- instead of from
    /// whatever the envelope's key names happen to be, which is what keeps
    /// this correct under an envelope shape the docs don't pin.
    #[test]
    fn words_come_back_time_ordered_even_when_the_envelope_keys_are_not() {
        // Speech order is zulu_word (t=0.0) then alpha_word (t=1.0); key
        // order is the reverse alphabetically, so a naive traversal would
        // yield "second" before "first".
        let response = json!({
            "zulu_word": { "type": "word_info", "text": "first", "start_offset": "0.0s", "end_offset": "0.5s" },
            "alpha_word": { "type": "word_info", "text": "second", "start_offset": "1.0s", "end_offset": "1.5s" }
        });
        let words = words_from_response(&response).unwrap();
        assert_eq!(
            words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
            vec!["first", "second"]
        );
    }

    #[test]
    fn shifting_moves_every_word_by_the_chunk_offset() {
        let mut words = vec![
            Word { text: "a".into(), start: 0.0, end: 1.0, speaker: None },
            Word { text: "b".into(), start: 1.0, end: 2.0, speaker: None },
        ];
        shift(&mut words, 1500.0);
        assert_eq!(words[0].start, 1500.0);
        assert_eq!(words[1].end, 1502.0);
    }

    /// Gemini's ids are sparse and arbitrary. `speakerLabel` in the frontend
    /// renders whatever number it is given, so densifying here is what makes
    /// the labels read SPEAKER_00, SPEAKER_01, SPEAKER_02 -- exactly what the
    /// deleted sherpa path produced, so the rename and export code is untouched.
    #[test]
    fn sparse_speaker_ids_are_densified_in_order_of_first_appearance() {
        let words = vec![
            Word { text: "a".into(), start: 0.0, end: 1.0, speaker: Some("spk_4".into()) },
            Word { text: "b".into(), start: 1.0, end: 2.0, speaker: Some("spk_1".into()) },
            Word { text: "c".into(), start: 2.0, end: 3.0, speaker: Some("spk_7".into()) },
            Word { text: "d".into(), start: 3.0, end: 4.0, speaker: Some("spk_1".into()) },
        ];
        let (turns, count) = turns_from_words(&words);
        assert_eq!(count, 3);
        assert_eq!(turns.iter().map(|t| t.speaker).collect::<Vec<_>>(), vec![0, 1, 2, 1]);
    }

    #[test]
    fn contiguous_words_by_one_speaker_collapse_into_a_single_turn() {
        let words = vec![
            Word { text: "a".into(), start: 0.0, end: 1.0, speaker: Some("spk_1".into()) },
            Word { text: "b".into(), start: 1.0, end: 2.0, speaker: Some("spk_1".into()) },
            Word { text: "c".into(), start: 2.0, end: 3.0, speaker: Some("spk_2".into()) },
        ];
        let (turns, count) = turns_from_words(&words);
        assert_eq!(count, 2);
        assert_eq!(turns.len(), 2);
        assert_eq!((turns[0].start, turns[0].end), (0.0, 2.0));
    }

    #[test]
    fn words_with_no_speaker_produce_no_turns() {
        let words = vec![Word { text: "a".into(), start: 0.0, end: 1.0, speaker: None }];
        let (turns, count) = turns_from_words(&words);
        assert!(turns.is_empty());
        assert_eq!(count, 0);
    }

    /// An unattributed word must END the current turn rather than being
    /// silently spanned by it: Gemini declined to say who was speaking then,
    /// so claiming the previous speaker occupied that timespan would be a
    /// fabrication. Same speaker on both sides of the gap must still produce
    /// two turns, not one merged one.
    #[test]
    fn an_unattributed_word_ends_the_turn_even_if_the_same_speaker_returns() {
        let words = vec![
            Word { text: "a".into(), start: 0.0, end: 1.0, speaker: Some("spk_1".into()) },
            Word { text: "b".into(), start: 1.0, end: 2.0, speaker: None },
            Word { text: "c".into(), start: 2.0, end: 3.0, speaker: Some("spk_1".into()) },
        ];
        let (turns, count) = turns_from_words(&words);
        assert_eq!(count, 1);
        assert_eq!(turns.len(), 2);
        assert_eq!((turns[0].start, turns[0].end), (0.0, 1.0));
        assert_eq!((turns[1].start, turns[1].end), (2.0, 3.0));
    }

    /// Same shape as above but with a different speaker after the gap, to
    /// prove the behaviour no longer depends on what comes next: both cases
    /// now produce a two-turn split with a silent gap over the unattributed
    /// word, rather than one merging and the other not.
    #[test]
    fn an_unattributed_word_before_a_different_speaker_still_leaves_a_gap() {
        let words = vec![
            Word { text: "a".into(), start: 0.0, end: 1.0, speaker: Some("spk_1".into()) },
            Word { text: "b".into(), start: 1.0, end: 2.0, speaker: None },
            Word { text: "c".into(), start: 2.0, end: 3.0, speaker: Some("spk_2".into()) },
        ];
        let (turns, count) = turns_from_words(&words);
        assert_eq!(count, 2);
        assert_eq!(turns.len(), 2);
        assert_eq!((turns[0].start, turns[0].end), (0.0, 1.0));
        assert_eq!((turns[1].start, turns[1].end), (2.0, 3.0));
    }

    /// full_text is joined from the WORDS, not taken from `output_text`, so
    /// the cues and full_text can never disagree -- something the local
    /// Whisper path explicitly cannot promise.
    #[test]
    fn full_text_is_the_words_joined_with_single_spaces() {
        let words = vec![
            Word { text: "Hello".into(), start: 0.0, end: 1.0, speaker: None },
            Word { text: "there".into(), start: 1.0, end: 2.0, speaker: None },
        ];
        assert_eq!(full_text(&words), "Hello there");
    }
}
