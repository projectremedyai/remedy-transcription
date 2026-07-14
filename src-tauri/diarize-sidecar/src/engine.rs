//! The sherpa-onnx pipeline. The only code in the tree that touches ONNX.
//!
//!   1. pyannote segmentation-3.0 finds speech regions and local speaker turns.
//!   2. WeSpeaker CAM++ embeds each region.
//!   3. fast clustering groups the embeddings into speakers.
//!
//! Everything here is allowed to die. See the crate docs.

use std::path::Path;

use serde::Serialize;
use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractorConfig, Wave,
};

use crate::cli::Args;
use crate::wav;

/// What the segmentation model was trained at. Checked against the model's own
/// reported rate below, so this constant cannot drift out of sync unnoticed.
const EXPECTED_SAMPLE_RATE: u32 = 16_000;

/// The stdout document. The parent parses exactly this and nothing else.
#[derive(Debug, Serialize)]
pub struct Output {
    pub turns: Vec<Turn>,
}

/// Wire-format twin of the app's `SpeakerTurn`. Deliberately duplicated rather
/// than shared through a common crate: sharing a crate would put this package in
/// the app's dependency graph, and the whole point is that it is not. The field
/// names are the contract; both sides have a test pinning the exact JSON.
#[derive(Debug, Serialize, PartialEq)]
pub struct Turn {
    pub start: f32,
    pub end: f32,
    /// Whatever the clusterer called this speaker. **Sparse**: a two-speaker
    /// file has come back as `{0, 3}`. Not an index. See the app's `SpeakerTurn`.
    pub speaker: u32,
}

pub fn run(args: &Args) -> Result<Vec<Turn>, String> {
    // Order matters. Everything cheap and everything that can produce a *useful*
    // error message happens before the 34 MB of models are touched.
    let format = wav::read_format(&args.wav)?;

    if format.channels != 1 {
        return Err(format!(
            "diarization needs mono audio, but {} has {} channels. Downmix it first \
             (ffmpeg -ac 1); interpreting interleaved stereo as mono would halve every \
             timestamp.",
            args.wav.display(),
            format.channels
        ));
    }
    if format.sample_rate != EXPECTED_SAMPLE_RATE {
        return Err(format!(
            "diarization needs {EXPECTED_SAMPLE_RATE} Hz audio, but {} is {} Hz. Resample \
             it first (ffmpeg -ar {EXPECTED_SAMPLE_RATE}); resampling here would silently \
             skew every timestamp.",
            args.wav.display(),
            format.sample_rate
        ));
    }

    // sherpa reports a missing OR corrupt model as a null pointer with no reason
    // -- and for a *corrupt* one, often does not get that far: ONNX Runtime
    // throws, and the C++ runtime aborts this process outright. Stat them first
    // so that at least the ordinary "not downloaded yet" case gets a real error.
    for (label, path) in [
        ("segmentation", &args.segmentation_model),
        ("speaker embedding", &args.embedding_model),
    ] {
        if !path.is_file() {
            return Err(format!(
                "the diarization {label} model is not at {}. Run scripts/fetch-sidecars.sh \
                 --models-only to download it.",
                path.display()
            ));
        }
    }

    let (num_clusters, threshold) = args.clustering();

    let config = OfflineSpeakerDiarizationConfig {
        segmentation: OfflineSpeakerSegmentationModelConfig {
            pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                model: Some(path_arg(&args.segmentation_model)?),
            },
            ..Default::default()
        },
        embedding: SpeakerEmbeddingExtractorConfig {
            model: Some(path_arg(&args.embedding_model)?),
            ..Default::default()
        },
        clustering: FastClusteringConfig {
            num_clusters,
            threshold,
        },
        min_duration_on: 0.3,
        min_duration_off: 0.5,
    };

    // >>> Past this line a bad model aborts the process. That is fine. <<<
    let diarizer = OfflineSpeakerDiarization::create(&config).ok_or_else(|| {
        "could not initialise the sherpa-onnx diarizer (are both model files valid ONNX?)"
            .to_string()
    })?;

    let reported = diarizer.sample_rate();
    if reported != EXPECTED_SAMPLE_RATE as i32 {
        return Err(format!(
            "the segmentation model expects {reported} Hz, but this build validates against \
             {EXPECTED_SAMPLE_RATE} Hz. The model changed; update EXPECTED_SAMPLE_RATE."
        ));
    }

    let wave = Wave::read(&path_arg(&args.wav)?)
        .ok_or_else(|| format!("could not decode the WAV at {}", args.wav.display()))?;

    let result = diarizer
        .process(wave.samples())
        .ok_or_else(|| format!("diarization failed on {}", args.wav.display()))?;

    // An empty result is a legitimate outcome: silence has no speaker turns.
    result
        .sort_by_start_time()
        .into_iter()
        .map(|s| to_turn(s.start, s.end, s.speaker))
        .collect()
}

/// sherpa's speaker label is an `i32`. Ours is a `u32`.
///
/// This used to be `s.speaker.max(0) as u32`, which folds a **negative** label
/// into speaker 0 -- silently merging a segment the clusterer refused to assign
/// with a real speaker's turns. That is exactly the quiet wrongness this whole
/// crate is written to avoid, so it fails loudly instead. A negative label has
/// never been observed; if one ever appears, that is a fact worth learning
/// rather than averaging away.
fn to_turn(start: f32, end: f32, speaker: i32) -> Result<Turn, String> {
    let speaker = u32::try_from(speaker).map_err(|_| {
        format!(
            "the clusterer returned a negative speaker label ({speaker}) for the turn at \
             {start:.2}s-{end:.2}s. That means the segment was not assigned to any speaker; \
             folding it into speaker 0 would silently merge it with a real one."
        )
    })?;
    Ok(Turn { start, end, speaker })
}

/// sherpa takes `String`, so a non-UTF-8 path would be mangled rather than
/// rejected. Refuse it instead.
fn path_arg(path: &Path) -> Result<String, String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("path is not valid UTF-8: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wire contract, pinned. The app has the mirror of this test on its
    /// `SpeakerTurn`; if either side is renamed, one of the two goes red.
    #[test]
    fn the_json_contract_is_exactly_this() {
        let out = Output {
            turns: vec![Turn {
                start: 1.5,
                end: 2.25,
                speaker: 3,
            }],
        };
        assert_eq!(
            serde_json::to_string(&out).unwrap(),
            r#"{"turns":[{"start":1.5,"end":2.25,"speaker":3}]}"#
        );
    }

    #[test]
    fn no_turns_serializes_as_an_empty_array_not_null() {
        // Silence is a success, and the parent must see a parseable success.
        assert_eq!(
            serde_json::to_string(&Output { turns: vec![] }).unwrap(),
            r#"{"turns":[]}"#
        );
    }

    #[test]
    fn ordinary_speaker_labels_pass_straight_through() {
        assert_eq!(
            to_turn(1.5, 2.25, 3).unwrap(),
            Turn {
                start: 1.5,
                end: 2.25,
                speaker: 3
            }
        );
        assert_eq!(to_turn(0.0, 1.0, 0).unwrap().speaker, 0);
    }

    /// A negative label means "this segment belongs to no speaker". The old code
    /// said `.max(0) as u32` and so quietly filed it under speaker 0 -- merging
    /// an unassigned segment into a real speaker's turns, in a file that is
    /// otherwise scrupulous about never being quietly wrong.
    #[test]
    fn a_negative_speaker_label_fails_loudly_instead_of_becoming_speaker_zero() {
        let err = to_turn(4.0, 5.0, -1).expect_err("a negative label must not become speaker 0");
        assert!(err.contains("negative speaker label"), "{err}");
        assert!(err.contains("-1"), "should quote the label: {err}");
        assert!(err.contains("merge"), "should say what the harm is: {err}");
    }
}
