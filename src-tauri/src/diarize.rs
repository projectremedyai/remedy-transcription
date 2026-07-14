//! Speaker diarization: "who spoke when".
//!
//! Two-model pipeline, run entirely on-device:
//!   1. pyannote segmentation-3.0 finds speech regions and local speaker turns.
//!   2. a speaker-embedding model (WeSpeaker CAM++) embeds each region.
//!   3. fast clustering groups the embeddings into speakers.
//!
//! Both models are mandatory. The engine lives behind [`Diarizer`] so it can be
//! swapped (e.g. for a Core ML implementation) without touching callers.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

/// One contiguous stretch of audio attributed to a single speaker.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpeakerTurn {
    /// Seconds from the start of the audio.
    pub start: f32,
    /// Seconds from the start of the audio.
    pub end: f32,
    /// Zero-based speaker index. Stable within one diarization run only.
    pub speaker: u32,
}

/// How to cluster the speaker embeddings.
///
/// The two fields are alternatives, not a pair: when `num_speakers` is set the
/// clusterer cuts the dendrogram to exactly that many clusters and
/// `cluster_threshold` is ignored. Otherwise it cuts by distance instead.
///
/// Measured on the test fixtures (see the real-model tests below), *neither*
/// mode is universally right:
///
/// - `num_speakers` recovers a conversation exactly, where auto-detect splits
///   one voice into several. It is the better answer whenever the count is
///   actually known -- so ask the user for it.
/// - but on audio with only a few long turns it can collapse everything into a
///   single speaker, where auto-detect gets it right.
///
/// Treat the result as a strong hint, not a guarantee: the engine may return
/// more or fewer speakers than requested.
#[derive(Debug, Clone, PartialEq)]
pub struct DiarizeOptions {
    /// Set when the user knows the count. Generally the more accurate mode, but
    /// not a hard guarantee -- see the type docs.
    pub num_speakers: Option<u32>,
    /// Used only when `num_speakers` is `None`. Lower = more speakers.
    pub cluster_threshold: f32,
}

impl Default for DiarizeOptions {
    fn default() -> Self {
        Self {
            num_speakers: None,
            cluster_threshold: 0.5,
        }
    }
}

impl DiarizeOptions {
    /// Map the options onto sherpa's `FastClusteringConfig` pair.
    ///
    /// sherpa treats `num_clusters < 0` as "decide for me, using `threshold`".
    /// A caller asking for 0 speakers is meaningless, so it is treated as
    /// auto-detect rather than silently producing an empty result.
    fn clustering(&self) -> (i32, f32) {
        match self.num_speakers {
            Some(n) if n > 0 => (n as i32, self.cluster_threshold),
            _ => (-1, self.cluster_threshold),
        }
    }
}

/// A speaker-diarization engine.
pub trait Diarizer: Send + Sync {
    /// Diarize a 16 kHz mono WAV file.
    fn diarize(&self, wav_path: &Path, opts: &DiarizeOptions) -> Result<Vec<SpeakerTurn>>;
}

/// The sherpa-onnx implementation. Holds only paths; the ONNX runtime is
/// constructed per call and dropped with the result, so an idle app carries no
/// model memory.
pub struct SherpaDiarizer {
    segmentation_model: PathBuf,
    embedding_model: PathBuf,
}

impl SherpaDiarizer {
    pub fn new(segmentation_model: PathBuf, embedding_model: PathBuf) -> Self {
        Self {
            segmentation_model,
            embedding_model,
        }
    }

    /// sherpa reports a null pointer for a missing/corrupt model, which gives a
    /// useless error. Check first so the caller learns which file is missing.
    fn check_models_present(&self) -> Result<()> {
        for (label, path) in [
            ("segmentation", &self.segmentation_model),
            ("speaker embedding", &self.embedding_model),
        ] {
            if !path.is_file() {
                return Err(anyhow!(
                    "diarization {label} model not found at {}. Run scripts/fetch-sidecars.sh to download it.",
                    path.display()
                ));
            }
        }
        Ok(())
    }
}

impl Diarizer for SherpaDiarizer {
    fn diarize(&self, wav_path: &Path, opts: &DiarizeOptions) -> Result<Vec<SpeakerTurn>> {
        use sherpa_onnx::{
            FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
            OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
            SpeakerEmbeddingExtractorConfig, Wave,
        };

        self.check_models_present()?;

        let wav = wav_path
            .to_str()
            .ok_or_else(|| anyhow!("WAV path is not valid UTF-8: {}", wav_path.display()))?;
        let wave = Wave::read(wav)
            .with_context(|| format!("failed to read WAV file at {}", wav_path.display()))?;

        let (num_clusters, threshold) = opts.clustering();

        let config = OfflineSpeakerDiarizationConfig {
            segmentation: OfflineSpeakerSegmentationModelConfig {
                pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                    model: Some(self.segmentation_model.to_string_lossy().into_owned()),
                },
                ..Default::default()
            },
            embedding: SpeakerEmbeddingExtractorConfig {
                model: Some(self.embedding_model.to_string_lossy().into_owned()),
                ..Default::default()
            },
            clustering: FastClusteringConfig {
                num_clusters,
                threshold,
            },
            min_duration_on: 0.3,
            min_duration_off: 0.5,
        };

        let diarizer = OfflineSpeakerDiarization::create(&config)
            .ok_or_else(|| anyhow!("failed to initialise the sherpa-onnx diarizer (check that the segmentation and embedding models are valid ONNX)"))?;

        // The segmentation model is trained at a fixed rate. Resampling here
        // would silently skew every timestamp, so refuse instead.
        let expected = diarizer.sample_rate();
        if wave.sample_rate() != expected {
            return Err(anyhow!(
                "diarization expects {expected} Hz mono audio but {} is {} Hz",
                wav_path.display(),
                wave.sample_rate()
            ));
        }

        let result = diarizer
            .process(wave.samples())
            .ok_or_else(|| anyhow!("diarization failed on {}", wav_path.display()))?;

        Ok(result
            .sort_by_start_time()
            .into_iter()
            .map(|s| SpeakerTurn {
                start: s.start,
                end: s.end,
                speaker: s.speaker.max(0) as u32,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in so command-level tests never load an ONNX runtime.
    pub struct MockDiarizer(pub Vec<SpeakerTurn>);

    impl Diarizer for MockDiarizer {
        fn diarize(&self, _: &Path, _: &DiarizeOptions) -> Result<Vec<SpeakerTurn>> {
            Ok(self.0.clone())
        }
    }

    fn turn(start: f32, end: f32, speaker: u32) -> SpeakerTurn {
        SpeakerTurn {
            start,
            end,
            speaker,
        }
    }

    #[test]
    fn mock_diarizer_returns_its_turns() {
        let turns = vec![turn(0.0, 1.0, 0), turn(1.0, 2.0, 1)];
        let d = MockDiarizer(turns.clone());
        let got = d
            .diarize(Path::new("/nonexistent.wav"), &DiarizeOptions::default())
            .unwrap();
        assert_eq!(got, turns);
    }

    #[test]
    fn mock_diarizer_is_usable_as_a_trait_object() {
        // Task 8 stores a Diarizer behind a trait object; prove it is object safe
        // and Send + Sync without dragging in the ONNX runtime.
        let d: Box<dyn Diarizer> = Box::new(MockDiarizer(vec![turn(0.0, 1.0, 0)]));
        fn assert_send_sync<T: Send + Sync + ?Sized>(_: &T) {}
        assert_send_sync(&*d);
        assert_eq!(d.diarize(Path::new("x.wav"), &DiarizeOptions::default()).unwrap().len(), 1);
    }

    #[test]
    fn default_options_auto_detect_speaker_count() {
        let opts = DiarizeOptions::default();
        assert!(opts.num_speakers.is_none());
        assert_eq!(opts.cluster_threshold, 0.5);
    }

    #[test]
    fn a_known_speaker_count_pins_the_cluster_count() {
        let opts = DiarizeOptions {
            num_speakers: Some(3),
            cluster_threshold: 0.5,
        };
        assert_eq!(opts.clustering(), (3, 0.5));
    }

    #[test]
    fn an_unknown_speaker_count_falls_back_to_the_threshold() {
        let opts = DiarizeOptions {
            num_speakers: None,
            cluster_threshold: 0.7,
        };
        // -1 is sherpa's sentinel for "cluster by threshold".
        assert_eq!(opts.clustering(), (-1, 0.7));
    }

    #[test]
    fn zero_speakers_is_treated_as_auto_detect_not_as_zero_clusters() {
        let opts = DiarizeOptions {
            num_speakers: Some(0),
            cluster_threshold: 0.5,
        };
        assert_eq!(opts.clustering().0, -1);
    }

    #[test]
    fn missing_models_fail_before_touching_the_onnx_runtime() {
        let d = SherpaDiarizer::new(
            PathBuf::from("/nonexistent/segmentation.onnx"),
            PathBuf::from("/nonexistent/embedding.onnx"),
        );
        let err = d
            .diarize(Path::new("/nonexistent.wav"), &DiarizeOptions::default())
            .expect_err("should fail when the models are absent");
        let msg = err.to_string();
        assert!(msg.contains("segmentation"), "unhelpful error: {msg}");
        assert!(msg.contains("fetch-sidecars"), "unhelpful error: {msg}");
    }

    #[test]
    fn speaker_turn_round_trips_through_serde() {
        let t = turn(1.5, 2.25, 1);
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, r#"{"start":1.5,"end":2.25,"speaker":1}"#);
        assert_eq!(serde_json::from_str::<SpeakerTurn>(&json).unwrap(), t);
    }

    // ---------------------------------------------------------------------
    // Real-model tests. The only ones that load ONNX. They require:
    //     ./scripts/fetch-sidecars.sh --models-only
    // Run with: cargo test -- --ignored
    //
    // Two fixtures, because the two clustering modes fail in opposite
    // directions and a single fixture would hide one of them:
    //
    //   two_speakers.wav        4 long turns  (Daniel/Samantha, macOS `say`)
    //       auto-detect     -> 2 speakers  (correct)
    //       num_speakers=2  -> 1 speaker   (collapses: too few embedding rows)
    //
    //   two_speakers_dense.wav  8 shorter turns, same two voices
    //       auto-detect     -> 4 speakers  (over-segments a single voice)
    //       num_speakers=2  -> 2 speakers  (correct, perfectly alternating)
    //
    // So neither mode is universally right. Telling the clusterer the speaker
    // count is the stronger option on realistic conversational audio, which is
    // why the UI should ask for it. See the task-7 report.
    // ---------------------------------------------------------------------

    fn fixture_diarizer(fixture: &str) -> (SherpaDiarizer, PathBuf) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let models = root.join("../models/diarization");
        let d = SherpaDiarizer::new(
            models.join("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
            models.join("wespeaker_en_voxceleb_CAM++.onnx"),
        );
        (d, root.join("tests/fixtures").join(fixture))
    }

    fn report(label: &str, turns: &[SpeakerTurn]) -> std::collections::HashSet<u32> {
        println!("--- {label} ---");
        for t in turns {
            println!("{:6.2}s - {:6.2}s  speaker {}", t.start, t.end, t.speaker);
        }
        turns
            .iter()
            .map(|t| t.speaker)
            .collect()
    }

    /// Ground truth for `two_speakers.wav`:
    ///     Daniel    0.00 -  7.01
    ///     Samantha  7.01 - 12.67
    ///     Daniel   12.67 - 18.93
    ///     Samantha 18.93 - 23.98
    #[test]
    #[ignore = "requires the diarization models; run with: cargo test -- --ignored"]
    fn sherpa_finds_two_speakers_in_the_fixture() {
        let (d, wav) = fixture_diarizer("two_speakers.wav");

        let turns = d
            .diarize(&wav, &DiarizeOptions::default())
            .expect("diarization should succeed on the fixture");

        let speakers = report("two_speakers.wav / auto-detect", &turns);
        assert!(!turns.is_empty(), "expected some speaker turns");
        assert_eq!(
            speakers.len(),
            2,
            "expected exactly two distinct speakers, got {speakers:?}"
        );

        // Turn boundaries should land near the real ones, not be smeared.
        assert!(
            turns[0].start < 0.5,
            "the first turn should start at the top of the file, got {:?}",
            turns[0]
        );
        assert!(
            (turns[0].end - 7.01).abs() < 0.5,
            "the first speaker change should be near 7.01s, got {:?}",
            turns[0]
        );
    }

    /// The `num_speakers` path -- the one we tell users to prefer -- against a
    /// real model. On conversational audio it recovers the speakers exactly,
    /// where auto-detect over-segments the same file into four.
    #[test]
    #[ignore = "requires the diarization models; run with: cargo test -- --ignored"]
    fn a_known_speaker_count_is_honoured_against_a_real_model() {
        let (d, wav) = fixture_diarizer("two_speakers_dense.wav");

        let turns = d
            .diarize(
                &wav,
                &DiarizeOptions {
                    num_speakers: Some(2),
                    ..Default::default()
                },
            )
            .expect("diarization should succeed on the fixture");

        let speakers = report("two_speakers_dense.wav / num_speakers = 2", &turns);
        assert_eq!(
            speakers.len(),
            2,
            "asking for 2 speakers should yield 2, got {speakers:?}"
        );

        // The fixture strictly alternates, so the labels should too.
        let labels: Vec<u32> = turns
            .iter()
            .map(|t| t.speaker)
            .collect();
        for pair in labels.windows(2) {
            assert_ne!(
                pair[0], pair[1],
                "the fixture alternates speakers, so no two adjacent turns should \
                 share a label: {labels:?}"
            );
        }
    }

    /// Guards the claim in `DiarizeOptions`' docs: auto-detect is NOT reliable
    /// on this file, so the UI must not silently default to it and call the
    /// answer authoritative. If a future model makes this pass, revisit the docs.
    #[test]
    #[ignore = "requires the diarization models; run with: cargo test -- --ignored"]
    fn auto_detect_over_segments_the_dense_fixture() {
        let (d, wav) = fixture_diarizer("two_speakers_dense.wav");

        let turns = d
            .diarize(&wav, &DiarizeOptions::default())
            .expect("diarization should succeed on the fixture");

        let speakers = report("two_speakers_dense.wav / auto-detect", &turns);
        assert!(
            speakers.len() > 2,
            "auto-detect was expected to over-segment this 2-speaker file, but it \
             found {}. If the engine improved, update the guidance in DiarizeOptions.",
            speakers.len()
        );
    }
}
