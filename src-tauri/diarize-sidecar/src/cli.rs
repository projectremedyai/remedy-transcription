//! Argument parsing. Hand-rolled: five flags do not justify a dependency, and
//! this crate's whole job is to stay small enough to be obviously correct.

use std::path::PathBuf;

pub const USAGE: &str = "\
usage: diarize-sidecar --wav <path>
                       --segmentation-model <path>
                       --embedding-model <path>
                       [--num-speakers <n>]
                       [--cluster-threshold <f>]

Prints {\"turns\":[{\"start\":<s>,\"end\":<s>,\"speaker\":<id>}, ...]} on stdout.
Exits non-zero with a reason on stderr on any failure.";

#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub wav: PathBuf,
    pub segmentation_model: PathBuf,
    pub embedding_model: PathBuf,
    /// `None` or `Some(0)` both mean auto-detect.
    pub num_speakers: Option<u32>,
    pub cluster_threshold: f32,
}

/// Mirrors `DiarizeOptions::default()` on the app side. Kept in sync by the
/// contract tests on both sides of the process boundary.
const DEFAULT_CLUSTER_THRESHOLD: f32 = 0.5;

/// The largest speaker count we accept. Mirrors `MAX_SPEAKERS` in the app's
/// `diarize.rs`, which rejects the same values before it even spawns us.
///
/// The bound is not fussiness, it is what makes the `i32` conversion below
/// **total**. sherpa's `num_clusters` is an `i32`, and `u32 as i32` fails in the
/// worst possible way -- silently, and into a *meaningful* value:
///
/// - `--num-speakers 4294967295` casts to `-1`, which sherpa reads as
///   "auto-detect". The flag is silently ignored and the caller gets a
///   confident, plausible answer to a question they did not ask.
/// - `--num-speakers 1000000` collapses the file to a single speaker.
///
/// Both measured against the real models. Rejected here rather than cast.
pub const MAX_SPEAKERS: u32 = 64;

impl Args {
    pub fn from_env() -> Result<Self, String> {
        Self::parse(std::env::args().skip(1))
    }

    pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Self, String> {
        let mut wav = None;
        let mut segmentation_model = None;
        let mut embedding_model = None;
        let mut num_speakers = None;
        let mut cluster_threshold = DEFAULT_CLUSTER_THRESHOLD;

        let mut it = args.into_iter();
        while let Some(flag) = it.next() {
            // A flag with a missing value is an error, never a silent skip.
            let mut next = || it.next().ok_or_else(|| format!("{flag} needs a value"));

            match flag.as_str() {
                "--wav" => wav = Some(PathBuf::from(next()?)),
                "--segmentation-model" => segmentation_model = Some(PathBuf::from(next()?)),
                "--embedding-model" => embedding_model = Some(PathBuf::from(next()?)),
                "--num-speakers" => {
                    let raw = next()?;
                    let n = raw
                        .parse::<u32>()
                        .map_err(|_| format!("--num-speakers must be a non-negative integer, got {raw:?}"))?;
                    if n > MAX_SPEAKERS {
                        return Err(format!(
                            "--num-speakers must be between 0 and {MAX_SPEAKERS} (0 means \
                             auto-detect), got {n}. Larger values are refused rather than cast \
                             into sherpa's i32, where they would silently become -1 \
                             ('auto-detect') or collapse the file to one speaker."
                        ));
                    }
                    num_speakers = Some(n);
                }
                "--cluster-threshold" => {
                    let raw = next()?;
                    cluster_threshold = raw
                        .parse::<f32>()
                        .map_err(|_| format!("--cluster-threshold must be a number, got {raw:?}"))?;
                    if !cluster_threshold.is_finite() || cluster_threshold <= 0.0 {
                        return Err(format!(
                            "--cluster-threshold must be a positive finite number, got {raw:?}"
                        ));
                    }
                }
                other => return Err(format!("unknown argument {other:?}")),
            }
        }

        Ok(Self {
            wav: wav.ok_or("--wav is required")?,
            segmentation_model: segmentation_model.ok_or("--segmentation-model is required")?,
            embedding_model: embedding_model.ok_or("--embedding-model is required")?,
            num_speakers,
            cluster_threshold,
        })
    }

    /// Map onto sherpa's `FastClusteringConfig` pair.
    ///
    /// sherpa treats `num_clusters < 0` as "decide for me, using `threshold`".
    /// Asking for 0 speakers is meaningless, so it is auto-detect rather than a
    /// silently empty result.
    pub fn clustering(&self) -> (i32, f32) {
        match self.num_speakers {
            // `min` rather than a bare cast. `parse` already bounds this at
            // MAX_SPEAKERS, so the clamp is unreachable -- it is here so that the
            // conversion is *provably* total at the point of use, and cannot
            // start wrapping to a negative "auto-detect" if the bound above is
            // ever loosened by someone who does not read this comment.
            Some(n) if n > 0 => (n.min(MAX_SPEAKERS) as i32, self.cluster_threshold),
            _ => (-1, self.cluster_threshold),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(extra: &[&str]) -> Result<Args, String> {
        let mut v: Vec<String> = ["--wav", "a.wav", "--segmentation-model", "s.onnx", "--embedding-model", "e.onnx"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        v.extend(extra.iter().map(|s| s.to_string()));
        Args::parse(v)
    }

    #[test]
    fn the_three_paths_are_required() {
        for missing in ["--wav", "--segmentation-model", "--embedding-model"] {
            let kept: Vec<String> = ["--wav", "a.wav", "--segmentation-model", "s.onnx", "--embedding-model", "e.onnx"]
                .chunks(2)
                .filter(|c| c[0] != missing)
                .flatten()
                .map(|s| s.to_string())
                .collect();
            let err = Args::parse(kept).expect_err("{missing} should be required");
            assert!(err.contains(missing), "unhelpful error for {missing}: {err}");
        }
    }

    #[test]
    fn defaults_match_the_app_side_diarize_options() {
        let a = args(&[]).unwrap();
        assert_eq!(a.num_speakers, None);
        assert_eq!(a.cluster_threshold, 0.5);
        assert_eq!(a.clustering(), (-1, 0.5));
    }

    #[test]
    fn a_known_speaker_count_pins_the_cluster_count() {
        assert_eq!(args(&["--num-speakers", "3"]).unwrap().clustering(), (3, 0.5));
    }

    #[test]
    fn zero_speakers_is_auto_detect_not_zero_clusters() {
        assert_eq!(args(&["--num-speakers", "0"]).unwrap().clustering().0, -1);
    }

    #[test]
    fn the_threshold_is_only_used_when_auto_detecting() {
        let a = args(&["--cluster-threshold", "0.7"]).unwrap();
        assert_eq!(a.clustering(), (-1, 0.7));
        let b = args(&["--num-speakers", "2", "--cluster-threshold", "0.7"]).unwrap();
        assert_eq!(b.clustering(), (2, 0.7)); // sherpa ignores the threshold when k >= 0
    }

    #[test]
    fn garbage_values_are_rejected_rather_than_coerced() {
        assert!(args(&["--num-speakers", "-1"]).is_err());
        assert!(args(&["--num-speakers", "two"]).is_err());
        assert!(args(&["--cluster-threshold", "nan"]).is_err());
        assert!(args(&["--cluster-threshold", "0"]).is_err());
        assert!(args(&["--cluster-threshold", "-0.5"]).is_err());
        assert!(args(&["--wat"]).is_err());
        assert!(args(&["--num-speakers"]).is_err(), "a dangling flag must not be silently ignored");
    }

    /// The lossy `u32 as i32` cast, pinned shut.
    ///
    /// This is the nastiest class of bug in the file, because the wrong answer
    /// looks right: `4294967295 as i32` is `-1`, and sherpa reads `-1` as
    /// "auto-detect". So the old code took an absurd request, silently discarded
    /// it, and returned a perfectly plausible auto-detected result. `1000000`
    /// collapsed the file to one speaker. Neither errored.
    #[test]
    fn an_absurd_speaker_count_is_rejected_not_cast_into_auto_detect() {
        for absurd in ["4294967295", "1000000", "65"] {
            let err = args(&["--num-speakers", absurd])
                .expect_err("{absurd} speakers must be rejected, not cast");
            assert!(err.contains("64"), "should name the bound: {err}");
        }

        // The bound itself still works, and still reaches sherpa as itself.
        assert_eq!(args(&["--num-speakers", "64"]).unwrap().clustering().0, 64);

        // The property that actually matters: whatever survives parsing maps to a
        // non-negative i32, so it can never be mistaken for "auto-detect".
        for n in ["0", "1", "2", "64"] {
            let k = args(&["--num-speakers", n]).unwrap().clustering().0;
            assert!(
                k == -1 || k > 0,
                "--num-speakers {n} produced num_clusters {k}; a value that is neither an \
                 explicit -1 (auto-detect) nor a real count means a cast went wrong"
            );
        }
    }
}
