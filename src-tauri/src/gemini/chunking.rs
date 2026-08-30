//! Splitting long audio into requests Gemini will accept.
//!
//! Everything here is PURE except `parse_silencedetect`'s caller. That is the
//! point: the arithmetic that decides whether a run gets speaker labels is
//! testable without ffmpeg, a network, or a key.

/// Gemini caps a request at 30 minutes once word timestamps or diarization are
/// enabled. 28 leaves margin for a duration probe that disagrees slightly with
/// what the API measures.
pub const MAX_CHUNK_SECS: f64 = 1680.0;

/// The length to aim for when splitting. Below the cap so that rounding can
/// never push a chunk over it.
pub const TARGET_CHUNK_SECS: f64 = 1500.0;

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkSpec {
    pub index: usize,
    pub start_secs: f64,
    pub end_secs: f64,
}

/// Equal-length chunks, not greedy 25-minute bites with a remainder.
///
/// A greedy split of 2h05m leaves a 5-minute tail; equal splitting gives five
/// 25-minute chunks instead. Same request count, no runt.
pub fn plan(duration_secs: f64) -> Vec<ChunkSpec> {
    if duration_secs <= 0.0 {
        return Vec::new();
    }
    if duration_secs <= MAX_CHUNK_SECS {
        return vec![ChunkSpec {
            index: 0,
            start_secs: 0.0,
            end_secs: duration_secs,
        }];
    }

    let count = (duration_secs / TARGET_CHUNK_SECS).ceil() as usize;
    let each = duration_secs / count as f64;

    (0..count)
        .map(|index| ChunkSpec {
            index,
            start_secs: each * index as f64,
            // The last chunk ends on the true duration, not on accumulated
            // float arithmetic, so the plan always covers the whole file.
            end_secs: if index + 1 == count {
                duration_secs
            } else {
                each * (index + 1) as f64
            },
        })
        .collect()
}

/// Move each boundary to the middle of the nearest silence, if one is close.
///
/// A hard cut lands mid-word and garbles a word on each side of the seam. This
/// is cheap insurance; when no silence is in range the hard boundary stands,
/// which is why `plan` alone is still a working design.
pub fn snap_to_silence(
    boundaries: &[f64],
    silences: &[(f64, f64)],
    window_secs: f64,
) -> Vec<f64> {
    boundaries
        .iter()
        .map(|&boundary| {
            silences
                .iter()
                .map(|&(start, end)| (start + end) / 2.0)
                .filter(|midpoint| (midpoint - boundary).abs() <= window_secs)
                .min_by(|a, b| {
                    (a - boundary)
                        .abs()
                        .partial_cmp(&(b - boundary).abs())
                        .expect("silence midpoints are finite")
                })
                .unwrap_or(boundary)
        })
        .collect()
}

/// Read `ffmpeg -af silencedetect`'s stderr into closed intervals.
///
/// Only PAIRED start/end lines become intervals. ffmpeg emits a trailing
/// `silence_start` with no `silence_end` when the file ends in silence, and
/// inventing an end for it could pull a boundary past the end of the audio.
pub fn parse_silencedetect(stderr: &str) -> Vec<(f64, f64)> {
    let mut intervals = Vec::new();
    let mut open: Option<f64> = None;

    for line in stderr.lines() {
        if let Some(rest) = line.split("silence_start:").nth(1) {
            open = rest.split_whitespace().next().and_then(|v| v.parse().ok());
        } else if let Some(rest) = line.split("silence_end:").nth(1) {
            if let (Some(start), Some(end)) = (
                open.take(),
                rest.split_whitespace().next().and_then(|v| v.parse::<f64>().ok()),
            ) {
                if end > start {
                    intervals.push((start, end));
                }
            }
        }
    }

    intervals
}

/// How far a cached chunk's boundaries may sit from the freshly planned ones
/// and still be treated as the same cut.
///
/// Not zero: `plan` rebuilds its boundaries with float division every run, and
/// `snap_to_silence` re-derives interior ones from a fresh ffmpeg pass, so
/// bit-exact equality would miss on drift far below the length of a syllable. A
/// false miss costs a chunk paid for twice, which is the exact thing resuming
/// exists to prevent. A millisecond is far tighter than any real re-cut and far
/// looser than float noise.
pub const BOUNDARY_EPSILON_SECS: f64 = 0.001;

/// Where a cached chunk claims to have been cut from.
///
/// The boundaries travel WITH the row rather than being trusted from the index,
/// because the index alone cannot tell "chunk 2 of the same plan" from "chunk 2
/// of a plan that has since moved". Reusing the second would splice the wrong
/// stretch of audio into the transcript and still report success.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CachedBounds {
    pub index: usize,
    pub start_secs: f64,
    pub end_secs: f64,
}

/// For each chunk in `plan`, which cached row (if any) may be reused for it.
///
/// Returns one entry per planned chunk, in plan order: `Some(i)` means
/// `cached[i]` describes the same cut and its words can stand in for a request;
/// `None` means the chunk has to be transcribed.
///
/// Pure, and deliberately free of any transcript type: it answers only "is this
/// the same cut?", and the caller maps the answer onto whatever payload it
/// stored. That is what lets the whole resume decision be tested without
/// ffmpeg, a network, or a key -- the same reason the rest of this module is
/// pure.
pub fn resolve_resume(plan: &[ChunkSpec], cached: &[CachedBounds]) -> Vec<Option<usize>> {
    plan.iter()
        .map(|spec| {
            cached.iter().position(|row| {
                row.index == spec.index
                    && (row.start_secs - spec.start_secs).abs() <= BOUNDARY_EPSILON_SECS
                    && (row.end_secs - spec.end_secs).abs() <= BOUNDARY_EPSILON_SECS
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Under the cap, ONE chunk -- which is the only shape that gets speaker
    /// labels, so this boundary decides whether diarization happens at all.
    #[test]
    fn audio_at_or_under_the_cap_is_a_single_chunk() {
        assert_eq!(plan(60.0).len(), 1);
        assert_eq!(plan(MAX_CHUNK_SECS).len(), 1);
        let only = &plan(1234.5)[0];
        assert_eq!((only.start_secs, only.end_secs), (0.0, 1234.5));
    }

    #[test]
    fn one_second_over_the_cap_splits_in_two() {
        let chunks = plan(MAX_CHUNK_SECS + 1.0);
        assert_eq!(chunks.len(), 2);
    }

    /// Equal-sized chunks, NOT "fill 25 minutes then take the remainder".
    /// A greedy split of a 2h05m file leaves a 5-minute tail chunk whose
    /// transcript quality is worse for no reason.
    #[test]
    fn chunks_are_equal_length_with_no_runt_tail() {
        let chunks = plan(7500.0);
        assert_eq!(chunks.len(), 5);
        for chunk in &chunks {
            assert!((chunk.end_secs - chunk.start_secs - 1500.0).abs() < 1e-6);
        }
    }

    #[test]
    fn every_chunk_stays_under_the_cap_at_the_two_hour_limit() {
        for chunk in plan(7200.0) {
            assert!(chunk.end_secs - chunk.start_secs <= MAX_CHUNK_SECS);
        }
    }

    #[test]
    fn chunks_are_contiguous_and_cover_the_whole_duration() {
        let chunks = plan(5000.0);
        assert_eq!(chunks[0].start_secs, 0.0);
        assert!((chunks.last().unwrap().end_secs - 5000.0).abs() < 1e-6);
        for pair in chunks.windows(2) {
            assert!((pair[0].end_secs - pair[1].start_secs).abs() < 1e-6);
        }
    }

    /// Zero-length audio yields NO chunks rather than one empty chunk: an
    /// empty upload is a request that costs money and cannot succeed.
    #[test]
    fn zero_or_negative_duration_yields_no_chunks() {
        assert!(plan(0.0).is_empty());
        assert!(plan(-1.0).is_empty());
    }

    #[test]
    fn boundaries_are_unchanged_when_there_is_no_silence_to_snap_to() {
        assert_eq!(snap_to_silence(&[1500.0], &[], 30.0), vec![1500.0]);
    }

    #[test]
    fn a_boundary_moves_to_the_midpoint_of_a_nearby_silence() {
        assert_eq!(snap_to_silence(&[1500.0], &[(1508.0, 1512.0)], 30.0), vec![1510.0]);
    }

    #[test]
    fn a_silence_outside_the_window_is_ignored() {
        assert_eq!(snap_to_silence(&[1500.0], &[(1600.0, 1604.0)], 30.0), vec![1500.0]);
    }

    #[test]
    fn the_nearest_of_several_silences_wins() {
        let silences = [(1470.0, 1474.0), (1504.0, 1506.0), (1520.0, 1524.0)];
        assert_eq!(snap_to_silence(&[1500.0], &silences, 30.0), vec![1505.0]);
    }

    #[test]
    fn silencedetect_output_is_parsed_into_intervals() {
        let stderr = "\
[silencedetect @ 0x7f] silence_start: 12.5
[silencedetect @ 0x7f] silence_end: 14.25 | silence_duration: 1.75
[silencedetect @ 0x7f] silence_start: 40
[silencedetect @ 0x7f] silence_end: 41.5 | silence_duration: 1.5
";
        assert_eq!(parse_silencedetect(stderr), vec![(12.5, 14.25), (40.0, 41.5)]);
    }

    /// ffmpeg emits a trailing `silence_start` with no `silence_end` when the
    /// file ends in silence. An unterminated interval must be DROPPED, not
    /// completed with a guessed end -- a bogus interval could pull a boundary
    /// past the end of the audio.
    #[test]
    fn an_unterminated_silence_is_dropped() {
        let stderr = "[silencedetect @ 0x7f] silence_start: 99.0\n";
        assert!(parse_silencedetect(stderr).is_empty());
    }

    /// An empty cache is the ordinary first run: everything is fetched.
    #[test]
    fn nothing_cached_means_every_chunk_is_fetched() {
        let plan = plan(5000.0);
        assert_eq!(resolve_resume(&plan, &[]), vec![None; plan.len()]);
    }

    /// The whole point: a chunk already paid for is not paid for twice.
    #[test]
    fn a_chunk_cut_at_the_same_boundaries_is_reused() {
        let plan = plan(5000.0);
        let cached: Vec<CachedBounds> = plan
            .iter()
            .map(|spec| CachedBounds {
                index: spec.index,
                start_secs: spec.start_secs,
                end_secs: spec.end_secs,
            })
            .collect();
        assert_eq!(
            resolve_resume(&plan, &cached),
            (0..plan.len()).map(Some).collect::<Vec<_>>()
        );
    }

    /// `plan` rebuilds boundaries with float arithmetic every run, and
    /// `snap_to_silence` re-derives them from a fresh ffmpeg pass. Bit-exact
    /// equality would miss on drift far too small to misalign a word, and the
    /// cost of a false miss is a chunk paid for twice.
    #[test]
    fn sub_millisecond_drift_still_counts_as_the_same_chunk() {
        let plan = plan(3000.0);
        let cached: Vec<CachedBounds> = plan
            .iter()
            .map(|spec| CachedBounds {
                index: spec.index,
                start_secs: spec.start_secs + 0.0001,
                end_secs: spec.end_secs - 0.0001,
            })
            .collect();
        assert!(resolve_resume(&plan, &cached).iter().all(Option::is_some));
    }

    /// THE MISALIGNMENT GUARD. A boundary that genuinely moved -- a different
    /// duration probe, a silence pass that landed elsewhere -- means the cached
    /// words describe DIFFERENT AUDIO than the chunk now planned at that index.
    /// Reusing it would splice one stretch of speech in where another belongs,
    /// and the run would succeed while being quietly wrong. Re-transcribing is
    /// the only honest answer, and it is why each row carries its own
    /// boundaries instead of trusting the index.
    #[test]
    fn a_chunk_whose_boundary_moved_is_refetched_not_reused() {
        let plan = plan(5000.0);
        let mut cached: Vec<CachedBounds> = plan
            .iter()
            .map(|spec| CachedBounds {
                index: spec.index,
                start_secs: spec.start_secs,
                end_secs: spec.end_secs,
            })
            .collect();
        cached[1].end_secs += 12.0;
        assert_eq!(resolve_resume(&plan, &cached)[1], None);
    }

    /// Per-row boundaries rather than one whole-plan hash, and this is what it
    /// buys: a plan that shifts near the END still resumes everything BEFORE
    /// the shift. Under a plan hash a single moved boundary would discard every
    /// chunk, which on a bad API day is exactly the money this feature exists
    /// to stop losing.
    #[test]
    fn a_late_boundary_shift_leaves_the_earlier_chunks_reusable() {
        let plan = plan(7500.0);
        let mut cached: Vec<CachedBounds> = plan
            .iter()
            .map(|spec| CachedBounds {
                index: spec.index,
                start_secs: spec.start_secs,
                end_secs: spec.end_secs,
            })
            .collect();
        cached[4].start_secs += 9.0;

        let resolved = resolve_resume(&plan, &cached);
        assert!(resolved[..4].iter().all(Option::is_some));
        assert_eq!(resolved[4], None);
    }

    /// A shorter re-probe leaves rows for chunks the plan no longer has. They
    /// must be ignored rather than indexed into.
    #[test]
    fn cached_rows_beyond_the_end_of_the_plan_are_ignored() {
        let plan = plan(1000.0);
        let cached = [
            CachedBounds { index: 0, start_secs: 0.0, end_secs: 1000.0 },
            CachedBounds { index: 1, start_secs: 1000.0, end_secs: 2000.0 },
        ];
        assert_eq!(resolve_resume(&plan, &cached), vec![Some(0)]);
    }
}
