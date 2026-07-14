//! A 40-line RIFF header reader.
//!
//! Two reasons this exists rather than deferring to sherpa's `Wave::read`:
//!
//! 1. **Order.** Validating the audio needs the sample rate and the channel
//!    count. sherpa only exposes the sample rate, and only *after* the models
//!    are loaded -- so a 48 kHz file used to pay a full 34 MB model load before
//!    being rejected. Reading 44 bytes of header first makes the rejection free.
//! 2. **Channels.** sherpa's `Wave` has no channel accessor at all, so a stereo
//!    file was silently accepted and its samples interpreted as mono: every
//!    timestamp comes out at half its true value. Wrong, quietly, which is the
//!    worst kind of wrong.

use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavFormat {
    pub channels: u16,
    pub sample_rate: u32,
}

/// Read just the RIFF/WAVE `fmt ` chunk. Does not decode any audio.
pub fn read_format(path: &Path) -> Result<WavFormat, String> {
    let mut buf = Vec::new();
    File::open(path)
        .and_then(|mut f| f.by_ref().take(4096).read_to_end(&mut buf))
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;

    parse_format(&buf).map_err(|e| format!("{} is not a usable WAV file: {e}", path.display()))
}

fn parse_format(buf: &[u8]) -> Result<WavFormat, String> {
    if buf.len() < 12 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        return Err("missing the RIFF/WAVE header".into());
    }

    // Walk the chunk list to `fmt `. It is normally first, but the spec does not
    // require it (LIST/JUNK chunks show up ahead of it in the wild).
    let mut pos = 12usize;
    while pos + 8 <= buf.len() {
        let id = &buf[pos..pos + 4];
        let size = u32::from_le_bytes([buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]]) as usize;
        let body = pos + 8;

        if id == b"fmt " {
            if size < 16 || body + 16 > buf.len() {
                return Err("the fmt chunk is truncated".into());
            }
            let channels = u16::from_le_bytes([buf[body + 2], buf[body + 3]]);
            let sample_rate =
                u32::from_le_bytes([buf[body + 4], buf[body + 5], buf[body + 6], buf[body + 7]]);
            if channels == 0 || sample_rate == 0 {
                return Err("the fmt chunk declares zero channels or a zero sample rate".into());
            }
            return Ok(WavFormat {
                channels,
                sample_rate,
            });
        }

        // Chunks are word-aligned; odd sizes carry a pad byte.
        pos = body + size + (size & 1);
    }

    Err("no fmt chunk found".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A well-formed 44-byte canonical WAV header with no samples.
    fn header(channels: u16, sample_rate: u32) -> Vec<u8> {
        let bits = 16u16;
        let block_align = channels * bits / 8;
        let byte_rate = sample_rate * block_align as u32;
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&36u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes()); // PCM
        b.extend_from_slice(&channels.to_le_bytes());
        b.extend_from_slice(&sample_rate.to_le_bytes());
        b.extend_from_slice(&byte_rate.to_le_bytes());
        b.extend_from_slice(&block_align.to_le_bytes());
        b.extend_from_slice(&bits.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&0u32.to_le_bytes());
        b
    }

    #[test]
    fn reads_channels_and_rate_from_a_canonical_header() {
        assert_eq!(
            parse_format(&header(1, 16_000)).unwrap(),
            WavFormat {
                channels: 1,
                sample_rate: 16_000
            }
        );
    }

    #[test]
    fn sees_stereo() {
        // The case that used to be accepted silently.
        assert_eq!(parse_format(&header(2, 44_100)).unwrap().channels, 2);
    }

    #[test]
    fn skips_chunks_that_precede_fmt() {
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&100u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"JUNK");
        b.extend_from_slice(&5u32.to_le_bytes()); // odd size -> one pad byte
        b.extend_from_slice(&[0u8; 6]);
        b.extend_from_slice(&header(1, 16_000)[12..]);
        assert_eq!(parse_format(&b).unwrap().sample_rate, 16_000);
    }

    #[test]
    fn rejects_things_that_are_not_wavs() {
        assert!(parse_format(b"").is_err());
        assert!(parse_format(b"not a wav file at all").is_err());
        // A RIFF container that is not WAVE.
        let mut avi = header(1, 16_000);
        avi[8..12].copy_from_slice(b"AVI ");
        assert!(parse_format(&avi).is_err());
    }

    #[test]
    fn rejects_a_truncated_fmt_chunk() {
        let full = header(1, 16_000);
        assert!(parse_format(&full[..20]).is_err());
    }

    #[test]
    fn rejects_a_header_with_no_fmt_chunk() {
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"data");
        b.extend_from_slice(&0u32.to_le_bytes());
        assert!(parse_format(&b).is_err());
    }
}
