//! A RIFF header reader.
//!
//! Three reasons this exists rather than deferring to sherpa's `Wave::read`:
//!
//! 1. **Order.** Validating the audio needs the sample rate and the channel
//!    count. sherpa only exposes the sample rate, and only *after* the models
//!    are loaded -- so a 48 kHz file used to pay a full 34 MB model load before
//!    being rejected. Reading the header first makes the rejection free.
//! 2. **Channels.** sherpa's `Wave` has no channel accessor at all, so a stereo
//!    file was silently accepted and its samples interpreted as mono: every
//!    timestamp comes out at half its true value. Wrong, quietly, which is the
//!    worst kind of wrong.
//! 3. **Encoding.** Nor does it check the format tag. A 16 kHz mono *IEEE-float*
//!    WAV passed validation and fell through to sherpa's decoder to be
//!    reinterpreted. Same failure mode, same silence.
//!
//! It reads by seeking, not by slurping a fixed prefix: an earlier version read
//! only the first 4096 bytes, so a perfectly valid WAV carrying a large leading
//! `LIST`/`INFO` chunk -- metadata, which plenty of encoders write -- was
//! rejected as "no fmt chunk found". ffmpeg does not produce those, so it never
//! bit us; it was still a bug, and a confusing one.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavFormat {
    pub channels: u16,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
}

/// `WAVE_FORMAT_PCM`. The only encoding this validator understands, and the only
/// one ffmpeg is asked for (`-acodec pcm_s16le`).
const FORMAT_PCM: u16 = 1;
/// `WAVE_FORMAT_IEEE_FLOAT`. Named only so the error can say so.
const FORMAT_IEEE_FLOAT: u16 = 3;
/// `WAVE_FORMAT_EXTENSIBLE`: the *real* tag is then the first two bytes of the
/// SubFormat GUID, 24 bytes into the `fmt ` body.
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// A `fmt ` body is 16, 18, or 40 bytes. Nothing beyond 40 is of any interest.
const MAX_FMT_BODY: u64 = 40;

/// Give up after this many chunks rather than walking a hostile file forever.
/// A real WAV has a handful.
const MAX_CHUNKS: usize = 64;

/// Read just the RIFF/WAVE `fmt ` chunk. Does not decode any audio.
pub fn read_format(path: &Path) -> Result<WavFormat, String> {
    let mut f = File::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse_format(&mut f).map_err(|e| format!("{} is not a usable WAV file: {e}", path.display()))
}

fn parse_format<R: Read + Seek>(r: &mut R) -> Result<WavFormat, String> {
    let mut riff = [0u8; 12];
    r.read_exact(&mut riff)
        .map_err(|_| "too short to be a RIFF file".to_string())?;
    if &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        return Err("missing the RIFF/WAVE header".into());
    }

    // Walk the chunk list to `fmt `. It is normally first, but the spec does not
    // require it (LIST/INFO/JUNK chunks show up ahead of it in the wild, and can
    // be arbitrarily large -- hence seeking rather than buffering).
    for _ in 0..MAX_CHUNKS {
        let mut head = [0u8; 8];
        if r.read_exact(&mut head).is_err() {
            break; // ran off the end of the file without finding it
        }
        let id = [head[0], head[1], head[2], head[3]];
        let size = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as u64;

        if &id != b"fmt " {
            // Chunks are word-aligned; an odd size carries a pad byte.
            let skip = size + (size & 1);
            r.seek(SeekFrom::Current(skip as i64)).map_err(|e| {
                format!(
                    "cannot seek past the {:?} chunk: {e}",
                    String::from_utf8_lossy(&id)
                )
            })?;
            continue;
        }

        if size < 16 {
            return Err("the fmt chunk is too short to hold a WAV format header".into());
        }
        let mut body = vec![0u8; size.min(MAX_FMT_BODY) as usize];
        r.read_exact(&mut body)
            .map_err(|_| "the fmt chunk is truncated".to_string())?;

        let mut tag = u16::from_le_bytes([body[0], body[1]]);
        let channels = u16::from_le_bytes([body[2], body[3]]);
        let sample_rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
        let bits_per_sample = u16::from_le_bytes([body[14], body[15]]);

        if tag == FORMAT_EXTENSIBLE {
            // The tag we were given is a placeholder; the real one lives in the
            // SubFormat GUID. A 16- or 18-byte body cannot carry one.
            if body.len() < 26 {
                return Err(
                    "the fmt chunk says WAVE_FORMAT_EXTENSIBLE but is too short to carry a \
                     SubFormat GUID"
                        .into(),
                );
            }
            tag = u16::from_le_bytes([body[24], body[25]]);
        }

        if channels == 0 || sample_rate == 0 {
            return Err("the fmt chunk declares zero channels or a zero sample rate".into());
        }

        // Waving an unrecognised encoding through to the decoder is how audio
        // gets silently reinterpreted -- the same class of bug as reading stereo
        // as mono. Refuse what we cannot vouch for.
        if tag != FORMAT_PCM || bits_per_sample != 16 {
            let float_note = if tag == FORMAT_IEEE_FLOAT {
                " (IEEE float)"
            } else {
                ""
            };
            return Err(format!(
                "expected 16-bit PCM, but this file declares WAVE format tag {tag}{float_note} \
                 at {bits_per_sample} bits per sample. Convert it first \
                 (ffmpeg -acodec pcm_s16le -ac 1 -ar 16000). This validator only understands \
                 16-bit PCM, and passing an encoding it cannot check through to the decoder is \
                 how audio gets misread without anyone noticing."
            ));
        }

        return Ok(WavFormat {
            channels,
            sample_rate,
            bits_per_sample,
        });
    }

    Err(format!(
        "no fmt chunk found in the first {MAX_CHUNKS} chunks"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse(bytes: &[u8]) -> Result<WavFormat, String> {
        parse_format(&mut Cursor::new(bytes))
    }

    /// A well-formed 44-byte canonical WAV header with no samples.
    fn header(channels: u16, sample_rate: u32) -> Vec<u8> {
        tagged_header(FORMAT_PCM, channels, sample_rate, 16)
    }

    fn tagged_header(tag: u16, channels: u16, sample_rate: u32, bits: u16) -> Vec<u8> {
        let block_align = channels * bits / 8;
        let byte_rate = sample_rate * block_align as u32;
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&36u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&tag.to_le_bytes());
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
            parse(&header(1, 16_000)).unwrap(),
            WavFormat {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16
            }
        );
    }

    #[test]
    fn sees_stereo() {
        // The case that used to be accepted silently.
        assert_eq!(parse(&header(2, 44_100)).unwrap().channels, 2);
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
        assert_eq!(parse(&b).unwrap().sample_rate, 16_000);
    }

    /// The 4096-byte-prefix bug. A valid WAV with a big leading `LIST`/`INFO`
    /// chunk -- metadata, which plenty of encoders write -- used to be rejected
    /// as "no fmt chunk found", because the reader only ever looked at the first
    /// 4096 bytes of the file. Seeking, rather than slurping a prefix, fixes it.
    #[test]
    fn finds_fmt_behind_a_leading_chunk_far_bigger_than_any_read_buffer() {
        let list_len = 100_000u32;
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + 8 + list_len).to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"LIST");
        b.extend_from_slice(&list_len.to_le_bytes());
        b.extend_from_slice(&vec![b'x'; list_len as usize]);
        b.extend_from_slice(&header(1, 16_000)[12..]);

        assert_eq!(
            parse(&b).unwrap(),
            WavFormat {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16
            },
            "a large leading LIST chunk is legal and must not hide the fmt chunk"
        );
    }

    /// The other silent-reinterpretation hole: a 16 kHz mono file that happens to
    /// be IEEE float passed every check and went straight to the decoder.
    #[test]
    fn rejects_ieee_float_by_name_rather_than_passing_it_to_the_decoder() {
        let err = parse(&tagged_header(FORMAT_IEEE_FLOAT, 1, 16_000, 32))
            .expect_err("float samples are not 16-bit PCM");
        assert!(err.contains("IEEE float"), "should name the encoding: {err}");
        assert!(err.contains("pcm_s16le"), "should say how to fix it: {err}");
    }

    #[test]
    fn rejects_sample_widths_it_cannot_vouch_for() {
        assert!(parse(&tagged_header(FORMAT_PCM, 1, 16_000, 8)).is_err());
        assert!(parse(&tagged_header(FORMAT_PCM, 1, 16_000, 24)).is_err());
    }

    /// WAVE_FORMAT_EXTENSIBLE is how a plain PCM file is often written once it
    /// has more than two channels or a channel mask. The real tag hides in the
    /// SubFormat GUID; reading the placeholder 0xFFFE as the tag would reject a
    /// perfectly good PCM file.
    #[test]
    fn accepts_extensible_pcm_by_reading_the_subformat_guid() {
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&60u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&40u32.to_le_bytes()); // extensible body
        b.extend_from_slice(&FORMAT_EXTENSIBLE.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes()); // channels
        b.extend_from_slice(&16_000u32.to_le_bytes()); // rate
        b.extend_from_slice(&32_000u32.to_le_bytes()); // byte rate
        b.extend_from_slice(&2u16.to_le_bytes()); // block align
        b.extend_from_slice(&16u16.to_le_bytes()); // bits
        b.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        b.extend_from_slice(&16u16.to_le_bytes()); // valid bits
        b.extend_from_slice(&4u32.to_le_bytes()); // channel mask
        b.extend_from_slice(&FORMAT_PCM.to_le_bytes()); // SubFormat GUID: PCM
        b.extend_from_slice(&[0u8; 14]); // ...rest of the GUID
        b.extend_from_slice(b"data");
        b.extend_from_slice(&0u32.to_le_bytes());

        assert_eq!(parse(&b).unwrap().sample_rate, 16_000);
    }

    #[test]
    fn rejects_things_that_are_not_wavs() {
        assert!(parse(b"").is_err());
        assert!(parse(b"not a wav file at all").is_err());
        // A RIFF container that is not WAVE.
        let mut avi = header(1, 16_000);
        avi[8..12].copy_from_slice(b"AVI ");
        assert!(parse(&avi).is_err());
    }

    #[test]
    fn rejects_a_truncated_fmt_chunk() {
        let full = header(1, 16_000);
        assert!(parse(&full[..20]).is_err());
    }

    #[test]
    fn rejects_a_header_with_no_fmt_chunk() {
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"data");
        b.extend_from_slice(&0u32.to_le_bytes());
        assert!(parse(&b).is_err());
    }
}
