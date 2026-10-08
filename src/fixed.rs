//! Fixed-point Layer II requantization with caller-owned synthesis.
//!
//! The Q23 scaling is ported from FFmpeg 2da55bf,
//! libavcodec/mpegaudiodec_template.c (decode_init_static, l1_unscale,
//! l2_unscale_group and mp_decode_layer2), as built by mpegaudiodec_fixed.c.
//! Copyright (c) 2001, 2002 Fabrice Bellard.
//! SPDX-License-Identifier: LGPL-2.1-or-later (see LICENSE-LGPL).
//!
//! Synthesis is supplied by the host so Layer I, Layer II and Musepack can
//! share one integer DCT/window implementation and its rounding contract.

use std::sync::LazyLock;

use oxideav_core::bits::BitReader;

use crate::audio_data::parse_audio_data_with_section_bits;
use crate::bitalloc::{class_of_quantization, QuantClass};
use crate::frame::{compute_layer2_crc, FrameError, PCM_SAMPLES_PER_CHANNEL};
use crate::header::FrameHeader;
use crate::requant::read_triplet_codes;

/// The synthesis half of FFmpeg's fixed-point MPEG audio decode contract.
/// Input is Q23, channel-major, 36 blocks of 32 subbands per channel.
/// Write every output sample, retaining filter history and the rounding
/// remainder across frames. FFmpeg processes all blocks of channel 0 before
/// channel 1 and shares one rounding remainder between both channels.
pub trait FixedSynthesis: Send {
    fn synthesize(&mut self, subbands: &[[[i32; 32]; 36]], pcm: &mut [[i16; 1152]]);
    fn reset(&mut self);
}

const FRAC_BITS: u32 = 23;
const FRAC_ONE: i64 = 1 << FRAC_BITS;
const SCALE_MOD: [f64; 3] = [1.0, 0.7937005259, 0.6299605249];

fn fixr(value: f64) -> i32 {
    (value * FRAC_ONE as f64 + 0.5) as i32
}

static SCALE_FACTOR_MULT: LazyLock<[[i32; 3]; 15]> = LazyLock::new(|| {
    std::array::from_fn(|i| {
        let n = i + 2;
        let norm = ((1i64 << n) * FRAC_ONE) / ((1i64 << n) - 1);
        SCALE_MOD.map(|factor| ((norm * i64::from(fixr(factor * 2.0))) >> FRAC_BITS) as i32)
    })
});

static SCALE_FACTOR_MULT2: LazyLock<[[i32; 3]; 3]> = LazyLock::new(|| {
    [3.0, 5.0, 9.0].map(|steps| SCALE_MOD.map(|factor| fixr(factor * (4.0 / steps))))
});

fn unscale(class: &QuantClass, code: u32, scale: u8) -> i32 {
    let shift = u32::from(scale / 3);
    let modulo = usize::from(scale % 3);
    if class.grouping {
        let value = (code as i32 - (class.nb_steps >> 1) as i32)
            * SCALE_FACTOR_MULT2[(class.nb_steps >> 2) as usize][modulo];
        if shift == 0 { value } else { (value + (1 << (shift - 1))) >> shift }
    } else {
        let n = class.bits_per_codeword - 1;
        let mantissa = code.wrapping_add(u32::MAX << n).wrapping_add(1) as i32;
        let value = i64::from(mantissa) * i64::from(SCALE_FACTOR_MULT[n as usize - 1][modulo]);
        let shift = shift + n;
        ((value + (1i64 << (shift - 1))) >> shift) as i32
    }
}

pub(crate) fn decode_frame(
    frame: &[u8],
    header: &FrameHeader,
    synthesis: &mut dyn FixedSynthesis,
) -> Result<Vec<Vec<u8>>, FrameError> {
    let start = if header.protection_bit { 4 } else { 6 };
    if frame.len() < start {
        return Err(FrameError::Truncated { have: frame.len(), need: start });
    }
    let mut reader = BitReader::with_position(frame, start);
    let alloc_start = reader.bit_position();
    let (audio, alloc_bits, scfsi_bits) = parse_audio_data_with_section_bits(header, &mut reader)?;
    if !header.protection_bit {
        let expected = u16::from_be_bytes([frame[4], frame[5]]);
        let computed = compute_layer2_crc(frame, alloc_start, alloc_bits + scfsi_bits);
        if computed != expected {
            return Err(FrameError::CrcMismatch { computed, expected });
        }
    }

    let mut subbands = [[[0i32; 32]; 36]; 2];
    for granule in 0..12 {
        for sb in 0..audio.sblimit {
            let coded_channels = if sb < audio.bound { audio.channels } else { 1 };
            for ch in 0..coded_channels {
                let steps = audio.nb_steps[ch][sb];
                if steps == 0 { continue; }
                let class = class_of_quantization(steps)
                    .ok_or(FrameError::UnknownQuantClass { ch, sb, nb_steps: steps })?;
                let codes = read_triplet_codes(&class, &mut reader)?;
                let end = if sb < audio.bound { ch + 1 } else { audio.channels };
                for channel in ch..end {
                    let scale = audio.scalefactor[channel][sb][granule / 4];
                    for (i, &code) in codes.iter().enumerate() {
                        subbands[channel][granule * 3 + i][sb] = unscale(&class, code, scale);
                    }
                }
            }
        }
    }

    let mut pcm = [[0i16; PCM_SAMPLES_PER_CHANNEL]; 2];
    synthesis.synthesize(&subbands[..audio.channels], &mut pcm[..audio.channels]);
    Ok(pcm[..audio.channels].iter().map(|plane| {
        plane.iter().flat_map(|sample| sample.to_le_bytes()).collect()
    }).collect())
}
