//! Progressive (spectral-selection) decoding for Huffman .bg streams (v4–v19).
//!
//! Each preview round decodes only the first K zigzag DCT coefficients per
//! 8×8 block (DC first, then progressively higher frequencies) and renders a
//! full-size frame, so a consumer can show a coarse preview early and refine
//! it round by round. Every round — including any interruption point — leaves
//! a complete, full-size image in the caller's output buffer. The final round
//! delegates to `decoder::decode`, so the end result is bit-identical to
//! one-shot decoding.
//!
//! Memory: preview rounds allocate only u8 working planes (Y + Cb + Cr at
//! 4:2:0 = 1.5·w·h bytes; alpha is rendered opaque until the final round),
//! i.e. ≤ 50% of the final RGB image (37.5% for RGBA). Coefficients are
//! streamed block-by-block with O(1) extra state — no per-plane `Vec<Block>`.
//!
//! The bitstream itself is unchanged: any .bg v4–v19 file decodes
//! progressively without re-encoding. Legacy RLE streams (v1–v3) fall back to
//! a single one-shot frame.

use crate::block::Block;
use crate::colorspace;
use crate::decoder::{self, dequantize, write_block_to_plane};
use crate::dct;
use crate::encoder;
use crate::huffman;

/// Default spectral schedule: zigzag coefficient budget per preview round
/// (1 = DC only … 28 = roughly the 5×5 low-frequency triangle). The final
/// full decode always runs after these preview rounds.
pub const DEFAULT_SCHEDULE: [usize; 4] = [1, 6, 15, 28];

/// Metadata for one delivered frame.
#[derive(Debug, Clone, Copy)]
pub struct ProgressiveFrameInfo {
    /// 0-based round index.
    pub round: usize,
    /// Total rounds that would run without interruption (previews + final).
    pub total_rounds: usize,
    /// Zigzag coefficients used for this frame (64 = full quality).
    pub coef_budget: usize,
    /// True for the final, bit-exact frame.
    pub is_final: bool,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

const HEADER_SIZE: usize = 3 + 4 + 4 + 1;
const HEADER_SIZE_OLD: usize = 3 + 4 + 4;

/// Read header and return (width, height, channels), or None if the buffer
/// is not a valid .bg stream. Channels are 1, 3, or 4.
pub fn probe(buffer: &[u8]) -> Option<(u32, u32, u32)> {
    if buffer.len() < HEADER_SIZE_OLD {
        return None;
    }
    if buffer[0] != b'B' || buffer[1] != b'G' {
        return None;
    }
    let version = buffer[2];
    if version == 0 || version > 19 {
        return None;
    }
    let width = u32::from_le_bytes(buffer[3..7].try_into().unwrap());
    let height = u32::from_le_bytes(buffer[7..11].try_into().unwrap());
    if width == 0 || height == 0 || width > 65536 || height > 65536 {
        return None;
    }
    let channels = match version {
        1 => 1,
        2 => 3,
        3 => 4,
        v if v % 2 == 1 => 4, // v5, v7, …, v19 carry an alpha plane
        _ => 3,
    };
    Some((width, height, channels))
}

/// Heap bytes the preview path allocates internally for a `width`×`height`
/// image: Y plane + two quarter-res chroma planes. Compare against
/// `width * height * channels` (the final image) — 50.0% for RGB, 37.5% for
/// RGBA. The per-block coefficient state is a single 128-byte stack block.
pub fn preview_working_bytes(width: usize, height: usize) -> usize {
    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;
    width * height + 2 * cw * ch
}

/// Quant tables + entropy profile for a Huffman-era stream version.
/// Mirrors the per-version mapping in `decoder::decode`.
fn profile_for_version(version: u8, q: u8) -> ([i16; 64], [i16; 64], bool, bool) {
    match version {
        4 | 5 => (
            encoder::quant_table_for_quality(q),
            encoder::chroma_quant_table_for_quality(q),
            false,
            false,
        ),
        6 | 7 => (
            encoder::quant_table_for_quality(q),
            encoder::chroma_quant_table_for_quality(q),
            true,
            false,
        ),
        8 | 9 => (
            encoder::quant_table_for_quality_perceptual(q),
            encoder::chroma_quant_table_for_quality_perceptual(q),
            true,
            false,
        ),
        10 | 11 => (
            encoder::quant_table_for_quality_perceptual(q),
            encoder::chroma_quant_table_for_quality_perceptual(q),
            true,
            true,
        ),
        12 | 13 => (
            encoder::quant_table_for_quality_perceptual_v2(q),
            encoder::chroma_quant_table_for_quality_perceptual_v2(q),
            true,
            true,
        ),
        14 | 15 => (
            encoder::quant_table_for_quality_perceptual_v3(q),
            encoder::chroma_quant_table_for_quality_perceptual_v3(q),
            true,
            true,
        ),
        _ => (
            // v16–v19
            encoder::quant_table_for_quality_perceptual_v4(q),
            encoder::chroma_quant_table_for_quality_perceptual_v4(q),
            true,
            true,
        ),
    }
}

/// Decode one plane with a spectral cap, streaming block-by-block:
/// dequantize + IDCT each block as it arrives and write it into `plane`.
fn decode_plane_preview(
    buffer: &[u8],
    pos: usize,
    w: usize,
    h: usize,
    quant: &[i16; 64],
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
    coef_cap: usize,
    plane: &mut [u8],
) -> Option<usize> {
    let bw = (w + 7) / 8;
    let n = bw * ((h + 7) / 8);
    let mut idx = 0usize;
    huffman::decode_plane_streaming_capped(
        buffer,
        pos,
        n,
        is_chroma,
        use_chroma_ac,
        use_dc_delta,
        coef_cap,
        |src: &Block| {
            let mut block = *src;
            dequantize(&mut block, quant);
            dct::idct(&mut block);
            let by = (idx / bw) * 8;
            let bx = (idx % bw) * 8;
            write_block_to_plane(&block, plane, w, h, bx, by);
            idx += 1;
        },
    )
}

/// Render one preview round into `out_pixels` (full-size RGB/RGBA).
/// Alpha, when present, is opaque in preview rounds.
#[allow(clippy::too_many_arguments)]
fn render_preview(
    buffer: &[u8],
    header_size: usize,
    version: u8,
    w: usize,
    h: usize,
    channels: usize,
    q: u8,
    coef_cap: usize,
    out_pixels: &mut [u8],
) -> bool {
    let (luma_q, chroma_q, use_chroma_ac, use_dc_delta) = profile_for_version(version, q);
    let cw = (w + 1) / 2;
    let ch = (h + 1) / 2;

    let mut y_plane = vec![0u8; w * h];
    let mut cb_plane = vec![0u8; cw * ch];
    let mut cr_plane = vec![0u8; cw * ch];

    let mut pos = header_size;
    // Luma always uses the luma AC table (use_chroma_ac = false), matching
    // `decoder::decode`; the profile's chroma-AC flag applies to Cb/Cr only.
    pos = match decode_plane_preview(
        buffer, pos, w, h, &luma_q, false, false, use_dc_delta, coef_cap, &mut y_plane,
    ) {
        Some(p) => p,
        None => return false,
    };
    pos = match decode_plane_preview(
        buffer, pos, cw, ch, &chroma_q, true, use_chroma_ac, use_dc_delta, coef_cap, &mut cb_plane,
    ) {
        Some(p) => p,
        None => return false,
    };
    let _pos = match decode_plane_preview(
        buffer, pos, cw, ch, &chroma_q, true, use_chroma_ac, use_dc_delta, coef_cap, &mut cr_plane,
    ) {
        Some(p) => p,
        None => return false,
    };
    // The alpha plane (odd versions ≥ 5) is intentionally not decoded here.

    let out = &mut out_pixels[..w * h * channels];
    if channels == 3 {
        colorspace::ycbcr420_to_rgb(&y_plane, &cb_plane, &cr_plane, w, h, out);
    } else {
        colorspace::ycbcr420_to_rgba_opaque(&y_plane, &cb_plane, &cr_plane, w, h, out);
    }
    true
}

/// Progressively decode a .bg stream, invoking `on_frame` after every round.
///
/// - `schedule`: zigzag coefficient budgets for preview rounds; `None` uses
///   [`DEFAULT_SCHEDULE`]. Budgets are clamped to 1..=63; the final full
///   decode always runs as the last round.
/// - `out_pixels`: caller-provided frame buffer, at least `w*h*channels`
///   bytes (see [`probe`]). Every delivered frame is full-size.
/// - `on_frame`: return `false` to stop after this frame (interruption); the
///   last delivered frame remains valid in `out_pixels`.
///
/// Returns false on malformed input or a too-small output buffer. Returns
/// true when decoding completed or was interrupted by the callback.
pub fn decode_progressive<F>(
    buffer: &[u8],
    schedule: Option<&[usize]>,
    out_pixels: &mut [u8],
    mut on_frame: F,
) -> bool
where
    F: FnMut(&ProgressiveFrameInfo, &[u8]) -> bool,
{
    let (width, height, channels) = match probe(buffer) {
        Some(v) => v,
        None => return false,
    };
    let version = buffer[2];
    let w = width as usize;
    let h = height as usize;
    let c = channels as usize;
    if out_pixels.len() < w * h * c {
        return false;
    }

    let (header_size, quality) = if buffer.len() >= HEADER_SIZE {
        (HEADER_SIZE, buffer[11])
    } else {
        (HEADER_SIZE_OLD, 50u8)
    };
    let q = if quality == 0 { 50 } else { quality };

    // Legacy RLE streams (v1–v3) and grayscale: no spectral structure to
    // refine — deliver a single one-shot frame.
    if version < 4 {
        let mut ow = 0u32;
        let mut oh = 0u32;
        let mut oc = 0u32;
        if !decoder::decode(buffer, &mut out_pixels[..w * h * c], &mut ow, &mut oh, &mut oc, None) {
            return false;
        }
        let info = ProgressiveFrameInfo {
            round: 0,
            total_rounds: 1,
            coef_budget: 64,
            is_final: true,
            width,
            height,
            channels,
        };
        on_frame(&info, &out_pixels[..w * h * c]);
        return true;
    }

    let default_schedule;
    let budgets: &[usize] = match schedule {
        Some(s) => s,
        None => {
            default_schedule = DEFAULT_SCHEDULE;
            &default_schedule
        }
    };
    let total_rounds = budgets.len() + 1;

    // Preview rounds: spectral cap, coarse full-size frames.
    for (round, &k) in budgets.iter().enumerate() {
        let cap = k.clamp(1, 63);
        if !render_preview(buffer, header_size, version, w, h, c, q, cap, out_pixels) {
            return false;
        }
        let info = ProgressiveFrameInfo {
            round,
            total_rounds,
            coef_budget: cap,
            is_final: false,
            width,
            height,
            channels,
        };
        if !on_frame(&info, &out_pixels[..w * h * c]) {
            return true; // interrupted: last frame stays valid in out_pixels
        }
    }

    // Final round: exact one-shot decode → bit-identical to `decoder::decode`.
    let mut ow = 0u32;
    let mut oh = 0u32;
    let mut oc = 0u32;
    if !decoder::decode(buffer, &mut out_pixels[..w * h * c], &mut ow, &mut oh, &mut oc, None) {
        return false;
    }
    let info = ProgressiveFrameInfo {
        round: total_rounds - 1,
        total_rounds,
        coef_budget: 64,
        is_final: true,
        width,
        height,
        channels,
    };
    on_frame(&info, &out_pixels[..w * h * c]);
    true
}
