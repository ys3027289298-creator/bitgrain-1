//! Decoder .bg → image.
//! Supports all versions:
//!   v1: grayscale, RLE
//!   v2: RGB planar, RLE
//!   v3: RGBA planar, RLE
//!   v4: YCbCr 4:2:0, Huffman  → RGB output
//!   v5: YCbCr 4:2:0 + A, Huffman → RGBA output
//!   v6: YCbCr 4:2:0, Huffman + chroma AC table → RGB output
//!   v7: YCbCr 4:2:0 + A, Huffman + chroma AC table → RGBA output
//!   v8: YCbCr 4:2:0, perceptual quant + chroma AC → RGB output
//!   v9: YCbCr 4:2:0 + A, perceptual quant + chroma AC → RGBA output
//!  v10: YCbCr 4:2:0, perceptual quant + chroma AC + DC delta → RGB output
//!  v11: YCbCr 4:2:0 + A, perceptual quant + chroma AC + DC delta → RGBA output
//!  v12: YCbCr 4:2:0, stronger perceptual quant + chroma AC + DC delta → RGB output
//!  v13: YCbCr 4:2:0 + A, stronger perceptual quant + chroma AC + DC delta → RGBA output
//!  v14: YCbCr 4:2:0, aggressive perceptual quant + chroma AC + DC delta → RGB output
//!  v15: YCbCr 4:2:0 + A, aggressive perceptual quant + chroma AC + DC delta → RGBA output
//!  v16: YCbCr 4:2:0, very aggressive perceptual quant + chroma AC + DC delta → RGB output
//!  v17: YCbCr 4:2:0 + A, very aggressive perceptual quant + chroma AC + DC delta → RGBA output
//!  v18: YCbCr 4:2:0, ultra perceptual + AC sparsify + chroma AC + DC delta → RGB output
//!  v19: YCbCr 4:2:0 + A, ultra perceptual + AC sparsify + chroma AC + DC delta → RGBA output
//!
//! Partial decode (opt-in):
//!   By default the decoder is strict: any corruption fails the whole decode
//!   (all-or-nothing, exactly the historical behavior). Setting the env var
//!   BITGRAIN_PARTIAL_DECODE=1 (also true/on/yes) enables partial output:
//!   decoding stops at the corruption point, fully decoded 8x8 blocks are
//!   written at their block-aligned positions, the remaining region is filled
//!   with gray (alpha: opaque), and the completed-block count plus a failure
//!   reason is reported via the bitgrain_partial_* getters below. Intact
//!   files decode byte-identically in both modes.

use crate::block::Block;
use crate::colorspace;
use crate::dct;
use crate::encoder;
use crate::ffi::dequantize_block;
use crate::huffman;
use crate::zigzag::ZIGZAG;
use rayon::prelude::*;
use std::cell::RefCell;
use std::ffi::CString;
const BLOCK_TILE_SIZE: usize = 512;
const PARALLEL_DEQUANT_BLOCKS_THRESHOLD: usize = 384;
const PARALLEL_DEQUANT_PIXELS_THRESHOLD: usize = 262_144;
const PARALLEL_WRITE_BLOCKS_THRESHOLD: usize = 2048;
const PARALLEL_WRITE_PIXELS_THRESHOLD: usize = 786_432;

const HEADER_SIZE:     usize = 3 + 4 + 4 + 1;
const HEADER_SIZE_OLD: usize = 3 + 4 + 4;
const EOB_RUN: u8 = 0xFF;

/// Fill value for undecoded luma/RGB regions (mid gray).
const GRAY_FILL: u8 = 128;
/// Fill value for undecoded alpha regions (opaque).
const ALPHA_FILL: u8 = 255;
/// Upper bound on entropy-coded bytes per 8x8 block; used only to tell a
/// corrupt plane length prefix apart from a truncated stream.
const MAX_PLANE_BYTES_PER_BLOCK: usize = 216;

#[inline]
fn should_parallel_dequant(n_blocks: usize, w: usize, h: usize) -> bool {
    n_blocks >= PARALLEL_DEQUANT_BLOCKS_THRESHOLD
        && w.saturating_mul(h) >= PARALLEL_DEQUANT_PIXELS_THRESHOLD
}

#[inline]
fn should_parallel_write(n_blocks: usize, w: usize, h: usize) -> bool {
    n_blocks >= PARALLEL_WRITE_BLOCKS_THRESHOLD
        && w.saturating_mul(h) >= PARALLEL_WRITE_PIXELS_THRESHOLD
}

// ---------------------------------------------------------------------------
// Partial decode status (thread-local) + additive read-only FFI getters
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct PartialDecodeInfo {
    /// Partial output was actually produced (corruption tolerated).
    occurred: bool,
    /// Fully decoded 8x8 blocks across all planes.
    completed_blocks: u64,
    /// Total 8x8 blocks across all planes.
    total_blocks: u64,
    /// Plane samples covered by completed blocks (clipped to plane bounds).
    valid_samples: u64,
    /// Total plane samples.
    total_samples: u64,
    /// Failure reason ("" when the decode was complete).
    reason: &'static str,
}

thread_local! {
    static LAST_DECODE_INFO: RefCell<PartialDecodeInfo> =
        RefCell::new(PartialDecodeInfo::default());
    static REASON_CSTRING: RefCell<CString> =
        RefCell::new(CString::new("").expect("static cstring"));
}

/// Non-zero when the last decode on this thread produced partial output
/// (only possible with BITGRAIN_PARTIAL_DECODE enabled).
#[no_mangle]
pub extern "C" fn bitgrain_partial_decode_occurred() -> i32 {
    LAST_DECODE_INFO.with(|i| i.borrow().occurred as i32)
}

/// Number of fully decoded 8x8 blocks in the last decode on this thread.
#[no_mangle]
pub extern "C" fn bitgrain_partial_completed_blocks() -> u64 {
    LAST_DECODE_INFO.with(|i| i.borrow().completed_blocks)
}

/// Total 8x8 blocks of the image decoded last on this thread.
#[no_mangle]
pub extern "C" fn bitgrain_partial_total_blocks() -> u64 {
    LAST_DECODE_INFO.with(|i| i.borrow().total_blocks)
}

/// Plane samples covered by completed blocks in the last decode.
#[no_mangle]
pub extern "C" fn bitgrain_partial_valid_samples() -> u64 {
    LAST_DECODE_INFO.with(|i| i.borrow().valid_samples)
}

/// Total plane samples of the image decoded last on this thread.
#[no_mangle]
pub extern "C" fn bitgrain_partial_total_samples() -> u64 {
    LAST_DECODE_INFO.with(|i| i.borrow().total_samples)
}

/// Failure reason of the last decode on this thread ("" when complete).
/// The returned pointer is valid until the next decode on the same thread.
#[no_mangle]
pub extern "C" fn bitgrain_partial_failure_reason() -> *const std::os::raw::c_char {
    LAST_DECODE_INFO.with(|i| {
        let reason = i.borrow().reason;
        REASON_CSTRING.with(|c| {
            *c.borrow_mut() = CString::new(reason).expect("static reason cstring");
            c.borrow().as_ptr()
        })
    })
}

/// Partial output is opt-in via BITGRAIN_PARTIAL_DECODE=1|true|on|yes.
#[inline]
fn partial_decode_enabled() -> bool {
    match std::env::var("BITGRAIN_PARTIAL_DECODE") {
        Ok(v) => matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes"),
        Err(_) => false,
    }
}

/// Samples of a w×h plane covered by the first `k` blocks (raster order),
/// clipped to the plane bounds.
fn covered_samples(k: usize, bw: usize, w: usize, h: usize) -> u64 {
    let full_rows = k / bw;
    let rem = k % bw;
    let mut s = (full_rows.saturating_mul(8).min(h) * w) as u64;
    if rem > 0 {
        let by = full_rows * 8;
        let band_h = h.saturating_sub(by).min(8);
        for i in 0..rem {
            let bw_rem = w.saturating_sub(i * 8).min(8);
            s += (bw_rem * band_h) as u64;
        }
    }
    s
}

// ---------------------------------------------------------------------------
// RLE decode (v1/v2/v3)
// ---------------------------------------------------------------------------

fn decode_rle_one_block(buffer: &[u8], mut pos: usize) -> Option<(Block, usize)> {
    let mut block = Block::new();
    if pos + 2 > buffer.len() { return None; }
    block.data[ZIGZAG[0]] = i16::from_le_bytes([buffer[pos], buffer[pos + 1]]);
    pos += 2;

    let mut ac_index = 1usize;
    loop {
        if pos + 3 > buffer.len() { return None; }
        let run   = buffer[pos];
        let level = i16::from_le_bytes([buffer[pos + 1], buffer[pos + 2]]);
        pos += 3;
        if run == EOB_RUN && level == 0 { break; }
        for _ in 0..run {
            if ac_index < 64 { block.data[ZIGZAG[ac_index]] = 0; ac_index += 1; }
        }
        if ac_index < 64 { block.data[ZIGZAG[ac_index]] = level; ac_index += 1; }
    }
    Some((block, pos))
}

fn decode_rle_to_blocks(buffer: &[u8], mut pos: usize, n: usize) -> Option<(Vec<Block>, usize)> {
    let mut blocks = Vec::with_capacity(n);
    for _ in 0..n {
        let (block, new_pos) = decode_rle_one_block(buffer, pos)?;
        pos = new_pos;
        blocks.push(block);
    }
    Some((blocks, pos))
}

#[inline]
fn write_block_to_plane(block: &Block, plane: &mut [u8], w: usize, h: usize, bx: usize, by: usize) {
    // Fast interior path: avoid bounds checks per pixel.
    if bx + 8 <= w && by + 8 <= h {
        for y in 0..8 {
            let dst = (by + y) * w + bx;
            let src = y * 8;
            for x in 0..8 {
                plane[dst + x] = (block.data[src + x] + 128).clamp(0, 255) as u8;
            }
        }
        return;
    }
    // Border blocks (partial at image edges).
    for y in 0..8 {
        for x in 0..8 {
            let py = by + y;
            let px = bx + x;
            if py < h && px < w {
                plane[py * w + px] = (block.data[y * 8 + x] + 128).clamp(0, 255) as u8;
            }
        }
    }
}

/// Dequant + IDCT + interleaved write of decoded blocks (RLE planes).
fn finish_blocks_interleaved(
    blocks: &mut [Block],
    quant: &[i16; 64],
    w: usize, h: usize,
    out: &mut [u8], stride: usize, offset: usize,
) {
    let n  = blocks.len();
    let bw = (w + 7) / 8;

    if should_parallel_dequant(n, w, h) {
        blocks.par_chunks_mut(BLOCK_TILE_SIZE).for_each(|chunk| {
            for block in chunk.iter_mut() {
                unsafe { dequantize_block(block.data.as_mut_ptr(), quant.as_ptr()); }
                dct::idct(block);
            }
        });
    } else {
        for block in blocks.iter_mut() {
            unsafe { dequantize_block(block.data.as_mut_ptr(), quant.as_ptr()); }
            dct::idct(block);
        }
    }

    for (idx, block) in blocks.iter().enumerate() {
        let by = (idx / bw) * 8;
        let bx = (idx % bw) * 8;
        // Interleaved write for legacy planar paths.
        if bx + 8 <= w && by + 8 <= h {
            for y in 0..8 {
                let src = y * 8;
                let row = (by + y) * w;
                for x in 0..8 {
                    out[(row + bx + x) * stride + offset] = (block.data[src + x] + 128).clamp(0, 255) as u8;
                }
            }
        } else {
            for y in 0..8 {
                for x in 0..8 {
                    let py = by + y;
                    let px = bx + x;
                    if py < h && px < w {
                        out[(py * w + px) * stride + offset] = (block.data[y * 8 + x] + 128).clamp(0, 255) as u8;
                    }
                }
            }
        }
    }
}

/// Fill one channel of an interleaved image with a constant value.
fn fill_interleaved(out: &mut [u8], w: usize, h: usize, stride: usize, offset: usize, fill: u8) {
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            out[(row + x) * stride + offset] = fill;
        }
    }
}

/// Result of decoding one plane.
enum PlaneDecode {
    /// Plane fully decoded; carries the byte position of the next plane.
    Full(usize),
    /// Partial mode: `decoded` leading blocks recovered, rest filled.
    Partial { decoded: usize, reason: &'static str },
    /// Strict mode failure.
    Failed(&'static str),
}

/// Decode one RLE plane. Strict mode matches the historical all-or-nothing
/// behavior; partial mode recovers the decodable block prefix and fills the
/// rest of the channel with `fill`.
#[allow(clippy::too_many_arguments)]
fn decode_plane_rle_report(
    buffer: &[u8], pos: usize,
    w: usize, h: usize,
    quant: &[i16; 64],
    out: &mut [u8], stride: usize, offset: usize,
    fill: u8,
    allow_partial: bool,
) -> PlaneDecode {
    let bw = (w + 7) / 8;
    let n  = bw * ((h + 7) / 8);

    if let Some((mut blocks, new_pos)) = decode_rle_to_blocks(buffer, pos, n) {
        finish_blocks_interleaved(&mut blocks, quant, w, h, out, stride, offset);
        return PlaneDecode::Full(new_pos);
    }

    let reason = "truncated or corrupt RLE block stream";
    if !allow_partial {
        return PlaneDecode::Failed(reason);
    }

    // RLE blocks are byte-aligned and independent: decode block by block
    // until the stream breaks.
    let mut blocks: Vec<Block> = Vec::with_capacity(n);
    let mut p = pos;
    while blocks.len() < n {
        match decode_rle_one_block(buffer, p) {
            Some((block, new_p)) => {
                blocks.push(block);
                p = new_p;
            }
            None => break,
        }
    }
    let decoded = blocks.len();
    fill_interleaved(out, w, h, stride, offset, fill);
    finish_blocks_interleaved(&mut blocks, quant, w, h, out, stride, offset);
    PlaneDecode::Partial { decoded, reason }
}

// ---------------------------------------------------------------------------
// Huffman decode (v4+)
// ---------------------------------------------------------------------------

/// Dequant + IDCT + write of decoded blocks into a flat plane buffer.
fn finish_blocks_to_plane(
    blocks: &mut [Block],
    quant: &[i16; 64],
    w: usize, h: usize,
    plane: &mut [u8],
) {
    let n  = blocks.len();
    let bw = (w + 7) / 8;

    // Parallel dequant + IDCT
    if should_parallel_dequant(n, w, h) {
        blocks.par_chunks_mut(BLOCK_TILE_SIZE).for_each(|chunk| {
            for block in chunk.iter_mut() {
                unsafe { dequantize_block(block.data.as_mut_ptr(), quant.as_ptr()); }
                dct::idct(block);
            }
        });
    } else {
        for block in blocks.iter_mut() {
            unsafe { dequantize_block(block.data.as_mut_ptr(), quant.as_ptr()); }
            dct::idct(block);
        }
    }

    // Write to flat plane
    if should_parallel_write(n, w, h) {
        let band_stride = w * 8;
        plane
            .par_chunks_mut(band_stride.max(1))
            .enumerate()
            .for_each(|(br, band)| {
                let by = br * 8;
                let band_h = (h.saturating_sub(by)).min(8);
                if band_h == 0 {
                    return;
                }
                let row_blocks_start = br * bw;
                for bx_i in 0..bw {
                    let idx = row_blocks_start + bx_i;
                    if idx >= blocks.len() {
                        break;
                    }
                    let block = &blocks[idx];
                    let bx = bx_i * 8;
                    if bx + 8 <= w && band_h == 8 {
                        for y in 0..8 {
                            let dst = y * w + bx;
                            let src = y * 8;
                            for x in 0..8 {
                                band[dst + x] = (block.data[src + x] + 128).clamp(0, 255) as u8;
                            }
                        }
                    } else {
                        for y in 0..band_h {
                            for x in 0..8 {
                                let px = bx + x;
                                if px < w {
                                    band[y * w + px] = (block.data[y * 8 + x] + 128).clamp(0, 255) as u8;
                                }
                            }
                        }
                    }
                }
            });
    } else {
        for (idx, block) in blocks.iter().enumerate() {
            let by = (idx / bw) * 8;
            let bx = (idx % bw) * 8;
            write_block_to_plane(block, plane, w, h, bx, by);
        }
    }
}

/// Classify why a Huffman plane failed to decode, without decoding again.
fn classify_plane_failure(buffer: &[u8], pos: usize, n_blocks: usize) -> &'static str {
    if pos + 4 > buffer.len() {
        return "truncated stream: plane length prefix missing";
    }
    let plane_len = u32::from_le_bytes(buffer[pos..pos + 4].try_into().unwrap()) as usize;
    // A length prefix beyond the worst-case entropy size cannot be genuine.
    if plane_len > n_blocks.saturating_mul(MAX_PLANE_BYTES_PER_BLOCK).saturating_add(8) {
        return "corrupt plane length prefix";
    }
    match (pos + 4).checked_add(plane_len) {
        Some(end) if end <= buffer.len() => "corrupt residual bitstream",
        _ => "truncated stream",
    }
}

/// Decode one Huffman plane. Strict mode matches the historical
/// all-or-nothing behavior; partial mode recovers the decodable block prefix
/// and fills the rest of the plane with `fill`.
#[allow(clippy::too_many_arguments)]
fn decode_plane_huffman_report(
    buffer: &[u8], pos: usize,
    w: usize, h: usize,
    quant: &[i16; 64],
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
    fill: u8,
    allow_partial: bool,
    plane: &mut [u8],
) -> PlaneDecode {
    let bw = (w + 7) / 8;
    let n  = bw * ((h + 7) / 8);

    if let Some((mut blocks, new_pos)) =
        huffman::decode_plane_with_profile(buffer, pos, n, is_chroma, use_chroma_ac, use_dc_delta)
    {
        finish_blocks_to_plane(&mut blocks, quant, w, h, plane);
        return PlaneDecode::Full(new_pos);
    }

    let reason = classify_plane_failure(buffer, pos, n);
    if !allow_partial {
        return PlaneDecode::Failed(reason);
    }

    if pos + 4 > buffer.len() {
        plane.fill(fill);
        return PlaneDecode::Partial { decoded: 0, reason };
    }

    // If the declared plane length runs past the buffer (truncation or a
    // corrupt length prefix), clamp it to the available bytes so the decoder
    // can recover whatever whole blocks are actually present.
    let plane_len = u32::from_le_bytes(buffer[pos..pos + 4].try_into().unwrap()) as usize;
    let mut patched: Vec<u8>;
    let buf2: &[u8] = match (pos + 4).checked_add(plane_len) {
        Some(end) if end <= buffer.len() => buffer,
        _ => {
            patched = buffer.to_vec();
            let avail = (buffer.len() - (pos + 4)) as u32;
            patched[pos..pos + 4].copy_from_slice(&avail.to_le_bytes());
            &patched
        }
    };

    // Largest decodable block prefix. Decoding k blocks is monotone in k:
    // any prefix of a decodable prefix also decodes.
    let decodable = |k: usize| {
        huffman::decode_plane_with_profile(buf2, pos, k, is_chroma, use_chroma_ac, use_dc_delta)
            .is_some()
    };
    let k_max = if decodable(n) {
        // Whole plane content recovered (e.g. only the length prefix was bad);
        // the next plane boundary is unknown, so higher layers stop here.
        n
    } else {
        let mut lo = 0usize; // decodable
        let mut hi = n; // not decodable
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if decodable(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };

    let (mut blocks, _) =
        huffman::decode_plane_with_profile(buf2, pos, k_max, is_chroma, use_chroma_ac, use_dc_delta)
            .expect("prefix known to decode");
    plane.fill(fill);
    finish_blocks_to_plane(&mut blocks, quant, w, h, plane);
    PlaneDecode::Partial { decoded: k_max, reason }
}

// ---------------------------------------------------------------------------
// ICC trailer
// ---------------------------------------------------------------------------

fn parse_icc_trailer(buffer: &[u8], pos: usize) -> Option<(Vec<u8>, usize)> {
    if pos + 8 > buffer.len() { return None; }
    if buffer[pos] != b'B' || buffer[pos+1] != b'G' || buffer[pos+2] != b'x' { return None; }
    let chunk_type = buffer[pos + 3];
    let len = u32::from_le_bytes(buffer[pos+4..pos+8].try_into().unwrap()) as usize;
    let data_pos = pos + 8;
    if chunk_type != 1 || data_pos + len > buffer.len() { return None; }
    Some((buffer[data_pos..data_pos+len].to_vec(), data_pos + len))
}

// ---------------------------------------------------------------------------
// Version profiles
// ---------------------------------------------------------------------------

/// Decode parameters shared by the Huffman-coded versions (v4..=v19).
struct HuffmanProfile {
    luma_q: [i16; 64],
    chroma_q: [i16; 64],
    use_chroma_ac: bool,
    use_dc_delta: bool,
    has_alpha: bool,
}

fn huffman_profile_for_version(version: u8, q: u8) -> Option<HuffmanProfile> {
    if !(4..=19).contains(&version) {
        return None;
    }
    let has_alpha = version % 2 == 1;
    let (luma_q, chroma_q) = match version {
        4..=7 => (
            encoder::quant_table_for_quality(q),
            encoder::chroma_quant_table_for_quality(q),
        ),
        8..=11 => (
            encoder::quant_table_for_quality_perceptual(q),
            encoder::chroma_quant_table_for_quality_perceptual(q),
        ),
        12 | 13 => (
            encoder::quant_table_for_quality_perceptual_v2(q),
            encoder::chroma_quant_table_for_quality_perceptual_v2(q),
        ),
        14 | 15 => (
            encoder::quant_table_for_quality_perceptual_v3(q),
            encoder::chroma_quant_table_for_quality_perceptual_v3(q),
        ),
        _ => (
            encoder::quant_table_for_quality_perceptual_v4(q),
            encoder::chroma_quant_table_for_quality_perceptual_v4(q),
        ),
    };
    Some(HuffmanProfile {
        luma_q,
        chroma_q,
        use_chroma_ac: version >= 6,
        use_dc_delta: version >= 10,
        has_alpha,
    })
}

/// Decode the Y/Cb/Cr(/A) planes of a Huffman-coded version and convert to
/// interleaved RGB(A). On partial output the undecoded planes are filled and
/// the colorspace conversion runs over real + filled data.
#[allow(clippy::too_many_arguments)]
fn decode_huffman_planes(
    buffer: &[u8],
    header_size: usize,
    w: usize, h: usize,
    profile: &HuffmanProfile,
    out_pixels: &mut [u8],
    out_icc: Option<&mut Vec<u8>>,
    allow_partial: bool,
    info: &mut PartialDecodeInfo,
) -> bool {
    let cw = (w + 1) / 2;
    let ch = (h + 1) / 2;
    let bw = (w + 7) / 8;
    let bh = (h + 7) / 8;
    let cbw = (cw + 7) / 8;
    let cbh = (ch + 7) / 8;

    let y_n = bw * bh;
    let c_n = cbw * cbh;
    info.total_blocks = (y_n + 2 * c_n + if profile.has_alpha { y_n } else { 0 }) as u64;
    info.total_samples = (w * h + 2 * cw * ch + if profile.has_alpha { w * h } else { 0 }) as u64;

    let mut y_plane  = vec![0u8; w * h];
    let mut cb_plane = vec![0u8; cw * ch];
    let mut cr_plane = vec![0u8; cw * ch];
    let mut a_plane  = if profile.has_alpha { vec![0u8; w * h] } else { Vec::new() };

    let mut pos = header_size;
    let mut stop_reason: Option<&'static str> = None;

    // Y plane (luma AC table, like the encoder)
    match decode_plane_huffman_report(
        buffer, pos, w, h, &profile.luma_q, false,
        false, profile.use_dc_delta,
        GRAY_FILL, allow_partial, &mut y_plane,
    ) {
        PlaneDecode::Full(p) => {
            pos = p;
            info.completed_blocks += y_n as u64;
            info.valid_samples += (w * h) as u64;
        }
        PlaneDecode::Partial { decoded, reason } => {
            info.completed_blocks += decoded as u64;
            info.valid_samples += covered_samples(decoded, bw, w, h);
            stop_reason = Some(reason);
        }
        PlaneDecode::Failed(reason) => {
            info.reason = reason;
            return false;
        }
    }

    // Cb plane
    match stop_reason {
        Some(_) => cb_plane.fill(GRAY_FILL),
        None => match decode_plane_huffman_report(
            buffer, pos, cw, ch, &profile.chroma_q, true,
            profile.use_chroma_ac, profile.use_dc_delta,
            GRAY_FILL, allow_partial, &mut cb_plane,
        ) {
            PlaneDecode::Full(p) => {
                pos = p;
                info.completed_blocks += c_n as u64;
                info.valid_samples += (cw * ch) as u64;
            }
            PlaneDecode::Partial { decoded, reason } => {
                info.completed_blocks += decoded as u64;
                info.valid_samples += covered_samples(decoded, cbw, cw, ch);
                stop_reason = Some(reason);
            }
            PlaneDecode::Failed(reason) => {
                info.reason = reason;
                return false;
            }
        },
    }

    // Cr plane
    match stop_reason {
        Some(_) => cr_plane.fill(GRAY_FILL),
        None => match decode_plane_huffman_report(
            buffer, pos, cw, ch, &profile.chroma_q, true,
            profile.use_chroma_ac, profile.use_dc_delta,
            GRAY_FILL, allow_partial, &mut cr_plane,
        ) {
            PlaneDecode::Full(p) => {
                pos = p;
                info.completed_blocks += c_n as u64;
                info.valid_samples += (cw * ch) as u64;
            }
            PlaneDecode::Partial { decoded, reason } => {
                info.completed_blocks += decoded as u64;
                info.valid_samples += covered_samples(decoded, cbw, cw, ch);
                stop_reason = Some(reason);
            }
            PlaneDecode::Failed(reason) => {
                info.reason = reason;
                return false;
            }
        },
    }

    // Alpha plane (odd versions; luma AC table, like the encoder)
    if profile.has_alpha {
        match stop_reason {
            Some(_) => a_plane.fill(ALPHA_FILL),
            None => match decode_plane_huffman_report(
                buffer, pos, w, h, &profile.luma_q, false,
                false, profile.use_dc_delta,
                ALPHA_FILL, allow_partial, &mut a_plane,
            ) {
                PlaneDecode::Full(p) => {
                    pos = p;
                    info.completed_blocks += y_n as u64;
                    info.valid_samples += (w * h) as u64;
                }
                PlaneDecode::Partial { decoded, reason } => {
                    info.completed_blocks += decoded as u64;
                    info.valid_samples += covered_samples(decoded, bw, w, h);
                    stop_reason = Some(reason);
                }
                PlaneDecode::Failed(reason) => {
                    info.reason = reason;
                    return false;
                }
            },
        }
    }

    if let Some(reason) = stop_reason {
        info.occurred = true;
        info.reason = reason;
    }

    if profile.has_alpha {
        colorspace::ycbcr420a_to_rgba(&y_plane, &cb_plane, &cr_plane, &a_plane, w, h, out_pixels);
    } else {
        colorspace::ycbcr420_to_rgb(&y_plane, &cb_plane, &cr_plane, w, h, out_pixels);
    }

    // ICC trailer only makes sense for a fully decoded stream.
    if stop_reason.is_none() {
        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
    }
    true
}

#[inline]
fn fill_for_channel(chs: usize, c: usize) -> u8 {
    if chs == 4 && c == 3 { ALPHA_FILL } else { GRAY_FILL }
}

// ---------------------------------------------------------------------------
// Public decode entry point
// ---------------------------------------------------------------------------

pub fn decode(
    buffer: &[u8],
    out_pixels: &mut [u8],
    out_width:  &mut u32,
    out_height: &mut u32,
    out_channels: &mut u32,
    out_icc: Option<&mut Vec<u8>>,
) -> bool {
    let allow_partial = partial_decode_enabled();
    let mut info = PartialDecodeInfo::default();

    let ok = decode_inner(
        buffer, out_pixels, out_width, out_height, out_channels, out_icc,
        allow_partial, &mut info,
    );

    if info.occurred {
        let pct = if info.total_samples > 0 {
            100.0 * info.valid_samples as f64 / info.total_samples as f64
        } else {
            0.0
        };
        eprintln!(
            "bitgrain: partial decode: {}/{} blocks complete ({:.1}% samples valid), reason: {}",
            info.completed_blocks, info.total_blocks, pct, info.reason
        );
    }
    LAST_DECODE_INFO.with(|slot| *slot.borrow_mut() = info);
    ok
}

#[allow(clippy::too_many_arguments)]
fn decode_inner(
    buffer: &[u8],
    out_pixels: &mut [u8],
    out_width:  &mut u32,
    out_height: &mut u32,
    out_channels: &mut u32,
    out_icc: Option<&mut Vec<u8>>,
    allow_partial: bool,
    info: &mut PartialDecodeInfo,
) -> bool {
    if buffer.len() < HEADER_SIZE_OLD {
        info.reason = "file too small";
        return false;
    }
    if buffer[0] != b'B' || buffer[1] != b'G' {
        info.reason = "bad magic";
        return false;
    }

    let version = buffer[2];
    if version == 0 || version > 19 {
        info.reason = "unsupported version";
        return false;
    }

    let width  = u32::from_le_bytes(buffer[3..7].try_into().unwrap());
    let height = u32::from_le_bytes(buffer[7..11].try_into().unwrap());
    let (header_size, quality) = if buffer.len() >= HEADER_SIZE {
        (HEADER_SIZE, buffer[11])
    } else {
        (HEADER_SIZE_OLD, 50u8)
    };
    let q = if quality == 0 { 50 } else { quality };

    if width == 0 || height == 0 || width > 65536 || height > 65536 {
        info.reason = "invalid dimensions";
        return false;
    }
    let w = width as usize;
    let h = height as usize;
    let bw = (w + 7) / 8;
    let bh = (h + 7) / 8;
    let n_blocks = bw * bh;

    // ---- v1: grayscale RLE ----
    if version == 1 {
        if out_pixels.len() < w * h {
            info.reason = "output buffer too small";
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 1;
        info.total_blocks = n_blocks as u64;
        info.total_samples = (w * h) as u64;
        let quant = encoder::quant_table_for_quality(q);
        return match decode_plane_rle_report(
            buffer, header_size, w, h, &quant, out_pixels, 1, 0, GRAY_FILL, allow_partial,
        ) {
            PlaneDecode::Full(pos) => {
                info.completed_blocks = n_blocks as u64;
                info.valid_samples = (w * h) as u64;
                if let Some(v) = out_icc {
                    if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
                }
                true
            }
            PlaneDecode::Partial { decoded, reason } => {
                info.completed_blocks = decoded as u64;
                info.valid_samples = covered_samples(decoded, bw, w, h);
                info.occurred = true;
                info.reason = reason;
                true
            }
            PlaneDecode::Failed(reason) => {
                info.reason = reason;
                false
            }
        };
    }

    // ---- v2: RGB planar RLE ---- / ---- v3: RGBA planar RLE ----
    if version == 2 || version == 3 {
        let chs: usize = if version == 2 { 3 } else { 4 };
        if out_pixels.len() < w * h * chs {
            info.reason = "output buffer too small";
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = chs as u32;
        info.total_blocks = (n_blocks * chs) as u64;
        info.total_samples = (w * h * chs) as u64;
        let quant = encoder::quant_table_for_quality(q);
        let mut pos = header_size;
        let mut stop_reason: Option<&'static str> = None;
        for c in 0..chs {
            if stop_reason.is_some() {
                fill_interleaved(out_pixels, w, h, chs, c, fill_for_channel(chs, c));
                continue;
            }
            match decode_plane_rle_report(
                buffer, pos, w, h, &quant, out_pixels, chs, c,
                fill_for_channel(chs, c), allow_partial,
            ) {
                PlaneDecode::Full(p) => {
                    pos = p;
                    info.completed_blocks += n_blocks as u64;
                    info.valid_samples += (w * h) as u64;
                }
                PlaneDecode::Partial { decoded, reason } => {
                    info.completed_blocks += decoded as u64;
                    info.valid_samples += covered_samples(decoded, bw, w, h);
                    stop_reason = Some(reason);
                }
                PlaneDecode::Failed(reason) => {
                    info.reason = reason;
                    return false;
                }
            }
        }
        if let Some(reason) = stop_reason {
            info.occurred = true;
            info.reason = reason;
            return true;
        }
        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v4..=v19: YCbCr 4:2:0(+A) + Huffman → RGB(A) ----
    if let Some(profile) = huffman_profile_for_version(version, q) {
        let chs: usize = if profile.has_alpha { 4 } else { 3 };
        if out_pixels.len() < w * h * chs {
            info.reason = "output buffer too small";
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = chs as u32;
        return decode_huffman_planes(
            buffer, header_size, w, h, &profile, out_pixels, out_icc, allow_partial, info,
        );
    }

    info.reason = "unsupported version";
    false
}

pub fn decode_grayscale(
    buffer: &[u8], out_pixels: &mut [u8],
    out_width: &mut u32, out_height: &mut u32,
) -> bool {
    let mut ch = 0u32;
    let ok = decode(buffer, out_pixels, out_width, out_height, &mut ch, None);
    ok && ch == 1
}

// ---------------------------------------------------------------------------
// Test-only shims for the C quant kernels
// ---------------------------------------------------------------------------

/// Release builds link quantize_block/dequantize_block from c/quant.c.
/// `cargo test` does not link the C objects, so test builds use these scalar
/// equivalents (same semantics as the scalar reference paths in c/quant.c).
#[cfg(test)]
mod c_quant_shims {
    #[no_mangle]
    pub extern "C" fn quantize_block(block: *mut i16, table: *const i16) {
        unsafe {
            for i in 0..64 {
                let b = *block.add(i) as i32;
                let t = *table.add(i) as i32;
                let q = (b as f32 / t as f32).round() as i32;
                *block.add(i) = q.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn dequantize_block(block: *mut i16, table: *const i16) {
        unsafe {
            for i in 0..64 {
                let v = (*block.add(i) as i32) * (*table.add(i) as i32);
                *block.add(i) = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod partial_decode_tests {
    use super::*;
    use std::sync::Mutex;

    /// Env vars are process-global; serialize the partial-mode tests.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct PartialEnvGuard;
    impl PartialEnvGuard {
        fn on() -> Self {
            std::env::set_var("BITGRAIN_PARTIAL_DECODE", "1");
            PartialEnvGuard
        }
        fn off() -> Self {
            std::env::remove_var("BITGRAIN_PARTIAL_DECODE");
            PartialEnvGuard
        }
    }
    impl Drop for PartialEnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("BITGRAIN_PARTIAL_DECODE");
        }
    }

    fn make_rgb_image(w: usize, h: usize) -> Vec<u8> {
        let mut img = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                img[i] = ((x * 255) / w.max(1)) as u8;
                img[i + 1] = ((y * 255) / h.max(1)) as u8;
                img[i + 2] = (((x + y) * 255) / (w + h).max(1)) as u8;
            }
        }
        img
    }

    fn encode_rgb_vec(img: &[u8], w: usize, h: usize, q: u8) -> Vec<u8> {
        let mut out = vec![0u8; w * h * 3 * 2 + 4096];
        let mut pos = 0i32;
        encoder::encode_rgb(img, w, h, q, &mut out, &mut pos, None);
        out.truncate(pos as usize);
        out
    }

    fn encode_gray_vec(img: &[u8], w: usize, h: usize, q: u8) -> Vec<u8> {
        let mut out = vec![0u8; w * h * 2 + 4096];
        let mut pos = 0i32;
        encoder::encode_grayscale(img, w, h, q, &mut out, &mut pos);
        out.truncate(pos as usize);
        out
    }

    fn dims(buf: &[u8]) -> (usize, usize) {
        (
            u32::from_le_bytes(buf[3..7].try_into().unwrap()) as usize,
            u32::from_le_bytes(buf[7..11].try_into().unwrap()) as usize,
        )
    }

    fn decode_to_vec(buf: &[u8], chs: usize) -> Option<Vec<u8>> {
        let (w, h) = dims(buf);
        let mut out = vec![0u8; w * h * chs];
        let (mut ow, mut oh, mut oc) = (0u32, 0u32, 0u32);
        if decode(buf, &mut out, &mut ow, &mut oh, &mut oc, None) {
            Some(out)
        } else {
            None
        }
    }

    fn last_info() -> PartialDecodeInfo {
        LAST_DECODE_INFO.with(|i| i.borrow().clone())
    }

    /// First plane (luma) byte range: [16, 16+y_len).
    fn luma_plane_len(buf: &[u8]) -> usize {
        u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize
    }

    /// Residual corruption: scribble 0xFF over the middle of the luma
    /// bitstream. No all-ones codeword exists in the JPEG Huffman tables, so
    /// the decoder is guaranteed to fail there.
    fn corrupt_residual(buf: &[u8]) -> Vec<u8> {
        let mut c = buf.to_vec();
        let y_len = luma_plane_len(buf);
        let at = 16 + y_len / 2;
        for b in &mut c[at..at + 8] {
            *b = 0xFF;
        }
        c
    }

    fn corrupt_plane_header(buf: &[u8]) -> Vec<u8> {
        let mut c = buf.to_vec();
        c[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        c
    }

    fn truncate_mid_luma(buf: &[u8]) -> Vec<u8> {
        let y_len = luma_plane_len(buf);
        buf[..16 + y_len / 2].to_vec()
    }

    #[test]
    fn strict_mode_is_default_and_fails_on_corruption() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = PartialEnvGuard::off();
        let (w, h) = (48usize, 32usize);
        let good = encode_rgb_vec(&make_rgb_image(w, h), w, h, 85);
        for bad in [
            corrupt_residual(&good),
            corrupt_plane_header(&good),
            truncate_mid_luma(&good),
        ] {
            assert!(decode_to_vec(&bad, 3).is_none(), "strict mode must fail");
            assert!(!last_info().occurred);
            assert!(!last_info().reason.is_empty());
        }
    }

    #[test]
    fn partial_mode_recovers_from_residual_corruption() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = PartialEnvGuard::on();
        let (w, h) = (48usize, 32usize);
        let good = encode_rgb_vec(&make_rgb_image(w, h), w, h, 85);
        let bad = corrupt_residual(&good);
        let out = decode_to_vec(&bad, 3).expect("partial mode must produce output");
        let info = last_info();
        assert!(info.occurred);
        assert_eq!(info.total_blocks, 36); // 24 Y + 6 Cb + 6 Cr
        assert!(info.completed_blocks > 0 && info.completed_blocks < info.total_blocks);
        assert_eq!(info.reason, "corrupt residual bitstream");
        assert!(info.valid_samples > 0 && info.valid_samples < info.total_samples);
        // Undecoded region is gray-filled: bottom-right corner must be ~128.
        let px = &out[((h - 1) * w + (w - 1)) * 3..];
        assert!((px[0] as i16 - 128).abs() <= 2, "expected gray fill, got {px:?}");
    }

    #[test]
    fn partial_mode_recovers_from_plane_header_corruption() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = PartialEnvGuard::on();
        let (w, h) = (48usize, 32usize);
        let good = encode_rgb_vec(&make_rgb_image(w, h), w, h, 85);
        let bad = corrupt_plane_header(&good);
        assert!(decode_to_vec(&bad, 3).is_some());
        let info = last_info();
        assert!(info.occurred);
        assert_eq!(info.total_blocks, 36);
        assert!(info.completed_blocks < info.total_blocks);
        assert_eq!(info.reason, "corrupt plane length prefix");
    }

    #[test]
    fn partial_mode_recovers_from_truncation() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = PartialEnvGuard::on();
        let (w, h) = (48usize, 32usize);
        let good = encode_rgb_vec(&make_rgb_image(w, h), w, h, 85);
        let bad = truncate_mid_luma(&good);
        assert!(decode_to_vec(&bad, 3).is_some());
        let info = last_info();
        assert!(info.occurred);
        assert!(info.completed_blocks < info.total_blocks);
        assert_eq!(info.reason, "truncated stream");
    }

    #[test]
    fn partial_mode_normal_file_is_byte_identical() {
        let _lock = ENV_LOCK.lock().unwrap();
        let (w, h) = (48usize, 32usize);
        let good = encode_rgb_vec(&make_rgb_image(w, h), w, h, 85);

        let strict_out = {
            let _env = PartialEnvGuard::off();
            decode_to_vec(&good, 3).expect("strict decode of intact file")
        };
        let partial_out = {
            let _env = PartialEnvGuard::on();
            decode_to_vec(&good, 3).expect("partial-mode decode of intact file")
        };
        assert_eq!(strict_out, partial_out, "intact file must decode byte-identically");
        let info = last_info();
        assert!(!info.occurred);
        assert_eq!(info.completed_blocks, info.total_blocks);
        assert_eq!(info.valid_samples, info.total_samples);
        assert!(info.reason.is_empty());
    }

    #[test]
    fn partial_mode_rle_v1_truncation() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = PartialEnvGuard::on();
        let (w, h) = (32usize, 24usize);
        let img: Vec<u8> = (0..w * h).map(|i| ((i * 7) % 256) as u8).collect();
        let good = encode_gray_vec(&img, w, h, 85);
        assert_eq!(good[2], 1, "expected v1 grayscale stream");
        let bad = good[..good.len() * 2 / 3].to_vec();
        assert!(decode_to_vec(&bad, 1).is_some());
        let info = last_info();
        assert!(info.occurred);
        assert!(info.completed_blocks < info.total_blocks);
        assert_eq!(info.reason, "truncated or corrupt RLE block stream");
    }

    #[test]
    fn partial_mode_rle_v1_normal_file_byte_identical() {
        let _lock = ENV_LOCK.lock().unwrap();
        let (w, h) = (32usize, 24usize);
        let img: Vec<u8> = (0..w * h).map(|i| ((i * 7) % 256) as u8).collect();
        let good = encode_gray_vec(&img, w, h, 85);
        let strict_out = {
            let _env = PartialEnvGuard::off();
            decode_to_vec(&good, 1).expect("strict decode")
        };
        let partial_out = {
            let _env = PartialEnvGuard::on();
            decode_to_vec(&good, 1).expect("partial-mode decode")
        };
        assert_eq!(strict_out, partial_out);
        assert!(!last_info().occurred);
    }

    #[test]
    fn partial_status_ffi_getters() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = PartialEnvGuard::on();
        let (w, h) = (48usize, 32usize);
        let good = encode_rgb_vec(&make_rgb_image(w, h), w, h, 85);
        let bad = truncate_mid_luma(&good);
        assert!(decode_to_vec(&bad, 3).is_some());
        assert_eq!(bitgrain_partial_decode_occurred(), 1);
        assert_eq!(bitgrain_partial_total_blocks(), 36);
        assert!(bitgrain_partial_completed_blocks() < 36);
        assert!(bitgrain_partial_valid_samples() < bitgrain_partial_total_samples());
        let reason = unsafe { std::ffi::CStr::from_ptr(bitgrain_partial_failure_reason()) };
        assert_eq!(reason.to_str().unwrap(), "truncated stream");
    }
}
