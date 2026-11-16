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

use crate::block::Block;
use crate::colorspace;
use crate::dct;
use crate::encoder;
use crate::ffi::dequantize_block;
use crate::huffman;
use crate::zigzag::ZIGZAG;
use rayon::prelude::*;
use std::sync::atomic::{AtomicPtr, Ordering};
const BLOCK_TILE_SIZE: usize = 512;
const PARALLEL_DEQUANT_BLOCKS_THRESHOLD: usize = 384;
const PARALLEL_DEQUANT_PIXELS_THRESHOLD: usize = 262_144;
const PARALLEL_WRITE_BLOCKS_THRESHOLD: usize = 2048;
const PARALLEL_WRITE_PIXELS_THRESHOLD: usize = 786_432;

const HEADER_SIZE:     usize = 3 + 4 + 4 + 1;
const HEADER_SIZE_OLD: usize = 3 + 4 + 4;
const EOB_RUN: u8 = 0xFF;

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

fn decode_rle_to_blocks(buffer: &[u8], mut pos: usize, n: usize, blocks: &mut Vec<Block>) -> Option<usize> {
    blocks.clear();
    blocks.reserve(n);
    for _ in 0..n {
        let (block, new_pos) = decode_rle_one_block(buffer, pos)?;
        pos = new_pos;
        blocks.push(block);
    }
    Some(pos)
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

/// Decode one plane (RLE), dequant+IDCT in parallel, write to interleaved output.
fn decode_plane_rle(
    buffer: &[u8], pos: usize,
    w: usize, h: usize,
    quant: &[i16; 64],
    out: &mut [u8], stride: usize, offset: usize,
    blocks: &mut Vec<Block>,
) -> Option<usize> {
    let bw = (w + 7) / 8;
    let n  = bw * ((h + 7) / 8);

    let new_pos = decode_rle_to_blocks(buffer, pos, n, blocks)?;

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
    Some(new_pos)
}

// ---------------------------------------------------------------------------
// Huffman decode (v4/v5)
// ---------------------------------------------------------------------------

/// Decode one plane using Huffman, dequant+IDCT in parallel, write to flat plane buffer.
fn decode_plane_huffman(
    buffer: &[u8], pos: usize,
    w: usize, h: usize,
    quant: &[i16; 64],
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
    plane: &mut [u8],
) -> Option<usize> {
    let bw = (w + 7) / 8;
    let bh = (h + 7) / 8;
    let n  = bw * bh;

    let (mut blocks, new_pos) =
        huffman::decode_plane_with_profile(buffer, pos, n, is_chroma, use_chroma_ac, use_dc_delta)?;

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
    Some(new_pos)
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
// Read-only concurrent decode: shared immutable context + per-thread workspace
// ---------------------------------------------------------------------------
//
// Server workloads decode the same .bg stream from many threads at once. The
// decode state splits cleanly in two:
//
//   * `DecodeContext` — everything derivable from the immutable bitstream
//     (header, version, quality, quantization tables). Built once, behind an
//     `&` reference, and shared by every worker thread. It never mutates.
//   * `DecodeWorkspace` — the scratch buffers a single decode needs (planes,
//     RLE block staging). One per thread; never shared. The owner thread is
//     tracked so that the misuse of handing one workspace to two threads is
//     caught deterministically instead of corrupting memory.
//
// The original single-threaded `decode` keeps its exact behavior; it simply
// wraps a fresh, private workspace around the shared inner routine.

/// Read-only decoding context built once from a .bg stream and safely shared
/// (`&DecodeContext: Sync`) across threads that decode the same stream.
pub struct DecodeContext {
    version: u8,
    width: u32,
    height: u32,
    channels: u32,
    header_size: usize,
    quality: u8,
    luma_q: [i16; 64],
    chroma_q: [i16; 64],
}

impl DecodeContext {
    /// Parse the immutable stream header and precompute the quantization
    /// tables. Returns None for a malformed/unsupported header.
    pub fn from_stream(buffer: &[u8]) -> Option<DecodeContext> {
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
        let (header_size, quality) = if buffer.len() >= HEADER_SIZE {
            (HEADER_SIZE, buffer[11])
        } else {
            (HEADER_SIZE_OLD, 50u8)
        };
        let q = if quality == 0 { 50 } else { quality };
        let (luma_q, chroma_q) = quant_tables_for(version, q);
        let channels = match version {
            1 => 1,
            2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 => 3,
            _ => 4, // 3, 5, 7, 9, 11, 13, 15, 17, 19
        };
        Some(DecodeContext { version, width, height, channels, header_size, quality: q, luma_q, chroma_q })
    }

    pub fn width(&self) -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }
    pub fn channels(&self) -> u32 { self.channels }
    pub fn quality(&self) -> u8 { self.quality }
}

/// Per-thread scratch space for one decode. Not `Sync` on purpose: each worker
/// owns its own workspace. Concurrent use of one workspace is detected in
/// `begin()` and aborts the misbehaving call with a panic.
pub struct DecodeWorkspace {
    p0: Vec<u8>,
    p1: Vec<u8>,
    p2: Vec<u8>,
    p3: Vec<u8>,
    rle_blocks: Vec<Block>,
    // Identity of the thread currently inside a decode with this workspace.
    // A null pointer means "idle". The address of a stack-local anchor is used
    // as a stable thread identity (ThreadId::as_u64 is still unstable); it is
    // only ever compared, never dereferenced.
    owner: AtomicPtr<u8>,
}

impl DecodeWorkspace {
    pub fn new() -> Self {
        DecodeWorkspace {
            p0: Vec::new(), p1: Vec::new(), p2: Vec::new(), p3: Vec::new(),
            rle_blocks: Vec::new(),
            owner: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Claim the workspace for the calling thread. Panics if another thread is
    /// already inside a decode with it (the shared-workspace misuse).
    fn begin(&self) {
        let anchor = &0u8;
        let id = anchor as *const u8 as *mut u8;
        if self.owner.compare_exchange(
            std::ptr::null_mut(), id, Ordering::Acquire, Ordering::Acquire,
        ).is_err() {
            panic!("bitgrain: DecodeWorkspace used concurrently from two threads; each thread needs its own workspace");
        }
    }

    /// Release the claim. Runs even on unwind so a panicked decode does not
    /// poison the workspace for later correct use.
    fn end(&self) {
        self.owner.store(std::ptr::null_mut(), Ordering::Release);
    }

    /// Grow the three Y/Cb/Cr plane buffers and hand back disjoint mutable
    /// views (all are live at once during color conversion).
    fn planes3(&mut self, luma_len: usize, chroma_len: usize)
        -> (&mut [u8], &mut [u8], &mut [u8])
    {
        self.p0.resize(luma_len, 0);
        self.p1.resize(chroma_len, 0);
        self.p2.resize(chroma_len, 0);
        // Disjoint field borrows: each Vec is a distinct struct field.
        (&mut self.p0, &mut self.p1, &mut self.p2)
    }

    /// Same as `planes3`, plus a fourth (alpha) plane of luma size.
    fn planes4(&mut self, luma_len: usize, chroma_len: usize)
        -> (&mut [u8], &mut [u8], &mut [u8], &mut [u8])
    {
        self.p0.resize(luma_len, 0);
        self.p1.resize(chroma_len, 0);
        self.p2.resize(chroma_len, 0);
        self.p3.resize(luma_len, 0);
        (&mut self.p0, &mut self.p1, &mut self.p2, &mut self.p3)
    }
}

impl Default for DecodeWorkspace {
    fn default() -> Self { Self::new() }
}

/// Select the quantization profile matching a stream version (mirrors the
/// per-version tables the legacy entry point used to compute inline).
fn quant_tables_for(version: u8, q: u8) -> ([i16; 64], [i16; 64]) {
    match version {
        1 | 2 | 3 | 4 | 5 | 6 | 7 => (
            encoder::quant_table_for_quality(q),
            encoder::chroma_quant_table_for_quality(q),
        ),
        8 | 9 | 10 | 11 => (
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
        // 16 | 17 | 18 | 19
        _ => (
            encoder::quant_table_for_quality_perceptual_v4(q),
            encoder::chroma_quant_table_for_quality_perceptual_v4(q),
        ),
    }
}

/// Shared, read-only decode of a stream with a prebuilt context and a
/// thread-private workspace. Safe to call concurrently as long as every call
/// has its own `DecodeWorkspace`.
pub fn decode_with(
    ctx: &DecodeContext,
    buffer: &[u8],
    ws: &mut DecodeWorkspace,
    out_pixels: &mut [u8],
    out_width: &mut u32,
    out_height: &mut u32,
    out_channels: &mut u32,
    out_icc: Option<&mut Vec<u8>>,
) -> bool {
    ws.begin();
    // Run the decode under catch_unwind solely so end() always releases the
    // owner marker, even on unwind; the panic is re-raised unchanged. Mirrors
    // the AssertUnwindSafe pattern already used in ffi.rs and adds no unsafe.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decode_inner(ctx, buffer, ws, out_pixels, out_width, out_height, out_channels, out_icc)
    }));
    ws.end();
    match result {
        Ok(ok) => ok,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

fn decode_inner(
    ctx: &DecodeContext,
    buffer: &[u8],
    ws: &mut DecodeWorkspace,
    out_pixels: &mut [u8],
    out_width:  &mut u32,
    out_height: &mut u32,
    out_channels: &mut u32,
    out_icc: Option<&mut Vec<u8>>,
) -> bool {
    let version = ctx.version;
    let width = ctx.width;
    let height = ctx.height;
    let header_size = ctx.header_size;

    let w = width as usize;
    let h = height as usize;

    // ---- v1: grayscale RLE ----
    if version == 1 {
        if out_pixels.len() < w * h { return false; }
        *out_width = width; *out_height = height; *out_channels = 1;
        let quant = &ctx.luma_q;
        let pos = match decode_plane_rle(buffer, header_size, w, h, quant, out_pixels, 1, 0, &mut ws.rle_blocks) {
            Some(p) => p, None => return false,
        };
        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v2: RGB planar RLE ----
    if version == 2 {
        if out_pixels.len() < w * h * 3 { return false; }
        *out_width = width; *out_height = height; *out_channels = 3;
        let quant = &ctx.luma_q;
        let mut pos = header_size;
        for c in 0..3 {
            pos = match decode_plane_rle(buffer, pos, w, h, quant, out_pixels, 3, c, &mut ws.rle_blocks) {
                Some(p) => p, None => return false,
            };
        }
        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v3: RGBA planar RLE ----
    if version == 3 {
        if out_pixels.len() < w * h * 4 { return false; }
        *out_width = width; *out_height = height; *out_channels = 4;
        let quant = &ctx.luma_q;
        let mut pos = header_size;
        for c in 0..4 {
            pos = match decode_plane_rle(buffer, pos, w, h, quant, out_pixels, 4, c, &mut ws.rle_blocks) {
                Some(p) => p, None => return false,
            };
        }
        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v4: YCbCr 4:2:0 + Huffman → RGB ----
    if version == 4 {
        if out_pixels.len() < w * h * 3 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 3;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane) = ws.planes3(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, false, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  false, false, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  false, false, &mut cr_plane) { Some(p) => p, None => return false };

        colorspace::ycbcr420_to_rgb(&y_plane, &cb_plane, &cr_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v5: YCbCr 4:2:0 + A + Huffman → RGBA ----
    if version == 5 {
        if out_pixels.len() < w * h * 4 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 4;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane, mut a_plane) = ws.planes4(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, false, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  false, false, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  false, false, &mut cr_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, false, &mut a_plane)  { Some(p) => p, None => return false };

        colorspace::ycbcr420a_to_rgba(&y_plane, &cb_plane, &cr_plane, &a_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v6: YCbCr 4:2:0 + Huffman (chroma AC) → RGB ----
    if version == 6 {
        if out_pixels.len() < w * h * 3 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 3;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane) = ws.planes3(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, false, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  false, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  false, &mut cr_plane) { Some(p) => p, None => return false };

        colorspace::ycbcr420_to_rgb(&y_plane, &cb_plane, &cr_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v7: YCbCr 4:2:0 + A + Huffman (chroma AC) → RGBA ----
    if version == 7 {
        if out_pixels.len() < w * h * 4 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 4;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane, mut a_plane) = ws.planes4(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, false, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  false, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  false, &mut cr_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, false, &mut a_plane)  { Some(p) => p, None => return false };

        colorspace::ycbcr420a_to_rgba(&y_plane, &cb_plane, &cr_plane, &a_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v8: YCbCr 4:2:0 + Huffman (perceptual quant + chroma AC) → RGB ----
    if version == 8 {
        if out_pixels.len() < w * h * 3 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 3;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane) = ws.planes3(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, false, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  false, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  false, &mut cr_plane) { Some(p) => p, None => return false };

        colorspace::ycbcr420_to_rgb(&y_plane, &cb_plane, &cr_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v9: YCbCr 4:2:0 + A + Huffman (perceptual quant + chroma AC) → RGBA ----
    if version == 9 {
        if out_pixels.len() < w * h * 4 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 4;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane, mut a_plane) = ws.planes4(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, false, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  false, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  false, &mut cr_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, false, &mut a_plane)  { Some(p) => p, None => return false };

        colorspace::ycbcr420a_to_rgba(&y_plane, &cb_plane, &cr_plane, &a_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v10: YCbCr 4:2:0 + Huffman (perceptual quant + chroma AC + DC delta) → RGB ----
    if version == 10 {
        if out_pixels.len() < w * h * 3 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 3;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane) = ws.planes3(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cr_plane) { Some(p) => p, None => return false };

        colorspace::ycbcr420_to_rgb(&y_plane, &cb_plane, &cr_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v11: YCbCr 4:2:0 + A + Huffman (perceptual quant + chroma AC + DC delta) → RGBA ----
    if version == 11 {
        if out_pixels.len() < w * h * 4 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 4;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane, mut a_plane) = ws.planes4(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cr_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut a_plane)  { Some(p) => p, None => return false };

        colorspace::ycbcr420a_to_rgba(&y_plane, &cb_plane, &cr_plane, &a_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v12: YCbCr 4:2:0 + Huffman (strong perceptual quant + chroma AC + DC delta) → RGB ----
    if version == 12 {
        if out_pixels.len() < w * h * 3 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 3;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane) = ws.planes3(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cr_plane) { Some(p) => p, None => return false };

        colorspace::ycbcr420_to_rgb(&y_plane, &cb_plane, &cr_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v13: YCbCr 4:2:0 + A + Huffman (strong perceptual quant + chroma AC + DC delta) → RGBA ----
    if version == 13 {
        if out_pixels.len() < w * h * 4 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 4;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane, mut a_plane) = ws.planes4(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cr_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut a_plane)  { Some(p) => p, None => return false };

        colorspace::ycbcr420a_to_rgba(&y_plane, &cb_plane, &cr_plane, &a_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v14: YCbCr 4:2:0 + Huffman (aggressive perceptual quant + chroma AC + DC delta) → RGB ----
    if version == 14 {
        if out_pixels.len() < w * h * 3 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 3;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane) = ws.planes3(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cr_plane) { Some(p) => p, None => return false };

        colorspace::ycbcr420_to_rgb(&y_plane, &cb_plane, &cr_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v15: YCbCr 4:2:0 + A + Huffman (aggressive perceptual quant + chroma AC + DC delta) → RGBA ----
    if version == 15 {
        if out_pixels.len() < w * h * 4 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 4;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane, mut a_plane) = ws.planes4(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cr_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut a_plane)  { Some(p) => p, None => return false };

        colorspace::ycbcr420a_to_rgba(&y_plane, &cb_plane, &cr_plane, &a_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v16: YCbCr 4:2:0 + Huffman (very aggressive perceptual quant + chroma AC + DC delta) → RGB ----
    if version == 16 {
        if out_pixels.len() < w * h * 3 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 3;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane) = ws.planes3(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cr_plane) { Some(p) => p, None => return false };

        colorspace::ycbcr420_to_rgb(&y_plane, &cb_plane, &cr_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v17: YCbCr 4:2:0 + A + Huffman (very aggressive perceptual quant + chroma AC + DC delta) → RGBA ----
    if version == 17 {
        if out_pixels.len() < w * h * 4 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 4;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane, mut a_plane) = ws.planes4(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cr_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut a_plane)  { Some(p) => p, None => return false };

        colorspace::ycbcr420a_to_rgba(&y_plane, &cb_plane, &cr_plane, &a_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v18: YCbCr 4:2:0 + Huffman (ultra perceptual + AC sparsify + chroma AC + DC delta) → RGB ----
    if version == 18 {
        if out_pixels.len() < w * h * 3 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 3;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane) = ws.planes3(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cr_plane) { Some(p) => p, None => return false };

        colorspace::ycbcr420_to_rgb(&y_plane, &cb_plane, &cr_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    // ---- v19: YCbCr 4:2:0 + A + Huffman (ultra perceptual + AC sparsify + chroma AC + DC delta) → RGBA ----
    if version == 19 {
        if out_pixels.len() < w * h * 4 {
            return false;
        }
        *out_width = width; *out_height = height; *out_channels = 4;

        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q   = &ctx.luma_q;
        let chroma_q = &ctx.chroma_q;

        let (mut y_plane, mut cb_plane, mut cr_plane, mut a_plane) = ws.planes4(w * h, cw * ch);

        let mut pos = header_size;
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut y_plane)  { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cb_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, cw, ch, &chroma_q, true,  true,  true, &mut cr_plane) { Some(p) => p, None => return false };
        pos = match decode_plane_huffman(buffer, pos, w,  h,  &luma_q,   false, false, true, &mut a_plane)  { Some(p) => p, None => return false };

        colorspace::ycbcr420a_to_rgba(&y_plane, &cb_plane, &cr_plane, &a_plane, w, h, out_pixels);

        if let Some(v) = out_icc {
            if let Some((icc, _)) = parse_icc_trailer(buffer, pos) { *v = icc; }
        }
        return true;
    }

    false
}

/// Original single-threaded entry point: parse the stream, build the shared
/// read-only context, and decode with a fresh private workspace. Behavior and
/// output bytes are unchanged from the pre-context implementation.
pub fn decode(
    buffer: &[u8],
    out_pixels: &mut [u8],
    out_width:  &mut u32,
    out_height: &mut u32,
    out_channels: &mut u32,
    out_icc: Option<&mut Vec<u8>>,
) -> bool {
    let Some(ctx) = DecodeContext::from_stream(buffer) else {
        return false;
    };
    let mut ws = DecodeWorkspace::new();
    decode_with(&ctx, buffer, &mut ws, out_pixels, out_width, out_height, out_channels, out_icc)
}

pub fn decode_grayscale(
    buffer: &[u8], out_pixels: &mut [u8],
    out_width: &mut u32, out_height: &mut u32,
) -> bool {
    let mut ch = 0u32;
    let ok = decode(buffer, out_pixels, out_width, out_height, &mut ch, None);
    ok && ch == 1
}

// ===========================================================================
// Tests for the shared read-only context / per-thread workspace decode path.
// Kept in this file so the feature and its tests ship in the one changed
// library file. The pre-existing suite lives in crate::tests.
// ===========================================================================
#[cfg(test)]
mod shared_context_tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::time::Instant;

    /// Deterministic pseudo-random pixels so every version gets real content.
    fn synth_pixels(w: usize, h: usize, ch: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * ch);
        for y in 0..h {
            for x in 0..w {
                for c in 0..ch {
                    let val = ((x * 37 + y * 17 + c * 91 + (x ^ y) * 7) & 0xff) as u8;
                    v.push(val);
                }
            }
        }
        v
    }

    /// Encode one of every family the codec actually produces:
    /// v1 grayscale RLE, v2 RGB RLE, v3 RGBA RLE, v18 YCbCr RGB, v19 YCbCrA RGBA.
    fn sample_streams() -> Vec<Vec<u8>> {
        let (w, h) = (96usize, 81usize); // odd height exercises chroma edge blocks
        let q = 85u8;
        let mut streams = Vec::new();

        let gray = synth_pixels(w, h, 1);
        let mut out = vec![0u8; 1 << 18];
        let mut pos = 0i32;
        crate::encoder::encode_grayscale(&gray, w, h, q, &mut out, &mut pos);
        out.truncate(pos as usize);
        streams.push(out);

        let rgb = synth_pixels(w, h, 3);
        let mut out = vec![0u8; 1 << 18];
        let mut pos = 0i32;
        crate::encoder::encode_rgb_rle(&rgb, w, h, q, &mut out, &mut pos, None);
        out.truncate(pos as usize);
        streams.push(out);

        let rgba = synth_pixels(w, h, 4);
        let mut out = vec![0u8; 1 << 18];
        let mut pos = 0i32;
        crate::encoder::encode_rgba_rle(&rgba, w, h, q, &mut out, &mut pos, None);
        out.truncate(pos as usize);
        streams.push(out);

        let mut out = vec![0u8; 1 << 18];
        let mut pos = 0i32;
        crate::encoder::encode_rgb(&rgb, w, h, q, &mut out, &mut pos, None);
        out.truncate(pos as usize);
        streams.push(out);

        let mut out = vec![0u8; 1 << 18];
        let mut pos = 0i32;
        crate::encoder::encode_rgba(&rgba, w, h, q, &mut out, &mut pos, None);
        out.truncate(pos as usize);
        streams.push(out);

        streams
    }

    /// Decode into an exactly-sized output buffer, as the FFI contract requires
    /// (`out_capacity == width*height*channels`). An oversized slice would let
    /// the color-conversion row loop index past the decoded planes.
    fn legacy_decode(buf: &[u8]) -> (Vec<u8>, u32, u32, u32) {
        let ctx = DecodeContext::from_stream(buf).expect("header");
        let (w, h, ch) = (ctx.width(), ctx.height(), ctx.channels());
        let mut px = vec![0u8; (w as usize) * (h as usize) * (ch as usize)];
        let (mut ow, mut oh, mut och) = (0u32, 0u32, 0u32);
        let ok = decode(buf, &mut px, &mut ow, &mut oh, &mut och, None);
        assert!(ok);
        assert_eq!((ow, oh, och), (w, h, ch));
        (px, w, h, ch)
    }

    fn shared_decode(ctx: &DecodeContext, buf: &[u8], ws: &mut DecodeWorkspace)
        -> (Vec<u8>, u32, u32, u32)
    {
        let (w, h, ch) = (ctx.width(), ctx.height(), ctx.channels());
        let mut px = vec![0u8; (w as usize) * (h as usize) * (ch as usize)];
        let (mut ow, mut oh, mut och) = (0u32, 0u32, 0u32);
        let ok = decode_with(ctx, buf, ws, &mut px, &mut ow, &mut oh, &mut och, None);
        assert!(ok);
        assert_eq!((ow, oh, och), (w, h, ch));
        (px, w, h, ch)
    }

    #[test]
    fn shared_path_matches_legacy_pixel_for_pixel_every_version() {
        for buf in sample_streams() {
            let (ref_px, rw, rh, rch) = legacy_decode(&buf);
            let ctx = DecodeContext::from_stream(&buf).expect("header must parse");
            assert_eq!(ctx.version, buf[2]);
            assert_eq!(ctx.width(), rw);
            assert_eq!(ctx.height(), rh);
            assert_eq!(ctx.channels(), rch);
            // First decode allocates; a second reuses the same workspace and
            // must produce identical bytes (buffer reuse path).
            for _ in 0..2 {
                let mut ws = DecodeWorkspace::new();
                let (px, w, h, ch) = shared_decode(&ctx, &buf, &mut ws);
                assert_eq!((w, h, ch), (rw, rh, rch));
                assert_eq!(px, ref_px, "version {} mismatch", buf[2]);
            }
        }
    }

    #[test]
    fn four_threads_sharing_one_context_produce_identical_pixels() {
        const THREADS: usize = 4;
        for buf in sample_streams() {
            let (ref_px, rw, rh, rch) = legacy_decode(&buf);
            let ref_px = Arc::new(ref_px);
            let ctx = Arc::new(DecodeContext::from_stream(&buf).expect("header"));
            let stream = Arc::new(buf);
            let barrier = Arc::new(Barrier::new(THREADS));
            let handles: Vec<_> = (0..THREADS).map(|t| {
                let ctx = Arc::clone(&ctx);
                let stream = Arc::clone(&stream);
                let barrier = Arc::clone(&barrier);
                let ref_px = Arc::clone(&ref_px);
                std::thread::spawn(move || {
                    let mut ws = DecodeWorkspace::new(); // private per thread
                    barrier.wait(); // maximize overlap
                    let (px, w, h, ch) = shared_decode(&ctx, &stream, &mut ws);
                    assert_eq!((w, h, ch), (rw, rh, rch), "thread {t} dims");
                    assert_eq!(px, *ref_px, "thread {t} pixels differ, version {}", stream[2]);
                })
            }).collect();
            for hd in handles { hd.join().expect("worker thread panicked"); }
        }
    }

    /// Test-only wrapper that deliberately opts a workspace into shared
    /// access across threads, modeling a buggy workspace pool. The library
    /// stays safe regardless: the ownership guard in `begin()` — the same
    /// claim `decode_with` performs on every call — rejects the second caller.
    struct SharedWs(DecodeWorkspace);
    unsafe impl Sync for SharedWs {}

    #[test]
    fn sharing_one_workspace_between_threads_is_detected() {
        let shared = Arc::new(SharedWs(DecodeWorkspace::new()));
        // Three barriers pin the interleaving so the rejection is deterministic:
        // both start together, A claims, B attempts while A still holds.
        let start = Arc::new(Barrier::new(2));
        let claimed = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));

        let (s_a, s_b) = (Arc::clone(&shared), Arc::clone(&shared));
        let (st_a, st_b) = (Arc::clone(&start), Arc::clone(&start));
        let (cl_a, cl_b) = (Arc::clone(&claimed), Arc::clone(&claimed));
        let (rl_a, rl_b) = (Arc::clone(&release), Arc::clone(&release));
        let owner = std::thread::spawn(move || {
            st_a.wait();
            s_a.0.begin();        // first claim succeeds
            cl_a.wait();          // signal: ownership held
            rl_a.wait();          // keep holding until B has attempted
            s_a.0.end();
        });
        let intruder = std::thread::spawn(move || {
            st_b.wait();
            cl_b.wait();          // A's begin() has definitely completed
            let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s_b.0.begin()));
            rl_b.wait();
            attempt
        });

        owner.join().expect("owner thread must not panic");
        let payload = intruder.join().expect("join")
            .err()
            .expect("the second concurrent claim must be rejected");
        let msg = payload.downcast_ref::<String>().map(|s| s.as_str())
            .or_else(|| payload.downcast_ref::<&'static str>().copied())
            .unwrap_or("");
        assert!(msg.contains("DecodeWorkspace used concurrently"),
                "unexpected panic message: {msg}");
    }

    #[test]
    fn workspace_is_reusable_after_the_owner_marker_clears() {
        let (w, h) = (64usize, 48usize);
        let rgb = synth_pixels(w, h, 3);
        let mut out = vec![0u8; 1 << 20];
        let mut pos = 0i32;
        crate::encoder::encode_rgb(&rgb, w, h, 85, &mut out, &mut pos, None);
        out.truncate(pos as usize);
        let (ref_px, _, _, _) = legacy_decode(&out);
        let ctx = DecodeContext::from_stream(&out).unwrap();

        let mut ws = DecodeWorkspace::new();
        // A begin/end pair returns the marker to idle — exactly what
        // decode_with guarantees via catch_unwind when a decode unwinds.
        ws.begin();
        ws.end();
        let (px, _, _, _) = shared_decode(&ctx, &out, &mut ws);
        assert_eq!(px, ref_px);
    }

    #[test]
    fn malformed_headers_are_rejected_without_panicking() {
        assert!(DecodeContext::from_stream(&[]).is_none());
        assert!(DecodeContext::from_stream(b"XX").is_none());
        let mut bad = b"BG\x02\x10\x00\x00\x00\x10\x00\x00\x00Q".to_vec();
        bad[2] = 0; // version 0
        assert!(DecodeContext::from_stream(&bad).is_none());
        bad[2] = 20; // unsupported
        assert!(DecodeContext::from_stream(&bad).is_none());
        bad[2] = 2;
        bad[3] = 0; // zero width
        assert!(DecodeContext::from_stream(&bad).is_none());
    }

    /// The shared context must select exactly the quantization profile each
    /// legacy branch computed inline, for every decode-reachable version
    /// (including the v4..v17 profiles the current encoder no longer emits).
    #[test]
    fn context_quant_tables_match_each_legacy_version() {
        let q = 77u8;
        let want = |v: u8| -> ([i16; 64], [i16; 64]) {
            match v {
                1..=7 => (
                    crate::encoder::quant_table_for_quality(q),
                    crate::encoder::chroma_quant_table_for_quality(q),
                ),
                8..=11 => (
                    crate::encoder::quant_table_for_quality_perceptual(q),
                    crate::encoder::chroma_quant_table_for_quality_perceptual(q),
                ),
                12..=13 => (
                    crate::encoder::quant_table_for_quality_perceptual_v2(q),
                    crate::encoder::chroma_quant_table_for_quality_perceptual_v2(q),
                ),
                14..=15 => (
                    crate::encoder::quant_table_for_quality_perceptual_v3(q),
                    crate::encoder::chroma_quant_table_for_quality_perceptual_v3(q),
                ),
                _ => (
                    crate::encoder::quant_table_for_quality_perceptual_v4(q),
                    crate::encoder::chroma_quant_table_for_quality_perceptual_v4(q),
                ),
            }
        };
        for v in 1..=19u8 {
            let (l, c) = quant_tables_for(v, q);
            let (wl, wc) = want(v);
            assert_eq!(l, wl, "luma table mismatch for v{v}");
            assert_eq!(c, wc, "chroma table mismatch for v{v}");
        }
    }

    #[test]
    fn context_is_send_sync_and_workspace_is_send_only() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<DecodeContext>();
        assert_sync::<DecodeContext>(); // shared read-only across threads
        assert_send::<DecodeWorkspace>(); // moves between threads, never shared
    }

    /// Resident-set size of the current process in bytes (Linux /proc), or 0.
    fn rss_bytes() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmRSS:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|n| n.parse::<u64>().ok())
            })
            .map(|kb| kb * 1024)
            .unwrap_or(0)
    }

    /// End-to-end measurement for the delivery report: 4 threads on one shared
    /// context, with the legacy single-thread path as the timing baseline and
    /// workspace RSS growth as the per-thread memory increment. Runs in one
    /// shot; prints a compact table (`cargo test -- --nocapture`).
    #[test]
    fn four_thread_consistency_timing_and_memory_report() {
        const THREADS: usize = 4;
        const ITERS: usize = 8;
        let (w, h) = (1024usize, 1024usize);
        let rgb = synth_pixels(w, h, 3);
        let mut out = vec![0u8; 1 << 22];
        let mut pos = 0i32;
        crate::encoder::encode_rgb(&rgb, w, h, 85, &mut out, &mut pos, None);
        out.truncate(pos as usize);
        let stream = Arc::new(out);
        let (ref_px, rw, rh, rch) = legacy_decode(&stream);
        let ref_px = Arc::new(ref_px);

        // Baseline: ITERS*THREADS sequential legacy decodes in this thread.
        let t0 = Instant::now();
        let mut last_len = 0usize;
        for _ in 0..(ITERS * THREADS) {
            let (px, _, _, _) = legacy_decode(&stream);
            last_len = px.len();
        }
        let seq_total = t0.elapsed();

        // Warm the shared context path once (quant tables, huffman OnceLocks).
        let warm_ctx = DecodeContext::from_stream(&stream).unwrap();
        let mut warm_ws = DecodeWorkspace::new();
        let (warm_px, _, _, _) = shared_decode(&warm_ctx, &stream, &mut warm_ws);
        assert_eq!(warm_px, *ref_px);

        // Memory: RSS before vs after four fully-grown workspaces.
        let rss_before = rss_bytes();
        let ctx = Arc::new(DecodeContext::from_stream(&stream).unwrap());
        let mut grown: Vec<DecodeWorkspace> = (0..THREADS).map(|_| DecodeWorkspace::new()).collect();
        for ws in grown.iter_mut() {
            let (px, _, _, _) = shared_decode(&ctx, &stream, ws);
            assert_eq!(px, *ref_px);
        }
        let rss_after_grown = rss_bytes();

        // Four threads, ITERS rounds each, all sharing one read-only context.
        let barrier = Arc::new(Barrier::new(THREADS));
        let mut worker_times = Vec::with_capacity(THREADS);
        let mut handles = Vec::new();
        for _ in 0..THREADS {
            let ctx = Arc::clone(&ctx);
            let stream = Arc::clone(&stream);
            let barrier = Arc::clone(&barrier);
            let ref_px = Arc::clone(&ref_px);
            handles.push(std::thread::spawn(move || {
                let mut ws = DecodeWorkspace::new();
                let mut local = Vec::with_capacity(ITERS);
                barrier.wait();
                for _ in 0..ITERS {
                    let start = Instant::now();
                    let (px, w2, h2, ch2) = shared_decode(&ctx, &stream, &mut ws);
                    local.push(start.elapsed());
                    assert_eq!((w2, h2, ch2), (rw, rh, rch));
                    assert_eq!(px, *ref_px); // pixel-identical, every iteration
                }
                local
            }));
        }
        let wall_start = Instant::now();
        for hd in handles.drain(..) {
            worker_times.push(hd.join().unwrap());
        }
        let wall = wall_start.elapsed();

        let per_thread_total: Vec<std::time::Duration> =
            (0..THREADS).map(|t| worker_times[t].iter().sum()).collect();
        let per_thread_avg_ms = per_thread_total.iter().map(|d| d.as_secs_f64() * 1000.0 / ITERS as f64).collect::<Vec<_>>();
        let seq_avg_ms = seq_total.as_secs_f64() * 1000.0 / (ITERS * THREADS) as f64;
        let rss_delta = rss_after_grown.saturating_sub(rss_before);

        println!("\n==== bitgrain read-only concurrent decode report ====");
        println!("stream        : v{} {}x{} {} bytes, {} output pixels/frame",
                 stream[2], rw, rh, stream.len(), last_len);
        println!("threads       : {THREADS} sharing one DecodeContext, private DecodeWorkspace each");
        println!("iterations    : {ITERS} per thread ({} frames total)", ITERS * THREADS);
        println!("legacy seq    : {:.3} ms/decode, {:.3} ms for {} decodes",
                 seq_avg_ms, seq_total.as_secs_f64() * 1000.0, ITERS * THREADS);
        for (t, avg) in per_thread_avg_ms.iter().enumerate() {
            println!("thread {t}      : {avg:.3} ms/decode (avg of {ITERS})");
        }
        println!("4-thread wall : {:.3} ms for {} concurrent frames ({:.2}x vs sequential {:.3} ms)",
                 wall.as_secs_f64() * 1000.0, ITERS * THREADS,
                 seq_total.as_secs_f64() / wall.as_secs_f64(), seq_total.as_secs_f64() * 1000.0);
        println!("memory        : RSS delta for 4 grown workspaces = {} KiB ({:.1} KiB/thread); \
                 shared context allocated once", rss_delta / 1024, rss_delta as f64 / 1024.0 / THREADS as f64);
        println!("consistency   : all {} frames byte-identical to legacy single-thread decode",
                 ITERS * THREADS);
        println!("======================================================");
    }
}

/// Test-only definitions of the C SIMD entry points (c/quant.c). The cargo
/// test binary links the crate without the C objects; dct.rs already swaps in
/// pure-Rust DCT under `cfg(test)`, and these mirror the scalar semantics of
/// `quantize_block`/`dequantize_block` (round-to-nearest-even division and
/// saturating i16 multiply). They are compiled into tests only and never
/// reach the release staticlib/shared library, whose C implementations win.
#[cfg(test)]
mod c_test_shims {
    #[no_mangle]
    pub extern "C" fn quantize_block(block: *mut i16, table: *const i16) {
        let b = unsafe { core::slice::from_raw_parts_mut(block, 64) };
        let t = unsafe { core::slice::from_raw_parts(table, 64) };
        for i in 0..64 {
            let v = (b[i] as f32 / t[i] as f32).round_ties_even() as i32;
            b[i] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }

    #[no_mangle]
    pub extern "C" fn dequantize_block(block: *mut i16, table: *const i16) {
        let b = unsafe { core::slice::from_raw_parts_mut(block, 64) };
        let t = unsafe { core::slice::from_raw_parts(table, 64) };
        for i in 0..64 {
            let v = b[i] as i32 * t[i] as i32;
            b[i] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }
}
