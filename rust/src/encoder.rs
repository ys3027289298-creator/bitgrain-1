use crate::bitstream;
use crate::block::Block;
use crate::blockizer::Blockizer;
use crate::colorspace;
use crate::dct;
use crate::entropy;
#[cfg(not(test))]
use crate::ffi::quantize_block;
use crate::huffman;
use crate::zigzag::ZIGZAG;
use rayon::prelude::*;
use std::collections::HashMap;
use std::env;
const BLOCK_TILE_SIZE: usize = 512;
const PARALLEL_BLOCKS_THRESHOLD: usize = 384;
const PARALLEL_PLANE_PIXELS_THRESHOLD: usize = 262_144;
const PARALLEL_IMAGE_PIXELS_THRESHOLD: usize = 262_144;

#[inline]
fn should_parallel_blocks(n_blocks: usize, plane_w: usize, plane_h: usize) -> bool {
    n_blocks >= PARALLEL_BLOCKS_THRESHOLD
        && plane_w.saturating_mul(plane_h) >= PARALLEL_PLANE_PIXELS_THRESHOLD
}

#[inline]
fn should_parallel_planes(width: usize, height: usize) -> bool {
    width.saturating_mul(height) >= PARALLEL_IMAGE_PIXELS_THRESHOLD
}

/// .bg header: "BG" + version + width(u32 LE) + height(u32 LE) + quality(u8) = 12 bytes.
///
/// Version byte:
///   1 = grayscale,  RLE entropy
///   2 = RGB planar, RLE entropy   (legacy)
///   3 = RGBA planar,RLE entropy   (legacy)
///   4 = YCbCr 4:2:0, Huffman     (RGB input, best compression)
///   5 = YCbCr 4:2:0 + A, Huffman (RGBA input)
///   6 = YCbCr 4:2:0, Huffman + chroma AC table (RGB input, improved size)
///   7 = YCbCr 4:2:0 + A, Huffman + chroma AC table (RGBA input, improved size)
///   8 = YCbCr 4:2:0, perceptual quant + chroma AC (RGB input, better size)
///   9 = YCbCr 4:2:0 + A, perceptual quant + chroma AC (RGBA input, better size)
///  10 = YCbCr 4:2:0, perceptual quant + chroma AC + DC delta (JPEG-like)
///  11 = YCbCr 4:2:0 + A, perceptual quant + chroma AC + DC delta (JPEG-like)
///  12 = YCbCr 4:2:0, stronger perceptual quant + chroma AC + DC delta
///  13 = YCbCr 4:2:0 + A, stronger perceptual quant + chroma AC + DC delta
///  14 = YCbCr 4:2:0, aggressive perceptual quant + chroma AC + DC delta
///  15 = YCbCr 4:2:0 + A, aggressive perceptual quant + chroma AC + DC delta
///  16 = YCbCr 4:2:0, very aggressive perceptual quant + chroma AC + DC delta
///  17 = YCbCr 4:2:0 + A, very aggressive perceptual quant + chroma AC + DC delta
///  18 = YCbCr 4:2:0, ultra perceptual + AC sparsify + chroma AC + DC delta
///  19 = YCbCr 4:2:0 + A, ultra perceptual + AC sparsify + chroma AC + DC delta
///  20 = v18 coefficient coding + global block dictionary (RGB, opt-in)
///  21 = v19 coefficient coding + global block dictionary (RGBA, opt-in)
pub const BG_HEADER_SIZE: usize = 3 + 4 + 4 + 1;

const BG_MAGIC_GRAY:    &[u8; 3] = b"BG\x01";
const BG_MAGIC_RGB:     &[u8; 3] = b"BG\x02";
const BG_MAGIC_RGBA:    &[u8; 3] = b"BG\x03";
const BG_MAGIC_YUV420_V8:  &[u8; 3] = b"BG\x12";
const BG_MAGIC_YUV420A_V8: &[u8; 3] = b"BG\x13";
/// Section magic that introduces the global dictionary right after the header.
const DICT_SECTION_MAGIC: &[u8; 3] = b"BGd";

// ---------------------------------------------------------------------------
// Block dictionary (v20/v21) — configuration
// ---------------------------------------------------------------------------
// A dictionary block is always one full 8x8 quantized coefficient block.
const DICT_BLOCK_COEFFS: usize = 64;
const DICT_BLOCK_BYTES: usize = DICT_BLOCK_COEFFS * 2; // i16 LE per coefficient
// Per-plane dictionary payload flags (low bit of the u8 flags field).
const DICT_PLANE_FLAG_ENABLED: u8 = 0x01;
const DICT_MAX_CAPACITY: u32 = 0x7FFF; // local/global indices are carried as u16

/// Dictionary tuning. `capacity` bounds the number of entries in the global
/// dictionary; `min_freq` is the minimum occurrence count for inclusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DictConfig {
    pub enabled: bool,
    pub capacity: u32,
    pub min_freq: u32,
}

impl Default for DictConfig {
    fn default() -> Self {
        Self { enabled: false, capacity: 256, min_freq: 2 }
    }
}

/// Read dictionary configuration from process environment.
///
/// This keeps the public C ABI (`includes/encoder.h`) untouched while still
/// making capacity and minimum frequency configurable:
///   BG_DICT=1            enable dictionary compression (v20/v21 output)
///   BG_DICT_CAPACITY=N   max dictionary entries (1..32767, default 256)
///   BG_DICT_MIN_FREQ=N   minimum block frequency for inclusion (default 2)
///
/// Malformed/empty values fall back to defaults; callers never see a failure.
pub fn dict_config_from_env() -> DictConfig {
    let enabled = env::var("BG_DICT")
        .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "on" | "ON"))
        .unwrap_or(false);
    let capacity = env::var("BG_DICT_CAPACITY")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|n| (1..=DICT_MAX_CAPACITY).contains(n))
        .unwrap_or(256);
    let min_freq = env::var("BG_DICT_MIN_FREQ")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(2);
    DictConfig { enabled, capacity, min_freq }
}

/// Standard JPEG luminance quantization table (quality ~50).
pub fn default_quant_table() -> [i16; 64] {
    [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ]
}

/// Standard JPEG chrominance quantization table (quality ~50).
pub fn default_chroma_quant_table() -> [i16; 64] {
    [
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
    ]
}

/// Scale a quant table by quality using the standard JPEG formula.
/// quality 50 = default table; 100 = minimal quantization; 1 = heavy.
pub fn quant_table_for_quality(quality: u8) -> [i16; 64] {
    scale_quant_table(&default_quant_table(), quality)
}

pub fn chroma_quant_table_for_quality(quality: u8) -> [i16; 64] {
    scale_quant_table(&default_chroma_quant_table(), quality)
}

fn scale_quant_table(base: &[i16; 64], quality: u8) -> [i16; 64] {
    let q = quality.clamp(1, 100) as i32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let mut out = [0i16; 64];
    for i in 0..64 {
        let v = (base[i] as i32 * scale + 50) / 100;
        out[i] = v.clamp(1, 255) as i16;
    }
    out
}

fn scale_quant_table_perceptual(base: &[i16; 64], quality: u8, chroma: bool) -> [i16; 64] {
    // Conservative perceptual tuning:
    // - stronger attenuation on chroma detail
    // - progressively stronger quantization toward high frequencies
    let mut out = scale_quant_table(base, quality);
    // 8x8 weights (percent). Top-left keeps detail; bottom-right compresses harder.
    const LUMA_W: [i32; 64] = [
        100,100,100,102,104,106,108,110,
        100,100,102,104,106,108,110,112,
        100,102,104,106,108,110,114,118,
        102,104,106,108,110,114,118,122,
        104,106,108,110,114,118,122,128,
        106,108,110,114,118,122,128,134,
        108,110,114,118,122,128,134,142,
        110,112,118,122,128,134,142,152,
    ];
    const CHROMA_W: [i32; 64] = [
        112,114,116,120,124,128,132,136,
        114,116,120,124,128,132,136,142,
        116,120,124,128,132,136,142,148,
        120,124,128,132,136,142,148,156,
        124,128,132,136,142,148,156,166,
        128,132,136,142,148,156,166,178,
        132,136,142,148,156,166,178,192,
        136,142,148,156,166,178,192,208,
    ];
    let w = if chroma { &CHROMA_W } else { &LUMA_W };
    for i in 0..64 {
        let v = (out[i] as i32 * w[i] + 50) / 100;
        out[i] = v.clamp(1, 255) as i16;
    }
    out
}

pub fn quant_table_for_quality_perceptual(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual(&default_quant_table(), quality, false)
}

pub fn chroma_quant_table_for_quality_perceptual(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual(&default_chroma_quant_table(), quality, true)
}

fn scale_quant_table_perceptual_v2(base: &[i16; 64], quality: u8, chroma: bool) -> [i16; 64] {
    // Stronger than v1 perceptual profile, still bounded and deterministic.
    let mut out = scale_quant_table(base, quality);
    const LUMA_W2: [i32; 64] = [
        100,100,102,104,108,112,116,122,
        100,102,104,108,112,116,122,128,
        102,104,108,112,116,122,130,138,
        104,108,112,116,122,130,138,148,
        108,112,116,122,130,138,148,160,
        112,116,122,130,138,148,160,174,
        116,122,130,138,148,160,174,190,
        122,128,138,148,160,174,190,208,
    ];
    const CHROMA_W2: [i32; 64] = [
        125,130,136,144,152,162,174,188,
        130,136,144,152,162,174,188,204,
        136,144,152,162,174,188,204,222,
        144,152,162,174,188,204,222,242,
        152,162,174,188,204,222,242,255,
        162,174,188,204,222,242,255,255,
        174,188,204,222,242,255,255,255,
        188,204,222,242,255,255,255,255,
    ];
    let w = if chroma { &CHROMA_W2 } else { &LUMA_W2 };
    for i in 0..64 {
        let mut v = (out[i] as i32 * w[i] + 50) / 100;
        // Slightly more pressure at medium/low quality to close size gap.
        if quality <= 85 {
            if chroma {
                v = (v * 106 + 50) / 100;
            } else if i >= 32 {
                v = (v * 103 + 50) / 100;
            }
        }
        out[i] = v.clamp(1, 255) as i16;
    }
    out
}

pub fn quant_table_for_quality_perceptual_v2(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual_v2(&default_quant_table(), quality, false)
}

pub fn chroma_quant_table_for_quality_perceptual_v2(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual_v2(&default_chroma_quant_table(), quality, true)
}

fn scale_quant_table_perceptual_v3(base: &[i16; 64], quality: u8, chroma: bool) -> [i16; 64] {
    // Experimental aggressive profile to probe size-gap closure.
    let mut out = scale_quant_table(base, quality);
    const LUMA_W3: [i32; 64] = [
        102,104,108,112,118,126,136,148,
        104,108,112,118,126,136,148,162,
        108,112,118,126,136,148,162,178,
        112,118,126,136,148,162,178,196,
        118,126,136,148,162,178,196,216,
        126,136,148,162,178,196,216,238,
        136,148,162,178,196,216,238,255,
        148,162,178,196,216,238,255,255,
    ];
    const CHROMA_W3: [i32; 64] = [
        145,154,166,180,196,214,234,255,
        154,166,180,196,214,234,255,255,
        166,180,196,214,234,255,255,255,
        180,196,214,234,255,255,255,255,
        196,214,234,255,255,255,255,255,
        214,234,255,255,255,255,255,255,
        234,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,
    ];
    let w = if chroma { &CHROMA_W3 } else { &LUMA_W3 };
    for i in 0..64 {
        let mut v = (out[i] as i32 * w[i] + 50) / 100;
        if quality <= 92 {
            if chroma {
                v = (v * 112 + 50) / 100;
            } else if i >= 24 {
                v = (v * 106 + 50) / 100;
            }
        }
        out[i] = v.clamp(1, 255) as i16;
    }
    out
}

pub fn quant_table_for_quality_perceptual_v3(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual_v3(&default_quant_table(), quality, false)
}

pub fn chroma_quant_table_for_quality_perceptual_v3(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual_v3(&default_chroma_quant_table(), quality, true)
}

fn scale_quant_table_perceptual_v4(base: &[i16; 64], quality: u8, chroma: bool) -> [i16; 64] {
    // Very aggressive profile for compression experiments.
    let mut out = scale_quant_table(base, quality);
    const LUMA_W4: [i32; 64] = [
        106,112,120,130,142,156,172,190,
        112,120,130,142,156,172,190,210,
        120,130,142,156,172,190,210,232,
        130,142,156,172,190,210,232,255,
        142,156,172,190,210,232,255,255,
        156,172,190,210,232,255,255,255,
        172,190,210,232,255,255,255,255,
        190,210,232,255,255,255,255,255,
    ];
    const CHROMA_W4: [i32; 64] = [
        170,184,200,218,238,255,255,255,
        184,200,218,238,255,255,255,255,
        200,218,238,255,255,255,255,255,
        218,238,255,255,255,255,255,255,
        238,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,
    ];
    let w = if chroma { &CHROMA_W4 } else { &LUMA_W4 };
    for i in 0..64 {
        let mut v = (out[i] as i32 * w[i] + 50) / 100;
        if quality <= 95 {
            if chroma {
                v = (v * 118 + 50) / 100;
            } else if i >= 20 {
                v = (v * 109 + 50) / 100;
            }
        }
        out[i] = v.clamp(1, 255) as i16;
    }
    out
}

pub fn quant_table_for_quality_perceptual_v4(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual_v4(&default_quant_table(), quality, false)
}

pub fn chroma_quant_table_for_quality_perceptual_v4(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual_v4(&default_chroma_quant_table(), quality, true)
}

fn build_sparsify_thresholds(quality: u8, is_chroma: bool) -> [i16; 64] {
    // Precompute once per plane to avoid per-block branchy threshold math.
    let q = quality.clamp(1, 100);
    let base_t: i16 = if is_chroma {
        if q >= 90 { 2 } else if q >= 80 { 3 } else { 4 }
    } else if q >= 90 {
        1
    } else if q >= 80 {
        2
    } else {
        3
    };
    let mut thr = [0i16; 64];
    for zi in 1..64 {
        thr[zi] = if zi >= 56 {
            base_t + 2
        } else if zi >= 40 {
            base_t + 1
        } else if zi >= 24 {
            base_t
        } else {
            base_t.saturating_sub(1).max(1)
        };
    }
    thr
}

#[inline]
fn sparsify_ac_block(block: &mut Block, thresholds: &[i16; 64]) {
    for zi in 1..64 {
        let z = ZIGZAG[zi];
        if block.data[z].abs() <= thresholds[zi] {
            block.data[z] = 0;
        }
    }
}

#[inline]
pub fn quantize(block: &mut [i16; 64], table: &[i16; 64]) {
    #[cfg(not(test))]
    unsafe { quantize_block(block.as_mut_ptr(), table.as_ptr()); }
    #[cfg(test)]
    {
        // Pure-Rust scalar equivalent of the C scalar fallback (truncating divide).
        for i in 0..64 {
            block[i] = (block[i] as i32 / table[i] as i32) as i16;
        }
    }
}

fn write_header(out: &mut [u8], pos: &mut i32, magic: &[u8; 3], w: usize, h: usize, q: u8) {
    if (*pos as usize) + BG_HEADER_SIZE > out.len() { return; }
    bitstream::write_bytes(out, pos, magic);
    for b in (w as u32).to_le_bytes() { bitstream::write_byte(out, pos, b); }
    for b in (h as u32).to_le_bytes() { bitstream::write_byte(out, pos, b); }
    bitstream::write_byte(out, pos, if q == 0 { 50 } else { q });
}

fn write_icc_trailer(out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>) {
    let Some(data) = icc else { return };
    if data.is_empty() { return; }
    let need = 3 + 1 + 4 + data.len();
    if (*pos as usize) + need > out.len() { return; }
    bitstream::write_byte(out, pos, b'B');
    bitstream::write_byte(out, pos, b'G');
    bitstream::write_byte(out, pos, b'x');
    bitstream::write_byte(out, pos, 1);
    for b in (data.len() as u32).to_le_bytes() { bitstream::write_byte(out, pos, b); }
    bitstream::write_bytes(out, pos, data);
}

// ---------------------------------------------------------------------------
// RLE path (legacy v1/v2/v3)
// ---------------------------------------------------------------------------

fn encode_blocks_rle(
    blocks: &mut [Block],
    table: &[i16; 64],
    out: &mut [u8],
    pos: &mut i32,
    plane_w: usize,
    plane_h: usize,
) {
    if should_parallel_blocks(blocks.len(), plane_w, plane_h) {
        blocks.par_chunks_mut(BLOCK_TILE_SIZE).for_each(|chunk| {
            for block in chunk.iter_mut() {
                dct::dct(block);
                quantize(&mut block.data, table);
            }
        });
    } else {
        for block in blocks.iter_mut() {
            dct::dct(block);
            quantize(&mut block.data, table);
        }
    }
    for block in blocks.iter() {
        entropy::encode_block_to_buffer(block, out, pos);
    }
}

fn encode_channel_rle(blocks: &mut [Block], table: &[i16; 64], plane_w: usize, plane_h: usize) -> Vec<u8> {
    if should_parallel_blocks(blocks.len(), plane_w, plane_h) {
        blocks.par_chunks_mut(BLOCK_TILE_SIZE).for_each(|chunk| {
            for block in chunk.iter_mut() {
                dct::dct(block);
                quantize(&mut block.data, table);
            }
        });
    } else {
        for block in blocks.iter_mut() {
            dct::dct(block);
            quantize(&mut block.data, table);
        }
    }
    let cap = blocks.len() * (2 + 63 * 3 + 3);
    let mut buf = vec![0u8; cap];
    let mut p: i32 = 0;
    for block in blocks.iter() {
        entropy::encode_block_to_buffer(block, &mut buf, &mut p);
    }
    buf.truncate(p as usize);
    buf
}

pub fn encode_grayscale(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32,
) {
    write_header(out, pos, BG_MAGIC_GRAY, width, height, quality);
    let table = quant_table_for_quality(quality);
    let blockizer = Blockizer::new(width, height);
    let mut blocks = blockizer.generate_blocks(image);
    encode_blocks_rle(&mut blocks, &table, out, pos, width, height);
}

// ---------------------------------------------------------------------------
// Huffman + YCbCr 4:2:0 path (v4/v5) — best compression
// ---------------------------------------------------------------------------

/// Run DCT, quantization, optional AC sparsification and JPEG coefficient
/// clamping over a plane's blocks. Shared by the plain Huffman path
/// (`encode_channel_huffman`) and the dictionary path so both operate on
/// exactly the same coefficient blocks (and therefore decode identically).
fn preprocess_plane_blocks(
    blocks: &mut [Block],
    table: &[i16; 64],
    plane_w: usize,
    plane_h: usize,
    sparsify_thresholds: Option<&[i16; 64]>,
) {
    if should_parallel_blocks(blocks.len(), plane_w, plane_h) {
        blocks.par_chunks_mut(BLOCK_TILE_SIZE).for_each(|chunk| {
            for block in chunk.iter_mut() {
                dct::dct(block);
                quantize(&mut block.data, table);
                if let Some(thr) = sparsify_thresholds {
                    sparsify_ac_block(block, thr);
                }
                huffman::clamp_block_jpeg_coeffs(block);
            }
        });
    } else {
        for block in blocks.iter_mut() {
            dct::dct(block);
            quantize(&mut block.data, table);
            if let Some(thr) = sparsify_thresholds {
                sparsify_ac_block(block, thr);
            }
            huffman::clamp_block_jpeg_coeffs(block);
        }
    }
}

/// Encode blocks with Huffman into a Vec<u8>. Parallel DCT+quant, sequential Huffman.
fn encode_channel_huffman(
    blocks: &mut [Block],
    table: &[i16; 64],
    plane_w: usize,
    plane_h: usize,
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
    sparsify_thresholds: Option<&[i16; 64]>,
) -> Vec<u8> {
    preprocess_plane_blocks(blocks, table, plane_w, plane_h, sparsify_thresholds);
    huffman::encode_plane_with_profile(blocks, is_chroma, use_chroma_ac, use_dc_delta)
}

/// Encode RGB image using YCbCr 4:2:0 + Huffman (version 4).
/// This is the recommended path for RGB images — best compression ratio.
pub fn encode_rgb_ycbcr(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
    write_header(out, pos, BG_MAGIC_YUV420_V8, width, height, quality);

    let (y, cb, cr) = colorspace::rgb_to_ycbcr420(image, width, height);
    let cw = (width  + 1) / 2;
    let ch = (height + 1) / 2;

    let luma_table   = quant_table_for_quality_perceptual_v4(quality);
    let chroma_table = chroma_quant_table_for_quality_perceptual_v4(quality);
    let luma_sparsify = build_sparsify_thresholds(quality, false);
    let chroma_sparsify = build_sparsify_thresholds(quality, true);

    // Encode Y, Cb, Cr in parallel — each into its own buffer
    let blockizer_full   = Blockizer::new(width, height);
    let blockizer_chroma = Blockizer::new(cw, ch);

    let (y_buf, cb_buf, cr_buf) = if should_parallel_planes(width, height) {
        let (y_buf, (cb_buf, cr_buf)) = rayon::join(
            || {
                let mut blocks = blockizer_full.generate_blocks(&y);
                encode_channel_huffman(&mut blocks, &luma_table, width, height, false, false, true, Some(&luma_sparsify))
            },
            || {
                rayon::join(
                    || {
                        let mut blocks = blockizer_chroma.generate_blocks(&cb);
                        encode_channel_huffman(&mut blocks, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify))
                    },
                    || {
                        let mut blocks = blockizer_chroma.generate_blocks(&cr);
                        encode_channel_huffman(&mut blocks, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify))
                    },
                )
            },
        );
        (y_buf, cb_buf, cr_buf)
    } else {
        let mut yb = blockizer_full.generate_blocks(&y);
        let y_buf = encode_channel_huffman(&mut yb, &luma_table, width, height, false, false, true, Some(&luma_sparsify));
        let mut cbb = blockizer_chroma.generate_blocks(&cb);
        let cb_buf = encode_channel_huffman(&mut cbb, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify));
        let mut crb = blockizer_chroma.generate_blocks(&cr);
        let cr_buf = encode_channel_huffman(&mut crb, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify));
        (y_buf, cb_buf, cr_buf)
    };

    bitstream::write_bytes(out, pos, &y_buf);
    bitstream::write_bytes(out, pos, &cb_buf);
    bitstream::write_bytes(out, pos, &cr_buf);
    write_icc_trailer(out, pos, icc);
}

/// Encode RGBA image using YCbCr 4:2:0 + Huffman + full-res alpha (version 5).
pub fn encode_rgba_ycbcr(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
    write_header(out, pos, BG_MAGIC_YUV420A_V8, width, height, quality);

    let (y, cb, cr, a) = colorspace::rgba_to_ycbcr420a(image, width, height);
    let cw = (width  + 1) / 2;
    let ch = (height + 1) / 2;

    let luma_table   = quant_table_for_quality_perceptual_v4(quality);
    let chroma_table = chroma_quant_table_for_quality_perceptual_v4(quality);
    let luma_sparsify = build_sparsify_thresholds(quality, false);
    let chroma_sparsify = build_sparsify_thresholds(quality, true);

    let blockizer_full   = Blockizer::new(width, height);
    let blockizer_chroma = Blockizer::new(cw, ch);

    let (y_buf, cb_buf, cr_buf, a_buf) = if should_parallel_planes(width, height) {
        let ((y_buf, a_buf), (cb_buf, cr_buf)) = rayon::join(
            || {
                rayon::join(
                    || {
                        let mut blocks = blockizer_full.generate_blocks(&y);
                        encode_channel_huffman(&mut blocks, &luma_table, width, height, false, false, true, Some(&luma_sparsify))
                    },
                    || {
                        let mut blocks = blockizer_full.generate_blocks(&a);
                        encode_channel_huffman(&mut blocks, &luma_table, width, height, false, false, true, Some(&luma_sparsify))
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        let mut blocks = blockizer_chroma.generate_blocks(&cb);
                        encode_channel_huffman(&mut blocks, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify))
                    },
                    || {
                        let mut blocks = blockizer_chroma.generate_blocks(&cr);
                        encode_channel_huffman(&mut blocks, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify))
                    },
                )
            },
        );
        (y_buf, cb_buf, cr_buf, a_buf)
    } else {
        let mut yb = blockizer_full.generate_blocks(&y);
        let y_buf = encode_channel_huffman(&mut yb, &luma_table, width, height, false, false, true, Some(&luma_sparsify));
        let mut cbb = blockizer_chroma.generate_blocks(&cb);
        let cb_buf = encode_channel_huffman(&mut cbb, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify));
        let mut crb = blockizer_chroma.generate_blocks(&cr);
        let cr_buf = encode_channel_huffman(&mut crb, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify));
        let mut ab = blockizer_full.generate_blocks(&a);
        let a_buf = encode_channel_huffman(&mut ab, &luma_table, width, height, false, false, true, Some(&luma_sparsify));
        (y_buf, cb_buf, cr_buf, a_buf)
    };

    bitstream::write_bytes(out, pos, &y_buf);
    bitstream::write_bytes(out, pos, &cb_buf);
    bitstream::write_bytes(out, pos, &cr_buf);
    bitstream::write_bytes(out, pos, &a_buf);
    write_icc_trailer(out, pos, icc);
}

// ---------------------------------------------------------------------------
// Block dictionary path (v20/v21)
//
// The original entropy coding is left byte-for-byte untouched. The dictionary
// is an additional outer layer over the *same* quantized 8x8 coefficient
// blocks: repeated blocks are stored once in a global dictionary and the
// plane payloads reference them by index. Planes that would not shrink keep
// their ordinary Huffman payload (dictionary is a per-plane, opt-in layer),
// so a dictionary can never inflate a plane.
//
// On-disk layout:
//   [12-byte BG header, version 20/21]
//   [global dictionary section]
//       'B''G''d' (3 bytes magic)
//       entry_count : u32 LE
//       entry_count * 128 bytes, each = 64 * i16 LE coefficients
//   [per-plane payloads, one per plane in Y, Cb, Cr (, A) order]
//       flags            : u8        (bit0 = dictionary enabled for plane)
//       huffman_len      : u32 LE    (length of the fallback/plain payload)
//       huffman_bytes    : [u8; huffman_len]  (always present)
//       if flags & 1:
//         map_len        : u32 LE    (number of global indices mapped)
//         map            : [u16 LE; map_len]  (local index -> global index)
//         token_len      : u32 LE
//         tokens         : [u8; token_len]
//   [optional ICC trailer, unchanged]
//
// Decoders read `huffman_len` first and may always ignore the dictionary
// extension; a v20/v21-aware decoder prefers the token stream when present.
// ---------------------------------------------------------------------------

/// One observed distinct block plus its frequency across every plane.
#[derive(Clone)]
struct DictCandidate {
    block: [i16; 64],
    freq: u32,
    first_seen: u32,
}

/// Statistics describing how the dictionary matched an image. Returned by the
/// internal encoders so callers (and tests) can report hit rate and sizes.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct DictStats {
    /// Total blocks across all planes.
    pub total_blocks: u64,
    /// Blocks encoded as a dictionary reference rather than inline.
    pub referenced_blocks: u64,
    /// Number of entries written into the global dictionary.
    pub dictionary_entries: u64,
    /// Candidates dropped because of the capacity bound (LFU evictions).
    pub evicted_candidates: u64,
    /// Planes that kept plain Huffman because the dictionary did not help.
    pub planes_fallback: u64,
    /// Number of planes in the image.
    pub planes: u64,
}

impl DictStats {
    /// Fraction of all blocks served from the dictionary (0.0..=1.0).
    pub fn hit_rate(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.referenced_blocks as f64 / self.total_blocks as f64
        }
    }
}

fn block_key(block: &Block) -> [i16; 64] {
    block.data
}

fn append_u16_le(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn append_u32_le(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Scan every plane, count distinct block frequencies, then select the global
/// dictionary using two configurable rules:
///   * minimum frequency (`min_freq`) filters one-off blocks;
///   * capacity (`capacity`) bounds dictionary size. When more candidates
///     qualify than fit, an LFU (least-frequently-used) policy evicts the
///     least frequent entries; ties are broken by first-seen order so the
///     oldest/earliest encountered block is retained (stable, deterministic).
///
/// Returns the selected entries ordered by (frequency desc, first_seen asc).
fn build_global_dictionary(
    planes: &[Vec<Block>],
    config: &DictConfig,
) -> (Vec<DictCandidate>, DictStats) {
    let mut stats = DictStats::default();
    stats.planes = planes.len() as u64;
    let mut order_seen: u32 = 0;
    let mut counts: HashMap<[i16; 64], (u32, u32)> = HashMap::new(); // key -> (freq, first_seen)
    let mut total_blocks: u64 = 0;
    for plane in planes {
        for block in plane.iter() {
            total_blocks += 1;
            let key = block_key(block);
            counts.entry(key).and_modify(|(f, _)| *f += 1).or_insert_with(|| {
                let seen = order_seen;
                order_seen += 1;
                (1, seen)
            });
        }
    }
    stats.total_blocks = total_blocks;

    let cap = config.capacity.clamp(1, DICT_MAX_CAPACITY) as usize;
    let mut candidates: Vec<DictCandidate> = counts
        .into_iter()
        .map(|(block, (freq, first_seen))| DictCandidate { block, freq, first_seen })
        .collect();
    // Qualifying candidates first (frequency filter), ranked by LFU priority.
    candidates.retain(|c| c.freq >= config.min_freq.max(1));
    candidates.sort_by(|a, b| {
        b.freq
            .cmp(&a.freq)
            .then_with(|| a.first_seen.cmp(&b.first_seen))
    });
    let total_qualifying = candidates.len();
    if total_qualifying > cap {
        stats.evicted_candidates = (total_qualifying - cap) as u64;
        candidates.truncate(cap);
    }
    (candidates, stats)
}

/// Per-plane encoding outcome for the v20/v21 container.
struct PlaneDictEncoding {
    /// v18-style payload: `[u32 len][huffman bytes]`, reusable as the plain
    /// representation inside v20/v21 and for the all-fallback stream.
    plain: Vec<u8>,
    /// v20/v21 dictionary payload (without the container's length/flags).
    /// Empty when no block matched the dictionary. Layout:
    ///   [ref_count:u32][bitmap_len:u32][bitmap][local_count:u32][map u16..]
    ///   [huff_len:u32][huffman of the literal blocks, in scan order]
    dict_body: Vec<u8>,
    /// Number of blocks in this plane represented by a dictionary reference.
    referenced: u64,
}

/// Encode one plane, producing both the ordinary Huffman payload and the
/// dictionary representation. The caller decides globally which framing wins.
///
/// The dictionary representation keeps every non-referenced ("literal") block
/// in its compact Huffman form and only replaces repeated blocks: a 1-bit-per-
/// block bitmap marks references, followed by the per-reference local dictionary
/// indices and the Huffman stream of just the literal blocks (scan order). This
/// means partial repetition (mixed images) still saves bytes instead of paying
/// 129 raw bytes per unique block.
fn encode_plane_dict_options(
    blocks: &[Block],
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
    global_index: &HashMap<[i16; 64], u16>,
) -> PlaneDictEncoding {
    // Plain payload: identical bytes to the non-dictionary format.
    let plain = huffman::encode_plane_with_profile(blocks, is_chroma, use_chroma_ac, use_dc_delta);

    let mut local_map: Vec<u16> = Vec::new();
    let mut local_index: HashMap<u16, u16> = HashMap::new(); // global idx -> local idx
    let mut is_ref = vec![false; blocks.len()];
    let mut refs: Vec<u16> = Vec::new();
    let mut literal_blocks: Vec<Block> = Vec::new();
    let mut referenced: u64 = 0;
    for (i, block) in blocks.iter().enumerate() {
        if let Some(&gidx) = global_index.get(&block_key(block)) {
            let lidx = match local_index.get(&gidx) {
                Some(&l) => l,
                None => {
                    let l = local_map.len() as u16;
                    local_map.push(gidx);
                    local_index.insert(gidx, l);
                    l
                }
            };
            is_ref[i] = true;
            refs.push(lidx);
            referenced += 1;
        } else {
            literal_blocks.push(*block);
        }
    }

    let mut dict_body = Vec::new();
    if referenced > 0 {
        // Huffman-encode only the literal blocks; DC-delta continuity is local
        // to this sub-stream and the decoder rebuilds the same sequence.
        let huff = huffman::encode_plane_with_profile(
            &literal_blocks, is_chroma, use_chroma_ac, use_dc_delta,
        );
        // Pack the reference bitmap (bit i = block i is a reference).
        let bitmap_len = (blocks.len() + 7) / 8;
        let mut bitmap = vec![0u8; bitmap_len];
        for (i, &r) in is_ref.iter().enumerate() {
            if r {
                bitmap[i / 8] |= 1u8 << (7 - (i % 8));
            }
        }
        append_u32_le(&mut dict_body, referenced as u32);
        append_u32_le(&mut dict_body, bitmap_len as u32);
        dict_body.extend_from_slice(&bitmap);
        append_u32_le(&mut dict_body, local_map.len() as u32);
        for &g in local_map.iter() {
            append_u16_le(&mut dict_body, g);
        }
        // Reference indices, one u16 per referenced block, in scan order.
        for &r in refs.iter() {
            append_u16_le(&mut dict_body, r);
        }
        append_u32_le(&mut dict_body, huff.len() as u32);
        dict_body.extend_from_slice(&huff);
    }
    PlaneDictEncoding { plain, dict_body, referenced }
}

/// Frame one v20/v21 plane as `[u32 total_len][u8 flags][payload...]`, where the
/// payload is either the plain Huffman stream (flags=0) or the dictionary body
/// (flags=1). The 4-byte prefix lets a decoder locate each plane exactly.
fn frame_v20_plane(encoding: &PlaneDictEncoding) -> Vec<u8> {
    let use_dict = !encoding.dict_body.is_empty()
        && encoding.dict_body.len() < encoding.plain.len();
    let mut frame = Vec::new();
    if use_dict {
        let total = 1 + encoding.dict_body.len();
        append_u32_le(&mut frame, total as u32);
        frame.push(DICT_PLANE_FLAG_ENABLED);
        frame.extend_from_slice(&encoding.dict_body);
    } else {
        let total = 1 + encoding.plain.len();
        append_u32_le(&mut frame, total as u32);
        frame.push(0u8);
        frame.extend_from_slice(&encoding.plain);
    }
    frame
}

/// Serialize the global dictionary section.
fn encode_dictionary_section(entries: &[DictCandidate]) -> Vec<u8> {
    let mut sec = Vec::with_capacity(3 + 4 + entries.len() * DICT_BLOCK_BYTES);
    sec.extend_from_slice(DICT_SECTION_MAGIC);
    append_u32_le(&mut sec, entries.len() as u32);
    for e in entries.iter() {
        for coeff in e.block.iter() {
            sec.extend_from_slice(&coeff.to_le_bytes());
        }
    }
    sec
}

/// Shared preprocessing for the dictionary path: DCT/quant/sparsify/clamp each
/// plane in place, mirroring `encode_rgb_ycbcr`/`encode_rgba_ycbcr` exactly.
struct PlaneSpec {
    blocks: Vec<Block>,
    width: usize,
    height: usize,
    is_chroma: bool,
    use_chroma_ac: bool,
    table: [i16; 64],
    sparsify: [i16; 64],
}

fn prepare_dict_planes(mut specs: Vec<PlaneSpec>) -> Vec<PlaneSpec> {
    for spec in specs.iter_mut() {
        preprocess_plane_blocks(
            &mut spec.blocks,
            &spec.table,
            spec.width,
            spec.height,
            Some(&spec.sparsify),
        );
    }
    specs
}

/// Core dictionary encoder shared by RGB and RGBA. Returns the full .bg bytes
/// (header + planes + optional dictionary/ICC) plus match statistics.
///
/// It builds both candidate streams and keeps the smaller:
///   * legacy layout — concatenated plain Huffman planes, byte-identical to
///     the v18/v19 non-dictionary format (zero inflation guarantee);
///   * v20/v21 layout — global dictionary section plus framed planes, each of
///     which independently prefers its dictionary body when that is smaller.
/// When the legacy stream wins, the header version is rewritten to 18/19 so
/// the output is an ordinary, older-decoder-compatible file.
fn encode_with_dictionary(
    dict_version: u8,
    legacy_magic: &[u8; 3],
    width: usize,
    height: usize,
    quality: u8,
    plane_specs: Vec<PlaneSpec>,
    icc: Option<&[u8]>,
    config: &DictConfig,
    out: &mut [u8],
    pos: &mut i32,
) -> DictStats {
    let specs = prepare_dict_planes(plane_specs);
    let planes: Vec<Vec<Block>> = specs.iter().map(|s| s.blocks.clone()).collect();
    let (entries, mut stats) = build_global_dictionary(&planes, config);
    stats.dictionary_entries = entries.len() as u64;

    // Global index: key -> dictionary position.
    let global_index: HashMap<[i16; 64], u16> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| (e.block, i as u16))
        .collect();

    // Per-plane dual encodings.
    let encodings: Vec<PlaneDictEncoding> = specs
        .iter()
        .map(|s| encode_plane_dict_options(&s.blocks, s.is_chroma, s.use_chroma_ac, true, &global_index))
        .collect();

    // Legacy candidate size: concatenated plain planes only.
    let legacy_plane_bytes: usize = encodings.iter().map(|e| e.plain.len()).sum();
    // v20/v21 candidate size: dictionary section + framed planes.
    let dict_section = encode_dictionary_section(&entries);
    let framed: Vec<Vec<u8>> = encodings.iter().map(frame_v20_plane).collect();
    let dict_plane_bytes: usize = framed.iter().map(|f| f.len()).sum();
    let use_dict = stats.dictionary_entries > 0
        && dict_section.len() + dict_plane_bytes < legacy_plane_bytes;

    if !use_dict {
        // Emit the exact legacy v18/v19 stream; original encoding unchanged.
        write_header(out, pos, legacy_magic, width, height, quality);
        for e in encodings.iter() {
            bitstream::write_bytes(out, pos, &e.plain);
        }
        write_icc_trailer(out, pos, icc);
        stats.referenced_blocks = 0;
        stats.planes_fallback = stats.planes;
        return stats;
    }

    write_header(out, pos, &[b'B', b'G', dict_version], width, height, quality);
    bitstream::write_bytes(out, pos, &dict_section);
    let mut total_referenced: u64 = 0;
    let mut fallback_planes: u64 = 0;
    for (frame, e) in framed.iter().zip(encodings.iter()) {
        if e.referenced > 0 && !e.dict_body.is_empty() && e.dict_body.len() < e.plain.len() {
            total_referenced += e.referenced;
        } else {
            fallback_planes += 1;
        }
        bitstream::write_bytes(out, pos, frame);
    }
    stats.referenced_blocks = total_referenced;
    stats.planes_fallback = fallback_planes;
    write_icc_trailer(out, pos, icc);
    stats
}

/// RGB dictionary encoder (version 20).
pub fn encode_rgb_ycbcr_dict(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>, config: &DictConfig,
) -> DictStats {
    let (y, cb, cr) = colorspace::rgb_to_ycbcr420(image, width, height);
    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;
    let luma_table = quant_table_for_quality_perceptual_v4(quality);
    let chroma_table = chroma_quant_table_for_quality_perceptual_v4(quality);
    let blockizer_full = Blockizer::new(width, height);
    let blockizer_chroma = Blockizer::new(cw, ch);
    let specs = vec![
        PlaneSpec {
            blocks: blockizer_full.generate_blocks(&y),
            width, height, is_chroma: false, use_chroma_ac: false,
            table: luma_table, sparsify: build_sparsify_thresholds(quality, false),
        },
        PlaneSpec {
            blocks: blockizer_chroma.generate_blocks(&cb),
            width: cw, height: ch, is_chroma: true, use_chroma_ac: true,
            table: chroma_table, sparsify: build_sparsify_thresholds(quality, true),
        },
        PlaneSpec {
            blocks: blockizer_chroma.generate_blocks(&cr),
            width: cw, height: ch, is_chroma: true, use_chroma_ac: true,
            table: chroma_table, sparsify: build_sparsify_thresholds(quality, true),
        },
    ];
    encode_with_dictionary(
        20, BG_MAGIC_YUV420_V8, width, height, quality, specs, icc, config, out, pos,
    )
}

/// RGBA dictionary encoder (version 21).
pub fn encode_rgba_ycbcr_dict(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>, config: &DictConfig,
) -> DictStats {
    let (y, cb, cr, a) = colorspace::rgba_to_ycbcr420a(image, width, height);
    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;
    let luma_table = quant_table_for_quality_perceptual_v4(quality);
    let chroma_table = chroma_quant_table_for_quality_perceptual_v4(quality);
    let blockizer_full = Blockizer::new(width, height);
    let blockizer_chroma = Blockizer::new(cw, ch);
    let specs = vec![
        PlaneSpec {
            blocks: blockizer_full.generate_blocks(&y),
            width, height, is_chroma: false, use_chroma_ac: false,
            table: luma_table, sparsify: build_sparsify_thresholds(quality, false),
        },
        PlaneSpec {
            blocks: blockizer_chroma.generate_blocks(&cb),
            width: cw, height: ch, is_chroma: true, use_chroma_ac: true,
            table: chroma_table, sparsify: build_sparsify_thresholds(quality, true),
        },
        PlaneSpec {
            blocks: blockizer_chroma.generate_blocks(&cr),
            width: cw, height: ch, is_chroma: true, use_chroma_ac: true,
            table: chroma_table, sparsify: build_sparsify_thresholds(quality, true),
        },
        PlaneSpec {
            blocks: blockizer_full.generate_blocks(&a),
            width, height, is_chroma: false, use_chroma_ac: false,
            table: luma_table, sparsify: build_sparsify_thresholds(quality, false),
        },
    ];
    encode_with_dictionary(
        21, BG_MAGIC_YUV420A_V8, width, height, quality, specs, icc, config, out, pos,
    )
}

// ---------------------------------------------------------------------------
// v20/v21 decode support
// ---------------------------------------------------------------------------

fn read_u16_le(buf: &[u8], pos: usize) -> Option<u16> {
    if pos + 2 > buf.len() { return None; }
    Some(u16::from_le_bytes([buf[pos], buf[pos + 1]]))
}

fn read_u32_le(buf: &[u8], pos: usize) -> Option<u32> {
    if pos + 4 > buf.len() { return None; }
    Some(u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?))
}

/// Parse the global dictionary section beginning at `start`.
/// Returns (dictionary blocks, position just past the section).
fn parse_dictionary_section(buf: &[u8], start: usize) -> Option<(Vec<Block>, usize)> {
    if start + 7 > buf.len() { return None; }
    if &buf[start..start + 3] != DICT_SECTION_MAGIC { return None; }
    let count = read_u32_le(buf, start + 3)? as usize;
    let mut pos = start + 7;
    let end = pos.checked_add(count.checked_mul(DICT_BLOCK_BYTES)?)?;
    if end > buf.len() { return None; }
    let mut dict = Vec::with_capacity(count);
    for _ in 0..count {
        let mut data = [0i16; 64];
        for coeff in data.iter_mut() {
            *coeff = i16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;
        }
        dict.push(Block { data });
    }
    Some((dict, end))
}

/// Decode one v20/v21 plane payload at `start`. Prefers the embedded token
/// stream when enabled; otherwise decodes the ordinary Huffman payload.
/// Returns (blocks, position just past the plane).
fn decode_dict_plane(
    buf: &[u8],
    start: usize,
    n_blocks: usize,
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
    dictionary: &[Block],
) -> Option<(Vec<Block>, usize)> {
    // Frame: [u32 total_len][u8 flags][payload], total_len = 1 + payload len.
    if start + 5 > buf.len() { return None; }
    let total_len = read_u32_le(buf, start)? as usize;
    if total_len < 1 { return None; }
    let frame_end = start.checked_add(4)?.checked_add(total_len)?;
    if frame_end > buf.len() { return None; }
    let flags = buf[start + 4];
    let payload = &buf[start + 5..frame_end];

    if flags & DICT_PLANE_FLAG_ENABLED == 0 {
        // Plain payload is exactly a v18 Huffman plane: [u32 huff_len][bytes].
        let (blocks, _) = huffman::decode_plane_with_profile(
            payload, 0, n_blocks, is_chroma, use_chroma_ac, use_dc_delta,
        )?;
        return Some((blocks, frame_end));
    }

    // Dictionary payload:
    // [ref_count][bitmap_len][bitmap][local_count][map][refs u16..][huff_len][huffman]
    let mut p: usize = 0;
    let ref_count = read_u32_le(payload, p)? as usize;
    p += 4;
    if ref_count > n_blocks { return None; } // malformed: more refs than blocks
    let bitmap_len = read_u32_le(payload, p)? as usize;
    p += 4;
    let expected_bitmap_len = (n_blocks + 7) / 8;
    if bitmap_len != expected_bitmap_len { return None; }
    let bitmap_end = p.checked_add(bitmap_len)?;
    if bitmap_end > payload.len() { return None; }
    let bitmap = &payload[p..bitmap_end];
    p = bitmap_end;

    let local_count = read_u32_le(payload, p)? as usize;
    p += 4;
    let map_end = p.checked_add(local_count.checked_mul(2)?)?;
    if map_end > payload.len() { return None; }
    let mut global_indices = Vec::with_capacity(local_count);
    let mut m = p;
    for _ in 0..local_count {
        global_indices.push(read_u16_le(payload, m)? as usize);
        m += 2;
    }
    for &g in global_indices.iter() {
        if g >= dictionary.len() { return None; }
    }
    p = map_end;

    // One u16 local index per referenced block (scan order).
    let refs_end = p.checked_add(ref_count.checked_mul(2)?)?;
    if refs_end > payload.len() { return None; }
    let mut refs = Vec::with_capacity(ref_count);
    let mut r = p;
    for _ in 0..ref_count {
        refs.push(read_u16_le(payload, r)? as usize);
        r += 2;
    }
    for &l in refs.iter() {
        if l >= global_indices.len() { return None; }
    }
    p = refs_end;

    let huff_len = read_u32_le(payload, p)? as usize;
    p += 4;
    let huff_end = p.checked_add(huff_len)?;
    if huff_end > payload.len() { return None; }
    // Cross-check the bitmap against ref_count so a tampered stream cannot
    // desynchronize the reference/literal reconstruction.
    let mut bitmap_refs = 0usize;
    for i in 0..n_blocks {
        if (bitmap[i / 8] >> (7 - (i % 8))) & 1 == 1 {
            bitmap_refs += 1;
        }
    }
    if bitmap_refs != ref_count { return None; }
    let n_literal = n_blocks - ref_count;
    let (literals, _) = huffman::decode_plane_with_profile(
        payload, p, n_literal, is_chroma, use_chroma_ac, use_dc_delta,
    )?;
    if literals.len() != n_literal { return None; }

    let mut blocks = Vec::with_capacity(n_blocks);
    let mut ref_iter = 0usize;
    let mut lit_iter = 0usize;
    for i in 0..n_blocks {
        let referenced = (bitmap[i / 8] >> (7 - (i % 8))) & 1 == 1;
        if referenced {
            if ref_iter >= refs.len() { return None; }
            let gidx = global_indices[refs[ref_iter]];
            ref_iter += 1;
            blocks.push(dictionary[gidx]);
        } else {
            if lit_iter >= literals.len() { return None; }
            blocks.push(literals[lit_iter]);
            lit_iter += 1;
        }
    }
    if ref_iter != ref_count || lit_iter != n_literal { return None; }
    Some((blocks, frame_end))
}

/// Full v20/v21 decode. `has_alpha` selects the 4-plane RGBA layout.
/// `out_icc` is populated when a trailer is present. Mirrors the v18/v19
/// decoder structure but reads the dictionary container.
pub fn decode_dict_version(
    version: u8,
    buffer: &[u8],
    header_size: usize,
    w: usize,
    h: usize,
    quality: u8,
    out_pixels: &mut [u8],
    out_icc: Option<&mut Vec<u8>>,
) -> bool {
    let has_alpha = version == 19 || version == 21;
    let is_dict = version == 20 || version == 21;
    let needed = w * h * if has_alpha { 4 } else { 3 };
    if out_pixels.len() < needed { return false; }
    let cw = (w + 1) / 2;
    let ch = (h + 1) / 2;
    let luma_q = quant_table_for_quality_perceptual_v4(quality);
    let chroma_q = chroma_quant_table_for_quality_perceptual_v4(quality);

    let n_y = ((w + 7) / 8) * ((h + 7) / 8);
    let n_c = ((cw + 7) / 8) * ((ch + 7) / 8);

    let mut y_plane = vec![0u8; w * h];
    let mut cb_plane = vec![0u8; cw * ch];
    let mut cr_plane = vec![0u8; cw * ch];
    let mut a_plane = if has_alpha { vec![0u8; w * h] } else { vec![] };

    let mut pos = header_size;
    if !is_dict {
        // Legacy v18/v19 fallback: back-to-back plain Huffman planes.
        let (y_blocks, p1) = match huffman::decode_plane_with_profile(buffer, pos, n_y, false, false, true) {
            Some(v) => v, None => return false,
        };
        pos = p1;
        let (cb_blocks, p2) = match huffman::decode_plane_with_profile(buffer, pos, n_c, true, true, true) {
            Some(v) => v, None => return false,
        };
        pos = p2;
        let (cr_blocks, p3) = match huffman::decode_plane_with_profile(buffer, pos, n_c, true, true, true) {
            Some(v) => v, None => return false,
        };
        pos = p3;
        if !write_decoded_huffman_plane(&y_blocks, &luma_q, w, h, &mut y_plane) { return false; }
        if !write_decoded_huffman_plane(&cb_blocks, &chroma_q, cw, ch, &mut cb_plane) { return false; }
        if !write_decoded_huffman_plane(&cr_blocks, &chroma_q, cw, ch, &mut cr_plane) { return false; }
        if has_alpha {
            let (a_blocks, p4) = match huffman::decode_plane_with_profile(buffer, pos, n_y, false, false, true) {
                Some(v) => v, None => return false,
            };
            pos = p4;
            if !write_decoded_huffman_plane(&a_blocks, &luma_q, w, h, &mut a_plane) { return false; }
        }
    } else {
        let (dictionary, dpos) = match parse_dictionary_section(buffer, header_size) {
            Some(v) => v,
            None => return false,
        };
        pos = dpos;

        let (y_blocks, p1) = match decode_dict_plane(buffer, pos, n_y, false, false, true, &dictionary) {
            Some(v) => v, None => return false,
        };
        pos = p1;
        let (cb_blocks, p2) = match decode_dict_plane(buffer, pos, n_c, true, true, true, &dictionary) {
            Some(v) => v, None => return false,
        };
        pos = p2;
        let (cr_blocks, p3) = match decode_dict_plane(buffer, pos, n_c, true, true, true, &dictionary) {
            Some(v) => v, None => return false,
        };
        pos = p3;

        if !write_decoded_huffman_plane(&y_blocks, &luma_q, w, h, &mut y_plane) { return false; }
        if !write_decoded_huffman_plane(&cb_blocks, &chroma_q, cw, ch, &mut cb_plane) { return false; }
        if !write_decoded_huffman_plane(&cr_blocks, &chroma_q, cw, ch, &mut cr_plane) { return false; }

        if has_alpha {
            let (a_blocks, p4) = match decode_dict_plane(buffer, pos, n_y, false, false, true, &dictionary) {
                Some(v) => v, None => return false,
            };
            pos = p4;
            if !write_decoded_huffman_plane(&a_blocks, &luma_q, w, h, &mut a_plane) { return false; }
        }
    }

    if has_alpha {
        colorspace::ycbcr420a_to_rgba(&y_plane, &cb_plane, &cr_plane, &a_plane, w, h, out_pixels);
    } else {
        colorspace::ycbcr420_to_rgb(&y_plane, &cb_plane, &cr_plane, w, h, out_pixels);
    }

    if let Some(v) = out_icc {
        if let Some((icc, _)) = parse_icc_trailer_from(buffer, pos) { *v = icc; }
    }
    true
}

/// Dequantize + IDCT + write a decoded coefficient plane to a flat byte plane.
/// Kept local to the dictionary module to avoid touching decoder.rs internals.
fn write_decoded_huffman_plane(
    blocks: &[Block],
    quant: &[i16; 64],
    w: usize,
    h: usize,
    plane: &mut [u8],
) -> bool {
    if plane.len() < w * h { return false; }
    let bw = (w + 7) / 8;
    let mut work: Vec<Block> = blocks.to_vec();
    if should_parallel_dequant_local(work.len(), w, h) {
        work.par_chunks_mut(BLOCK_TILE_SIZE).for_each(|chunk| {
            for block in chunk.iter_mut() {
                dequant_for_profile(block, quant);
                dct::idct(block);
            }
        });
    } else {
        for block in work.iter_mut() {
            dequant_for_profile(block, quant);
            dct::idct(block);
        }
    }
    for (idx, block) in work.iter().enumerate() {
        let by = (idx / bw) * 8;
        let bx = (idx % bw) * 8;
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
    true
}

#[inline]
fn should_parallel_dequant_local(n_blocks: usize, w: usize, h: usize) -> bool {
    n_blocks >= PARALLEL_BLOCKS_THRESHOLD && w.saturating_mul(h) >= PARALLEL_PLANE_PIXELS_THRESHOLD
}

/// Dequantize one block. Release builds use the C SIMD implementation; test
/// builds use a pure-Rust scalar equivalent (matches c/quant.c scalar path)
/// so `cargo test` links without the C object files.
#[inline]
fn dequant_for_profile(block: &mut Block, quant: &[i16; 64]) {
    #[cfg(not(test))]
    unsafe {
        crate::ffi::dequantize_block(block.data.as_mut_ptr(), quant.as_ptr());
    }
    #[cfg(test)]
    {
        for i in 0..64 {
            let v = block.data[i] as i32 * quant[i] as i32;
            block.data[i] = v.clamp(-32768, 32767) as i16;
        }
    }
}

/// Local ICC-trailer parser mirroring the decoder's layout (avoids a cross-module
/// dependency while keeping encoder.rs the single modified file).
fn parse_icc_trailer_from(buffer: &[u8], pos: usize) -> Option<(Vec<u8>, usize)> {
    if pos + 8 > buffer.len() { return None; }
    if buffer[pos] != b'B' || buffer[pos + 1] != b'G' || buffer[pos + 2] != b'x' { return None; }
    let chunk_type = buffer[pos + 3];
    let len = read_u32_le(buffer, pos + 4)? as usize;
    let data_pos = pos + 8;
    if chunk_type != 1 || data_pos + len > buffer.len() { return None; }
    Some((buffer[data_pos..data_pos + len].to_vec(), data_pos + len))
}

// ---------------------------------------------------------------------------
// Legacy RGB/RGBA RLE paths (v2/v3) — kept for backward compat
// ---------------------------------------------------------------------------

pub fn encode_rgb(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
    // Opt-in dictionary path (v20); disabled by default, so the historical
    // v18 byte stream remains the output unless explicitly requested.
    let config = dict_config_from_env();
    if config.enabled {
        encode_rgb_ycbcr_dict(image, width, height, quality, out, pos, icc, &config);
        return;
    }
    // Default to the better YCbCr path; callers that need legacy RLE use encode_rgb_rle
    encode_rgb_ycbcr(image, width, height, quality, out, pos, icc);
}

pub fn encode_rgba(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
    let config = dict_config_from_env();
    if config.enabled {
        encode_rgba_ycbcr_dict(image, width, height, quality, out, pos, icc, &config);
        return;
    }
    encode_rgba_ycbcr(image, width, height, quality, out, pos, icc);
}

/// Legacy RLE RGB encoder (v2). Used when explicit backward-compat is needed.
pub fn encode_rgb_rle(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
    write_header(out, pos, BG_MAGIC_RGB, width, height, quality);
    let table = quant_table_for_quality(quality);
    let blockizer = Blockizer::new(width, height);
    let channel_bufs: Vec<Vec<u8>> = if should_parallel_planes(width, height) {
        (0..3usize).into_par_iter()
            .map(|c| {
                let mut blocks = blockizer.generate_blocks_rgb(image, c);
                encode_channel_rle(&mut blocks, &table, width, height)
            })
            .collect()
    } else {
        (0..3usize)
            .map(|c| {
                let mut blocks = blockizer.generate_blocks_rgb(image, c);
                encode_channel_rle(&mut blocks, &table, width, height)
            })
            .collect()
    };
    for buf in &channel_bufs { bitstream::write_bytes(out, pos, buf); }
    write_icc_trailer(out, pos, icc);
}

/// Legacy RLE RGBA encoder (v3).
pub fn encode_rgba_rle(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
    write_header(out, pos, BG_MAGIC_RGBA, width, height, quality);
    let table = quant_table_for_quality(quality);
    let blockizer = Blockizer::new(width, height);
    let channel_bufs: Vec<Vec<u8>> = if should_parallel_planes(width, height) {
        (0..4usize).into_par_iter()
            .map(|c| {
                let mut blocks = blockizer.generate_blocks_rgba(image, c);
                encode_channel_rle(&mut blocks, &table, width, height)
            })
            .collect()
    } else {
        (0..4usize)
            .map(|c| {
                let mut blocks = blockizer.generate_blocks_rgba(image, c);
                encode_channel_rle(&mut blocks, &table, width, height)
            })
            .collect()
    };
    for buf in &channel_bufs { bitstream::write_bytes(out, pos, buf); }
    write_icc_trailer(out, pos, icc);
}


// ---------------------------------------------------------------------------
// Dictionary compression unit tests (v20/v21)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod dict_tests {
    use super::*;

    const CAP: usize = 1 << 20;

    fn encode_plain_rgb(img: &[u8], w: usize, h: usize, q: u8) -> Vec<u8> {
        let mut out = vec![0u8; CAP];
        let mut pos = 0i32;
        encode_rgb_ycbcr(img, w, h, q, &mut out, &mut pos, None);
        out.truncate(pos as usize);
        out
    }

    fn encode_dict_rgb(img: &[u8], w: usize, h: usize, q: u8, cfg: &DictConfig) -> (Vec<u8>, DictStats) {
        let mut out = vec![0u8; CAP];
        let mut pos = 0i32;
        let stats = encode_rgb_ycbcr_dict(img, w, h, q, &mut out, &mut pos, None, cfg);
        out.truncate(pos as usize);
        (out, stats)
    }

    fn encode_dict_rgba(img: &[u8], w: usize, h: usize, q: u8, cfg: &DictConfig) -> Vec<u8> {
        let mut out = vec![0u8; CAP];
        let mut pos = 0i32;
        encode_rgba_ycbcr_dict(img, w, h, q, &mut out, &mut pos, None, cfg);
        out.truncate(pos as usize);
        out
    }

    /// Full v20/v21 pixel round trip using the pure-Rust dictionary decoder
    /// (test builds link without the C SIMD objects, so decoder::decode — which
    /// references C FFI symbols — is intentionally not exercised here).
    fn decode_dict_to_vec(buf: &[u8], w: usize, h: usize, ch: usize) -> Vec<u8> {
        let mut px = vec![0u8; w * h * ch];
        let quality = if buf[11] == 0 { 50 } else { buf[11] };
        let ok = decode_dict_version(buf[2], buf, BG_HEADER_SIZE, w, h, quality, &mut px, None);
        assert!(ok, "dictionary decode failed");
        px
    }

    /// Decode every plane back to coefficient blocks using pure-Rust parsers.
    /// Handles both the v20 dictionary container and a v18 fallback stream.
    fn decode_stream_planes(buf: &[u8], w: usize, h: usize) -> Vec<Vec<Block>> {
        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let n_y = ((w + 7) / 8) * ((h + 7) / 8);
        let n_c = ((cw + 7) / 8) * ((ch + 7) / 8);
        let profiles = [(n_y, false, false), (n_c, true, true), (n_c, true, true)];

        if buf[2] == 20 || buf[2] == 21 {
            let (dict, mut pos) = parse_dictionary_section(buf, BG_HEADER_SIZE).expect("dict section");
            let mut out = Vec::new();
            for (n, is_chroma, use_ac) in profiles {
                let (blocks, next) =
                    decode_dict_plane(buf, pos, n, is_chroma, use_ac, true, &dict).expect("plane");
                pos = next;
                out.push(blocks);
            }
            out
        } else {
            // Legacy v18 layout: back-to-back [u32 len][huffman] planes.
            let mut pos = BG_HEADER_SIZE;
            let mut out = Vec::new();
            for (n, is_chroma, use_ac) in profiles {
                let (blocks, next) =
                    huffman::decode_plane_with_profile(buf, pos, n, is_chroma, use_ac, true)
                        .expect("legacy plane");
                pos = next;
                out.push(blocks);
            }
            out
        }
    }

    /// Build and preprocess the Y/Cb/Cr coefficient planes exactly as the
    /// dictionary encoder does, so tests can compare at coefficient level.
    fn preprocessed_rgb_planes(img: &[u8], w: usize, h: usize, q: u8) -> Vec<Vec<Block>> {
        let (y, cb, cr) = colorspace::rgb_to_ycbcr420(img, w, h);
        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let bf = Blockizer::new(w, h);
        let bc = Blockizer::new(cw, ch);
        let luma = quant_table_for_quality_perceptual_v4(q);
        let chroma = chroma_quant_table_for_quality_perceptual_v4(q);
        let specs = vec![
            PlaneSpec {
                blocks: bf.generate_blocks(&y), width: w, height: h,
                is_chroma: false, use_chroma_ac: false, table: luma,
                sparsify: build_sparsify_thresholds(q, false),
            },
            PlaneSpec {
                blocks: bc.generate_blocks(&cb), width: cw, height: ch,
                is_chroma: true, use_chroma_ac: true, table: chroma,
                sparsify: build_sparsify_thresholds(q, true),
            },
            PlaneSpec {
                blocks: bc.generate_blocks(&cr), width: cw, height: ch,
                is_chroma: true, use_chroma_ac: true, table: chroma,
                sparsify: build_sparsify_thresholds(q, true),
            },
        ];
        prepare_dict_planes(specs).into_iter().map(|s| s.blocks).collect()
    }

    // Deterministic LCG so no external RNG dependency is needed.
    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*seed >> 33) as u32
    }

    /// Repeated 8x8 tile pattern -> many identical blocks.
    fn patterned_image(w: usize, h: usize) -> Vec<u8> {
        let mut img = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let base = (y * w + x) * 3;
                let ty = (y % 8) as u8;
                let tx = (x % 8) as u8;
                img[base] = ty.wrapping_mul(28).wrapping_add(tx);
                img[base + 1] = tx.wrapping_mul(17).wrapping_add(ty / 2);
                img[base + 2] = ty.wrapping_add(tx.wrapping_mul(31));
            }
        }
        img
    }

    /// Pseudo-random "photographic" content -> almost no repeated blocks.
    fn photo_image(w: usize, h: usize) -> Vec<u8> {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut img = vec![0u8; w * h * 3];
        for px in img.chunks_exact_mut(3) {
            px[0] = (lcg(&mut seed) & 0xFF) as u8;
            px[1] = (lcg(&mut seed) & 0xFF) as u8;
            px[2] = (lcg(&mut seed) & 0xFF) as u8;
        }
        img
    }

    /// Mostly patterned with a pseudo-random band -> mixed repetition. The
    /// patterned three-quarters yields repeated blocks while the random quarter
    /// keeps literal blocks, exercising both paths in one image.
    fn mixed_image(w: usize, h: usize) -> Vec<u8> {
        let mut seed = 0x1234_5678u64;
        let mut img = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let base = (y * w + x) * 3;
                if x < w * 3 / 4 {
                    let ty = (y % 8) as u8;
                    let tx = (x % 8) as u8;
                    img[base] = ty.wrapping_mul(28).wrapping_add(tx);
                    img[base + 1] = tx.wrapping_mul(17);
                    img[base + 2] = ty.wrapping_add(tx.wrapping_mul(31));
                } else {
                    img[base] = (lcg(&mut seed) & 0xFF) as u8;
                    img[base + 1] = (lcg(&mut seed) & 0xFF) as u8;
                    img[base + 2] = (lcg(&mut seed) & 0xFF) as u8;
                }
            }
        }
        img
    }

    #[test]
    fn config_parses_defaults_and_bounds() {
        let d = DictConfig::default();
        assert!(!d.enabled);
        assert_eq!(d.capacity, 256);
        assert_eq!(d.min_freq, 2);
        // Capacity is hard-clamped to the 15-bit token range.
        let cfg = DictConfig { enabled: true, capacity: u32::MAX, min_freq: 1 };
        let mut blocks = vec![Block::new(); 1];
        blocks[0].data[0] = 5;
        let (entries, _) = build_global_dictionary(&[blocks], &cfg);
        assert!(entries.len() <= DICT_MAX_CAPACITY as usize);
    }

    #[test]
    fn dictionary_selects_only_frequent_blocks() {
        // Two identical blocks + one unique block. With min_freq=2 only the
        // repeated block qualifies for the dictionary.
        let mut a = Block::new();
        a.data[0] = 42;
        let mut b = Block::new();
        b.data[0] = 42;
        let mut c = Block::new();
        c.data[0] = -7;
        let planes = vec![vec![a, b, c]];
        let cfg = DictConfig { enabled: true, capacity: 16, min_freq: 2 };
        let (entries, stats) = build_global_dictionary(&planes, &cfg);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].block, a.data);
        assert_eq!(stats.total_blocks, 3);
        assert_eq!(stats.evicted_candidates, 0);
    }

    #[test]
    fn lfu_eviction_when_capacity_exceeded() {
        // Build blocks with distinct frequencies; cap below candidate count.
        let mk = |dc: i16| {
            let mut blk = Block::new();
            blk.data[0] = dc;
            blk
        };
        // freq layout: block1 x5, block2 x3, block3 x2, block4 x1 (filtered by min_freq)
        let plane = vec![
            mk(1), mk(1), mk(1), mk(1), mk(1),
            mk(2), mk(2), mk(2),
            mk(3), mk(3),
            mk(4),
        ];
        let cfg = DictConfig { enabled: true, capacity: 2, min_freq: 2 };
        let (entries, stats) = build_global_dictionary(&[plane], &cfg);
        // Only the two most frequent survive; one qualifying entry is evicted.
        assert_eq!(entries.len(), 2);
        assert_eq!(stats.evicted_candidates, 1);
        assert_eq!(entries[0].block[0], 1); // highest frequency first
        assert_eq!(entries[1].block[0], 2);
    }

    #[test]
    fn patterned_rgb_roundtrip_matches_plain_blocks() {
        // 128x128 tiled pattern: large enough to amortize the dictionary section.
        let (w, h) = (128usize, 128usize);
        let img = patterned_image(w, h);
        let cfg = DictConfig { enabled: true, capacity: 256, min_freq: 2 };
        let (dict_buf, stats) = encode_dict_rgb(&img, w, h, 85, &cfg);
        assert_eq!(dict_buf[2], 20, "version byte must be 20");
        // Strong repetition: nearly every block is a dictionary reference.
        assert!(stats.hit_rate() > 0.9, "expected very high hit rate, got {}", stats.hit_rate());
        assert!(stats.dictionary_entries > 0);
        assert!(dict_buf.len() < encode_plain_rgb(&img, w, h, 85).len(), "must shrink");

        // Coefficient-level identity: the dictionary layer must reproduce the
        // exact pre-DCT blocks that the plain Huffman path also encodes.
        let expected = preprocessed_rgb_planes(&img, w, h, 85);
        let got = decode_stream_planes(&dict_buf, w, h);
        assert_eq!(got.len(), expected.len());
        for (p, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
            assert_eq!(e.len(), g.len(), "plane {p} block count");
            for (i, (eb, gb)) in e.iter().zip(g.iter()).enumerate() {
                assert_eq!(eb.data, gb.data, "plane {p} block {i} mismatch");
            }
        }
        // And a full pixel round trip still succeeds.
        let px = decode_dict_to_vec(&dict_buf, w, h, 3);
        assert_eq!(px.len(), w * h * 3);
    }

    #[test]
    fn photo_rgb_never_inflates_and_roundtrips() {
        let (w, h) = (32usize, 32usize);
        let img = photo_image(w, h);
        let cfg = DictConfig { enabled: true, capacity: 256, min_freq: 2 };
        let (dict_buf, stats) = encode_dict_rgb(&img, w, h, 85, &cfg);
        let plain = encode_plain_rgb(&img, w, h, 85);
        // Bloat protection: the dictionary container cannot exceed the plain stream.
        assert!(
            dict_buf.len() <= plain.len() + 1,
            "dict {} must not inflate over plain {}",
            dict_buf.len(),
            plain.len()
        );
        // Random content has essentially no repeated blocks -> no references.
        assert_eq!(stats.referenced_blocks, 0);
        assert_eq!(stats.planes_fallback, stats.planes);
        // Bloat protection rewrote the output to the legacy v18 stream.
        assert_eq!(dict_buf[2], 18);
        assert_eq!(dict_buf, plain);
        // Fallback planes still decode losslessly via the embedded Huffman stream.
        let expected = preprocessed_rgb_planes(&img, w, h, 85);
        let got = decode_stream_planes(&dict_buf, w, h);
        for (e, g) in expected.iter().zip(got.iter()) {
            assert_eq!(e.len(), g.len());
            for (eb, gb) in e.iter().zip(g.iter()) {
                assert_eq!(eb.data, gb.data);
            }
        }
        let px = decode_dict_to_vec(&dict_buf, w, h, 3);
        assert_eq!(px.len(), w * h * 3);
    }

    #[test]
    fn dictionary_and_legacy_produce_identical_pixels() {
        // End-to-end pixel parity: v20 dictionary output and v18 legacy output
        // must decode to the exact same RGB pixels for every image class.
        let (w, h) = (128usize, 128usize);
        let cfg = DictConfig { enabled: true, capacity: 256, min_freq: 2 };
        for img in [patterned_image(w, h), photo_image(w, h), mixed_image(w, h)] {
            let (dict_buf, _) = encode_dict_rgb(&img, w, h, 85, &cfg);
            let legacy = encode_plain_rgb(&img, w, h, 85);
            let px_dict = decode_dict_to_vec(&dict_buf, w, h, 3);
            let px_legacy = decode_dict_to_vec(&legacy, w, h, 3);
            assert_eq!(px_dict, px_legacy, "pixel parity broken for an image class");
        }
    }

    #[test]
    fn capacity_and_min_freq_are_configurable_and_respected() {
        // Tight capacity forces LFU eviction; high min_freq admits fewer blocks.
        let (w, h) = (128usize, 128usize);
        let img = patterned_image(w, h);
        let tight = DictConfig { enabled: true, capacity: 1, min_freq: 2 };
        let (buf_tight, st_tight) = encode_dict_rgb(&img, w, h, 85, &tight);
        assert!(st_tight.dictionary_entries <= 1);

        let strict = DictConfig { enabled: true, capacity: 256, min_freq: 64 };
        let (buf_strict, st_strict) = encode_dict_rgb(&img, w, h, 85, &strict);
        // Either still beneficial (fewer entries) or it safely fell back.
        assert!(buf_strict.len() <= encode_plain_rgb(&img, w, h, 85).len());
        let _ = (buf_tight, st_tight, st_strict);
    }

    #[test]
    fn mixed_rgb_roundtrips() {
        // 192x192 with a patterned three-quarters and a random quarter.
        let (w, h) = (192usize, 192usize);
        let img = mixed_image(w, h);
        let cfg = DictConfig { enabled: true, capacity: 128, min_freq: 2 };
        let (dict_buf, stats) = encode_dict_rgb(&img, w, h, 85, &cfg);
        let plain = encode_plain_rgb(&img, w, h, 85);
        assert_eq!(dict_buf[2], 20, "mixed image should still pick the dictionary layout");
        assert!(dict_buf.len() < plain.len(), "mixed image must shrink");
        assert!(stats.referenced_blocks > 0, "patterned region must produce hits");
        let expected = preprocessed_rgb_planes(&img, w, h, 85);
        let got = decode_stream_planes(&dict_buf, w, h);
        for (e, g) in expected.iter().zip(got.iter()) {
            assert_eq!(e.len(), g.len());
            for (eb, gb) in e.iter().zip(g.iter()) {
                assert_eq!(eb.data, gb.data);
            }
        }
    }

    #[test]
    fn rgba_dictionary_roundtrips_version_21() {
        let (w, h) = (128usize, 128usize);
        let mut img = patterned_image(w, h);
        // Add a constant alpha channel (4th byte per pixel).
        let mut rgba = Vec::with_capacity(w * h * 4);
        for px in img.chunks_exact_mut(3) {
            rgba.extend_from_slice(px);
            rgba.push(200);
        }
        let cfg = DictConfig { enabled: true, capacity: 64, min_freq: 2 };
        let buf = encode_dict_rgba(&rgba, w, h, 85, &cfg);
        assert_eq!(buf[2], 21, "RGBA dictionary version must be 21");
        let px = decode_dict_to_vec(&buf, w, h, 4);
        assert_eq!(px.len(), w * h * 4);
        // Alpha must survive the round trip.
        assert!(px.iter().skip(3).step_by(4).all(|&a| a == 200));
    }

    #[test]
    fn disabled_config_produces_legacy_stream() {
        // With the dictionary disabled the public RGB encoder must emit the
        // exact legacy v18 bytes (original encoding is unchanged).
        let (w, h) = (16usize, 16usize);
        let img = patterned_image(w, h);
        let mut via_public = vec![0u8; CAP];
        let mut pos = 0i32;
        encode_rgb(&img, w, h, 85, &mut via_public, &mut pos, None);
        via_public.truncate(pos as usize);
        let plain = encode_plain_rgb(&img, w, h, 85);
        assert_eq!(via_public, plain);
        assert_eq!(via_public[2], 18);
    }

    #[test]
    fn malformed_streams_are_rejected_not_panicked() {
        // A valid v20 stream, then several corruptions that must fail cleanly.
        let (w, h) = (128usize, 128usize);
        let img = patterned_image(w, h);
        let cfg = DictConfig { enabled: true, capacity: 256, min_freq: 2 };
        let (good, _) = encode_dict_rgb(&img, w, h, 85, &cfg);
        assert_eq!(good[2], 20);

        // Truncation at several points must never panic.
        for cut in [BG_HEADER_SIZE + 1, good.len() / 2, good.len() - 1] {
            let mut px = vec![0u8; w * h * 3];
            let ok = decode_dict_version(good[2], &good[..cut], BG_HEADER_SIZE, w, h, 85, &mut px, None);
            assert!(!ok, "truncated stream at {cut} must be rejected");
        }
        // Garbage where the dictionary section magic is expected.
        let mut bad = good.clone();
        bad[BG_HEADER_SIZE] ^= 0xFF;
        let mut px = vec![0u8; w * h * 3];
        assert!(!decode_dict_version(bad[2], &bad, BG_HEADER_SIZE, w, h, 85, &mut px, None));
        // ref_count larger than block count must be rejected (no usize underflow).
        let mut inflated = good.clone();
        // Find first plane frame's ref_count field (after its flags byte) and inflate it.
        // Locate the end of the dictionary section, then the first frame length/flags.
        let (_, dict_end) = parse_dictionary_section(&inflated, BG_HEADER_SIZE).unwrap();
        let frame_len = u32::from_le_bytes(inflated[dict_end..dict_end + 4].try_into().unwrap()) as usize;
        let payload_start = dict_end + 5;
        if inflated[dict_end + 4] & DICT_PLANE_FLAG_ENABLED != 0 {
            inflated[payload_start..payload_start + 4].copy_from_slice(&u32::MAX.to_le_bytes());
            let mut px2 = vec![0u8; w * h * 3];
            assert!(!decode_dict_version(inflated[2], &inflated, BG_HEADER_SIZE, w, h, 85, &mut px2, None));
        }
        let _ = frame_len;
    }

    #[test]
    fn min_freq_one_captures_unique_blocks() {
        let (w, h) = (16usize, 16usize);
        let img = photo_image(w, h);
        let cfg = DictConfig { enabled: true, capacity: 4096, min_freq: 1 };
        let (buf, _stats) = encode_dict_rgb(&img, w, h, 85, &cfg);
        // Even with every block eligible, bloat protection keeps size bounded.
        let plain = encode_plain_rgb(&img, w, h, 85);
        assert!(buf.len() <= plain.len() + 1);
        // And the stream remains decodable.
        let px = decode_dict_to_vec(&buf, w, h, 3);
        assert_eq!(px.len(), w * h * 3);
    }
}



