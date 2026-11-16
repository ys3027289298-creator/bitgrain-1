use crate::bitstream;
use crate::block::Block;
use crate::blockizer::Blockizer;
use crate::colorspace;
use crate::dct;
use crate::entropy;
use crate::huffman;
use crate::zigzag::ZIGZAG;
use rayon::prelude::*;
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
pub const BG_HEADER_SIZE: usize = 3 + 4 + 4 + 1;

const BG_MAGIC_GRAY:    &[u8; 3] = b"BG\x01";
const BG_MAGIC_RGB:     &[u8; 3] = b"BG\x02";
const BG_MAGIC_RGBA:    &[u8; 3] = b"BG\x03";
const BG_MAGIC_YUV420_V8:  &[u8; 3] = b"BG\x12";
const BG_MAGIC_YUV420A_V8: &[u8; 3] = b"BG\x13";

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
    // Release uses the C quantizer; test builds link no C objects, so the
    // scalar integer-division fallback (matching quant.c's scalar path) is used.
    #[cfg(not(test))]
    unsafe { crate::ffi::quantize_block(block.as_mut_ptr(), table.as_ptr()); }

    #[cfg(test)]
    for (v, t) in block.iter_mut().zip(table.iter()) {
        *v = *v / *t;
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
// Legacy RGB/RGBA RLE paths (v2/v3) — kept for backward compat
// ---------------------------------------------------------------------------

pub fn encode_rgb(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
    // Default to the better YCbCr path; callers that need legacy RLE use encode_rgb_rle
    encode_rgb_ycbcr(image, width, height, quality, out, pos, icc);
}

pub fn encode_rgba(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
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

// ===========================================================================
// Grayscale animation: append-style incremental encoding (.bga container)
//
// New frames are compared block-by-block against the previous frame in the
// quantized-DCT-coefficient domain. Frame 0 is a complete standard v1 .bg
// byte stream; every later frame stores only the changed blocks (each as a
// normal RLE block record) preceded by a per-frame changed-block table.
// Decoding patches the previous frame's coefficient blocks, so a delta frame
// reconstructs exactly the pixels of a full per-frame encode of that image.
//
// On-disk layout (all integers little-endian):
//
//   offset  size  field
//   0       3     magic "BGA"
//   3       1     container version (1)
//   4       4     width  (u32)
//   8       4     height (u32)
//   12      1     quality (1-100)
//   13      4     frame_count (u32, must be >= 1)
//   17      8*N   frame table: per frame (payload_offset: u32, payload_len: u32)
//   ...           frame payloads, contiguous in frame order
//
// Frame 0 payload = standard v1 grayscale .bg stream (12-byte BG header +
// RLE blocks), byte-for-byte identical to bitgrain_encode_grayscale output.
//
// Delta frame payload = repeated changed-block records:
//   block_index: u32 (scan order, 0..num_blocks-1)
//   block_len:   u32 (byte length of the RLE record that follows)
//   block bytes: DC(2) + (run,level)* + EOB(3), exactly the v1 RLE encoding
// Unchanged blocks are omitted and inherited from the previous frame.
// ===========================================================================

const BGA_MAGIC: &[u8; 3] = b"BGA";
const BGA_VERSION: u8 = 1;
const BGA_HEADER_SIZE: usize = 3 + 1 + 4 + 4 + 1 + 4; // 17
const BGA_FRAME_RECORD_SIZE: usize = 8;
const BGA_RLE_BLOCK_MAX: usize = 2 + 63 * 3 + 3;

/// Errors returned by the grayscale animation codec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnimError {
    /// Null/empty/size-mismatched argument.
    InvalidArgument,
    /// Output buffer cannot hold the serialized container.
    BufferTooSmall,
    /// Malformed container, table entry, delta record or RLE block.
    BadStream,
    /// Frame 0 is absent or cannot be decoded (delta frames need it).
    MissingFirstFrame,
}

// Quantization/dequantization wrappers. Release builds use the exact same C
// entry points as the single-frame path so outputs stay identical; test
// builds link no C objects, so they use equivalent pure-Rust arithmetic
// (integer division truncates toward zero in both).
#[cfg(not(test))]
#[inline]
fn anim_quantize_block(block: &mut [i16; 64], table: &[i16; 64]) {
    unsafe { crate::ffi::quantize_block(block.as_mut_ptr(), table.as_ptr()) }
}

#[cfg(test)]
#[inline]
fn anim_quantize_block(block: &mut [i16; 64], table: &[i16; 64]) {
    for (v, t) in block.iter_mut().zip(table.iter()) {
        *v = *v / *t;
    }
}

#[cfg(not(test))]
#[inline]
fn anim_dequantize_block(block: &mut Block, table: &[i16; 64]) {
    unsafe { crate::ffi::dequantize_block(block.data.as_mut_ptr(), table.as_ptr()) }
}

#[cfg(test)]
#[inline]
fn anim_dequantize_block(block: &mut Block, table: &[i16; 64]) {
    for (v, t) in block.data.iter_mut().zip(table.iter()) {
        let r = *v as i32 * *t as i32;
        *v = r.clamp(-32768, 32767) as i16;
    }
}

/// Forward DCT + quantization for one grayscale frame, in scan order.
fn anim_frame_coefficients(
    image: &[u8], width: usize, height: usize, table: &[i16; 64],
) -> Vec<Block> {
    let blockizer = Blockizer::new(width, height);
    let mut blocks = blockizer.generate_blocks(image);
    if should_parallel_blocks(blocks.len(), width, height) {
        blocks.par_chunks_mut(BLOCK_TILE_SIZE).for_each(|chunk| {
            for block in chunk.iter_mut() {
                dct::dct(block);
                anim_quantize_block(&mut block.data, table);
            }
        });
    } else {
        for block in blocks.iter_mut() {
            dct::dct(block);
            anim_quantize_block(&mut block.data, table);
        }
    }
    blocks
}

/// Build a standard v1 grayscale .bg payload (header + RLE blocks).
/// Byte-for-byte identical to encode_grayscale() for the same inputs.
fn anim_build_full_payload(
    blocks: &[Block], width: usize, height: usize, quality: u8,
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(BG_HEADER_SIZE + blocks.len() * BGA_RLE_BLOCK_MAX);
    payload.extend_from_slice(BG_MAGIC_GRAY);
    payload.extend_from_slice(&(width as u32).to_le_bytes());
    payload.extend_from_slice(&(height as u32).to_le_bytes());
    payload.push(quality);
    let mut block_buf = [0u8; BGA_RLE_BLOCK_MAX];
    for block in blocks {
        let mut block_pos: i32 = 0;
        entropy::encode_block_to_buffer(block, &mut block_buf, &mut block_pos);
        payload.extend_from_slice(&block_buf[..block_pos as usize]);
    }
    payload
}

/// Build a delta payload: one (index, len, RLE bytes) record per block whose
/// quantized coefficients differ from the previous frame.
fn anim_build_delta_payload(prev: &[Block], current: &[Block]) -> Vec<u8> {
    let mut payload = Vec::new();
    let mut block_buf = [0u8; BGA_RLE_BLOCK_MAX];
    for (idx, (a, b)) in prev.iter().zip(current.iter()).enumerate() {
        if a.data == b.data {
            continue;
        }
        let mut block_pos: i32 = 0;
        entropy::encode_block_to_buffer(b, &mut block_buf, &mut block_pos);
        payload.extend_from_slice(&(idx as u32).to_le_bytes());
        payload.extend_from_slice(&(block_pos as u32).to_le_bytes());
        payload.extend_from_slice(&block_buf[..block_pos as usize]);
    }
    payload
}

/// Encode a sequence of same-size grayscale frames into one .bga container.
///
/// `quality` follows the FFI convention: 0 means default 85, otherwise 1-100.
/// On success `*pos` receives the container length; on error no partial
/// container is committed (`*pos` is set to 0).
pub fn encode_grayscale_animation(
    frames: &[&[u8]],
    width: usize,
    height: usize,
    quality: u8,
    out: &mut [u8],
    pos: &mut i32,
) -> Result<(), AnimError> {
    *pos = 0;
    if frames.is_empty() || width == 0 || height == 0 || out.is_empty() {
        return Err(AnimError::InvalidArgument);
    }
    let frame_pixels = width.checked_mul(height).ok_or(AnimError::InvalidArgument)?;
    let blocks_wide = (width + 7) / 8;
    let blocks_high = (height + 7) / 8;
    let num_blocks = blocks_wide
        .checked_mul(blocks_high)
        .ok_or(AnimError::InvalidArgument)?;
    if frames.len() > u32::MAX as usize || num_blocks > u32::MAX as usize {
        return Err(AnimError::InvalidArgument);
    }
    if frames.iter().any(|f| f.len() < frame_pixels) {
        return Err(AnimError::InvalidArgument);
    }
    let q = if quality == 0 { 85 } else { quality.clamp(1, 100) };
    let table = quant_table_for_quality(q);

    // Frame 0: full payload; later frames: delta payloads against the previous.
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    let mut previous: Option<Vec<Block>> = None;
    for frame in frames {
        let blocks = anim_frame_coefficients(frame, width, height, &table);
        let payload = match &previous {
            None => anim_build_full_payload(&blocks, width, height, q),
            Some(prev) => anim_build_delta_payload(prev, &blocks),
        };
        payloads.push(payload);
        previous = Some(blocks);
    }

    let table_base = BGA_HEADER_SIZE + frames.len() * BGA_FRAME_RECORD_SIZE;
    let payload_total: usize = payloads.iter().map(Vec::len).sum();
    let total = table_base
        .checked_add(payload_total)
        .ok_or(AnimError::BufferTooSmall)?;
    if total > out.len() || total > i32::MAX as usize {
        return Err(AnimError::BufferTooSmall);
    }

    // Container header.
    out[0..3].copy_from_slice(BGA_MAGIC);
    out[3] = BGA_VERSION;
    out[4..8].copy_from_slice(&(width as u32).to_le_bytes());
    out[8..12].copy_from_slice(&(height as u32).to_le_bytes());
    out[12] = q;
    out[13..17].copy_from_slice(&(frames.len() as u32).to_le_bytes());

    // Frame table.
    let mut cursor = BGA_HEADER_SIZE;
    let mut offset = table_base as u32;
    for payload in &payloads {
        out[cursor..cursor + 4].copy_from_slice(&offset.to_le_bytes());
        out[cursor + 4..cursor + 8].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        cursor += BGA_FRAME_RECORD_SIZE;
        offset = offset.checked_add(payload.len() as u32).ok_or(AnimError::BufferTooSmall)?;
    }

    // Payloads.
    for payload in &payloads {
        out[cursor..cursor + payload.len()].copy_from_slice(payload);
        cursor += payload.len();
    }
    *pos = cursor as i32;
    Ok(())
}

/// Read one v1 RLE block record. Mirrors decode_rle_one_block() in decoder.rs.
fn anim_read_rle_block(buf: &[u8], mut pos: usize) -> Result<(Block, usize), AnimError> {
    let mut block = Block::new();
    if pos + 2 > buf.len() {
        return Err(AnimError::BadStream);
    }
    block.data[ZIGZAG[0]] = i16::from_le_bytes([buf[pos], buf[pos + 1]]);
    pos += 2;

    let mut ac_index = 1usize;
    loop {
        if pos + 3 > buf.len() {
            return Err(AnimError::BadStream);
        }
        let run = buf[pos];
        let level = i16::from_le_bytes([buf[pos + 1], buf[pos + 2]]);
        pos += 3;
        if run == 0xFF && level == 0 {
            break;
        }
        for _ in 0..run {
            if ac_index < 64 {
                block.data[ZIGZAG[ac_index]] = 0;
                ac_index += 1;
            }
        }
        if ac_index < 64 {
            block.data[ZIGZAG[ac_index]] = level;
            ac_index += 1;
        }
    }
    Ok((block, pos))
}

/// Dequantize + IDCT one block and write its pixels into a flat plane.
fn anim_emit_block_pixels(
    block: &mut Block, plane: &mut [u8],
    width: usize, height: usize, blocks_wide: usize, block_index: usize,
    table: &[i16; 64],
) {
    anim_dequantize_block(block, table);
    dct::idct(block);
    let bx = (block_index % blocks_wide) * 8;
    let by = (block_index / blocks_wide) * 8;
    for y in 0..8 {
        for x in 0..8 {
            let py = by + y;
            let px = bx + x;
            if py < height && px < width {
                plane[py * width + px] =
                    (block.data[y * 8 + x] + 128).clamp(0, 255) as u8;
            }
        }
    }
}

/// Decode a .bga container produced by encode_grayscale_animation().
///
/// `frames_out` is cleared and receives one w*h grayscale buffer per frame,
/// in frame order. Returns AnimError::MissingFirstFrame when frame 0 is
/// absent or unreadable, since delta frames can only be rebuilt on top of it.
pub fn decode_grayscale_animation(
    stream: &[u8],
    out_width: &mut u32,
    out_height: &mut u32,
    frames_out: &mut Vec<Vec<u8>>,
) -> Result<(), AnimError> {
    frames_out.clear();
    if stream.len() < BGA_HEADER_SIZE || &stream[0..3] != BGA_MAGIC {
        return Err(AnimError::BadStream);
    }
    if stream[3] != BGA_VERSION {
        return Err(AnimError::BadStream);
    }
    let width = u32::from_le_bytes(stream[4..8].try_into().unwrap()) as usize;
    let height = u32::from_le_bytes(stream[8..12].try_into().unwrap()) as usize;
    let quality = stream[12];
    let frame_count = u32::from_le_bytes(stream[13..17].try_into().unwrap()) as usize;
    if width == 0 || height == 0 || frame_count == 0 {
        // No frame 0 to anchor reconstruction.
        return Err(AnimError::MissingFirstFrame);
    }
    let q = if quality == 0 { 50 } else { quality };
    let table = quant_table_for_quality(q);
    let blocks_wide = (width + 7) / 8;
    let num_blocks = blocks_wide * ((height + 7) / 8);
    let frame_pixels = width
        .checked_mul(height)
        .ok_or(AnimError::BadStream)?;

    let table_base = BGA_HEADER_SIZE
        .checked_add(frame_count.checked_mul(BGA_FRAME_RECORD_SIZE).ok_or(AnimError::BadStream)?)
        .ok_or(AnimError::BadStream)?;
    if table_base > stream.len() {
        return Err(AnimError::BadStream);
    }

    // Validate the whole frame table up front: bounded, in-range, contiguous.
    let mut entries: Vec<(usize, usize)> = Vec::with_capacity(frame_count);
    let mut expected_offset = table_base;
    for i in 0..frame_count {
        let rec = BGA_HEADER_SIZE + i * BGA_FRAME_RECORD_SIZE;
        let offset = u32::from_le_bytes(stream[rec..rec + 4].try_into().unwrap()) as usize;
        let len = u32::from_le_bytes(stream[rec + 4..rec + 8].try_into().unwrap()) as usize;
        if offset != expected_offset {
            return Err(AnimError::BadStream);
        }
        let end = offset.checked_add(len).ok_or(AnimError::BadStream)?;
        if end > stream.len() {
            return Err(if i == 0 { AnimError::MissingFirstFrame } else { AnimError::BadStream });
        }
        entries.push((offset, len));
        expected_offset = end;
    }

    // ---- Frame 0: complete v1 .bg payload ----
    let (f0_off, f0_len) = entries[0];
    let f0 = &stream[f0_off..f0_off + f0_len];
    if f0_len < BG_HEADER_SIZE || &f0[0..2] != b"BG" || f0[2] != 1 {
        return Err(AnimError::MissingFirstFrame);
    }
    let f0_width = u32::from_le_bytes(f0[3..7].try_into().unwrap()) as usize;
    let f0_height = u32::from_le_bytes(f0[7..11].try_into().unwrap()) as usize;
    if f0_width != width || f0_height != height {
        return Err(AnimError::MissingFirstFrame);
    }
    let mut coefficient_blocks: Vec<Block> = Vec::with_capacity(num_blocks);
    let mut rle_pos = BG_HEADER_SIZE;
    for _ in 0..num_blocks {
        let (block, next) = anim_read_rle_block(f0, rle_pos)
            .map_err(|_| AnimError::MissingFirstFrame)?;
        rle_pos = next;
        coefficient_blocks.push(block);
    }
    if rle_pos != f0_len {
        return Err(AnimError::MissingFirstFrame);
    }

    let mut current_pixels = vec![0u8; frame_pixels];
    for (idx, block) in coefficient_blocks.iter_mut().enumerate() {
        anim_emit_block_pixels(block, &mut current_pixels, width, height, blocks_wide, idx, &table);
    }
    *out_width = width as u32;
    *out_height = height as u32;
    frames_out.push(current_pixels.clone());

    // ---- Delta frames: patch inherited coefficient blocks ----
    for (frame_idx, &(off, len)) in entries.iter().enumerate().skip(1) {
        let payload = &stream[off..off + len];
        let mut p = 0usize;
        while p < len {
            if p + BGA_FRAME_RECORD_SIZE > len {
                return Err(AnimError::BadStream);
            }
            let block_index =
                u32::from_le_bytes(payload[p..p + 4].try_into().unwrap()) as usize;
            let block_len =
                u32::from_le_bytes(payload[p + 4..p + 8].try_into().unwrap()) as usize;
            p += BGA_FRAME_RECORD_SIZE;
            if block_index >= num_blocks || block_len < 5 || p + block_len > len {
                return Err(AnimError::BadStream);
            }
            let (block, next) = anim_read_rle_block(&payload[..p + block_len], p)?;
            if next != p + block_len {
                return Err(AnimError::BadStream);
            }
            p = next;
            coefficient_blocks[block_index] = block;
            let mut patched = coefficient_blocks[block_index];
            anim_emit_block_pixels(
                &mut patched, &mut current_pixels, width, height,
                blocks_wide, block_index, &table,
            );
            coefficient_blocks[block_index] = patched;
        }
        if p != len {
            return Err(AnimError::BadStream);
        }
        frames_out.push(current_pixels.clone());
        let _ = frame_idx;
    }

    Ok(())
}

#[cfg(test)]
mod animation_tests {
    use super::*;

    /// Independent reference: decode a full v1 .bg payload straight into pixels.
    fn reference_full_decode(payload: &[u8], width: usize, height: usize, q: u8) -> Vec<u8> {
        assert_eq!(&payload[0..2], b"BG");
        assert_eq!(payload[2], 1);
        let table = quant_table_for_quality(q);
        let blocks_wide = (width + 7) / 8;
        let num_blocks = blocks_wide * ((height + 7) / 8);
        let mut pos = BG_HEADER_SIZE;
        let mut pixels = vec![0u8; width * height];
        for idx in 0..num_blocks {
            let (mut block, next) = anim_read_rle_block(payload, pos).unwrap();
            pos = next;
            anim_emit_block_pixels(&mut block, &mut pixels, width, height, blocks_wide, idx, &table);
        }
        assert_eq!(pos, payload.len());
        pixels
    }

    /// Three small 16x16 frames (4 blocks). Frame 1 changes block 1 only;
    /// frame 2 keeps that change and changes block 2 too.
    fn sample_frames() -> Vec<Vec<u8>> {
        let (w, h) = (16usize, 16usize);
        let base: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                ((x * 17 + y * 13) & 0xFF) as u8
            })
            .collect();
        let paint = |frame: &mut Vec<u8>, bx: usize, by: usize, value: u8| {
            for y in by..by + 8 {
                for x in bx..bx + 8 {
                    frame[y * 16 + x] = value;
                }
            }
        };
        let mut f1 = base.clone();
        paint(&mut f1, 8, 0, 200);
        let mut f2 = f1.clone();
        paint(&mut f2, 0, 8, 40);
        vec![base, f1, f2]
    }

    #[test]
    fn frame0_payload_is_identical_to_single_frame_encode() {
        let frames = sample_frames();
        let (w, h, q) = (16usize, 16usize, 85u8);
        let table = quant_table_for_quality(q);
        let blocks = anim_frame_coefficients(&frames[0], w, h, &table);

        // Full payload built by the animation path.
        let anim_payload = anim_build_full_payload(&blocks, w, h, q);

        // Full payload built through the existing public single-frame encoder.
        let mut single = vec![0u8; 4096];
        let mut single_pos: i32 = 0;
        encode_grayscale(&frames[0], w, h, q, &mut single, &mut single_pos);
        single.truncate(single_pos as usize);

        assert_eq!(anim_payload, single);
    }

    #[test]
    fn three_frame_roundtrip_matches_full_per_frame_encoding() {
        let frames = sample_frames();
        let (w, h, q) = (16usize, 16usize, 85u8);
        let refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();

        let mut container = vec![0u8; 1 << 16];
        let mut pos: i32 = 0;
        encode_grayscale_animation(&refs, w, h, q, &mut container, &mut pos).unwrap();
        container.truncate(pos as usize);

        let mut got_width = 0u32;
        let mut got_height = 0u32;
        let mut decoded: Vec<Vec<u8>> = Vec::new();
        decode_grayscale_animation(&container, &mut got_width, &mut got_height, &mut decoded)
            .unwrap();
        assert_eq!(got_width as usize, w);
        assert_eq!(got_height as usize, h);
        assert_eq!(decoded.len(), 3);

        let table = quant_table_for_quality(q);
        for (i, frame) in frames.iter().enumerate() {
            // Independent full encode + decode of this same frame.
            let blocks = anim_frame_coefficients(frame, w, h, &table);
            let full_payload = anim_build_full_payload(&blocks, w, h, q);
            let reference = reference_full_decode(&full_payload, w, h, q);
            assert_eq!(
                decoded[i], reference,
                "frame {i} differs from full per-frame encoding"
            );
        }
    }

    #[test]
    fn delta_payloads_contain_only_intended_changed_blocks() {
        let frames = sample_frames();
        let (w, h, q) = (16usize, 16usize, 85u8);
        let table = quant_table_for_quality(q);
        let b0 = anim_frame_coefficients(&frames[0], w, h, &table);
        let b1 = anim_frame_coefficients(&frames[1], w, h, &table);
        let b2 = anim_frame_coefficients(&frames[2], w, h, &table);

        let d1 = anim_build_delta_payload(&b0, &b1);
        let d2 = anim_build_delta_payload(&b1, &b2);

        let changed = |payload: &[u8]| -> Vec<u32> {
            let mut indices = Vec::new();
            let mut p = 0;
            while p < payload.len() {
                let idx = u32::from_le_bytes(payload[p..p + 4].try_into().unwrap());
                let len = u32::from_le_bytes(payload[p + 4..p + 8].try_into().unwrap()) as usize;
                indices.push(idx);
                p += 8 + len;
            }
            assert_eq!(p, payload.len());
            indices
        };
        // 16x16 => blocks: [0 (top-left), 1 (top-right), 2 (bottom-left), 3].
        assert_eq!(changed(&d1), vec![1]);
        assert_eq!(changed(&d2), vec![2]);

        // Sanity: exactly those coefficient blocks actually differ.
        assert!(b0[0].data == b1[0].data && b0[2].data == b1[2].data && b0[3].data == b1[3].data);
        assert!(b1[0].data == b2[0].data && b1[1].data == b2[1].data && b1[3].data == b2[3].data);
    }

    #[test]
    fn animation_is_smaller_than_three_full_encodes() {
        let frames = sample_frames();
        let (w, h, q) = (16usize, 16usize, 85u8);
        let refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();

        let mut container = vec![0u8; 1 << 16];
        let mut pos: i32 = 0;
        encode_grayscale_animation(&refs, w, h, q, &mut container, &mut pos).unwrap();
        let anim_size = pos as usize;

        let table = quant_table_for_quality(q);
        let mut full_total = 0usize;
        let mut full_sizes = Vec::new();
        for frame in &frames {
            let blocks = anim_frame_coefficients(frame, w, h, &table);
            let payload = anim_build_full_payload(&blocks, w, h, q);
            full_sizes.push(payload.len());
            full_total += payload.len();
        }

        // Per-frame payload sizes straight from the container table.
        let frame_count =
            u32::from_le_bytes(container[13..17].try_into().unwrap()) as usize;
        let table_base = BGA_HEADER_SIZE + frame_count * BGA_FRAME_RECORD_SIZE;
        let mut anim_frame_sizes = Vec::new();
        for i in 0..frame_count {
            let rec = BGA_HEADER_SIZE + i * BGA_FRAME_RECORD_SIZE;
            let len = u32::from_le_bytes(container[rec + 4..rec + 8].try_into().unwrap()) as usize;
            anim_frame_sizes.push(len);
        }
        assert_eq!(anim_frame_sizes[0], full_sizes[0]);
        assert!(anim_frame_sizes[1] < full_sizes[1]);
        assert!(anim_frame_sizes[2] < full_sizes[2]);

        println!("--- 16x16 grayscale, 3 frames, quality {q} ---");
        println!("full encode per frame (bytes): {full_sizes:?}");
        println!("full encodes total:            {full_total} bytes");
        println!("bga frame payloads (bytes):    {anim_frame_sizes:?}");
        println!("bga container total:           {anim_size} bytes (table base {table_base})");
        println!(
            "saving vs full encodes:        {} bytes ({:.1}%)",
            full_total - anim_size,
            100.0 * (full_total - anim_size) as f64 / full_total as f64
        );
        assert!(anim_size < full_total);
    }

    #[test]
    fn missing_or_corrupt_first_frame_is_reported() {
        let frames = sample_frames();
        let (w, h, q) = (16usize, 16usize, 85u8);
        let refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();
        let mut container = vec![0u8; 1 << 16];
        let mut pos: i32 = 0;
        encode_grayscale_animation(&refs, w, h, q, &mut container, &mut pos).unwrap();

        // frame_count = 0: no first frame at all.
        let mut no_frames = container.clone();
        no_frames[13..17].copy_from_slice(&0u32.to_le_bytes());
        let mut out = Vec::new();
        let err = decode_grayscale_animation(&no_frames, &mut 0, &mut 0, &mut out).unwrap_err();
        assert_eq!(err, AnimError::MissingFirstFrame);

        // Truncated stream: header + table intact, but frame 0's payload bytes
        // are gone, so the first frame cannot be recovered.
        let table_base = BGA_HEADER_SIZE + 3 * BGA_FRAME_RECORD_SIZE;
        let truncated = container[..table_base].to_vec();
        let err = decode_grayscale_animation(&truncated, &mut 0, &mut 0, &mut out).unwrap_err();
        assert_eq!(err, AnimError::MissingFirstFrame);

        // Frame-0 payload with a bad inner magic.
        let mut bad_magic = container.clone();
        let f0_off = u32::from_le_bytes(
            bad_magic[BGA_HEADER_SIZE..BGA_HEADER_SIZE + 4].try_into().unwrap(),
        ) as usize;
        bad_magic[f0_off] = b'X';
        let err = decode_grayscale_animation(&bad_magic, &mut 0, &mut 0, &mut out).unwrap_err();
        assert_eq!(err, AnimError::MissingFirstFrame);
    }

    #[test]
    fn corrupt_delta_record_is_reported() {
        let frames = sample_frames();
        let (w, h, q) = (16usize, 16usize, 85u8);
        let refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();
        let mut container = vec![0u8; 1 << 16];
        let mut pos: i32 = 0;
        encode_grayscale_animation(&refs, w, h, q, &mut container, &mut pos).unwrap();

        // Frame 1 table entry: point its block index out of range.
        let f1_rec = BGA_HEADER_SIZE + BGA_FRAME_RECORD_SIZE;
        let f1_off = u32::from_le_bytes(container[f1_rec..f1_rec + 4].try_into().unwrap()) as usize;
        container[f1_off..f1_off + 4].copy_from_slice(&999u32.to_le_bytes());
        let mut out = Vec::new();
        let err = decode_grayscale_animation(&container, &mut 0, &mut 0, &mut out).unwrap_err();
        assert_eq!(err, AnimError::BadStream);
    }

    #[test]
    fn unchanged_frames_produce_empty_deltas_and_still_roundtrip() {
        let frames = sample_frames();
        let (w, h, q) = (16usize, 16usize, 85u8);
        let repeated = vec![frames[0].clone(), frames[0].clone(), frames[0].clone()];
        let refs: Vec<&[u8]> = repeated.iter().map(Vec::as_slice).collect();

        let mut container = vec![0u8; 1 << 16];
        let mut pos: i32 = 0;
        encode_grayscale_animation(&refs, w, h, q, &mut container, &mut pos).unwrap();
        container.truncate(pos as usize);

        let frame_count =
            u32::from_le_bytes(container[13..17].try_into().unwrap()) as usize;
        for i in 1..frame_count {
            let rec = BGA_HEADER_SIZE + i * BGA_FRAME_RECORD_SIZE;
            let len = u32::from_le_bytes(container[rec + 4..rec + 8].try_into().unwrap());
            assert_eq!(len, 0, "unchanged frame {i} must have an empty delta payload");
        }

        let mut decoded = Vec::new();
        decode_grayscale_animation(&container, &mut 0, &mut 0, &mut decoded).unwrap();
        assert_eq!(decoded.len(), 3);
        assert!(decoded.iter().all(|f| *f == decoded[0]));
    }

    #[test]
    fn encoding_is_deterministic_and_validates_arguments() {
        let frames = sample_frames();
        let (w, h, q) = (16usize, 16usize, 85u8);
        let refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();

        let mut a = vec![0u8; 1 << 16];
        let mut pa: i32 = 0;
        encode_grayscale_animation(&refs, w, h, q, &mut a, &mut pa).unwrap();
        let mut b = vec![0u8; 1 << 16];
        let mut pb: i32 = 0;
        encode_grayscale_animation(&refs, w, h, q, &mut b, &mut pb).unwrap();
        assert_eq!(pa, pb);
        assert_eq!(a[..pa as usize], b[..pb as usize]);

        let mut pos = 0i32;
        assert_eq!(
            encode_grayscale_animation(&[], w, h, q, &mut a, &mut pos).unwrap_err(),
            AnimError::InvalidArgument
        );
        let mut tiny = vec![0u8; 4];
        assert_eq!(
            encode_grayscale_animation(&refs, w, h, q, &mut tiny, &mut pos).unwrap_err(),
            AnimError::BufferTooSmall
        );
        let short_frame = vec![vec![0u8; w * h - 1]];
        let short_refs: Vec<&[u8]> = short_frame.iter().map(Vec::as_slice).collect();
        assert_eq!(
            encode_grayscale_animation(&short_refs, w, h, q, &mut a, &mut pos).unwrap_err(),
            AnimError::InvalidArgument
        );
    }

    #[test]
    fn non_multiple_of_eight_dimensions_roundtrip() {
        // 10x6 => 2x1 blocks, exercising border replication.
        let (w, h, q) = (10usize, 6usize, 85u8);
        let f0: Vec<u8> = (0..w * h).map(|i| (i * 9 % 256) as u8).collect();
        let mut f1 = f0.clone();
        for x in 8..10 {
            f1[x] = 230;
        }
        let frames = vec![f0, f1];
        let refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();

        let mut container = vec![0u8; 1 << 16];
        let mut pos: i32 = 0;
        encode_grayscale_animation(&refs, w, h, q, &mut container, &mut pos).unwrap();
        container.truncate(pos as usize);

        let mut decoded = Vec::new();
        decode_grayscale_animation(&container, &mut 0, &mut 0, &mut decoded).unwrap();
        let table = quant_table_for_quality(q);
        for (i, frame) in frames.iter().enumerate() {
            let blocks = anim_frame_coefficients(frame, w, h, &table);
            let payload = anim_build_full_payload(&blocks, w, h, q);
            assert_eq!(decoded[i], reference_full_decode(&payload, w, h, q));
        }
    }
}

