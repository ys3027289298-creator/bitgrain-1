use crate::bitstream;
use crate::block::Block;
use crate::blockizer::Blockizer;
use crate::colorspace;
use crate::dct;
use crate::entropy;
use crate::ffi::quantize_block;
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
    unsafe { quantize_block(block.as_mut_ptr(), table.as_ptr()); }
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
// 16-bit input support (module `bitdepth16`)
// ---------------------------------------------------------------------------
// The legacy pipeline above is hard-wired to 8-bit samples: u8 planes, i16
// blocks centered at 128, i16 quant tables and the JPEG Huffman alphabet
// (DC category <= 11, AC category <= 10). This module adds a fully
// self-contained 16-bit path without touching any of that code:
//
//   * samples are u16 (0..=65535), blocks are i32 centered at 32768;
//   * DCT/IDCT run in f64 over i32 coefficients (same DCT-II definition as
//     `dct::dct_reference`, widened so 16-bit ranges cannot overflow);
//   * quantization tables are computed over the 16-bit range:
//       Keep16 -> same step sizes as the 8-bit tables, full 16-bit fidelity;
//       Down8  -> tables scaled x256 plus 8-bit-visibility sparsification,
//                 so the stream only keeps detail an 8-bit output can show;
//   * an extended fixed-length Huffman alphabet carries i32 levels
//     (DC category <= 18, AC category <= 19);
//   * two new .bg profiles reuse the 12-byte header plus a 1-byte flags
//     trailer: version 20 = YCbCr 4:2:0 (or grayscale), version 21 = +alpha.
//
// The 8-bit functions above are not modified or called differently, so their
// byte output is unchanged. Everything here is additive.
// ===========================================================================

pub mod bitdepth16 {
    use crate::encoder::{
        chroma_quant_table_for_quality_perceptual_v4, quant_table_for_quality_perceptual_v4,
    };
    use crate::zigzag::ZIGZAG;
    use std::f64::consts::PI;
    use std::fmt;

    /// Output depth for a 16-bit encode.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum OutputMode {
        /// Preserve the full 16-bit range in the reconstructed stream.
        Keep16,
        /// Internally process at 16 bits, then quantize/downshift to 8 bits.
        Down8,
    }

    /// Supported input bit depths.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BitDepth {
        B8,
        B16,
    }

    /// Errors reported by the 16-bit API. Bit-depth and quality violations are
    /// rejected up front instead of being silently clamped.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum BitgrainError {
        /// Only 8 and 16 bit inputs are legal; carries the offending value.
        InvalidBitDepth(u32),
        /// Quality must be 0 (default 85) or 1..=100; carries the offending value.
        InvalidQuality(u8),
        /// Zero/oversized dimensions or a buffer whose length does not match
        /// width * height * channels.
        InvalidDimensions,
        /// A .bg v20/v21 stream failed structural validation.
        DecodeFailed,
    }

    impl fmt::Display for BitgrainError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                BitgrainError::InvalidBitDepth(b) => {
                    write!(f, "invalid bit depth {b}: only 8 and 16 are supported")
                }
                BitgrainError::InvalidQuality(q) => {
                    write!(f, "invalid quality {q}: expected 0 (default) or 1..=100")
                }
                BitgrainError::InvalidDimensions => {
                    write!(f, "invalid image dimensions or buffer length mismatch")
                }
                BitgrainError::DecodeFailed => write!(f, "malformed 16-bit .bg stream"),
            }
        }
    }

    impl std::error::Error for BitgrainError {}

    pub type BgResult<T> = Result<T, BitgrainError>;

    /// Reject any bit depth other than 8 or 16.
    pub fn validate_bit_depth(bits: u32) -> BgResult<BitDepth> {
        match bits {
            8 => Ok(BitDepth::B8),
            16 => Ok(BitDepth::B16),
            other => Err(BitgrainError::InvalidBitDepth(other)),
        }
    }

    fn normalize_quality(quality: u8) -> BgResult<u8> {
        match quality {
            0 => Ok(85),
            1..=100 => Ok(quality),
            other => Err(BitgrainError::InvalidQuality(other)),
        }
    }

    fn check_dims(w: usize, h: usize, channels: usize, len: usize) -> BgResult<()> {
        if w == 0 || h == 0 || w > 65536 || h > 65536 {
            return Err(BitgrainError::InvalidDimensions);
        }
        let need = w
            .checked_mul(h)
            .and_then(|n| n.checked_mul(channels))
            .ok_or(BitgrainError::InvalidDimensions)?;
        if len != need {
            return Err(BitgrainError::InvalidDimensions);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Header / profile constants
    // -----------------------------------------------------------------------

    const MAGIC: [u8; 2] = *b"BG";
    const VERSION_RGB_OR_GRAY: u8 = 20;
    const VERSION_RGBA: u8 = 21;
    const FLAG_KEEP16: u8 = 0x01;
    const FLAG_GRAY: u8 = 0x02;
    const HEADER_LEN: usize = 13; // 3 magic/version + 4 w + 4 h + 1 quality + 1 flags

    // -----------------------------------------------------------------------
    // i32 block + widened DCT/IDCT (same separable DCT-II as dct::dct_reference)
    // -----------------------------------------------------------------------

    #[derive(Clone, Copy)]
    pub(crate) struct Block32 {
        pub(crate) data: [i32; 64],
    }

    impl Block32 {
        fn zero() -> Self {
            Self { data: [0i32; 64] }
        }
    }

    pub(crate) fn dct_i32(block: &[i32; 64]) -> [i32; 64] {
        let mut result = [0f64; 64];
        for u in 0..8 {
            for v in 0..8 {
                let mut sum = 0.0f64;
                for x in 0..8 {
                    for y in 0..8 {
                        let pixel = block[y * 8 + x] as f64;
                        sum += pixel
                            * ((2 * x + 1) as f64 * u as f64 * PI / 16.0).cos()
                            * ((2 * y + 1) as f64 * v as f64 * PI / 16.0).cos();
                    }
                }
                let cu = if u == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                let cv = if v == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                result[v * 8 + u] = 0.25 * cu * cv * sum;
            }
        }
        let mut out = [0i32; 64];
        for i in 0..64 {
            out[i] = result[i].round_ties_even() as i64 as i32;
        }
        out
    }

    pub(crate) fn idct_f64(coef: &[f64; 64]) -> [i32; 64] {
        let mut result = [0f64; 64];
        for y in 0..8 {
            for x in 0..8 {
                let mut sum = 0.0f64;
                for u in 0..8 {
                    for v in 0..8 {
                        let c = coef[v * 8 + u];
                        let cu = if u == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                        let cv = if v == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                        sum += cu * cv * c
                            * ((2 * x + 1) as f64 * u as f64 * PI / 16.0).cos()
                            * ((2 * y + 1) as f64 * v as f64 * PI / 16.0).cos();
                    }
                }
                result[y * 8 + x] = 0.25 * sum;
            }
        }
        let mut out = [0i32; 64];
        for i in 0..64 {
            let r = result[i].round_ties_even() as i64;
            out[i] = r.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
        out
    }

    // -----------------------------------------------------------------------
    // 16-bit-range quantization tables
    // -----------------------------------------------------------------------

    // Perceptual v4 weights (see scale_quant_table_perceptual_v4 in this file).
    const LUMA_W4: [i32; 64] = [
        106, 112, 120, 130, 142, 156, 172, 190,
        112, 120, 130, 142, 156, 172, 190, 210,
        120, 130, 142, 156, 172, 190, 210, 232,
        130, 142, 156, 172, 190, 210, 232, 255,
        142, 156, 172, 190, 210, 232, 255, 255,
        156, 172, 190, 210, 232, 255, 255, 255,
        172, 190, 210, 232, 255, 255, 255, 255,
        190, 210, 232, 255, 255, 255, 255, 255,
    ];
    const CHROMA_W4: [i32; 64] = [
        170, 184, 200, 218, 238, 255, 255, 255,
        184, 200, 218, 238, 255, 255, 255, 255,
        200, 218, 238, 255, 255, 255, 255, 255,
        218, 238, 255, 255, 255, 255, 255, 255,
        238, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
    ];

    /// Build a 16-bit quantization table.
    ///
    /// `Keep16` divides the 8-bit perceptual-v4 steps by 16. DCT coefficient
    /// magnitude scales linearly with sample range, and 16-bit samples span
    /// 256x the 8-bit range; applying the *same* per-level precision as the
    /// 8-bit path therefore requires steps scaled by 256, while steps/16 keep
    /// roughly 12 bits of effective precision — genuinely 16-bit-class fidelity
    /// with practical entropy (step 1 would be fully lossless but verbose).
    ///
    /// `Down8` applies the v4 perceptual weights and scales every step by 256,
    /// making quantization strength equivalent to the 8-bit path; the
    /// visibility sparsification applied afterwards then removes what an 8-bit
    /// sample cannot show.
    pub(crate) fn quant_table_16(quality: u8, mode: OutputMode, is_chroma: bool) -> [i32; 64] {
        let base8 = if is_chroma {
            chroma_quant_table_for_quality_perceptual_v4(quality)
        } else {
            quant_table_for_quality_perceptual_v4(quality)
        };
        let mut out = [0i32; 64];
        match mode {
            OutputMode::Keep16 => {
                for i in 0..64 {
                    // Round the /16 so small steps collapse to 1, never 0.
                    out[i] = ((base8[i] as i32 + 8) / 16).max(1);
                }
            }
            OutputMode::Down8 => {
                let weights = if is_chroma { &CHROMA_W4 } else { &LUMA_W4 };
                for i in 0..64 {
                    let mut v = (base8[i] as i32 * weights[i] + 50) / 100;
                    // Same medium/low-quality pressure as scale_quant_table_perceptual_v4.
                    if quality <= 95 {
                        if is_chroma {
                            v = (v * 118 + 50) / 100;
                        } else if i >= 20 {
                            v = (v * 109 + 50) / 100;
                        }
                    }
                    out[i] = v.clamp(1, 255) * 256;
                }
            }
        }
        out
    }

    /// Round-to-nearest, ties away from zero, integer division by a positive table.
    #[inline]
    fn divide_quant(v: i32, q: i32) -> i32 {
        let q = q.max(1);
        let quot = v / q;
        let rem = v % q;
        if rem.abs() * 2 >= q {
            quot + if v >= 0 { 1 } else { -1 }
        } else {
            quot
        }
    }

    // Absolute coefficient bounds matched to the extended Huffman alphabet
    // (DC category <= 18, AC category <= 19). Natural 16-bit signals stay well
    // inside these limits; the clamp only guards pathological constructed input.
    const DC_COEF_MAX: i32 = (1i32 << 18) - 1;
    const AC_COEF_MAX: i32 = (1i32 << 19) - 1;

    fn clamp_block16_coeffs(block: &mut Block32) {
        block.data[ZIGZAG[0]] = block.data[ZIGZAG[0]].clamp(-DC_COEF_MAX, DC_COEF_MAX);
        for zi in 1..64 {
            let z = ZIGZAG[zi];
            block.data[z] = block.data[z].clamp(-AC_COEF_MAX, AC_COEF_MAX);
        }
    }

    pub(crate) fn quantize_block32(block: &mut Block32, table: &[i32; 64]) {
        for i in 0..64 {
            block.data[i] = divide_quant(block.data[i], table[i]);
        }
        clamp_block16_coeffs(block);
    }

    // JPEG coefficient clamp, expressed in 8-bit-domain levels (Down8 only).
    const DC_LEVEL_MAX: i32 = 2047;
    const AC_LEVEL_MAX: i32 = 1023;

    /// Down8 sparsification: drop levels whose reconstructed magnitude stays
    /// below one 8-bit code value after the x256 table scaling. The base
    /// thresholds mirror `build_sparsify_thresholds` in the 8-bit encoder.
    fn build_sparsify_thresholds_8(quality: u8, is_chroma: bool) -> [i32; 64] {
        let q = quality.clamp(1, 100);
        let base_t: i32 = if is_chroma {
            if q >= 90 { 2 } else if q >= 80 { 3 } else { 4 }
        } else if q >= 90 {
            1
        } else if q >= 80 {
            2
        } else {
            3
        };
        let mut thr = [0i32; 64];
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

    fn sparsify_and_clamp_block(block: &mut Block32, thresholds: &[i32; 64]) {
        block.data[ZIGZAG[0]] = block.data[ZIGZAG[0]].clamp(-DC_LEVEL_MAX, DC_LEVEL_MAX);
        for zi in 1..64 {
            let z = ZIGZAG[zi];
            let v = block.data[z];
            if v.abs() <= thresholds[zi] {
                block.data[z] = 0;
            } else {
                block.data[z] = v.clamp(-AC_LEVEL_MAX, AC_LEVEL_MAX);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Blockizer (edge replication, center 32768) and 16-bit color space
    // -----------------------------------------------------------------------

    pub(crate) fn blocks_from_plane(plane: &[u16], w: usize, h: usize) -> Vec<Block32> {
        let blocks_wide = (w + 7) / 8;
        let blocks_high = (h + 7) / 8;
        let num_blocks = blocks_wide * blocks_high;
        let mut blocks = Vec::with_capacity(num_blocks);
        for idx in 0..num_blocks {
            let bx = (idx % blocks_wide) * 8;
            let by = (idx / blocks_wide) * 8;
            let mut block = Block32::zero();
            for y in 0..8 {
                let iy = (by + y).min(h - 1);
                let row_base = iy * w;
                for x in 0..8 {
                    let ix = (bx + x).min(w - 1);
                    block.data[y * 8 + x] = plane[row_base + ix] as i32 - 32768;
                }
            }
            blocks.push(block);
        }
        blocks
    }

    #[inline]
    fn clamp_u16(v: i64) -> u16 {
        v.clamp(0, 65535) as u16
    }

    /// Interleaved RGB16 -> (Y full res, Cb/Cr at 4:2:0). BT.601 full range,
    /// fixed point, 2x2 box-average chroma — the 16-bit widening of
    /// `colorspace::rgb_to_ycbcr420`.
    pub(crate) fn rgb16_to_ycbcr420(image: &[u16], w: usize, h: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let npix = w * h;
        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let mut y_plane = vec![0u16; npix];
        let mut cb_plane = vec![0u16; cw * ch];
        let mut cr_plane = vec![0u16; cw * ch];

        for p in 0..npix {
            let base = p * 3;
            let (r, g, b) = (image[base] as i64, image[base + 1] as i64, image[base + 2] as i64);
            let y = (19595 * r + 38470 * g + 7471 * b + 32768) >> 16;
            y_plane[p] = clamp_u16(y);
        }
        for cy in 0..ch {
            for cx in 0..cw {
                let (mut sum_cb, mut sum_cr, mut count) = (0i64, 0i64, 0i64);
                for dy in 0..2usize {
                    for dx in 0..2usize {
                        let px = cx * 2 + dx;
                        let py = cy * 2 + dy;
                        if px < w && py < h {
                            let base = (py * w + px) * 3;
                            let r = image[base] as i64;
                            let g = image[base + 1] as i64;
                            let b = image[base + 2] as i64;
                            let cb_c = ((-11058 * r - 21709 * g + 32768 * b + 32768) >> 16) + 32768;
                            let cr_c = ((32768 * r - 27439 * g - 5329 * b + 32768) >> 16) + 32768;
                            sum_cb += cb_c;
                            sum_cr += cr_c;
                            count += 1;
                        }
                    }
                }
                let off = cy * cw + cx;
                cb_plane[off] = clamp_u16((sum_cb + count / 2) / count);
                cr_plane[off] = clamp_u16((sum_cr + count / 2) / count);
            }
        }
        (y_plane, cb_plane, cr_plane)
    }

    pub(crate) fn rgba16_to_ycbcr420a(
        image: &[u16],
        w: usize,
        h: usize,
    ) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let npix = w * h;
        let mut a_plane = vec![0u16; npix];
        let mut rgb = Vec::with_capacity(npix * 3);
        for p in 0..npix {
            let base = p * 4;
            rgb.push(image[base]);
            rgb.push(image[base + 1]);
            rgb.push(image[base + 2]);
            a_plane[p] = image[base + 3];
        }
        let (y, cb, cr) = rgb16_to_ycbcr420(&rgb, w, h);
        (y, cb, cr, a_plane)
    }

    pub(crate) fn ycbcr420_to_rgb16(
        y: &[u16],
        cb: &[u16],
        cr: &[u16],
        w: usize,
        h: usize,
        out: &mut [u16],
    ) {
        let cw = (w + 1) / 2;
        for py in 0..h {
            let c_row = (py / 2) * cw;
            for px in 0..w {
                let yi = y[py * w + px] as i64;
                let cb_c = cb[c_row + px / 2] as i64 - 32768;
                let cr_c = cr[c_row + px / 2] as i64 - 32768;
                // BT.601 inverse at 16-bit scale (coeffs * 65536, >>16). The
                // chroma delta spans +-32768, so 8-bit-domain >>15 constants
                // would double the correction and saturate saturated colors.
                let r = yi + ((91881 * cr_c + 32768) >> 16);
                let g = yi - ((22554 * cb_c + 46802 * cr_c + 32768) >> 16);
                let b = yi + ((116130 * cb_c + 32768) >> 16);
                let o = (py * w + px) * 3;
                out[o] = clamp_u16(r);
                out[o + 1] = clamp_u16(g);
                out[o + 2] = clamp_u16(b);
            }
        }
    }

    pub(crate) fn ycbcr420a_to_rgba16(
        y: &[u16],
        cb: &[u16],
        cr: &[u16],
        a: &[u16],
        w: usize,
        h: usize,
        out: &mut [u16],
    ) {
        let cw = (w + 1) / 2;
        for py in 0..h {
            let c_row = (py / 2) * cw;
            for px in 0..w {
                let yi = y[py * w + px] as i64;
                let cb_c = cb[c_row + px / 2] as i64 - 32768;
                let cr_c = cr[c_row + px / 2] as i64 - 32768;
                let r = yi + ((91881 * cr_c + 32768) >> 16);
                let g = yi - ((22554 * cb_c + 46802 * cr_c + 32768) >> 16);
                let b = yi + ((116130 * cb_c + 32768) >> 16);
                let o = (py * w + px) * 4;
                out[o] = clamp_u16(r);
                out[o + 1] = clamp_u16(g);
                out[o + 2] = clamp_u16(b);
                out[o + 3] = a[py * w + px];
            }
        }
    }

    // -----------------------------------------------------------------------
    // Extended Huffman entropy coding (fixed-length canonical alphabet)
    //
    // DC symbols 0..=18  -> 5-bit codes (category 0..=18)
    // AC symbols 0..=260 -> 9-bit codes:
    //   0 = EOB, 1 = ZRL (16 zero run), otherwise 2 + run*20 + (cat-1),
    //   run 0..=15, cat 1..=19.
    // Plane framing matches the 8-bit path: 4-byte LE payload length then the
    // MSB-first bitstream with JPEG 0xFF byte stuffing.
    // -----------------------------------------------------------------------

    const DC_BITS: u8 = 5;
    const AC_BITS: u8 = 9;
    const AC_EOB: u32 = 0;
    const AC_ZRL: u32 = 1;
    const AC_CAT_MAX: u8 = 19;
    const DC_CAT_MAX: u8 = 18;

    struct BitWriter16 {
        buf: Vec<u8>,
        acc: u64,
        nbits: u8,
    }

    impl BitWriter16 {
        fn new() -> Self {
            Self { buf: Vec::with_capacity(4096), acc: 0, nbits: 0 }
        }

        fn push_byte(&mut self, byte: u8) {
            self.buf.push(byte);
            if byte == 0xFF {
                self.buf.push(0x00);
            }
        }

        fn write_bits(&mut self, code: u32, n: u8) {
            if n == 0 {
                return;
            }
            self.acc = (self.acc << n) | ((code as u64) & ((1u64 << n) - 1));
            self.nbits += n;
            while self.nbits >= 8 {
                let shift = self.nbits - 8;
                let byte = (self.acc >> shift) as u8;
                self.push_byte(byte);
                self.nbits -= 8;
                if self.nbits == 0 {
                    self.acc = 0;
                } else {
                    self.acc &= (1u64 << self.nbits) - 1;
                }
            }
        }

        fn flush(mut self) -> Vec<u8> {
            if self.nbits > 0 {
                let pad = 8 - self.nbits;
                self.write_bits((1u32 << pad) - 1, pad);
            }
            self.buf
        }
    }

    struct BitReader16<'a> {
        buf: &'a [u8],
        pos: usize,
        acc: u64,
        nbits: u8,
    }

    impl<'a> BitReader16<'a> {
        fn new(buf: &'a [u8]) -> Self {
            Self { buf, pos: 0, acc: 0, nbits: 0 }
        }

        fn refill(&mut self) {
            while self.nbits <= 56 && self.pos < self.buf.len() {
                let byte = self.buf[self.pos];
                self.pos += 1;
                if byte == 0xFF && self.pos < self.buf.len() && self.buf[self.pos] == 0x00 {
                    self.pos += 1;
                }
                self.acc = (self.acc << 8) | byte as u64;
                self.nbits += 8;
            }
        }

        fn read_bits(&mut self, n: u8) -> Option<u32> {
            if n == 0 {
                return Some(0);
            }
            if self.nbits < n {
                self.refill();
            }
            if self.nbits < n {
                return None;
            }
            let shift = self.nbits - n;
            let out = ((self.acc >> shift) & ((1u64 << n) - 1)) as u32;
            self.nbits -= n;
            if self.nbits == 0 {
                self.acc = 0;
            } else {
                self.acc &= (1u64 << self.nbits) - 1;
            }
            Some(out)
        }
    }

    #[inline]
    fn category_i32(v: i32) -> u8 {
        debug_assert!(v != 0);
        (32 - v.unsigned_abs().leading_zeros()) as u8
    }

    #[inline]
    fn magnitude_i32(v: i32, cat: u8) -> u32 {
        if v >= 0 {
            v as u32
        } else {
            (1u32 << cat).wrapping_sub(1).wrapping_add(v as u32)
        }
    }

    #[inline]
    fn demagnitude_i32(bits: u32, cat: u8) -> i32 {
        let threshold = 1u32 << (cat - 1);
        if bits >= threshold {
            bits as i32
        } else {
            bits as i32 - (1i32 << cat) + 1
        }
    }

    pub(crate) fn encode_plane16(blocks: &[Block32]) -> Vec<u8> {
        let mut w = BitWriter16::new();
        for block in blocks {
            // DC: absolute value (no delta coding), category 0..=18.
            let dc = block.data[ZIGZAG[0]];
            if dc == 0 {
                w.write_bits(0, DC_BITS);
            } else {
                let cat = category_i32(dc).min(DC_CAT_MAX);
                w.write_bits(cat as u32, DC_BITS);
                w.write_bits(magnitude_i32(dc, cat), cat);
            }

            // AC.
            let mut last_nz = 63usize;
            while last_nz > 0 && block.data[ZIGZAG[last_nz]] == 0 {
                last_nz -= 1;
            }
            if last_nz == 0 {
                w.write_bits(AC_EOB, AC_BITS);
                continue;
            }
            let mut zero_run: u32 = 0;
            for i in 1..=last_nz {
                let val = block.data[ZIGZAG[i]];
                if val == 0 {
                    zero_run += 1;
                    continue;
                }
                while zero_run >= 16 {
                    w.write_bits(AC_ZRL, AC_BITS);
                    zero_run -= 16;
                }
                let cat = category_i32(val).min(AC_CAT_MAX);
                let sym = 2 + zero_run * 20 + (cat as u32 - 1);
                w.write_bits(sym, AC_BITS);
                w.write_bits(magnitude_i32(val, cat), cat);
                zero_run = 0;
            }
            w.write_bits(AC_EOB, AC_BITS);
        }
        let payload = w.flush();
        let mut out = Vec::with_capacity(4 + payload.len());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }

    pub(crate) fn decode_plane16(buf: &[u8], start: usize, n_blocks: usize) -> Option<(Vec<Block32>, usize)> {
        if start + 4 > buf.len() {
            return None;
        }
        let plane_len = u32::from_le_bytes(buf[start..start + 4].try_into().ok()?) as usize;
        let data_start = start + 4;
        let data_end = data_start.checked_add(plane_len)?;
        if data_end > buf.len() {
            return None;
        }
        let mut reader = BitReader16::new(&buf[data_start..data_end]);
        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let mut block = Block32::zero();

            let dc_cat = reader.read_bits(DC_BITS)? as u8;
            if dc_cat > DC_CAT_MAX {
                return None;
            }
            if dc_cat > 0 {
                let bits = reader.read_bits(dc_cat)?;
                block.data[ZIGZAG[0]] = demagnitude_i32(bits, dc_cat);
            }

            let mut ac_idx = 1usize;
            loop {
                let sym = reader.read_bits(AC_BITS)?;
                if sym == AC_EOB {
                    break;
                }
                if sym == AC_ZRL {
                    ac_idx += 16;
                    if ac_idx > 64 {
                        return None;
                    }
                    continue;
                }
                if sym < 2 {
                    return None;
                }
                let v = sym - 2;
                let run = (v / 20) as usize;
                let cat = ((v % 20) + 1) as u8;
                if cat > AC_CAT_MAX || ac_idx + run >= 64 {
                    return None;
                }
                ac_idx += run;
                let bits = reader.read_bits(cat)?;
                block.data[ZIGZAG[ac_idx]] = demagnitude_i32(bits, cat);
                ac_idx += 1;
                if ac_idx > 64 {
                    return None;
                }
            }
            blocks.push(block);
        }
        Some((blocks, data_end))
    }

    // -----------------------------------------------------------------------
    // Encode pipeline
    // -----------------------------------------------------------------------

    struct PlaneSpec {
        samples: Vec<u16>,
        w: usize,
        h: usize,
        is_chroma: bool,
    }

    fn write_header(out: &mut Vec<u8>, version: u8, w: usize, h: usize, quality: u8, flags: u8) {
        out.extend_from_slice(&MAGIC);
        out.push(version);
        out.extend_from_slice(&(w as u32).to_le_bytes());
        out.extend_from_slice(&(h as u32).to_le_bytes());
        out.push(quality);
        out.push(flags);
    }

    fn encode_planes(specs: &[PlaneSpec], quality: u8, mode: OutputMode) -> Vec<u8> {
        let mut out = Vec::new();
        for spec in specs {
            let table = quant_table_16(quality, mode, spec.is_chroma);
            let thresholds = build_sparsify_thresholds_8(quality, spec.is_chroma);
            let mut blocks = blocks_from_plane(&spec.samples, spec.w, spec.h);
            for block in blocks.iter_mut() {
                block.data = dct_i32(&block.data);
                quantize_block32(block, &table);
                if mode == OutputMode::Down8 {
                    sparsify_and_clamp_block(block, &thresholds);
                }
            }
            out.extend_from_slice(&encode_plane16(&blocks));
        }
        out
    }

    fn flags_for(mode: OutputMode, gray: bool) -> u8 {
        let mut flags = 0u8;
        if mode == OutputMode::Keep16 {
            flags |= FLAG_KEEP16;
        }
        if gray {
            flags |= FLAG_GRAY;
        }
        flags
    }

    /// Encode a 16-bit grayscale image to a .bg v20 stream.
    pub fn encode_grayscale16(
        image: &[u16],
        width: usize,
        height: usize,
        quality: u8,
        mode: OutputMode,
    ) -> BgResult<Vec<u8>> {
        let q = normalize_quality(quality)?;
        check_dims(width, height, 1, image.len())?;
        let mut out = Vec::with_capacity(HEADER_LEN + 64);
        write_header(&mut out, VERSION_RGB_OR_GRAY, width, height, q, flags_for(mode, true));
        let specs = [PlaneSpec { samples: image.to_vec(), w: width, h: height, is_chroma: false }];
        out.extend_from_slice(&encode_planes(&specs, q, mode));
        Ok(out)
    }

    /// Encode a 16-bit RGB image (3 u16 per pixel) to a .bg v20 stream.
    pub fn encode_rgb16(
        image: &[u16],
        width: usize,
        height: usize,
        quality: u8,
        mode: OutputMode,
    ) -> BgResult<Vec<u8>> {
        let q = normalize_quality(quality)?;
        check_dims(width, height, 3, image.len())?;
        let (y, cb, cr) = rgb16_to_ycbcr420(image, width, height);
        let cw = (width + 1) / 2;
        let ch = (height + 1) / 2;
        let mut out = Vec::with_capacity(HEADER_LEN + 64);
        write_header(&mut out, VERSION_RGB_OR_GRAY, width, height, q, flags_for(mode, false));
        let specs = [
            PlaneSpec { samples: y, w: width, h: height, is_chroma: false },
            PlaneSpec { samples: cb, w: cw, h: ch, is_chroma: true },
            PlaneSpec { samples: cr, w: cw, h: ch, is_chroma: true },
        ];
        out.extend_from_slice(&encode_planes(&specs, q, mode));
        Ok(out)
    }

    /// Encode a 16-bit RGBA image (4 u16 per pixel) to a .bg v21 stream.
    pub fn encode_rgba16(
        image: &[u16],
        width: usize,
        height: usize,
        quality: u8,
        mode: OutputMode,
    ) -> BgResult<Vec<u8>> {
        let q = normalize_quality(quality)?;
        check_dims(width, height, 4, image.len())?;
        let (y, cb, cr, a) = rgba16_to_ycbcr420a(image, width, height);
        let cw = (width + 1) / 2;
        let ch = (height + 1) / 2;
        let mut out = Vec::with_capacity(HEADER_LEN + 64);
        write_header(&mut out, VERSION_RGBA, width, height, q, flags_for(mode, false));
        let specs = [
            PlaneSpec { samples: y, w: width, h: height, is_chroma: false },
            PlaneSpec { samples: cb, w: cw, h: ch, is_chroma: true },
            PlaneSpec { samples: cr, w: cw, h: ch, is_chroma: true },
            PlaneSpec { samples: a, w: width, h: height, is_chroma: false },
        ];
        out.extend_from_slice(&encode_planes(&specs, q, mode));
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Decode pipeline
    // -----------------------------------------------------------------------

    /// Decoded 16-bit stream. `pixels` are u16; for `Down8` each value occupies
    /// the low 8 bits (0..=255), for `Keep16` the full 16-bit range is used.
    #[derive(Debug, Clone)]
    pub struct Decoded16 {
        pub width: usize,
        pub height: usize,
        pub channels: usize,
        pub mode: OutputMode,
        pub pixels: Vec<u16>,
    }

    pub(crate) fn reconstruct_plane(
        blocks: &[Block32],
        table: &[i32; 64],
        w: usize,
        h: usize,
        mode: OutputMode,
    ) -> Vec<u16> {
        let bw = (w + 7) / 8;
        let mut plane = vec![0u16; w * h];
        // For Down8 the stored levels are 8-bit-domain (the encode table was the
        // 8-bit step x256). Reconstruct with the table divided back by 256 and an
        // 8-bit center, so the output genuinely occupies 0..=255.
        let deq_scale: i32 = match mode {
            OutputMode::Keep16 => 1,
            OutputMode::Down8 => 256,
        };
        let center: i64 = match mode {
            OutputMode::Keep16 => 32768,
            OutputMode::Down8 => 128,
        };
        let max_out: i64 = match mode {
            OutputMode::Keep16 => 65535,
            OutputMode::Down8 => 255,
        };
        for (idx, levels) in blocks.iter().enumerate() {
            let mut coef = [0f64; 64];
            for i in 0..64 {
                coef[i] = levels.data[i] as f64 * (table[i] / deq_scale).max(1) as f64;
            }
            let centered = idct_f64(&coef);
            let bx = (idx % bw) * 8;
            let by = (idx / bw) * 8;
            for yy in 0..8 {
                for xx in 0..8 {
                    let px = bx + xx;
                    let py = by + yy;
                    if px < w && py < h {
                        let sample = (centered[yy * 8 + xx] as i64 + center).clamp(0, max_out);
                        plane[py * w + px] = sample as u16;
                    }
                }
            }
        }
        plane
    }

    /// Decode a .bg v20/v21 stream produced by this module.
    pub fn decode16(buf: &[u8]) -> BgResult<Decoded16> {
        if buf.len() < HEADER_LEN || buf[0] != MAGIC[0] || buf[1] != MAGIC[1] {
            return Err(BitgrainError::DecodeFailed);
        }
        let version = buf[2];
        let gray;
        let channels = match version {
            VERSION_RGB_OR_GRAY => 3,
            VERSION_RGBA => 4,
            _ => return Err(BitgrainError::DecodeFailed),
        };
        let width = u32::from_le_bytes(buf[3..7].try_into().map_err(|_| BitgrainError::DecodeFailed)?) as usize;
        let height = u32::from_le_bytes(buf[7..11].try_into().map_err(|_| BitgrainError::DecodeFailed)?) as usize;
        if width == 0 || height == 0 || width > 65536 || height > 65536 {
            return Err(BitgrainError::DecodeFailed);
        }
        let quality = buf[11];
        if quality == 0 || quality > 100 {
            return Err(BitgrainError::DecodeFailed);
        }
        let flags = buf[12];
        if flags & !(FLAG_KEEP16 | FLAG_GRAY) != 0 {
            return Err(BitgrainError::DecodeFailed);
        }
        let mode = if flags & FLAG_KEEP16 != 0 { OutputMode::Keep16 } else { OutputMode::Down8 };
        gray = flags & FLAG_GRAY != 0;
        if gray && version != VERSION_RGB_OR_GRAY {
            return Err(BitgrainError::DecodeFailed);
        }

        let cw = (width + 1) / 2;
        let ch = (height + 1) / 2;
        let n_full = ((width + 7) / 8) * ((height + 7) / 8);
        let n_chroma = ((cw + 7) / 8) * ((ch + 7) / 8);
        let mut pos = HEADER_LEN;

        let luma_table = quant_table_16(quality, mode, false);
        let chroma_table = quant_table_16(quality, mode, true);

        if gray {
            let (blocks, _) = decode_plane16(buf, pos, n_full).ok_or(BitgrainError::DecodeFailed)?;
            let plane = reconstruct_plane(&blocks, &luma_table, width, height, mode);
            return Ok(Decoded16 { width, height, channels: 1, mode, pixels: plane });
        }

        let (y_blocks, next) = decode_plane16(buf, pos, n_full).ok_or(BitgrainError::DecodeFailed)?;
        pos = next;
        let (cb_blocks, next) = decode_plane16(buf, pos, n_chroma).ok_or(BitgrainError::DecodeFailed)?;
        pos = next;
        let (cr_blocks, next) = decode_plane16(buf, pos, n_chroma).ok_or(BitgrainError::DecodeFailed)?;
        pos = next;

        let y_plane = reconstruct_plane(&y_blocks, &luma_table, width, height, mode);
        let cb_plane = reconstruct_plane(&cb_blocks, &chroma_table, cw, ch, mode);
        let cr_plane = reconstruct_plane(&cr_blocks, &chroma_table, cw, ch, mode);

        if channels == 3 {
            // No further planes: nothing may trail the last plane payload.
            if pos != buf.len() {
                return Err(BitgrainError::DecodeFailed);
            }
            let mut pixels = vec![0u16; width * height * 3];
            match mode {
                OutputMode::Keep16 => {
                    ycbcr420_to_rgb16(&y_plane, &cb_plane, &cr_plane, width, height, &mut pixels);
                }
                OutputMode::Down8 => {
                    // Planes hold 0..=255; run the 16-bit color math at upscaled
                    // scale, then downshift the RGB result back to 0..=255.
                    let (y16, cb16, cr16) = (
                        upscale8_to_16(&y_plane),
                        upscale8_to_16(&cb_plane),
                        upscale8_to_16(&cr_plane),
                    );
                    let mut rgb16 = vec![0u16; width * height * 3];
                    ycbcr420_to_rgb16(&y16, &cb16, &cr16, width, height, &mut rgb16);
                    for (o, v) in rgb16.iter().enumerate() {
                        pixels[o] = (((*v as u32) + 128) >> 8).min(255) as u16;
                    }
                }
            }
            Ok(Decoded16 { width, height, channels, mode, pixels })
        } else {
            let (a_blocks, next_a) = decode_plane16(buf, pos, n_full).ok_or(BitgrainError::DecodeFailed)?;
            pos = next_a;
            if pos != buf.len() {
                return Err(BitgrainError::DecodeFailed);
            }
            let a_plane = reconstruct_plane(&a_blocks, &luma_table, width, height, mode);
            let mut pixels = vec![0u16; width * height * 4];
            match mode {
                OutputMode::Keep16 => {
                    ycbcr420a_to_rgba16(&y_plane, &cb_plane, &cr_plane, &a_plane, width, height, &mut pixels);
                }
                OutputMode::Down8 => {
                    let (y16, cb16, cr16, a16) = (
                        upscale8_to_16(&y_plane),
                        upscale8_to_16(&cb_plane),
                        upscale8_to_16(&cr_plane),
                        upscale8_to_16(&a_plane),
                    );
                    let mut rgba16 = vec![0u16; width * height * 4];
                    ycbcr420a_to_rgba16(&y16, &cb16, &cr16, &a16, width, height, &mut rgba16);
                    for (o, v) in rgba16.iter().enumerate() {
                        pixels[o] = (((*v as u32) + 128) >> 8).min(255) as u16;
                    }
                }
            }
            Ok(Decoded16 { width, height, channels, mode, pixels })
        }
    }

    /// Upscale an 8-bit-domain plane (0..=255) to 16-bit scale (0..=65535)
    /// using left-shift by 8, matching the input downshift convention.
    fn upscale8_to_16(plane: &[u16]) -> Vec<u16> {
        plane.iter().map(|&v| (v as u32 * 257).min(65535) as u16).collect()
    }
}
