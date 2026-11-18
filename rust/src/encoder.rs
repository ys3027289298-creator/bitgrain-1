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
// Partial re-encode ("patch a small region without re-encoding the image")
//
// Three stages, mirroring the request:
//   1. Locate affected areas through a block index (`.bgi` sidecar).
//   2. Recompute residuals (DCT+quant, or RLE coefficients) and entropy codes
//      only for the blocks touched by the dirty rectangles.
//   3. Splice the new code stream back together, update plane length prefixes
//      and rebuild the index.
//
// The source .bg file is never modified; callers write the result to a new
// buffer. The block tile size (multiple of 8) and whether the old entropy
// stream of untouched blocks is preserved (`preserve_old_stream`) are
// configurable. A sidecar index with CRC32 protection makes any corruption a
// hard error.
//
// Index file layout (all multi-byte integers little-endian):
//   offset  bytes  field
//   0       4      magic "BGI1"
//   4       4      index length `idx_len` (u32) = number of bytes through CRC
//   8       2      source width (u16)
//   10      2      source height (u16)
//   12      1      .bg version byte
//   13      1      quality
//   14      2      tile size in pixels (u16; always a multiple of 8)
//   16      1      number of planes P (u8)
//   17      4      source .bg length (u32)
//   21      P * (4 + 4*Np)
//                  per plane: Np blocks (u32 LE), then Np u32 LE block start
//                  offsets — bytes from the plane's first payload byte for RLE
//                  planes; bit offsets inside the *unstuffed* entropy payload
//                  for Huffman planes.
//   ...     P * 2  per plane: Np u16 LE DC coefficient values
//   end     4      CRC32 (IEEE) of all preceding index bytes
// ===========================================================================
pub mod partial {
    use crate::block::Block;
    use crate::colorspace;
    use crate::dct;
    use crate::entropy;
    use crate::huffman;
    use crate::jpeg_luma_ac_ht::JPEG_LUMA_AC_HT;
    use crate::zigzag::ZIGZAG;
    use std::time::Duration;

    pub const INDEX_MAGIC: &[u8; 4] = b"BGI1";
    const BG_HEADER_LEN: usize = super::BG_HEADER_SIZE; // 12
    const EOB_RUN: u8 = 0xFF;

    /// Everything that can go wrong while indexing or partially re-encoding.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum PartialError {
        NotABgFile,
        UnsupportedVersion(u8),
        TruncatedStream,
        BadEntropyStream,
        BadIndexMagic,
        IndexTruncated,
        IndexChecksumMismatch,
        IndexSourceMismatch,
        BadConfig(&'static str),
        ImageSizeMismatch,
    }

    impl std::fmt::Display for PartialError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                PartialError::NotABgFile => write!(f, "not a .bg file"),
                PartialError::UnsupportedVersion(v) => write!(f, "unsupported .bg version {v}"),
                PartialError::TruncatedStream => write!(f, "truncated .bg stream"),
                PartialError::BadEntropyStream => write!(f, "undecodable entropy stream"),
                PartialError::BadIndexMagic => write!(f, "bad block-index magic"),
                PartialError::IndexTruncated => write!(f, "truncated block index"),
                PartialError::IndexChecksumMismatch => write!(f, "block index CRC mismatch"),
                PartialError::IndexSourceMismatch => write!(f, "index does not match source .bg"),
                PartialError::BadConfig(m) => write!(f, "bad config: {m}"),
                PartialError::ImageSizeMismatch => write!(f, "new image dimensions differ from index"),
            }
        }
    }

    impl std::error::Error for PartialError {}

    type Result<T> = std::result::Result<T, PartialError>;

    // -- CRC32 (IEEE 802.3), bitwise — no external crate needed. -------------
    fn crc32_ieee(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }

    // -- Profile resolution per .bg version ----------------------------------
    #[derive(Clone, Copy)]
    struct PlaneProfile {
        plane_w: usize,
        plane_h: usize,
        is_chroma: bool,
        use_chroma_ac: bool,
        use_dc_delta: bool,
    }

    #[derive(Clone, Copy)]
    struct FileProfile {
        version: u8,
        width: usize,
        height: usize,
        quality: u8,
        channels: usize, // 1 = gray RLE, 3 = RGB, 4 = RGBA
        is_huffman: bool,
    }

    fn parse_header(bg: &[u8]) -> Result<FileProfile> {
        if bg.len() < BG_HEADER_LEN || &bg[0..2] != b"BG" {
            return Err(PartialError::NotABgFile);
        }
        let version = bg[2];
        if version == 0 || version > 19 {
            return Err(PartialError::UnsupportedVersion(version));
        }
        let width = u32::from_le_bytes(bg[3..7].try_into().unwrap()) as usize;
        let height = u32::from_le_bytes(bg[7..11].try_into().unwrap()) as usize;
        let quality = bg[11];
        if width == 0 || height == 0 || width > u16::MAX as usize || height > u16::MAX as usize {
            return Err(PartialError::BadConfig("image dimensions out of range"));
        }
        let channels = match version {
            1 => 1,
            2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 => 3,
            3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 => 4,
            _ => return Err(PartialError::UnsupportedVersion(version)),
        };
        Ok(FileProfile {
            version,
            width,
            height,
            quality,
            channels,
            is_huffman: version >= 4,
        })
    }

    fn plane_profiles(fp: &FileProfile) -> Vec<PlaneProfile> {
        let (w, h) = (fp.width, fp.height);
        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let full = PlaneProfile { plane_w: w, plane_h: h, is_chroma: false, use_chroma_ac: false, use_dc_delta: false };
        if !fp.is_huffman {
            // RLE v1/v2/v3: one full-resolution plane per channel, no Huffman flags.
            return (0..fp.channels).map(|_| full).collect();
        }
        let chroma_ac = fp.version >= 6;
        let dc_delta = fp.version >= 10;
        let chroma = PlaneProfile {
            plane_w: cw, plane_h: ch, is_chroma: true,
            use_chroma_ac: chroma_ac, use_dc_delta: dc_delta,
        };
        let luma = PlaneProfile {
            plane_w: w, plane_h: h, is_chroma: false,
            use_chroma_ac: false, use_dc_delta: dc_delta,
        };
        match fp.channels {
            3 => vec![luma, chroma, chroma],
            _ => vec![luma, chroma, chroma, luma], // RGBA: Y Cb Cr A
        }
    }

    fn quant_tables(fp: &FileProfile) -> Vec<[i16; 64]> {
        let q = fp.quality;
        if !fp.is_huffman {
            let t = super::quant_table_for_quality(q);
            return (0..fp.channels).map(|_| t).collect();
        }
        let (luma, chroma) = match fp.version {
            4..=5 | 6..=7 => (super::quant_table_for_quality(q), super::chroma_quant_table_for_quality(q)),
            8..=11 => (
                super::quant_table_for_quality_perceptual(q),
                super::chroma_quant_table_for_quality_perceptual(q),
            ),
            12..=13 => (
                super::quant_table_for_quality_perceptual_v2(q),
                super::chroma_quant_table_for_quality_perceptual_v2(q),
            ),
            _ => (
                super::quant_table_for_quality_perceptual_v4(q),
                super::chroma_quant_table_for_quality_perceptual_v4(q),
            ),
        };
        match fp.channels {
            3 => vec![luma, chroma, chroma],
            _ => vec![luma, chroma, chroma, luma],
        }
    }

    fn sparsify_thresholds(fp: &FileProfile) -> Vec<Option<[i16; 64]>> {
        // Only v18/v19 apply AC sparsification in the production encoder.
        let enabled = fp.version == 18 || fp.version == 19;
        plane_profiles(fp)
            .iter()
            .map(|p| enabled.then(|| super::build_sparsify_thresholds(fp.quality, p.is_chroma)))
            .collect()
    }

    // -- Pure-Rust quantization ---------------------------------------------
    // Matches the release C SIMD path (float divide, round-to-nearest-even,
    // saturating i16 cast). Kept local so `cargo test` builds without linking
    // the C objects; release builds of this module call the same function the
    // production encoder uses, so patched streams stay byte-identical.
    #[cfg(test)]
    fn quantize_block_rust(block: &mut [i16; 64], table: &[i16; 64]) {
        for i in 0..64 {
            let r = (block[i] as f32) / (table[i] as f32);
            block[i] = r.round_ties_even() as i16;
        }
    }

    #[cfg(not(test))]
    #[inline]
    fn quantize_block_local(block: &mut [i16; 64], table: &[i16; 64]) {
        super::quantize(block, table);
    }

    #[cfg(test)]
    #[inline]
    fn quantize_block_local(block: &mut [i16; 64], table: &[i16; 64]) {
        quantize_block_rust(block, table);
    }

    // -- Local block extraction (mirrors Blockizer edge replication) --------
    fn block_from_flat(plane: &[u8], pw: usize, ph: usize, bx: usize, by: usize) -> Block {
        let mut data = [0i16; 64];
        for y in 0..8 {
            let iy = (by + y).min(ph.saturating_sub(1));
            for x in 0..8 {
                let ix = (bx + x).min(pw.saturating_sub(1));
                data[y * 8 + x] = plane[iy * pw + ix] as i16 - 128;
            }
        }
        Block { data }
    }

    fn block_from_interleaved(
        image: &[u8], pw: usize, ph: usize, stride: usize, channel: usize, bx: usize, by: usize,
    ) -> Block {
        let mut data = [0i16; 64];
        for y in 0..8 {
            let iy = (by + y).min(ph.saturating_sub(1));
            for x in 0..8 {
                let ix = (bx + x).min(pw.saturating_sub(1));
                data[y * 8 + x] = image[(iy * pw + ix) * stride + channel] as i16 - 128;
            }
        }
        Block { data }
    }

    /// Full per-block transform used both for changed blocks and for the
    /// deterministic reference encoder in tests.
    fn transform_block(block: &mut Block, table: &[i16; 64], huffman: bool, thr: Option<&[i16; 64]>) {
        dct::dct(block);
        quantize_block_local(&mut block.data, table);
        if huffman {
            if let Some(t) = thr {
                super::sparsify_ac_block(block, t);
            }
            huffman::clamp_block_jpeg_coeffs(block);
        }
    }

    // -- Huffman tables (canonical JPEG, re-declared locally so the scanner
    //    and block entropy writer stay inside this single library file) -----
    type AcTable = [(u8, u16); 256];

    fn luma_ac_table() -> AcTable {
        let mut t = [(0u8, 0u16); 256];
        for (sym, entry) in JPEG_LUMA_AC_HT.iter().enumerate() {
            // jpeg_luma_ac_ht stores [code, length]; zero length = unused.
            if entry[1] > 0 {
                t[sym] = (entry[1] as u8, entry[0]);
            }
        }
        t
    }

    fn chroma_ac_table() -> AcTable {
        // ISO 10918-1 Annex K chrominance AC table (same constants the
        // production huffman module uses).
        const NR_CODES: [u8; 17] = [0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
        const VALUES: [u8; 162] = [
            0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
            0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
            0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
            0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
            0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
            0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
            0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
            0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
            0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
            0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
            0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
            0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
            0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
            0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
            0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
            0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
            0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
            0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
            0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
            0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
            0xf9, 0xfa,
        ];
        let mut t = [(0u8, 0u16); 256];
        let mut code: u16 = 0;
        let mut k = 0usize;
        for len in 1..=16usize {
            for _ in 0..NR_CODES[len] as usize {
                let sym = VALUES[k] as usize;
                t[sym] = (len as u8, code);
                code = code.wrapping_add(1);
                k += 1;
            }
            code <<= 1;
        }
        t
    }

    const LUMA_DC_TABLE: [(u8, u16); 12] = [
        (2, 0b00), (3, 0b010), (3, 0b011), (3, 0b100), (3, 0b101), (3, 0b110),
        (4, 0b1110), (5, 0b11110), (6, 0b111110), (7, 0b1111110),
        (8, 0b11111110), (9, 0b111111110),
    ];
    const CHROMA_DC_TABLE: [(u8, u16); 12] = [
        (2, 0b00), (2, 0b01), (2, 0b10), (3, 0b110), (4, 0b1110), (5, 0b11110),
        (6, 0b111110), (7, 0b1111110), (8, 0b11111110), (9, 0b111111110),
        (10, 0b1111111110), (11, 0b11111111110),
    ];

    #[inline]
    fn category(v: i16) -> u8 {
        if v == 0 { 0 } else { (16 - v.unsigned_abs().leading_zeros()) as u8 }
    }

    #[inline]
    fn magnitude_bits(v: i16, cat: u8) -> u16 {
        if v >= 0 { v as u16 } else { ((1u16 << cat) - 1).wrapping_add(v as u16) }
    }

    #[inline]
    fn magnitude_decode(bits: u16, cat: u8) -> i16 {
        let threshold = 1u16 << (cat - 1);
        if bits >= threshold { bits as i16 } else { bits as i16 - (1i16 << cat) + 1 }
    }

    /// Minimal MSB-first reader over the *unstuffed* payload (0xFF00 already
    /// removed), used only to recover per-block bit boundaries.
    struct BitCursor<'a> {
        data: &'a [u8],
        bit_pos: usize,
    }

    impl<'a> BitCursor<'a> {
        fn new(data: &'a [u8]) -> Self { Self { data, bit_pos: 0 } }
        fn read_bits(&mut self, n: u8) -> Option<u16> {
            if n == 0 { return Some(0); }
            let end = self.bit_pos + n as usize;
            if end > self.data.len() * 8 { return None; }
            let mut v: u16 = 0;
            for _ in 0..n {
                let b = (self.data[self.bit_pos >> 3] >> (7 - (self.bit_pos & 7))) & 1;
                v = (v << 1) | b as u16;
                self.bit_pos += 1;
            }
            Some(v)
        }
        fn decode_sym(&mut self, table: &DecodeTree) -> Option<u8> {
            let mut idx = 0usize;
            for _ in 0..16 {
                let bit = self.read_bits(1)? as usize;
                idx = table.nodes[idx].child[bit] as usize;
                let sym = table.nodes[idx].sym;
                if sym >= 0 {
                    return Some(sym as u8);
                }
            }
            None
        }
    }

    #[derive(Clone, Copy)]
    struct TreeNode { child: [i16; 2], sym: i16 }
    struct DecodeTree { nodes: Vec<TreeNode> }

    impl DecodeTree {
        fn from_table(table: &[(u8, u16)]) -> Option<Self> {
            let mut nodes = vec![TreeNode { child: [-1, -1], sym: -1 }];
            for (sym, &(len, code)) in table.iter().enumerate() {
                if len == 0 { continue; }
                let mut idx = 0usize;
                for shift in (0..len).rev() {
                    if nodes[idx].sym >= 0 { return None; }
                    let bit = ((code >> shift) & 1) as usize;
                    let next = nodes[idx].child[bit];
                    idx = if next >= 0 {
                        next as usize
                    } else {
                        nodes.push(TreeNode { child: [-1, -1], sym: -1 });
                        let ni = (nodes.len() - 1) as i16;
                        nodes[idx].child[bit] = ni;
                        nodes.len() - 1
                    };
                }
                if nodes[idx].sym >= 0 || nodes[idx].child[0] >= 0 || nodes[idx].child[1] >= 0 {
                    return None;
                }
                nodes[idx].sym = sym as i16;
            }
            Some(Self { nodes })
        }
    }

    fn remove_byte_stuffing(payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(payload.len());
        let mut i = 0usize;
        while i < payload.len() {
            let b = payload[i];
            out.push(b);
            if b == 0xFF && i + 1 < payload.len() && payload[i + 1] == 0x00 {
                i += 2;
            } else {
                i += 1;
            }
        }
        out
    }

    /// Scan one Huffman plane and return (block bit starts, block DC values,
    /// total valid payload bits excluding the trailing 1-pad).
    fn scan_huffman_plane(
        payload: &[u8], n_blocks: usize, pp: PlaneProfile,
    ) -> Result<(Vec<u32>, Vec<i16>, usize)> {
        let dc_table: [(u8, u16); 12] = if pp.is_chroma { CHROMA_DC_TABLE } else { LUMA_DC_TABLE };
        let ac_owned: AcTable = if pp.use_chroma_ac { chroma_ac_table() } else { luma_ac_table() };
        let dc_tree = DecodeTree::from_table(&dc_table).ok_or(PartialError::BadEntropyStream)?;
        let ac_tree = DecodeTree::from_table(&ac_owned).ok_or(PartialError::BadEntropyStream)?;
        let raw = remove_byte_stuffing(payload);
        let mut cur = BitCursor::new(&raw);
        let mut starts = Vec::with_capacity(n_blocks);
        let mut dcs = Vec::with_capacity(n_blocks);
        let mut prev_dc = 0i16;
        for _ in 0..n_blocks {
            starts.push(cur.bit_pos as u32);
            let dc_cat = cur.decode_sym(&dc_tree).ok_or(PartialError::BadEntropyStream)?;
            let dc_diff = if dc_cat == 0 {
                0i16
            } else {
                let bits = cur.read_bits(dc_cat).ok_or(PartialError::BadEntropyStream)?;
                magnitude_decode(bits, dc_cat)
            };
            let dc = if pp.use_dc_delta {
                prev_dc = prev_dc.wrapping_add(dc_diff);
                prev_dc
            } else {
                dc_diff
            };
            dcs.push(dc);
            let mut ac_idx = 1usize;
            loop {
                let sym = cur.decode_sym(&ac_tree).ok_or(PartialError::BadEntropyStream)?;
                if sym == 0x00 { break; }
                if sym == 0xF0 {
                    if ac_idx + 16 > 64 { return Err(PartialError::BadEntropyStream); }
                    ac_idx += 16;
                    continue;
                }
                let run = (sym >> 4) as usize;
                let cat = sym & 0x0F;
                if cat == 0 || ac_idx + run >= 64 {
                    return Err(PartialError::BadEntropyStream);
                }
                ac_idx += run;
                cur.read_bits(cat).ok_or(PartialError::BadEntropyStream)?;
                ac_idx += 1;
                if ac_idx > 64 { return Err(PartialError::BadEntropyStream); }
            }
        }
        let valid_bits = cur.bit_pos;
        Ok((starts, dcs, valid_bits))
    }

    /// Full Huffman block decode over an *unstuffed* payload. This mirrors
    /// `huffman::decode_plane_with_profile` but never applies 0xFF00 removal,
    /// which would corrupt a payload that was unstuffed already.
    fn decode_huffman_blocks_unstuffed(
        raw: &[u8], n_blocks: usize, pp: PlaneProfile,
    ) -> Result<Vec<Block>> {
        let dc_table: [(u8, u16); 12] = if pp.is_chroma { CHROMA_DC_TABLE } else { LUMA_DC_TABLE };
        let ac_owned: AcTable = if pp.use_chroma_ac { chroma_ac_table() } else { luma_ac_table() };
        let dc_tree = DecodeTree::from_table(&dc_table).ok_or(PartialError::BadEntropyStream)?;
        let ac_tree = DecodeTree::from_table(&ac_owned).ok_or(PartialError::BadEntropyStream)?;
        let mut cur = BitCursor::new(raw);
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut prev_dc = 0i16;
        for _ in 0..n_blocks {
            let mut block = Block::new();
            let dc_cat = cur.decode_sym(&dc_tree).ok_or(PartialError::BadEntropyStream)?;
            let dc_diff = if dc_cat == 0 {
                0i16
            } else {
                magnitude_decode(cur.read_bits(dc_cat).ok_or(PartialError::BadEntropyStream)?, dc_cat)
            };
            let dc = if pp.use_dc_delta {
                prev_dc = prev_dc.wrapping_add(dc_diff);
                prev_dc
            } else {
                dc_diff
            };
            block.data[ZIGZAG[0]] = dc;

            let mut ac_idx = 1usize;
            loop {
                let sym = cur.decode_sym(&ac_tree).ok_or(PartialError::BadEntropyStream)?;
                if sym == 0x00 { break; }
                if sym == 0xF0 {
                    if ac_idx + 16 > 64 { return Err(PartialError::BadEntropyStream); }
                    ac_idx += 16;
                    continue;
                }
                let run = (sym >> 4) as usize;
                let cat = (sym & 0x0F) as usize;
                if cat == 0 || ac_idx + run >= 64 {
                    return Err(PartialError::BadEntropyStream);
                }
                ac_idx += run;
                let bits = cur.read_bits(cat as u8).ok_or(PartialError::BadEntropyStream)?;
                block.data[ZIGZAG[ac_idx]] = magnitude_decode(bits, cat as u8);
                ac_idx += 1;
                if ac_idx > 64 { return Err(PartialError::BadEntropyStream); }
            }
            blocks.push(block);
        }
        Ok(blocks)
    }

    // -- RLE scanner: each block ends in (0xFF, 0x00, 0x00) -----------------
    fn scan_rle_plane(payload: &[u8], n_blocks: usize) -> Result<(Vec<u32>, Vec<i16>)> {
        let mut starts = Vec::with_capacity(n_blocks);
        let mut dcs = Vec::with_capacity(n_blocks);
        let mut pos = 0usize;
        for _ in 0..n_blocks {
            if pos + 2 > payload.len() {
                return Err(PartialError::TruncatedStream);
            }
            starts.push(pos as u32);
            dcs.push(i16::from_le_bytes([payload[pos], payload[pos + 1]]));
            pos += 2;
            loop {
                if pos + 3 > payload.len() {
                    return Err(PartialError::TruncatedStream);
                }
                let run = payload[pos];
                let level = i16::from_le_bytes([payload[pos + 1], payload[pos + 2]]);
                pos += 3;
                if run == EOB_RUN && level == 0 {
                    break;
                }
            }
        }
        Ok((starts, dcs))
    }

    // -- On-disk index --------------------------------------------------------
    /// Parsed block index for one .bg file.
    #[derive(Clone)]
    pub struct BlockIndex {
        pub width: usize,
        pub height: usize,
        pub version: u8,
        pub quality: u8,
        pub tile_size: usize,
        pub source_len: usize,
        /// Per plane: (block start offsets, block DC values).
        pub planes: Vec<PlaneIndex>,
    }

    #[derive(Clone)]
    pub struct PlaneIndex {
        pub offsets: Vec<u32>,
        pub dcs: Vec<i16>,
        /// Huffman planes: valid payload bits excluding trailing 1-pad.
        /// RLE planes: 0 (boundaries are byte offsets delimited by EOB).
        pub valid_bits: u32,
    }

    /// Build the block index for an existing .bg byte stream.
    /// `tile_size` must be a positive multiple of 8 (the codec block size).
    pub fn build_index(bg: &[u8], tile_size: usize) -> Result<BlockIndex> {
        if tile_size == 0 || tile_size % 8 != 0 || tile_size > u16::MAX as usize {
            return Err(PartialError::BadConfig("tile_size must be a multiple of 8"));
        }
        let fp = parse_header(bg)?;
        let profiles = plane_profiles(&fp);
        let mut planes = Vec::with_capacity(profiles.len());
        let mut pos = BG_HEADER_LEN;
        for pp in &profiles {
            let n = ((pp.plane_w + 7) / 8) * ((pp.plane_h + 7) / 8);
            if fp.is_huffman {
                if pos + 4 > bg.len() { return Err(PartialError::TruncatedStream); }
                let payload_len = u32::from_le_bytes(bg[pos..pos + 4].try_into().unwrap()) as usize;
                let data_start = pos + 4;
                let data_end = data_start + payload_len;
                if data_end > bg.len() { return Err(PartialError::TruncatedStream); }
                let (offsets, dcs, valid_bits) = scan_huffman_plane(&bg[data_start..data_end], n, *pp)?;
                planes.push(PlaneIndex { offsets, dcs, valid_bits: valid_bits as u32 });
                pos = data_end;
            } else {
                // RLE planes have no length prefix; the scanner walks EOBs.
                let (offsets, dcs) = scan_rle_plane(&bg[pos..], n)?;
                // Advance past the last block using its self-delimiting EOB.
                let last_start = *offsets.last().unwrap_or(&0) as usize;
                let consumed = rle_consumed_len(&bg[pos + last_start..])?;
                pos += last_start + consumed;
                planes.push(PlaneIndex { offsets, dcs, valid_bits: 0 });
            }
        }
        Ok(BlockIndex {
            width: fp.width,
            height: fp.height,
            version: fp.version,
            quality: fp.quality,
            tile_size,
            source_len: bg.len(),
            planes,
        })
    }

    /// Bytes consumed by one RLE block starting at `buf[0]`, including EOB.
    fn rle_consumed_len(buf: &[u8]) -> Result<usize> {
        let mut pos = 2usize; // DC
        loop {
            if pos + 3 > buf.len() { return Err(PartialError::TruncatedStream); }
            let run = buf[pos];
            let level = i16::from_le_bytes([buf[pos + 1], buf[pos + 2]]);
            pos += 3;
            if run == EOB_RUN && level == 0 {
                return Ok(pos);
            }
        }
    }

    /// Serialize the index to the `.bgi` sidecar format (CRC32 protected).
    pub fn write_index(index: &BlockIndex) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(INDEX_MAGIC);
        // Placeholder for total index length (filled in at the end).
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(index.width as u16).to_le_bytes());
        out.extend_from_slice(&(index.height as u16).to_le_bytes());
        out.push(index.version);
        out.push(index.quality);
        out.extend_from_slice(&(index.tile_size as u16).to_le_bytes());
        out.push(index.planes.len() as u8);
        out.extend_from_slice(&(index.source_len as u32).to_le_bytes());
        for p in &index.planes {
            out.extend_from_slice(&(p.offsets.len() as u32).to_le_bytes());
            for &o in &p.offsets {
                out.extend_from_slice(&o.to_le_bytes());
            }
        }
        for p in &index.planes {
            for &d in &p.dcs {
                out.extend_from_slice(&d.to_le_bytes());
            }
        }
        for p in &index.planes {
            out.extend_from_slice(&p.valid_bits.to_le_bytes());
        }
        let total = (out.len() + 4) as u32;
        out[4..8].copy_from_slice(&total.to_le_bytes());
        let crc = crc32_ieee(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        out
    }

    /// Read and validate a `.bgi` index. Any corruption — bad magic,
    /// truncation, checksum mismatch, or source mismatch — is an error.
    pub fn read_index(bytes: &[u8], bg: &[u8]) -> Result<BlockIndex> {
        if bytes.len() < 25 || &bytes[0..4] != INDEX_MAGIC {
            return Err(PartialError::BadIndexMagic);
        }
        let idx_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if idx_len < 25 || idx_len != bytes.len() {
            return Err(PartialError::IndexTruncated);
        }
        let stored_crc = u32::from_le_bytes(bytes[idx_len - 4..idx_len].try_into().unwrap());
        if crc32_ieee(&bytes[..idx_len - 4]) != stored_crc {
            return Err(PartialError::IndexChecksumMismatch);
        }
        let width = u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize;
        let height = u16::from_le_bytes(bytes[10..12].try_into().unwrap()) as usize;
        let version = bytes[12];
        let quality = bytes[13];
        let tile_size = u16::from_le_bytes(bytes[14..16].try_into().unwrap()) as usize;
        let n_planes = bytes[16] as usize;
        let source_len = u32::from_le_bytes(bytes[17..21].try_into().unwrap()) as usize;

        let fp = parse_header(bg)?;
        if fp.version != version
            || fp.width != width
            || fp.height != height
            || fp.quality != quality
            || bg.len() != source_len
        {
            return Err(PartialError::IndexSourceMismatch);
        }
        let profiles = plane_profiles(&fp);
        if profiles.len() != n_planes {
            return Err(PartialError::IndexSourceMismatch);
        }

        let mut cursor = 21usize;
        let mut counts = Vec::with_capacity(n_planes);
        // Layout: every plane's (count + offset table) comes first, DC tables after.
        let mut all_offsets: Vec<Vec<u32>> = Vec::with_capacity(n_planes);
        for _ in 0..n_planes {
            if cursor + 4 > idx_len - 4 { return Err(PartialError::IndexTruncated); }
            let n = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            if cursor + n * 4 > idx_len - 4 { return Err(PartialError::IndexTruncated); }
            counts.push(n);
            let mut offsets = Vec::with_capacity(n);
            for _ in 0..n {
                offsets.push(u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()));
                cursor += 4;
            }
            all_offsets.push(offsets);
        }
        let mut planes = Vec::with_capacity(n_planes);
        for (pi, &n) in counts.iter().enumerate() {
            let expected = ((profiles[pi].plane_w + 7) / 8) * ((profiles[pi].plane_h + 7) / 8);
            if n != expected {
                return Err(PartialError::IndexSourceMismatch);
            }
            if cursor + n * 2 > idx_len - 4 { return Err(PartialError::IndexTruncated); }
            let mut dcs = Vec::with_capacity(n);
            for _ in 0..n {
                dcs.push(i16::from_le_bytes(bytes[cursor..cursor + 2].try_into().unwrap()));
                cursor += 2;
            }
            planes.push(PlaneIndex {
                offsets: all_offsets[pi].clone(),
                dcs,
                valid_bits: 0, // filled from the trailing table below
            });
        }
        // Trailing per-plane valid-bit counts.
        for pi in 0..n_planes {
            if cursor + 4 > idx_len - 4 { return Err(PartialError::IndexTruncated); }
            planes[pi].valid_bits =
                u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
        }
        if cursor + 4 != idx_len {
            return Err(PartialError::IndexTruncated);
        }
        Ok(BlockIndex { width, height, version, quality, tile_size, source_len, planes })
    }

    // -- Affected-block mapping ----------------------------------------------
    /// Rectangle of changed pixels in the full-resolution image coordinate
    /// system. Coordinates are clipped against image bounds.
    #[derive(Clone, Copy, Debug)]
    pub struct DirtyRect {
        pub x0: usize,
        pub y0: usize,
        pub x1: usize, // exclusive
        pub y1: usize, // exclusive
    }

    impl DirtyRect {
        pub fn new(x0: usize, y0: usize, x1: usize, y1: usize) -> Result<Self> {
            if x1 <= x0 || y1 <= y0 {
                return Err(PartialError::BadConfig("dirty rectangle must be non-empty"));
            }
            Ok(Self { x0, y0, x1, y1 })
        }
    }

    /// Parameters of a partial re-encode.
    #[derive(Clone, Copy, Debug)]
    pub struct PartialConfig {
        /// Block tile size in pixels; a positive multiple of 8.
        pub tile_size: usize,
        /// When true, untouched blocks keep their original entropy bytes/bits
        /// (only touched blocks get new residuals + entropy codes). When false,
        /// affected planes are entropy-encoded end-to-end again, but DCT +
        /// quantization still runs only on touched blocks.
        pub preserve_old_stream: bool,
    }

    impl PartialConfig {
        pub fn new(tile_size: usize, preserve_old_stream: bool) -> Self {
            Self { tile_size, preserve_old_stream }
        }
    }

    /// Statistics reported by [`reencode`].
    #[derive(Clone, Debug, Default)]
    pub struct PartialStats {
        pub output_len: usize,
        pub full_planes: usize,
        /// Number of blocks recomputed (residuals + entropy) per plane.
        pub affected_blocks: Vec<usize>,
        /// Total blocks across all planes.
        pub total_blocks: usize,
        /// Size of the serialized block index in bytes.
        pub index_bytes: usize,
        pub elapsed: Duration,
    }

    fn blocks_in_rect(
        rect: &DirtyRect, img_w: usize, img_h: usize,
        plane_w: usize, plane_h: usize, subsampled: bool, tile: usize,
    ) -> Result<Vec<usize>> {
        let x0 = rect.x0.min(img_w);
        let x1 = rect.x1.min(img_w);
        let y0 = rect.y0.min(img_h);
        let y1 = rect.y1.min(img_h);
        if x1 <= x0 || y1 <= y0 {
            return Err(PartialError::BadConfig("dirty rectangle outside image"));
        }
        // Map full-resolution pixels onto the plane's sample grid.
        let (sx0, sx1, sy0, sy1) = if subsampled {
            (x0 / 2, (x1 + 1) / 2, y0 / 2, (y1 + 1) / 2)
        } else {
            (x0, x1, y0, y1)
        };
        let pbw = (plane_w + 7) / 8;
        let pbh = (plane_h + 7) / 8;
        let bx0 = (sx0 / tile) * (tile / 8);
        let by0 = (sy0 / tile) * (tile / 8);
        let bx1 = (((sx1 - 1) / tile) + 1) * (tile / 8);
        let by1 = (((sy1 - 1) / tile) + 1) * (tile / 8);
        let mut out = Vec::new();
        for by in by0..by1.min(pbh) {
            for bx in bx0..bx1.min(pbw) {
                out.push(by * pbw + bx);
            }
        }
        Ok(out)
    }

    fn union_affected(
        rects: &[DirtyRect], fp: &FileProfile, pp: PlaneProfile, tile: usize,
    ) -> Result<Vec<usize>> {
        let subsampled = pp.plane_w != fp.width || pp.plane_h != fp.height;
        let mut set = std::collections::BTreeSet::new();
        for r in rects {
            for b in blocks_in_rect(r, fp.width, fp.height, pp.plane_w, pp.plane_h, subsampled, tile)? {
                set.insert(b);
            }
        }
        Ok(set.into_iter().collect())
    }


    fn rle_encode_block(block: &Block) -> Vec<u8> {
        let cap = 2 + 63 * 3 + 3;
        let mut buf = vec![0u8; cap];
        let mut pos: i32 = 0;
        entropy::encode_block_to_buffer(block, &mut buf, &mut pos);
        buf.truncate(pos as usize);
        buf
    }

    // -- Plane source extraction ---------------------------------------------
    struct PlaneSources {
        /// One flat sample buffer per plane.
        planes: Vec<Vec<u8>>,
        /// For RLE planar paths, true when the plane samples are interleaved
        /// inside `interleaved` instead of flat `planes`.
        interleaved: Option<(Vec<u8>, usize)>, // (image, stride)
    }

    fn build_sources(fp: &FileProfile, new_image: &[u8]) -> Result<PlaneSources> {
        let npix = fp.width * fp.height;
        if new_image.len() < npix * fp.channels {
            return Err(PartialError::ImageSizeMismatch);
        }
        if !fp.is_huffman {
            // RLE: keep the interleaved image; blocks are extracted per channel.
            let img = new_image[..npix * fp.channels].to_vec();
            return Ok(PlaneSources { planes: Vec::new(), interleaved: Some((img, fp.channels)) });
        }
        let planes = if fp.channels == 3 {
            let (y, cb, cr) = colorspace::rgb_to_ycbcr420(&new_image[..npix * 3], fp.width, fp.height);
            vec![y, cb, cr]
        } else {
            let (y, cb, cr, a) =
                colorspace::rgba_to_ycbcr420a(&new_image[..npix * 4], fp.width, fp.height);
            vec![y, cb, cr, a]
        };
        Ok(PlaneSources { planes, interleaved: None })
    }

    // -- Core entry point -----------------------------------------------------
    /// Partially re-encode `bg` after the pixels in `dirty_rects` changed.
    ///
    /// `new_image` is the complete post-edit image in the same channel layout
    /// as the original (1/3/4 bytes per pixel). The original `bg` bytes are
    /// never modified; the new stream is returned together with a fresh block
    /// index and run statistics.
    pub fn reencode(
        bg: &[u8],
        index: &BlockIndex,
        new_image: &[u8],
        dirty_rects: &[DirtyRect],
        config: PartialConfig,
    ) -> Result<(Vec<u8>, BlockIndex, PartialStats)> {
        let started = std::time::Instant::now();
        if config.tile_size == 0 || config.tile_size % 8 != 0 || config.tile_size > u16::MAX as usize {
            return Err(PartialError::BadConfig("tile_size must be a multiple of 8"));
        }
        let fp = parse_header(bg)?;
        if fp.width != index.width
            || fp.height != index.height
            || fp.version != index.version
            || fp.quality != index.quality
            || bg.len() != index.source_len
        {
            return Err(PartialError::IndexSourceMismatch);
        }
        for r in dirty_rects {
            if r.x1 > fp.width || r.y1 > fp.height {
                return Err(PartialError::BadConfig("dirty rectangle exceeds image bounds"));
            }
        }
        let profiles = plane_profiles(&fp);
        if profiles.len() != index.planes.len() {
            return Err(PartialError::IndexSourceMismatch);
        }
        let tables = quant_tables(&fp);
        let thresholds = sparsify_thresholds(&fp);
        let sources = build_sources(&fp, new_image)?;

        let mut stats = PartialStats::default();
        let mut out: Vec<u8> = Vec::with_capacity(bg.len() + 1024);
        out.extend_from_slice(&bg[..BG_HEADER_LEN]);

        let mut new_plane_index: Vec<PlaneIndex> = Vec::with_capacity(profiles.len());
        let mut src_pos = BG_HEADER_LEN;

        for (pi, pp) in profiles.iter().enumerate() {
            let pbw = (pp.plane_w + 7) / 8;
            let n = pbw * ((pp.plane_h + 7) / 8);
            if index.planes[pi].offsets.len() != n || index.planes[pi].dcs.len() != n {
                return Err(PartialError::IndexSourceMismatch);
            }
            let affected = union_affected(dirty_rects, &fp, *pp, config.tile_size)?;
            let affected_set: std::collections::BTreeSet<usize> = affected.iter().copied().collect();
            stats.affected_blocks.push(affected.len());
            stats.total_blocks += n;

            if fp.is_huffman {
                if src_pos + 4 > bg.len() { return Err(PartialError::TruncatedStream); }
                let payload_len =
                    u32::from_le_bytes(bg[src_pos..src_pos + 4].try_into().unwrap()) as usize;
                let data_start = src_pos + 4;
                let data_end = data_start + payload_len;
                if data_end > bg.len() { return Err(PartialError::TruncatedStream); }
                let raw = remove_byte_stuffing(&bg[data_start..data_end]);

                let (new_payload, new_offsets, new_dcs, new_valid_bits) = reencode_huffman_plane(
                    &raw,
                    &index.planes[pi],
                    &affected_set,
                    &sources.planes[pi],
                    *pp,
                    &tables[pi],
                    thresholds[pi].as_ref(),
                    config.preserve_old_stream,
                )?;
                src_pos = data_end;

                out.extend_from_slice(&(new_payload.len() as u32).to_le_bytes());
                out.extend_from_slice(&new_payload);
                new_plane_index.push(PlaneIndex {
                    offsets: new_offsets, dcs: new_dcs, valid_bits: new_valid_bits,
                });
            } else {
                // RLE plane: scan boundaries from the source on demand.
                let (old_offsets, _) = scan_rle_plane(&bg[src_pos..], n)?;
                let (img, stride) = sources.interleaved.as_ref().unwrap();
                let (new_payload, new_offsets, new_dcs) = reencode_rle_plane(
                    bg,
                    src_pos,
                    &old_offsets,
                    &affected_set,
                    img,
                    *stride,
                    pi,
                    *pp,
                    &tables[pi],
                    config.preserve_old_stream,
                )?;
                src_pos += *old_offsets.last().unwrap_or(&0) as usize
                    + rle_consumed_len(&bg[src_pos + *old_offsets.last().unwrap_or(&0) as usize..])?;
                out.extend_from_slice(&new_payload);
                new_plane_index.push(PlaneIndex { offsets: new_offsets, dcs: new_dcs, valid_bits: 0 });
            }
        }

        // Anything after the planes (ICC trailer) is copied byte-for-byte.
        if src_pos < bg.len() {
            out.extend_from_slice(&bg[src_pos..]);
        }

        let mut new_index = BlockIndex {
            width: fp.width,
            height: fp.height,
            version: fp.version,
            quality: fp.quality,
            tile_size: config.tile_size,
            source_len: out.len(),
            planes: new_plane_index,
        };
        let index_bytes = write_index(&new_index).len();
        new_index.source_len = out.len();
        stats.output_len = out.len();
        stats.full_planes = profiles.len();
        stats.index_bytes = index_bytes;
        stats.elapsed = started.elapsed();
        Ok((out, new_index, stats))
    }


    /// Local MSB-first entropy writer with exact byte semantics of the
    /// production `huffman::BitWriter` (0xFF stuffing, 1-pad flush) plus a
    /// running count of emitted logical bits for block-boundary bookkeeping.
    struct LocalBitWriter {
        buf: Vec<u8>,
        bit_buf: u64,
        bits_in: u8,
        /// Total logical bits emitted (pre-stuffing). Never decreases.
        logical_bits: usize,
    }

    impl LocalBitWriter {
        fn new() -> Self {
            Self { buf: Vec::with_capacity(4096), bit_buf: 0, bits_in: 0, logical_bits: 0 }
        }
        fn push_byte(&mut self, byte: u8) {
            self.buf.push(byte);
            if byte == 0xFF {
                self.buf.push(0x00);
            }
        }
        fn write_bits(&mut self, code: u16, n: u8) {
            if n == 0 { return; }
            let mask = (1u64 << n) - 1;
            self.bit_buf = (self.bit_buf << n) | ((code as u64) & mask);
            self.bits_in += n;
            self.logical_bits += n as usize;
            while self.bits_in >= 8 {
                let shift = self.bits_in - 8;
                let byte = (self.bit_buf >> shift) as u8;
                self.push_byte(byte);
                self.bits_in -= 8;
                if self.bits_in == 0 {
                    self.bit_buf = 0;
                } else {
                    self.bit_buf &= (1u64 << self.bits_in) - 1;
                }
            }
        }
        fn copy_raw_bits(&mut self, src: &[u8], start: usize, n: usize) {
            for i in 0..n {
                let p = start + i;
                let b = (src[p >> 3] >> (7 - (p & 7))) & 1;
                self.write_bits(b as u16, 1);
            }
        }
        fn flush(&mut self) {
            if self.bits_in > 0 {
                let pad = 8 - self.bits_in;
                self.write_bits((1u16 << pad) - 1, pad);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn reencode_huffman_plane(
        raw: &[u8],
        old: &PlaneIndex,
        affected: &std::collections::BTreeSet<usize>,
        plane_samples: &[u8],
        pp: PlaneProfile,
        table: &[i16; 64],
        thr: Option<&[i16; 64]>,
        preserve_old_stream: bool,
    ) -> Result<(Vec<u8>, Vec<u32>, Vec<i16>, u32)> {
        let n = old.offsets.len();
        let pbw = (pp.plane_w + 7) / 8;
        let dc_table: [(u8, u16); 12] = if pp.is_chroma { CHROMA_DC_TABLE } else { LUMA_DC_TABLE };
        let ac_owned: AcTable = if pp.use_chroma_ac { chroma_ac_table() } else { luma_ac_table() };

        // Stage 2: residuals (DCT+quant+sparsify+clamp) only for touched blocks.
        // Keep block ids attached — `affected` is a BTreeSet, so ids and the
        // transformed blocks must not be paired by positional index.
        let mut touched: Vec<(usize, Block)> = Vec::with_capacity(affected.len());
        for &bi in affected {
            let by = (bi / pbw) * 8;
            let bx = (bi % pbw) * 8;
            let mut b = block_from_flat(plane_samples, pp.plane_w, pp.plane_h, bx, by);
            transform_block(&mut b, table, true, thr);
            touched.push((bi, b));
        }

        // Blocks whose entropy symbols must be (re)written: every touched
        // block, plus — under DC-delta profiles — the immediate successor of
        // each affected run. The successor's *absolute* DC is unchanged, but
        // its stored delta references the previous block's DC, so its symbols
        // have to be regenerated; blocks after the successor stay valid.
        let mut reemit: std::collections::BTreeSet<usize> = affected.clone();
        if pp.use_dc_delta {
            for &bi in affected {
                if bi + 1 < n && !affected.contains(&(bi + 1)) {
                    reemit.insert(bi + 1);
                }
            }
        }

        // Decode the old plane whenever some block is re-emitted, or whenever
        // the caller asked for a full plane entropy re-encode: in non-preserve
        // mode every block is re-emitted, so its coefficients must be
        // materialized even when no block was actually changed.
        let need_decoded = !preserve_old_stream || !reemit.is_empty();
        let mut blocks = if need_decoded {
            decode_huffman_blocks_unstuffed(raw, n, pp)?
        } else {
            vec![Block::new(); n]
        };
        for (bi, b) in touched.iter() {
            blocks[*bi] = *b;
        }

        // Stage 3: splice or re-emit; record every block's bit start and DC.
        let mut w = LocalBitWriter::new();
        let mut prev_dc = 0i16;
        let mut new_offsets = Vec::with_capacity(n);
        let mut new_dcs = Vec::with_capacity(n);

        for bi in 0..n {
            new_offsets.push(w.logical_bits as u32);
            if preserve_old_stream && !reemit.contains(&bi) {
                let start = old.offsets[bi] as usize;
                // The final block ends at the payload's valid bit count, not
                // the byte boundary — copying the trailing 1-pad would emit
                // phantom symbols before our own flush.
                let end = if bi + 1 < n {
                    old.offsets[bi + 1] as usize
                } else {
                    old.valid_bits as usize
                };
                if end > raw.len() * 8 || start > end {
                    return Err(PartialError::BadEntropyStream);
                }
                w.copy_raw_bits(raw, start, end - start);
                if pp.use_dc_delta {
                    prev_dc = old.dcs[bi];
                }
                new_dcs.push(old.dcs[bi]);
            } else {
                write_huffman_block_local(
                    &mut w, &blocks[bi], &mut prev_dc,
                    &dc_table, &ac_owned, pp.use_dc_delta,
                );
                new_dcs.push(blocks[bi].data[ZIGZAG[0]]);
            }
        }
        let valid_bits = w.logical_bits as u32;
        w.flush();
        Ok((w.buf, new_offsets, new_dcs, valid_bits))
    }

    /// Per-block Huffman emission identical to
    /// `huffman::encode_plane_with_profile`, targeting [`LocalBitWriter`].
    fn write_huffman_block_local(
        w: &mut LocalBitWriter, block: &Block, prev_dc: &mut i16,
        dc_table: &[(u8, u16); 12], ac_table: &AcTable, use_dc_delta: bool,
    ) {
        let dc_val = block.data[ZIGZAG[0]];
        let dc_emit = if use_dc_delta {
            let d = dc_val.wrapping_sub(*prev_dc);
            *prev_dc = dc_val;
            d
        } else {
            dc_val
        };
        let dc_cat = category(dc_emit);
        let (dc_len, dc_code) = dc_table[dc_cat as usize];
        w.write_bits(dc_code, dc_len);
        if dc_cat > 0 {
            w.write_bits(magnitude_bits(dc_emit, dc_cat), dc_cat);
        }
        let eob = ac_table[0x00];
        let zrl = ac_table[0xF0];
        let mut last_nz = 63usize;
        while last_nz > 0 && block.data[ZIGZAG[last_nz]] == 0 {
            last_nz -= 1;
        }
        if last_nz == 0 {
            w.write_bits(eob.1, eob.0);
            return;
        }
        let mut zero_run: u8 = 0;
        for i in 1..=last_nz {
            let val = block.data[ZIGZAG[i]];
            if val == 0 {
                zero_run += 1;
                continue;
            }
            while zero_run >= 16 {
                w.write_bits(zrl.1, zrl.0);
                zero_run -= 16;
            }
            let cat = category(val);
            let sym = ((zero_run as usize) << 4) | cat as usize;
            let (al, ac) = ac_table[sym];
            debug_assert!(al > 0, "missing AC Huffman symbol {sym:#04x}");
            if al == 0 {
                let (el, ec) = ac_table[0x00];
                w.write_bits(ec, el);
                return;
            }
            w.write_bits(ac, al);
            w.write_bits(magnitude_bits(val, cat), cat);
            zero_run = 0;
        }
        w.write_bits(eob.1, eob.0);
    }

    #[allow(clippy::too_many_arguments)]
    fn reencode_rle_plane(
        bg: &[u8],
        plane_start: usize,
        old_offsets: &[u32],
        affected: &std::collections::BTreeSet<usize>,
        image: &[u8],
        stride: usize,
        channel: usize,
        pp: PlaneProfile,
        table: &[i16; 64],
        preserve_old_stream: bool,
    ) -> Result<(Vec<u8>, Vec<u32>, Vec<i16>)> {
        let n = old_offsets.len();
        let pbw = (pp.plane_w + 7) / 8;
        let mut out: Vec<u8> = Vec::new();
        let mut new_offsets = Vec::with_capacity(n);
        let mut new_dcs = Vec::with_capacity(n);

        // Non-preserve mode still only transforms touched blocks; untouched
        // blocks are copied from the old stream byte-for-byte (RLE blocks are
        // self-delimiting, so both modes splice identically here).
        let _ = preserve_old_stream;
        for bi in 0..n {
            new_offsets.push(out.len() as u32);
            if affected.contains(&bi) {
                let by = (bi / pbw) * 8;
                let bx = (bi % pbw) * 8;
                let mut b = block_from_interleaved(
                    image, pp.plane_w, pp.plane_h, stride, channel, bx, by,
                );
                transform_block(&mut b, table, false, None);
                out.extend_from_slice(&rle_encode_block(&b));
                new_dcs.push(b.data[ZIGZAG[0]]);
            } else {
                let start = plane_start + old_offsets[bi] as usize;
                let len = rle_consumed_len(&bg[start..])?;
                if start + len > bg.len() {
                    return Err(PartialError::TruncatedStream);
                }
                out.extend_from_slice(&bg[start..start + len]);
                new_dcs.push(i16::from_le_bytes([bg[start], bg[start + 1]]));
            }
        }
        Ok((out, new_offsets, new_dcs))
    }


    // =======================================================================
    // Tests. These use the pure-Rust DCT/quant path so `cargo test` links
    // without the C objects; a deterministic reference full-image encoder
    // built from the same primitives is the byte-identity oracle.
    // =======================================================================
    #[cfg(test)]
    mod partial_tests {
        use super::*;
        use crate::huffman;
        use std::time::Instant;

        const W: usize = 32;
        const H: usize = 24; // deliberately not a multiple of 8 (edge blocks)

        fn sample_image(seed: u8) -> Vec<u8> {
            let mut img = vec![0u8; W * H * 3];
            for y in 0..H {
                for x in 0..W {
                    let i = (y * W + x) * 3;
                    img[i]     = ((x * 7 + y * 3 + seed as usize) & 0xFF) as u8;
                    img[i + 1] = ((x * 5 + y * 11 + seed as usize * 2) & 0xFF) as u8;
                    img[i + 2] = ((x * 13 + y * 17 + seed as usize * 5) & 0xFF) as u8;
                }
            }
            img
        }

        fn sample_gray(seed: u8) -> Vec<u8> {
            let mut img = vec![0u8; W * H];
            for y in 0..H {
                for x in 0..W {
                    img[y * W + x] = ((x * 7 + y * 3 + seed as usize) & 0xFF) as u8;
                }
            }
            img
        }

        /// Test-only full RGB .bg (v18) encoder using the exact same building
        /// blocks the partial path uses, so identity is meaningful.
        fn reference_encode_rgb(image: &[u8], quality: u8) -> Vec<u8> {
            let (y, cb, cr) = colorspace::rgb_to_ycbcr420(image, W, H);
            let cw = (W + 1) / 2;
            let ch = (H + 1) / 2;
            let luma_table = super::super::quant_table_for_quality_perceptual_v4(quality);
            let chroma_table = super::super::chroma_quant_table_for_quality_perceptual_v4(quality);
            let luma_thr = super::super::build_sparsify_thresholds(quality, false);
            let chroma_thr = super::super::build_sparsify_thresholds(quality, true);

            let mut out = Vec::new();
            out.extend_from_slice(b"BG\x12");
            out.extend_from_slice(&(W as u32).to_le_bytes());
            out.extend_from_slice(&(H as u32).to_le_bytes());
            out.push(quality);

            let planes: [(&[u8], usize, usize, [i16; 64], [i16; 64], bool, bool, bool, Option<[i16; 64]>); 3] = [
                (&y, W, H, luma_table, luma_table, false, false, true, Some(luma_thr)),
                (&cb, cw, ch, chroma_table, chroma_table, true, true, true, Some(chroma_thr)),
                (&cr, cw, ch, chroma_table, chroma_table, true, true, true, Some(chroma_thr)),
            ];
            for (samples, pw, ph, table, _, is_chroma, use_chroma_ac, dc_delta, thr) in planes {
                let bw = (pw + 7) / 8;
                let bh = (ph + 7) / 8;
                let mut blocks = Vec::with_capacity(bw * bh);
                for by in 0..bh {
                    for bx in 0..bw {
                        let mut b = block_from_flat(samples, pw, ph, bx * 8, by * 8);
                        transform_block(&mut b, &table, true, thr.as_ref());
                        blocks.push(b);
                    }
                }
                let buf = huffman::encode_plane_with_profile(&blocks, is_chroma, use_chroma_ac, dc_delta);
                out.extend_from_slice(&buf);
            }
            out
        }

        /// Test-only full grayscale RLE .bg (v1) encoder.
        fn reference_encode_gray(image: &[u8], quality: u8) -> Vec<u8> {
            let table = super::super::quant_table_for_quality(quality);
            let bw = (W + 7) / 8;
            let bh = (H + 7) / 8;
            let mut out = Vec::new();
            out.extend_from_slice(b"BG\x01");
            out.extend_from_slice(&(W as u32).to_le_bytes());
            out.extend_from_slice(&(H as u32).to_le_bytes());
            out.push(quality);
            for by in 0..bh {
                for bx in 0..bw {
                    let mut b = block_from_flat(image, W, H, bx * 8, by * 8);
                    transform_block(&mut b, &table, false, None);
                    out.extend_from_slice(&rle_encode_block(&b));
                }
            }
            out
        }

        fn patch(img: &mut [u8], x0: usize, y0: usize, x1: usize, y1: usize, val: u8) {
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * W + x) * 3;
                    img[i] = val;
                    img[i + 1] = val.wrapping_add(40);
                    img[i + 2] = val.wrapping_add(90);
                }
            }
        }

        fn patch_gray(img: &mut [u8], x0: usize, y0: usize, x1: usize, y1: usize, val: u8) {
            for y in y0..y1 {
                for x in x0..x1 {
                    img[y * W + x] = val;
                }
            }
        }

        fn scenarios() -> Vec<(&'static str, DirtyRect)> {
            vec![
                ("single-block", DirtyRect::new(8, 8, 12, 12).unwrap()),
                ("edge-block",   DirtyRect::new(28, 20, 32, 24).unwrap()), // bottom-right edge
                ("multi-block",  DirtyRect::new(6, 5, 20, 14).unwrap()),  // spans several 8x8 blocks
                ("full-image",   DirtyRect::new(0, 0, W, H).unwrap()),
            ]
        }

        #[test]
        fn index_roundtrip_and_byte_identity_rgb() {
            let quality = 75;
            let original = sample_image(1);
            let bg = reference_encode_rgb(&original, quality);
            let index = build_index(&bg, 8).unwrap();
            let serialized = write_index(&index);
            let reparsed = read_index(&serialized, &bg).unwrap();
            assert_eq!(reparsed.planes.len(), 3);
            assert_eq!(reparsed.planes[0].offsets.len(), ((W + 7) / 8) * ((H + 7) / 8));

            for (name, rect) in scenarios() {
                let mut edited = original.clone();
                patch(&mut edited, rect.x0, rect.y0, rect.x1, rect.y1, 200);
                let oracle = reference_encode_rgb(&edited, quality);

                for preserve in [true, false] {
                    let (out, new_index, stats) = reencode(
                        &bg, &reparsed, &edited, &[rect],
                        PartialConfig::new(8, preserve),
                    )
                    .unwrap_or_else(|e| panic!("{name} preserve={preserve} failed: {e}"));

                    // Deterministic codec: a correct partial rebuild is
                    // byte-identical to a full re-encode of the edited image.
                    assert_eq!(out, oracle, "{name} preserve={preserve} not byte-identical to full re-encode");
                    // Fresh index describes the new stream and re-parses.
                    let bytes = write_index(&new_index);
                    let re2 = read_index(&bytes, &out).unwrap();
                    assert_eq!(re2.source_len, out.len());
                    assert!(stats.affected_blocks.iter().sum::<usize>() > 0 || name == "full-image");
                }
            }
        }

        #[test]
        fn rgba_four_plane_path_is_byte_identical() {
            // v19: Y Cb Cr A — exercises the fourth (full-res alpha) plane.
            let quality = 70;
            let mut original = vec![0u8; W * H * 4];
            for y in 0..H {
                for x in 0..W {
                    let i = (y * W + x) * 4;
                    original[i]     = ((x * 7 + y * 3) & 0xFF) as u8;
                    original[i + 1] = ((x * 5 + y * 11) & 0xFF) as u8;
                    original[i + 2] = ((x * 13 + y * 17) & 0xFF) as u8;
                    original[i + 3] = ((x + y * 9) & 0xFF) as u8;
                }
            }
            let (y, cb, cr, a) = colorspace::rgba_to_ycbcr420a(&original, W, H);
            let cw = (W + 1) / 2;
            let ch = (H + 1) / 2;
            let luma_table = super::super::quant_table_for_quality_perceptual_v4(quality);
            let chroma_table = super::super::chroma_quant_table_for_quality_perceptual_v4(quality);
            let luma_thr = super::super::build_sparsify_thresholds(quality, false);
            let chroma_thr = super::super::build_sparsify_thresholds(quality, true);

            let encode = |yy: &[u8], cbb: &[u8], crr: &[u8], aa: &[u8]| -> Vec<u8> {
                let mut out = Vec::new();
                out.extend_from_slice(b"BG\x13");
                out.extend_from_slice(&(W as u32).to_le_bytes());
                out.extend_from_slice(&(H as u32).to_le_bytes());
                out.push(quality);
                let planes: [(&[u8], usize, usize, [i16; 64], bool, bool, bool, Option<[i16; 64]>); 4] = [
                    (yy, W, H, luma_table, false, false, true, Some(luma_thr)),
                    (cbb, cw, ch, chroma_table, true, true, true, Some(chroma_thr)),
                    (crr, cw, ch, chroma_table, true, true, true, Some(chroma_thr)),
                    (aa, W, H, luma_table, false, false, true, Some(luma_thr)),
                ];
                for (samples, pw, ph, table, is_chroma, use_chroma_ac, dc_delta, thr) in planes {
                    let bw = (pw + 7) / 8;
                    let bh = (ph + 7) / 8;
                    let mut blocks = Vec::new();
                    for by in 0..bh {
                        for bx in 0..bw {
                            let mut b = block_from_flat(samples, pw, ph, bx * 8, by * 8);
                            transform_block(&mut b, &table, true, thr.as_ref());
                            blocks.push(b);
                        }
                    }
                    out.extend_from_slice(
                        &huffman::encode_plane_with_profile(&blocks, is_chroma, use_chroma_ac, dc_delta),
                    );
                }
                out
            };

            let bg = encode(&y, &cb, &cr, &a);
            let index = build_index(&bg, 8).unwrap();
            assert_eq!(index.planes.len(), 4);
            let mut edited = original.clone();
            // Edit RGB and alpha inside one interior block.
            for yp in 8..12 {
                for xp in 8..12 {
                    let i = (yp * W + xp) * 4;
                    edited[i] = 230;
                    edited[i + 1] = 10;
                    edited[i + 2] = 60;
                    edited[i + 3] = 140;
                }
            }
            let (ey, ecb, ecr, ea) = colorspace::rgba_to_ycbcr420a(&edited, W, H);
            let oracle = encode(&ey, &ecb, &ecr, &ea);
            for preserve in [true, false] {
                let (out, new_index, _) = reencode(
                    &bg, &index, &edited,
                    &[DirtyRect::new(8, 8, 12, 12).unwrap()],
                    PartialConfig::new(8, preserve),
                )
                .unwrap_or_else(|e| panic!("rgba preserve={preserve} failed: {e}"));
                assert_eq!(out, oracle, "rgba preserve={preserve} mismatch");
                // The rebuilt index must describe the new stream and re-parse.
                read_index(&write_index(&new_index), &out).unwrap();
            }
        }

        #[test]
        fn truncated_entropy_stream_is_rejected() {
            let original = sample_image(1);
            let bg = reference_encode_rgb(&original, 75);
            // Cut the file mid-plane: the scanner must fail, not panic.
            let cut = &bg[..bg.len() - 6];
            assert!(matches!(
                build_index(cut, 8).err(),
                Some(PartialError::TruncatedStream) | Some(PartialError::BadEntropyStream)
            ));
            // A non-BG buffer is a hard error too.
            assert_eq!(build_index(b"not a bg file at all", 8).err(), Some(PartialError::NotABgFile));
        }

        #[test]
        fn partial_preserves_original_and_rle_path() {
            let quality = 60;
            let original = sample_gray(3);
            let bg = reference_encode_gray(&original, quality);
            let bg_before = bg.clone();
            let index = build_index(&bg, 8).unwrap();
            let idx_bytes = write_index(&index);

            for (name, rect) in scenarios() {
                let mut edited = original.clone();
                patch_gray(&mut edited, rect.x0, rect.y0, rect.x1, rect.y1, 210);
                let oracle = reference_encode_gray(&edited, quality);
                for preserve in [true, false] {
                    let (out, _, _) = reencode(
                        &bg, &index, &edited, &[rect],
                        PartialConfig::new(8, preserve),
                    )
                    .unwrap_or_else(|e| panic!("rle {name} failed: {e}"));
                    assert_eq!(out, oracle, "rle {name} preserve={preserve} mismatch");
                }
            }
            // The source .bg must be untouched byte-for-byte.
            assert_eq!(bg, bg_before, "source .bg was modified");
            let _ = idx_bytes;
        }

        #[test]
        fn tile_size_is_configurable_and_validated() {
            let original = sample_image(1);
            let bg = reference_encode_rgb(&original, 75);
            // tile 16 covers the image in a coarser grid.
            let index16 = build_index(&bg, 16).unwrap();
            assert_eq!(index16.tile_size, 16);
            let mut edited = original.clone();
            patch(&mut edited, 8, 8, 10, 10, 50);
            let rect = DirtyRect::new(8, 8, 10, 10).unwrap();
            let oracle = reference_encode_rgb(&edited, 75);
            let (out, _, _) =
                reencode(&bg, &index16, &edited, &[rect], PartialConfig::new(16, true)).unwrap();
            assert_eq!(out, oracle);

            assert_eq!(
                build_index(&bg, 7).err(),
                Some(PartialError::BadConfig("tile_size must be a multiple of 8"))
            );
            assert_eq!(
                reencode(&bg, &index16, &edited, &[rect], PartialConfig::new(0, true)).err(),
                Some(PartialError::BadConfig("tile_size must be a multiple of 8"))
            );
        }

        #[test]
        fn corrupted_index_is_rejected() {
            let original = sample_image(1);
            let bg = reference_encode_rgb(&original, 75);
            let index = build_index(&bg, 8).unwrap();
            let mut bytes = write_index(&index);

            // 1. Flipped payload byte breaks the CRC.
            let mut crc_bad = bytes.clone();
            crc_bad[30] ^= 0x01;
            assert_eq!(read_index(&crc_bad, &bg).err(), Some(PartialError::IndexChecksumMismatch));

            // 2. Truncation is detected.
            let truncated = bytes[..bytes.len() - 5].to_vec();
            assert!(matches!(
                read_index(&truncated, &bg).err(),
                Some(PartialError::IndexTruncated) | Some(PartialError::IndexChecksumMismatch)
            ));

            // 3. Bad magic.
            bytes[0] = b'X';
            // Recompute CRC so magic, not checksum, is the reported failure.
            let len = bytes.len();
            let crc = crc32_ieee(&bytes[..len - 4]);
            bytes[len - 4..].copy_from_slice(&crc.to_le_bytes());
            assert_eq!(read_index(&bytes, &bg).err(), Some(PartialError::BadIndexMagic));

            // 4. Index built for a different .bg (length mismatch).
            let other = reference_encode_rgb(&sample_image(9), 75);
            let other_index = build_index(&other, 8).unwrap();
            let other_bytes = write_index(&other_index);
            assert_eq!(read_index(&other_bytes, &bg).err(), Some(PartialError::IndexSourceMismatch));
        }

        #[test]
        fn no_change_reproduces_original_stream() {
            let original = sample_image(7);
            let bg = reference_encode_rgb(&original, 80);
            let index = build_index(&bg, 8).unwrap();
            // Empty dirty set: nothing to recompute. Both splice modes must
            // reproduce the original stream byte-for-byte.
            for preserve in [true, false] {
                let (out, _, stats) =
                    reencode(&bg, &index, &original, &[], PartialConfig::new(8, preserve)).unwrap();
                assert_eq!(out, bg, "no-change preserve={preserve} changed the stream");
                assert_eq!(stats.affected_blocks.iter().sum::<usize>(), 0);
            }
        }

        /// Prints the four-scorecard metrics required by the task. Run with
        /// `--nocapture`; the eval script greps the `PARTIAL_METRIC` lines.
        #[test]
        fn metrics_scorecard() {
            let quality = 75;
            let original = sample_image(1);
            let bg = reference_encode_rgb(&original, quality);
            let index = build_index(&bg, 8).unwrap();
            let idx_bytes = write_index(&index).len();
            let full_len = bg.len();

            // Warm up so timing is stable.
            let warm = original.clone();
            let _ = reencode(&bg, &index, &warm, &[DirtyRect::new(0, 0, 1, 1).unwrap()],
                PartialConfig::new(8, true)).unwrap();

            println!();
            println!("PARTIAL_METRIC image={W}x{H} rgb v18 quality={quality} full_stream={full_len}B index_overhead={idx_bytes}B");
            println!("{:<12} {:>10} {:>12} {:>14} {:>14} {:>12} {:>10}",
                "scenario", "blocks", "stream_B", "delta_vs_full", "index_overh_B", "time_us", "full_us");
            for (name, rect) in scenarios() {
                let mut edited = original.clone();
                patch(&mut edited, rect.x0, rect.y0, rect.x1, rect.y1, 200);
                let oracle = reference_encode_rgb(&edited, quality);

                let mut best_partial = u128::MAX;
                let mut best_full = u128::MAX;
                let mut last_stats = None;
                for _ in 0..5 {
                    let t0 = Instant::now();
                    let (_, _, stats) = reencode(
                        &bg, &index, &edited, &[rect], PartialConfig::new(8, true),
                    )
                    .unwrap();
                    best_partial = best_partial.min(t0.elapsed().as_micros());
                    last_stats = Some(stats);

                    let t1 = Instant::now();
                    let _full = reference_encode_rgb(&edited, quality);
                    best_full = best_full.min(t1.elapsed().as_micros());
                }
                let stats = last_stats.unwrap();
                let n_blocks = stats.affected_blocks.iter().sum::<usize>();
                // Stream size: partial output must equal the full re-encode size.
                assert_eq!(stats.output_len, oracle.len());
                println!("{:<12} {:>10} {:>12} {:>14} {:>14} {:>12} {:>10}",
                    name, n_blocks, stats.output_len,
                    stats.output_len as isize - full_len as isize,
                    stats.index_bytes, best_partial, best_full);
            }
            println!("PARTIAL_METRIC consistency=byte-identical-to-full-reencode pixel_consistency=exact");
        }

        #[test]
        fn local_decoder_matches_production_decoder() {
            // Round-trip random blocks through the production encoder, then
            // decode the unstuffed payload with the local scanner/decoder.
            fn next_u32(state: &mut u64) -> u32 {
                *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (*state >> 32) as u32
            }
            let mut seed = 0xabcdefu64;
            let n = 64usize;
            let mut blocks = Vec::with_capacity(n);
            for _ in 0..n {
                let mut b = Block::new();
                b.data[ZIGZAG[0]] = ((next_u32(&mut seed) % 2047) as i16) - 1023;
                for zi in 1..64 {
                    let r = next_u32(&mut seed) % 100;
                    if r >= 65 {
                        b.data[ZIGZAG[zi]] = (next_u32(&mut seed) % 2047) as i16 - 1023;
                    }
                }
                huffman::clamp_block_jpeg_coeffs(&mut b);
                blocks.push(b);
            }
            for (is_chroma, use_chroma_ac, dc_delta) in
                [(false, false, true), (true, true, true), (false, false, false)]
            {
                let encoded = huffman::encode_plane_with_profile(&blocks, is_chroma, use_chroma_ac, dc_delta);
                let payload_len =
                    u32::from_le_bytes(encoded[0..4].try_into().unwrap()) as usize;
                let stuffed = &encoded[4..4 + payload_len];
                let raw = remove_byte_stuffing(stuffed);
                let pp = PlaneProfile {
                    plane_w: 8 * 8, plane_h: 8, is_chroma, use_chroma_ac, use_dc_delta: dc_delta,
                };
                let local = decode_huffman_blocks_unstuffed(&raw, n, pp).unwrap();
                for (i, (o, d)) in blocks.iter().zip(local.iter()).enumerate() {
                    assert_eq!(o.data, d.data, "profile {is_chroma}/{use_chroma_ac}/{dc_delta} block {i} mismatch");
                }
            }
        }
    }
} // pub mod partial
