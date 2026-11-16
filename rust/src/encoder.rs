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

// ---------------------------------------------------------------------------
// Variable block-size path (experimental, grayscale)
//
// The image is tiled into 16×16 superblocks. Each superblock is encoded
// either as one 16×16 DCT block (flat content) or as four 8×8 DCT blocks
// (textured content). The choice is driven by the u8 pixel variance inside
// the superblock and is recorded in a 1-byte block header per superblock
// (0 = four 8×8, 1 = one 16×16), emitted in scan order before that
// superblock's coefficient payload.
//
// This path is fully self-contained: it does not touch the fixed-block
// streams above, and its own stream carries version 0x80 so existing
// decoders reject it instead of mis-parsing. Everything below is additive.
// ---------------------------------------------------------------------------

/// Version byte of the experimental variable-block grayscale stream.
/// Outside the fixed-block range (1..=19) on purpose.
pub const VARBLOCK_VERSION: u8 = 0x80;

/// Variance threshold (u8 pixel variance over a 16×16 superblock).
/// Superblocks at or below this are encoded as one 16×16 block.
pub const VARBLOCK_VAR_THRESHOLD: f64 = 100.0;

/// Allowed block sizes for the variable-block path.
pub const VARBLOCK_ALLOWED_SIZES: [u8; 2] = [8, 16];

/// Validate the caller-supplied block size set.
/// Legal sets: {8}, {16}, {8,16} (any order, no duplicates).
fn validate_varblock_sizes(block_sizes: &[u8]) -> Result<Vec<u8>, String> {
    if block_sizes.is_empty() {
        return Err("invalid block size set: empty (allowed sizes are 8 and 16)".to_string());
    }
    let mut sizes = block_sizes.to_vec();
    sizes.sort_unstable();
    sizes.dedup();
    if sizes.len() != block_sizes.len() {
        return Err(format!(
            "invalid block size set {block_sizes:?}: duplicate entries"
        ));
    }
    for &s in &sizes {
        if !VARBLOCK_ALLOWED_SIZES.contains(&s) {
            return Err(format!(
                "invalid block size set {block_sizes:?}: size {s} not supported (allowed sizes are 8 and 16)"
            ));
        }
    }
    Ok(sizes)
}

/// Result of the variance-driven partition pass.
pub struct VarBlockPartition {
    /// Superblock grid dimensions (16×16 superblocks, ceil).
    pub sb_w: usize,
    pub sb_h: usize,
    /// One entry per superblock, scan order: 1 = one 16×16 block, 0 = four 8×8 blocks.
    pub modes: Vec<u8>,
    /// Wall time of the decision pass itself (excludes DCT/entropy coding).
    pub partition_time: std::time::Duration,
}

/// Decide 8 vs 16 per 16×16 superblock from intra-block variance.
/// `block_sizes` must be a valid subset of {8, 16}; illegal sets are rejected.
pub fn partition_by_variance(
    image: &[u8],
    width: usize,
    height: usize,
    block_sizes: &[u8],
) -> Result<VarBlockPartition, String> {
    let sizes = validate_varblock_sizes(block_sizes)?;
    if width == 0 || height == 0 || image.len() < width * height {
        return Err(format!(
            "invalid image: {width}x{height} with {} bytes",
            image.len()
        ));
    }
    let sb_w = width.div_ceil(16);
    let sb_h = height.div_ceil(16);
    let n_sb = sb_w * sb_h;

    let t0 = std::time::Instant::now();
    let mut modes = vec![0u8; n_sb];
    let allow8 = sizes.contains(&8);
    let allow16 = sizes.contains(&16);
    if allow16 && !allow8 {
        modes.fill(1);
    } else if allow16 && allow8 {
        for sb in 0..n_sb {
            let sbx = (sb % sb_w) * 16;
            let sby = (sb / sb_w) * 16;
            // Mean and variance of the 16×16 superblock (edge-replicated).
            let mut sum = 0u64;
            for y in 0..16 {
                let iy = (sby + y).min(height - 1);
                let row = iy * width;
                for x in 0..16 {
                    sum += image[row + (sbx + x).min(width - 1)] as u64;
                }
            }
            let mean = sum as f64 / 256.0;
            let mut acc = 0.0f64;
            for y in 0..16 {
                let iy = (sby + y).min(height - 1);
                let row = iy * width;
                for x in 0..16 {
                    let d = image[row + (sbx + x).min(width - 1)] as f64 - mean;
                    acc += d * d;
                }
            }
            if acc / 256.0 <= VARBLOCK_VAR_THRESHOLD {
                modes[sb] = 1;
            }
        }
    }
    // sizes == {8}: all modes stay 0 (fixed 8×8 through the same writer).
    Ok(VarBlockPartition {
        sb_w,
        sb_h,
        modes,
        partition_time: t0.elapsed(),
    })
}

/// Extract an n×n block as centered samples (-128..127), edge-replicated.
fn varblock_extract(image: &[u8], width: usize, height: usize, bx: usize, by: usize, n: usize) -> Vec<i16> {
    let mut out = vec![0i16; n * n];
    for y in 0..n {
        let iy = (by + y).min(height - 1);
        let row = iy * width;
        for x in 0..n {
            out[y * n + x] = image[row + (bx + x).min(width - 1)] as i16 - 128;
        }
    }
    out
}

/// Generic separable DCT-II for n×n blocks (same normalization as
/// `dct::dct_reference`: factor 2/N with cu=cv=1/sqrt(2) at DC).
fn varblock_dct(pixels: &[i16], n: usize) -> Vec<i16> {
    let nf = n as f64;
    let mut result = vec![0f64; n * n];
    for u in 0..n {
        for v in 0..n {
            let mut sum = 0.0f64;
            for x in 0..n {
                for y in 0..n {
                    sum += pixels[y * n + x] as f64
                        * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / (2.0 * nf)).cos()
                        * ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / (2.0 * nf)).cos();
                }
            }
            let cu = if u == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
            let cv = if v == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
            result[v * n + u] = (2.0 / nf) * cu * cv * sum;
        }
    }
    result.iter().map(|&c| c.round() as i16).collect()
}

/// Generic separable inverse DCT for n×n blocks. Coefficients → centered pixels.
fn varblock_idct(coef: &[i16], n: usize) -> Vec<i16> {
    let nf = n as f64;
    let mut result = vec![0f64; n * n];
    for y in 0..n {
        for x in 0..n {
            let mut sum = 0.0f64;
            for u in 0..n {
                for v in 0..n {
                    let cu = if u == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                    let cv = if v == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                    sum += cu * cv * coef[v * n + u] as f64
                        * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / (2.0 * nf)).cos()
                        * ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / (2.0 * nf)).cos();
                }
            }
            result[y * n + x] = (2.0 / nf) * sum;
        }
    }
    result.iter().map(|&c| c.round() as i16).collect()
}

/// Zigzag scan order for an n×n block (JPEG diagonal order).
fn varblock_zigzag(n: usize) -> Vec<usize> {
    let mut order = Vec::with_capacity(n * n);
    for s in 0..(2 * n - 1) {
        let lo = s.saturating_sub(n - 1);
        let hi = s.min(n - 1);
        if s % 2 == 0 {
            // even diagonal: bottom-left → top-right
            for y in (lo..=hi).rev() {
                order.push(y * n + (s - y));
            }
        } else {
            for y in lo..=hi {
                order.push(y * n + (s - y));
            }
        }
    }
    order
}

/// 16×16 luma quant table: 2× nearest-neighbor expansion of the scaled 8×8 table.
fn varblock_quant_table_16(quality: u8) -> Vec<i16> {
    let t8 = quant_table_for_quality(quality);
    let mut t = vec![0i16; 256];
    for y in 0..16 {
        for x in 0..16 {
            t[y * 16 + x] = t8[(y / 2) * 8 + (x / 2)];
        }
    }
    t
}

/// Scalar round-to-nearest quantize (self-contained; the release SIMD
/// quantizer is only used by the fixed-block paths above).
fn varblock_quantize(coefs: &mut [i16], table: &[i16]) {
    for (c, &t) in coefs.iter_mut().zip(table.iter()) {
        let t = t.max(1) as i32;
        let v = *c as i32;
        *c = (if v >= 0 { (v + t / 2) / t } else { -((-v + t / 2) / t) }) as i16;
    }
}

fn varblock_dequantize(coefs: &mut [i16], table: &[i16]) {
    for (c, &t) in coefs.iter_mut().zip(table.iter()) {
        *c = (*c as i32 * t.max(1) as i32) as i16;
    }
}

/// RLE-encode `count` coefficients in `order`: DC as i16 LE, then
/// (run: u8, level: i16 LE) pairs, EOB = 0xFF 0x00 0x00. Same wire
/// format as the fixed-block RLE path, generalized past 64 coefficients.
fn varblock_rle_write(coefs: &[i16], order: &[usize], out: &mut Vec<u8>) {
    out.extend_from_slice(&coefs[order[0]].to_le_bytes());
    let mut run: u8 = 0;
    for &idx in &order[1..] {
        let level = coefs[idx];
        if level == 0 {
            if run == 0xFE {
                out.push(0xFE);
                out.extend_from_slice(&0i16.to_le_bytes());
                run = 0;
            }
            run += 1;
        } else {
            out.push(run);
            out.extend_from_slice(&level.to_le_bytes());
            run = 0;
        }
    }
    out.push(0xFF);
    out.extend_from_slice(&0i16.to_le_bytes());
}

/// Inverse of `varblock_rle_write`. Returns `count` coefficients in
/// natural (row-major) order.
fn varblock_rle_read(
    stream: &[u8],
    pos: &mut usize,
    order: &[usize],
) -> Result<Vec<i16>, String> {
    let count = order.len();
    let mut coefs = vec![0i16; count];
    if *pos + 2 > stream.len() {
        return Err("corrupt varblock stream: truncated DC".to_string());
    }
    coefs[order[0]] = i16::from_le_bytes([stream[*pos], stream[*pos + 1]]);
    *pos += 2;
    let mut zi = 1usize;
    // The writer always terminates a block with EOB, even when every
    // coefficient was emitted — keep reading until EOB.
    loop {
        if *pos + 3 > stream.len() {
            return Err("corrupt varblock stream: truncated RLE pair".to_string());
        }
        let run = stream[*pos];
        let level = i16::from_le_bytes([stream[*pos + 1], stream[*pos + 2]]);
        *pos += 3;
        if run == 0xFF && level == 0 {
            break; // EOB
        }
        if run == 0xFE && level == 0 {
            zi += 0xFE; // max-run zero flush
            continue;
        }
        zi += run as usize;
        if zi >= count {
            return Err("corrupt varblock stream: coefficient overrun".to_string());
        }
        coefs[order[zi]] = level;
        zi += 1;
    }
    Ok(coefs)
}

/// Output of the variable-block encoder.
pub struct VarBlockEncode {
    /// Complete stream: 12-byte header + per-superblock (mode byte + payload).
    pub stream: Vec<u8>,
    /// Bytes spent on block headers (one mode byte per 16×16 superblock).
    pub header_bytes: usize,
    /// Bytes spent on coefficient payloads (stream minus file header minus block headers).
    pub payload_bytes: usize,
    /// Number of 8×8 blocks emitted.
    pub blocks8: usize,
    /// Number of 16×16 blocks emitted.
    pub blocks16: usize,
    /// Wall time of the variance partition pass.
    pub partition_time: std::time::Duration,
}

/// Encode a grayscale image with variable block sizes chosen per 16×16
/// superblock by intra-block variance. `block_sizes` must be a subset of
/// {8, 16}: {8} reproduces a fixed-8×8 tiling through this same writer,
/// {8, 16} enables the variance-driven choice. Illegal sets are rejected.
pub fn encode_grayscale_varblock(
    image: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    block_sizes: &[u8],
) -> Result<VarBlockEncode, String> {
    let part = partition_by_variance(image, width, height, block_sizes)?;
    let q = if quality == 0 { 50 } else { quality };
    let table8 = quant_table_for_quality(q);
    let table16 = varblock_quant_table_16(q);
    let zz8 = varblock_zigzag(8);
    let zz16 = varblock_zigzag(16);

    // File header: same 12-byte layout as the fixed-block streams.
    let mut stream = Vec::with_capacity(BG_HEADER_SIZE + part.modes.len() * 8);
    stream.extend_from_slice(b"BG");
    stream.push(VARBLOCK_VERSION);
    stream.extend_from_slice(&(width as u32).to_le_bytes());
    stream.extend_from_slice(&(height as u32).to_le_bytes());
    stream.push(q);

    let mut payload = Vec::new();
    let mut blocks8 = 0usize;
    let mut blocks16 = 0usize;
    for sb in 0..part.modes.len() {
        let mode = part.modes[sb];
        // Block header: 1 byte recording the chosen block size.
        payload.push(mode);
        let sbx = (sb % part.sb_w) * 16;
        let sby = (sb / part.sb_w) * 16;
        if mode == 1 {
            let px = varblock_extract(image, width, height, sbx, sby, 16);
            let mut coefs = varblock_dct(&px, 16);
            varblock_quantize(&mut coefs, &table16);
            varblock_rle_write(&coefs, &zz16, &mut payload);
            blocks16 += 1;
        } else {
            for sub in 0..4 {
                let ox = sbx + (sub % 2) * 8;
                let oy = sby + (sub / 2) * 8;
                let px = varblock_extract(image, width, height, ox, oy, 8);
                let mut coefs = varblock_dct(&px, 8);
                varblock_quantize(&mut coefs, &table8);
                varblock_rle_write(&coefs, &zz8, &mut payload);
                blocks8 += 1;
            }
        }
    }

    let header_bytes = part.modes.len();
    let payload_bytes = payload.len() - header_bytes;
    stream.extend_from_slice(&payload);
    Ok(VarBlockEncode {
        stream,
        header_bytes,
        payload_bytes,
        blocks8,
        blocks16,
        partition_time: part.partition_time,
    })
}

/// Decode a stream produced by `encode_grayscale_varblock`.
/// Returns (pixels, width, height).
pub fn decode_grayscale_varblock(stream: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    if stream.len() < BG_HEADER_SIZE {
        return Err("not a varblock stream: too short".to_string());
    }
    if &stream[0..2] != b"BG" || stream[2] != VARBLOCK_VERSION {
        return Err(format!(
            "not a varblock stream (version {:#04x}, expected {:#04x})",
            stream[2], VARBLOCK_VERSION
        ));
    }
    let width = u32::from_le_bytes([stream[3], stream[4], stream[5], stream[6]]) as usize;
    let height = u32::from_le_bytes([stream[7], stream[8], stream[9], stream[10]]) as usize;
    let quality = stream[11];
    if width == 0 || height == 0 {
        return Err("corrupt varblock stream: zero dimensions".to_string());
    }
    let table8 = quant_table_for_quality(quality);
    let table16 = varblock_quant_table_16(quality);
    let zz8 = varblock_zigzag(8);
    let zz16 = varblock_zigzag(16);
    let sb_w = width.div_ceil(16);
    let sb_h = height.div_ceil(16);

    let mut img = vec![0u8; width * height];
    let mut pos = BG_HEADER_SIZE;
    let mut place = |bx: usize, by: usize, n: usize, px: &[i16]| {
        for y in 0..n {
            let iy = by + y;
            if iy >= height {
                break;
            }
            for x in 0..n {
                let ix = bx + x;
                if ix >= width {
                    break;
                }
                img[iy * width + ix] = (px[y * n + x] + 128).clamp(0, 255) as u8;
            }
        }
    };
    for sb in 0..sb_w * sb_h {
        if pos >= stream.len() {
            return Err("corrupt varblock stream: missing block header".to_string());
        }
        let mode = stream[pos];
        pos += 1;
        let sbx = (sb % sb_w) * 16;
        let sby = (sb / sb_w) * 16;
        match mode {
            1 => {
                let mut coefs = varblock_rle_read(stream, &mut pos, &zz16)?;
                varblock_dequantize(&mut coefs, &table16);
                let px = varblock_idct(&coefs, 16);
                place(sbx, sby, 16, &px);
            }
            0 => {
                for sub in 0..4 {
                    let ox = sbx + (sub % 2) * 8;
                    let oy = sby + (sub / 2) * 8;
                    let mut coefs = varblock_rle_read(stream, &mut pos, &zz8)?;
                    varblock_dequantize(&mut coefs, &table8);
                    let px = varblock_idct(&coefs, 8);
                    place(ox, oy, 8, &px);
                }
            }
            other => {
                return Err(format!(
                    "corrupt varblock stream: invalid block header {other} at superblock {sb}"
                ));
            }
        }
    }
    Ok((img, width, height))
}

#[cfg(test)]
mod varblock_tests {
    use super::*;

    /// Deterministic xorshift64* PRNG so samples need no external files.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo + 1) as u64)) as i32
        }
    }

    /// Flat image: smooth, slowly varying content (low intra-block variance).
    fn flat_image(w: usize, h: usize) -> Vec<u8> {
        (0..h)
            .flat_map(|y| (0..w).map(move |x| (120 + ((x * 3 + y * 2) % 9)) as u8))
            .collect()
    }

    /// Textured image: strong deterministic noise (high intra-block variance).
    fn textured_image(w: usize, h: usize) -> Vec<u8> {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        (0..w * h)
            .map(|_| (128 + rng.range(-70, 70)).clamp(0, 255) as u8)
            .collect()
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let mse = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x as f64 - y as f64;
                d * d
            })
            .sum::<f64>()
            / a.len() as f64;
        if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        }
    }

    #[test]
    fn varblock_rejects_invalid_size_sets() {
        let img = flat_image(32, 32);
        for set in [
            &[][..],
            &[4][..],
            &[0][..],
            &[32][..],
            &[8, 32][..],
            &[8, 8][..],
            &[16, 16][..],
            &[8, 16, 8][..],
        ] {
            assert!(
                encode_grayscale_varblock(&img, 32, 32, 50, set).is_err(),
                "set {set:?} must be rejected"
            );
        }
        for set in [&[8][..], &[16][..], &[8, 16][..], &[16, 8][..]] {
            assert!(
                encode_grayscale_varblock(&img, 32, 32, 50, set).is_ok(),
                "set {set:?} must be accepted"
            );
        }
    }

    #[test]
    fn varblock_roundtrip_non_multiple_dims() {
        let img = textured_image(100, 60);
        let enc = encode_grayscale_varblock(&img, 100, 60, 90, &[8, 16]).unwrap();
        let (dec, w, h) = decode_grayscale_varblock(&enc.stream).unwrap();
        assert_eq!((w, h), (100, 60));
        assert!(psnr(&img, &dec) > 25.0);
    }

    #[test]
    fn varblock_decoder_rejects_foreign_stream() {
        // A fixed-block stream (version 1..=19) must not parse as varblock.
        let mut fake = vec![0u8; BG_HEADER_SIZE];
        fake[0] = b'B';
        fake[1] = b'G';
        fake[2] = 1;
        assert!(decode_grayscale_varblock(&fake).is_err());
    }

    /// One-shot metrics run: flat vs textured, fixed {8} vs variable {8,16}.
    /// Prints size ratio, PSNR delta, partition time and block-header share.
    #[test]
    fn varblock_metrics_report() {
        let (w, h, quality) = (128usize, 128usize, 50u8);
        println!("\nimage   | fixed8 B | var B | var/fixed | dPSNR (dB) | part. time | hdr%  | 16/8 blocks");
        for (name, img) in [
            ("flat", flat_image(w, h)),
            ("textured", textured_image(w, h)),
        ] {
            let fixed = encode_grayscale_varblock(&img, w, h, quality, &[8]).unwrap();
            let var = encode_grayscale_varblock(&img, w, h, quality, &[8, 16]).unwrap();
            let (dec_fixed, _, _) = decode_grayscale_varblock(&fixed.stream).unwrap();
            let (dec_var, _, _) = decode_grayscale_varblock(&var.stream).unwrap();
            let psnr_fixed = psnr(&img, &dec_fixed);
            let psnr_var = psnr(&img, &dec_var);
            let hdr_pct = 100.0 * var.header_bytes as f64 / var.stream.len() as f64;
            println!(
                "{name:8}| {:8} | {:5} | {:9.3} | {:+10.2} | {:10.1?} | {:5.2} | {}/{}",
                fixed.stream.len(),
                var.stream.len(),
                var.stream.len() as f64 / fixed.stream.len() as f64,
                psnr_var - psnr_fixed,
                var.partition_time,
                hdr_pct,
                var.blocks16,
                var.blocks8,
            );
            // Sanity: both paths must reconstruct the image. Heavy noise at
            // q50 quantizes hard, so the bound is loose on purpose.
            assert!(psnr_fixed > 20.0, "{name} fixed8 PSNR {psnr_fixed}");
            assert!(psnr_var > 20.0, "{name} var PSNR {psnr_var}");
        }
    }
}

