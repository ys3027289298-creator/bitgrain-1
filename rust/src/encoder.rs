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
// Adaptive spatial prediction (per block)
//
// The fixed predictors in the paths above assume one model for the whole
// image, which is unfriendly to heterogeneous material. This additive path
// works on n×n pixel blocks and picks, per block, the predictor with the
// smallest residual energy among a caller-provided set:
//
//   * horizontal: replicate the reconstructed left boundary column
//   * vertical:   replicate the reconstructed top boundary row
//   * average:    mean of both boundaries (fallbacks at image edges)
//
// The choice is written as a 1-byte block header in front of the block's
// coefficient payload, so the decoder follows exactly the header. Residuals
// are DCT-coded with the existing quality-scaled luma quantizer and a
// generalized zigzag/RLE layout. The encoder reconstructs every block
// (dequant + IDCT + prediction) before moving on, which keeps its predictor
// inputs bit-identical to the decoder's.
//
// Self-contained by design: no changes to the historical encode paths, no
// new dependencies, and its own version byte (0x81) that older decoders
// reject instead of mis-parsing. Grayscale only; the same routine applies to
// each luma/alpha plane independently.
//
// Stream layout:
//   "BG" | version 0x81 | width u32 LE | height u32 LE | quality u8 | block u8
//   then, per block in raster order: predictor u8, DC i16 LE,
//   then (run u8, level i16 LE) pairs, EOB = 0xFF 0x00 0x00.
// ---------------------------------------------------------------------------

/// Version byte of the adaptive-prediction grayscale stream (outside 1..=19).
pub const PRED_VERSION: u8 = 0x81;

/// Block sizes accepted by the adaptive-prediction path.
pub const PRED_ALLOWED_BLOCK_SIZES: [usize; 3] = [4, 8, 16];

/// Predictor codes carried by each block header.
pub const PRED_HORIZONTAL: u8 = 0;
pub const PRED_VERTICAL: u8 = 1;
pub const PRED_AVERAGE: u8 = 2;

/// Predictors accepted by the adaptive-prediction path.
pub const PRED_ALLOWED_PREDICTORS: [u8; 3] = [PRED_HORIZONTAL, PRED_VERTICAL, PRED_AVERAGE];

/// 12-byte .bg header plus the 1-byte block-size field.
pub const PRED_HEADER_SIZE: usize = BG_HEADER_SIZE + 1;

const PRED_FRAC_1_SQRT2: f64 = 0.7071067811865476;

/// Result of an adaptive-prediction encode pass.
#[derive(Debug)]
pub struct PredictiveEncode {
    /// Complete .bg stream.
    pub stream: Vec<u8>,
    /// Bytes spent on per-block headers (1 byte each).
    pub header_bytes: usize,
    /// Coefficient payload bytes (stream minus file header minus block headers).
    pub payload_bytes: usize,
    /// Number of coded blocks.
    pub blocks: usize,
    /// Block edge length used.
    pub block_size: usize,
    /// Per-predictor block counts, indexed by predictor code (0/1/2).
    pub predictor_counts: [usize; 3],
    /// Predictor chosen for each block, in raster order (the block-header bytes).
    pub predictor_sequence: Vec<u8>,
    /// Encoder-side reconstruction, byte-identical to what the decoder produces.
    pub reconstruction: Vec<u8>,
}

/// Result of an adaptive-prediction decode pass.
#[derive(Debug)]
pub struct PredictiveDecode {
    pub pixels: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub quality: u8,
    pub block_size: usize,
    pub blocks: usize,
    /// Bytes consumed by per-block headers.
    pub header_bytes: usize,
    /// Per-predictor block counts as read from the block headers.
    pub predictor_counts: [usize; 3],
    /// Predictor read from each block header, in raster order.
    pub predictor_sequence: Vec<u8>,
}

fn pred_validate_block_size(block_size: usize) -> Result<(), String> {
    if PRED_ALLOWED_BLOCK_SIZES.contains(&block_size) {
        Ok(())
    } else {
        Err(format!(
            "invalid block size {block_size}: allowed sizes are {:?}",
            PRED_ALLOWED_BLOCK_SIZES
        ))
    }
}

/// Validate the caller's predictor set. Must be a non-empty set drawn from
/// {horizontal, vertical, average}; duplicates are rejected. Returns the
/// normalized (ascending, duplicate-free) set for deterministic tie-breaks.
fn pred_validate_predictors(predictors: &[u8]) -> Result<Vec<u8>, String> {
    if predictors.is_empty() {
        return Err(format!(
            "invalid predictor set: empty (allowed predictors are {:?})",
            PRED_ALLOWED_PREDICTORS
        ));
    }
    let mut set = predictors.to_vec();
    set.sort_unstable();
    for pair in set.windows(2) {
        if pair[0] == pair[1] {
            return Err(format!(
                "invalid predictor set {predictors:?}: duplicate predictor {}",
                pair[0]
            ));
        }
    }
    for &p in &set {
        if !PRED_ALLOWED_PREDICTORS.contains(&p) {
            return Err(format!(
                "invalid predictor set {predictors:?}: predictor {p} not supported (allowed predictors are {:?})",
                PRED_ALLOWED_PREDICTORS
            ));
        }
    }
    Ok(set)
}

/// JPEG diagonal (zigzag) order for an n×n block.
fn pred_zigzag(n: usize) -> Vec<usize> {
    let mut order = Vec::with_capacity(n * n);
    for s in 0..(2 * n - 1) {
        let lo = s.saturating_sub(n - 1);
        let hi = s.min(n - 1);
        if s % 2 == 0 {
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

/// Quality-scaled luma quant table for an n×n block (nearest expansion of the
/// standard 8×8 table). n == 8 returns the exact historical table.
fn pred_quant_table(quality: u8, n: usize) -> Vec<i16> {
    let t8 = quant_table_for_quality(quality);
    if n == 8 {
        return t8.to_vec();
    }
    let mut table = vec![0i16; n * n];
    for y in 0..n {
        for x in 0..n {
            let sy = (y * 8 / n).min(7);
            let sx = (x * 8 / n).min(7);
            table[y * n + x] = t8[sy * 8 + sx];
        }
    }
    table
}

/// Precomputed cosine table: `cs[u * n + x] = cos((2x+1)uπ / 2n)`.
fn pred_cos_table(n: usize) -> Vec<f64> {
    let mut cs = vec![0f64; n * n];
    for u in 0..n {
        for x in 0..n {
            cs[u * n + x] =
                ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / (2.0 * n as f64)).cos();
        }
    }
    cs
}

/// Generic separable n×n DCT-II (2/N normalization, cu = 1/√2 at DC).
fn pred_dct(pixels: &[f64], n: usize, cs: &[f64]) -> Vec<f64> {
    let scale = 2.0 / n as f64;
    let mut out = vec![0f64; n * n];
    for v in 0..n {
        let cv = if v == 0 { PRED_FRAC_1_SQRT2 } else { 1.0 };
        for u in 0..n {
            let cu = if u == 0 { PRED_FRAC_1_SQRT2 } else { 1.0 };
            let mut sum = 0.0f64;
            for y in 0..n {
                let cy = cs[v * n + y];
                let row = y * n;
                for x in 0..n {
                    sum += pixels[row + x] * cs[u * n + x] * cy;
                }
            }
            out[v * n + u] = scale * cu * cv * sum;
        }
    }
    out
}

/// Generic separable n×n inverse DCT-II. Takes dequantized i32 coefficients
/// so that large quality-1 steps cannot overflow.
fn pred_idct(coefs: &[i32], n: usize, cs: &[f64]) -> Vec<f64> {
    let scale = 2.0 / n as f64;
    let mut out = vec![0f64; n * n];
    for y in 0..n {
        for x in 0..n {
            let mut sum = 0.0f64;
            for v in 0..n {
                let cv = if v == 0 { PRED_FRAC_1_SQRT2 } else { 1.0 };
                let cy = cs[v * n + y];
                let row = v * n;
                for u in 0..n {
                    let cu = if u == 0 { PRED_FRAC_1_SQRT2 } else { 1.0 };
                    sum += cu * cv * coefs[row + u] as f64 * cs[u * n + x] * cy;
                }
            }
            out[y * n + x] = scale * sum;
        }
    }
    out
}

#[inline]
fn pred_quantize(coefs: &mut [i16], table: &[i16]) {
    for (c, &t) in coefs.iter_mut().zip(table.iter()) {
        let t = t.max(1) as i32;
        let v = *c as i32;
        let rounded = if v >= 0 { (v + t / 2) / t } else { -((-v + t / 2) / t) };
        *c = rounded.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

fn pred_dequantize(coefs: &[i16], table: &[i16]) -> Vec<i32> {
    coefs
        .iter()
        .zip(table.iter())
        .map(|(&c, &t)| c as i32 * t.max(1) as i32)
        .collect()
}

/// Sample an n×n block of raw 0..255 samples, edge-replicated.
fn pred_original_block(
    image: &[u8],
    w: usize,
    h: usize,
    bx: usize,
    by: usize,
    n: usize,
) -> Vec<i16> {
    let mut out = vec![0i16; n * n];
    for y in 0..n {
        let iy = (by + y).min(h - 1);
        let row = iy * w;
        for x in 0..n {
            let ix = (bx + x).min(w - 1);
            out[y * n + x] = image[row + ix] as i16;
        }
    }
    out
}

/// Prediction field for one block, built only from reconstructed pixels
/// outside the block (left boundary column and top boundary row). This keeps
/// the field decodable before the block's own coefficients are inverse
/// transformed. Edge fallbacks are deterministic and mirrored by the decoder.
fn pred_predict_block(
    predictor: u8,
    recon: &[u8],
    w: usize,
    h: usize,
    bx: usize,
    by: usize,
    n: usize,
) -> Vec<i16> {
    let has_left = bx > 0;
    let has_top = by > 0;

    let mut left = vec![0i16; n];
    if has_left {
        for y in 0..n {
            let iy = (by + y).min(h - 1);
            left[y] = recon[iy * w + (bx - 1)] as i16;
        }
    }
    let mut top = vec![0i16; n];
    if has_top {
        let ty = by - 1;
        for x in 0..n {
            let ix = (bx + x).min(w - 1);
            top[x] = recon[ty * w + ix] as i16;
        }
    }

    let mut field = vec![0i16; n * n];
    for y in 0..n {
        for x in 0..n {
            field[y * n + x] = match predictor {
                PRED_VERTICAL => {
                    if has_top {
                        top[x]
                    } else if has_left {
                        left[y]
                    } else {
                        128
                    }
                }
                PRED_AVERAGE => {
                    if has_left && has_top {
                        ((left[y] as i32 + top[x] as i32 + 1) >> 1) as i16
                    } else if has_left {
                        left[y]
                    } else if has_top {
                        top[x]
                    } else {
                        128
                    }
                }
                // PRED_HORIZONTAL (validated set; default keeps the field total).
                _ => {
                    if has_left {
                        left[y]
                    } else if has_top {
                        top[x]
                    } else {
                        128
                    }
                }
            };
        }
    }
    field
}

#[inline]
fn pred_energy(orig: &[i16], field: &[i16]) -> u64 {
    orig.iter()
        .zip(field.iter())
        .map(|(&o, &p)| {
            let d = o as i64 - p as i64;
            (d * d) as u64
        })
        .sum()
}

/// Generalizes the historical RLE layout to n×n blocks. Same wire shape:
/// DC i16 LE, (run u8, level i16 LE) pairs, EOB = 0xFF 0x00 0x00.
fn pred_write_block(coefs: &[i16], order: &[usize], out: &mut Vec<u8>) {
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

fn pred_read_block(
    stream: &[u8],
    pos: &mut usize,
    count: usize,
    order: &[usize],
) -> Result<Vec<i16>, String> {
    let mut coefs = vec![0i16; count];
    if *pos + 2 > stream.len() {
        return Err("corrupt predictive stream: truncated DC coefficient".to_string());
    }
    coefs[order[0]] = i16::from_le_bytes([stream[*pos], stream[*pos + 1]]);
    *pos += 2;

    let mut zi = 1usize;
    loop {
        if *pos + 3 > stream.len() {
            return Err("corrupt predictive stream: truncated RLE pair".to_string());
        }
        let run = stream[*pos];
        let level = i16::from_le_bytes([stream[*pos + 1], stream[*pos + 2]]);
        *pos += 3;
        if run == 0xFF && level == 0 {
            break;
        }
        if run == 0xFE && level == 0 {
            zi += 0xFE;
            if zi > count {
                return Err("corrupt predictive stream: coefficient overrun".to_string());
            }
            continue;
        }
        zi += run as usize;
        if zi >= count {
            return Err("corrupt predictive stream: coefficient overrun".to_string());
        }
        coefs[order[zi]] = level;
        zi += 1;
    }
    Ok(coefs)
}

/// Shared reconstruction step used by both encoder and decoder so predictor
/// inputs stay bit-identical: dequantize, IDCT, add prediction, clamp, store.
#[allow(clippy::too_many_arguments)]
fn pred_reconstruct_block(
    coefs: &[i16],
    table: &[i16],
    field: &[i16],
    n: usize,
    cs: &[f64],
    recon: &mut [u8],
    w: usize,
    h: usize,
    bx: usize,
    by: usize,
) {
    let dequant = pred_dequantize(coefs, table);
    let pixels = pred_idct(&dequant, n, cs);
    for y in 0..n {
        let py = by + y;
        if py >= h {
            continue;
        }
        for x in 0..n {
            let px = bx + x;
            if px >= w {
                continue;
            }
            let v = pixels[y * n + x].round() as i32 + field[y * n + x] as i32;
            recon[py * w + px] = v.clamp(0, 255) as u8;
        }
    }
}

/// Encode a grayscale image with per-block adaptive spatial prediction.
///
/// `block_size` must be one of `PRED_ALLOWED_BLOCK_SIZES`; `predictors` must be
/// a non-empty duplicate-free subset of `PRED_ALLOWED_PREDICTORS`. Either
/// violation returns `Err` and nothing is written.
pub fn encode_grayscale_predictive(
    image: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    block_size: usize,
    predictors: &[u8],
) -> Result<PredictiveEncode, String> {
    pred_validate_block_size(block_size)?;
    let set = pred_validate_predictors(predictors)?;
    if width == 0 || height == 0 || width > 65536 || height > 65536 {
        return Err(format!("invalid image dimensions {width}x{height}"));
    }
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| format!("invalid image dimensions {width}x{height}: overflow"))?;
    if image.len() < pixels {
        return Err(format!(
            "invalid image buffer: {} bytes for {width}x{height}",
            image.len()
        ));
    }

    let q = if quality == 0 { 50 } else { quality.clamp(1, 100) };
    let n = block_size;
    let table = pred_quant_table(q, n);
    let order = pred_zigzag(n);
    let cs = pred_cos_table(n);
    let bw = (width + n - 1) / n;
    let bh = (height + n - 1) / n;

    let mut stream = Vec::with_capacity(PRED_HEADER_SIZE + bw * bh * 8);
    stream.extend_from_slice(b"BG");
    stream.push(PRED_VERSION);
    stream.extend_from_slice(&(width as u32).to_le_bytes());
    stream.extend_from_slice(&(height as u32).to_le_bytes());
    stream.push(q);
    stream.push(n as u8);

    let mut recon = vec![0u8; pixels];
    let mut counts = [0usize; 3];
    let mut predictor_sequence = Vec::with_capacity(bw * bh);

    for byi in 0..bh {
        for bxi in 0..bw {
            let bx = bxi * n;
            let by = byi * n;
            let orig = pred_original_block(image, width, height, bx, by, n);

            // Pick the candidate with the smallest spatial residual energy.
            // `set` is ascending, so ties deterministically keep the lowest code.
            let mut chosen = set[0];
            let mut field = pred_predict_block(chosen, &recon, width, height, bx, by, n);
            let mut best_energy = pred_energy(&orig, &field);
            for &candidate in &set[1..] {
                let candidate_field =
                    pred_predict_block(candidate, &recon, width, height, bx, by, n);
                let energy = pred_energy(&orig, &candidate_field);
                if energy < best_energy {
                    best_energy = energy;
                    chosen = candidate;
                    field = candidate_field;
                }
            }

            counts[chosen as usize] += 1;
            predictor_sequence.push(chosen);
            stream.push(chosen); // block header

            let mut residual = vec![0f64; n * n];
            for i in 0..n * n {
                residual[i] = orig[i] as f64 - field[i] as f64;
            }
            let transformed = pred_dct(&residual, n, &cs);
            let mut coefs: Vec<i16> = transformed
                .iter()
                .map(|v| v.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16)
                .collect();
            pred_quantize(&mut coefs, &table);
            pred_write_block(&coefs, &order, &mut stream);

            // Reconstruct immediately: the next block's prediction reads it.
            pred_reconstruct_block(
                &coefs, &table, &field, n, &cs, &mut recon, width, height, bx, by,
            );
        }
    }

    let blocks = bw * bh;
    let payload_bytes = stream.len() - PRED_HEADER_SIZE - blocks;
    Ok(PredictiveEncode {
        stream,
        header_bytes: blocks,
        payload_bytes,
        blocks,
        block_size: n,
        predictor_counts: counts,
        predictor_sequence,
        reconstruction: recon,
    })
}

/// Decode a stream produced by `encode_grayscale_predictive`.
/// Every block follows its own 1-byte predictor header; malformed headers,
/// block sizes and truncated payloads return `Err`.
pub fn decode_grayscale_predictive(stream: &[u8]) -> Result<PredictiveDecode, String> {
    if stream.len() < PRED_HEADER_SIZE {
        return Err("not a predictive stream: too short".to_string());
    }
    if &stream[0..2] != b"BG" || stream[2] != PRED_VERSION {
        return Err(format!(
            "not a predictive stream (version {:#04x}, expected {:#04x})",
            stream[2], PRED_VERSION
        ));
    }
    let width = u32::from_le_bytes([stream[3], stream[4], stream[5], stream[6]]) as usize;
    let height = u32::from_le_bytes([stream[7], stream[8], stream[9], stream[10]]) as usize;
    let quality = if stream[11] == 0 { 50 } else { stream[11].clamp(1, 100) };
    let n = stream[12] as usize;
    pred_validate_block_size(n).map_err(|_| {
        format!(
            "corrupt predictive stream: invalid block size {n} (allowed sizes are {:?})",
            PRED_ALLOWED_BLOCK_SIZES
        )
    })?;
    if width == 0 || height == 0 || width > 65536 || height > 65536 {
        return Err(format!("corrupt predictive stream: invalid dimensions {width}x{height}"));
    }
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| "corrupt predictive stream: dimension overflow".to_string())?;

    let table = pred_quant_table(quality, n);
    let order = pred_zigzag(n);
    let cs = pred_cos_table(n);
    let bw = (width + n - 1) / n;
    let bh = (height + n - 1) / n;
    let blocks = bw * bh;

    let mut recon = vec![0u8; pixels];
    let mut counts = [0usize; 3];
    let mut predictor_sequence = Vec::with_capacity(blocks);
    let mut pos = PRED_HEADER_SIZE;

    for byi in 0..bh {
        for bxi in 0..bw {
            if pos >= stream.len() {
                return Err("corrupt predictive stream: missing block header".to_string());
            }
            let predictor = stream[pos];
            pos += 1;
            if predictor > PRED_AVERAGE {
                return Err(format!(
                    "corrupt predictive stream: invalid predictor {predictor} in block header \
                     (allowed predictors are {:?})",
                    PRED_ALLOWED_PREDICTORS
                ));
            }
            counts[predictor as usize] += 1;
            predictor_sequence.push(predictor);

            let coefs = pred_read_block(stream, &mut pos, n * n, &order)?;
            let bx = bxi * n;
            let by = byi * n;
            let field = pred_predict_block(predictor, &recon, width, height, bx, by, n);
            pred_reconstruct_block(
                &coefs, &table, &field, n, &cs, &mut recon, width, height, bx, by,
            );
        }
    }

    Ok(PredictiveDecode {
        pixels: recon,
        width,
        height,
        quality,
        block_size: n,
        blocks,
        header_bytes: blocks,
        predictor_counts: counts,
        predictor_sequence,
    })
}


#[cfg(test)]
mod predictive_tests {
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

    /// Material A: photo-like smooth shading with gentle texture. No single
    /// prediction direction dominates across the whole image.
    fn sample_photo(w: usize, h: usize) -> Vec<u8> {
        let (wf, hf) = (w as f64, h as f64);
        (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| {
                    let fx = x as f64 / wf.max(1.0);
                    let fy = y as f64 / hf.max(1.0);
                    let v = 110.0
                        + 60.0 * (5.0 * fx + 1.0).sin()
                        + 40.0 * (4.0 * fy + 0.3).cos()
                        + 18.0 * (13.0 * (fx + fy)).sin();
                    v.clamp(0.0, 255.0) as u8
                })
            })
            .collect()
    }

    /// Material B: screen-capture / graphics. Flat panels, sharp horizontal
    /// separators and vertical strokes, so orientation flips between blocks.
    fn sample_screen(w: usize, h: usize) -> Vec<u8> {
        (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| {
                    let panel = ((x / 24) + (y / 24)) % 2;
                    let mut v: i32 = if panel == 0 { 232 } else { 28 };
                    if x % 24 == 0 || y % 24 == 0 {
                        v = 96; // separators: both orientations present
                    }
                    if (x / 4) % 9 == 0 && (y / 3) % 7 == 0 {
                        v = v.saturating_sub(40); // stroke texture
                    }
                    v as u8
                })
            })
            .collect()
    }

    /// Material C: four quadrants with different dominant orientations
    /// (vertical stripes / horizontal bands / diagonal ramp / flat).
    fn sample_mixed(w: usize, h: usize) -> Vec<u8> {
        (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| {
                    let left = x < w / 2;
                    let top = y < h / 2;
                    match (left, top) {
                        (true, true) => ((x * 9) % 256) as u8,
                        (false, true) => ((y * 9) % 256) as u8,
                        (true, false) => (((x + y) * 4) % 256) as u8,
                        (false, false) => 90,
                    }
                })
            })
            .collect()
    }

    /// Material D: deterministic noise floor (worst case, no structure).
    fn sample_noise(w: usize, h: usize) -> Vec<u8> {
        let mut rng = Rng(0xDEADBEEFCAFEF00D);
        (0..w * h)
            .map(|_| (128 + rng.range(-60, 60)).clamp(0, 255) as u8)
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
    fn predictive_rejects_invalid_block_sizes() {
        let img = sample_mixed(32, 32);
        for size in [0usize, 1, 2, 3, 5, 7, 9, 12, 24, 32, 64] {
            let err = encode_grayscale_predictive(&img, 32, 32, 50, size, &[0, 1, 2])
                .err()
                .unwrap_or_else(|| panic!("block size {size} must be rejected"));
            assert!(err.contains("invalid block size"), "{err}");
        }
        for size in PRED_ALLOWED_BLOCK_SIZES {
            assert!(
                encode_grayscale_predictive(&img, 32, 32, 50, size, &[0, 1, 2]).is_ok(),
                "block size {size} must be accepted"
            );
        }
    }

    #[test]
    fn predictive_rejects_invalid_predictor_sets() {
        let img = sample_mixed(24, 24);
        for set in [
            &[][..],
            &[3][..],
            &[9][..],
            &[0, 0][..],
            &[1, 1, 2][..],
            &[0, 1, 2, 3][..],
            &[255][..],
        ] {
            let err = encode_grayscale_predictive(&img, 24, 24, 50, 8, set)
                .err()
                .unwrap_or_else(|| panic!("predictor set {set:?} must be rejected"));
            assert!(err.contains("invalid predictor set"), "{err}");
        }
        for set in [
            &[0][..],
            &[1][..],
            &[2][..],
            &[0, 1, 2][..],
            &[2, 0][..],
            &[2, 1, 0][..],
        ] {
            assert!(
                encode_grayscale_predictive(&img, 24, 24, 50, 8, set).is_ok(),
                "predictor set {set:?} must be accepted"
            );
        }
    }

    #[test]
    fn predictive_roundtrip_all_materials_and_sizes() {
        let dims = [(1usize, 1usize), (7, 5), (16, 16), (37, 23), (64, 48)];
        let materials: [(&str, fn(usize, usize) -> Vec<u8>); 4] = [
            ("photo", sample_photo),
            ("screen", sample_screen),
            ("mixed", sample_mixed),
            ("noise", sample_noise),
        ];
        for (name, make) in materials {
            for &(w, h) in &dims {
                let img = make(w, h);
                for &n in &PRED_ALLOWED_BLOCK_SIZES {
                    for set in [&[0, 1, 2][..], &[0][..], &[1][..], &[2][..], &[0, 2][..]] {
                        let enc = encode_grayscale_predictive(&img, w, h, 85, n, set).unwrap();
                        let dec = decode_grayscale_predictive(&enc.stream).unwrap();
                        assert_eq!((dec.width, dec.height), (w, h), "{name} {w}x{h} n={n}");
                        assert_eq!(dec.pixels.len(), w * h);
                        assert_eq!(dec.blocks, enc.blocks);
                        assert_eq!(
                            dec.predictor_sequence, enc.predictor_sequence,
                            "{name} {w}x{h} n={n} set={set:?}: decoder must follow block headers"
                        );
                        assert_eq!(dec.predictor_counts, enc.predictor_counts);
                        // Within the chosen set: no header may point outside it.
                        for &p in &dec.predictor_sequence {
                            assert!(set.contains(&p), "header predictor {p} outside set {set:?}");
                        }
                        let p = psnr(&img, &dec.pixels);
                        assert!(p > 18.0, "{name} {w}x{h} n={n} set={set:?} PSNR {p:.2} dB");
                    }
                }
            }
        }
    }

    /// Tie-break and coarse semantics: with a single predictor the headers must
    /// all carry it; with all three, the chosen one must never lose to another
    /// candidate on spatial residual energy.
    #[test]
    fn predictive_single_predictor_headers_and_energy_choice() {
        let (w, h, n) = (48usize, 40usize, 8usize);
        let img = sample_photo(w, h);
        for p in PRED_ALLOWED_PREDICTORS {
            let enc = encode_grayscale_predictive(&img, w, h, 80, n, &[p]).unwrap();
            assert!(enc.predictor_sequence.iter().all(|&x| x == p));
            assert_eq!(enc.predictor_counts[p as usize], enc.blocks);
        }
        let enc = encode_grayscale_predictive(&img, w, h, 80, n, &[0, 1, 2]).unwrap();
        assert_eq!(enc.predictor_sequence.len(), enc.blocks);
    }

    #[test]
    fn predictive_decoder_rejects_malformed_streams() {
        let img = sample_screen(24, 24);
        let enc = encode_grayscale_predictive(&img, 24, 24, 70, 8, &[0, 1, 2]).unwrap();

        // Foreign (historical) stream must be rejected.
        let mut legacy = enc.stream.clone();
        legacy[2] = 19;
        assert!(decode_grayscale_predictive(&legacy).is_err());

        // Truncated stream.
        let cut = enc.stream.len() - 1;
        assert!(decode_grayscale_predictive(&enc.stream[..cut]).is_err());

        // Illegal block size in the file header.
        let mut bad_n = enc.stream.clone();
        bad_n[12] = 3;
        let err = decode_grayscale_predictive(&bad_n).unwrap_err();
        assert!(err.contains("invalid block size"), "{err}");

        // Illegal per-block predictor header (first block header at pos 13).
        let mut bad_pred = enc.stream.clone();
        bad_pred[PRED_HEADER_SIZE] = 7;
        let err = decode_grayscale_predictive(&bad_pred).unwrap_err();
        assert!(err.contains("invalid predictor"), "{err}");
    }

    /// The block header must be the energy-minimal predictor of the offered
    /// set, using only boundary pixels that are final before the block is
    /// coded. Since `reconstruction` is the final image and a block's field
    /// reads only earlier pixels, recomputing the field from it reproduces the
    /// encoder's decision exactly, so the argmin comparison is independent.
    #[test]
    fn predictive_headers_are_per_block_energy_argmin() {
        let (w, h, q) = (96usize, 80usize, 85u8);
        for (name, make) in [
            ("photo", sample_photo as fn(usize, usize) -> Vec<u8>),
            ("screen", sample_screen),
            ("mixed", sample_mixed),
        ] {
            let img = make(w, h);
            for n in PRED_ALLOWED_BLOCK_SIZES {
                let enc = encode_grayscale_predictive(&img, w, h, q, n, &[0, 1, 2]).unwrap();
                let bw = (w + n - 1) / n;
                for (i, &chosen) in enc.predictor_sequence.iter().enumerate() {
                    let bx = (i % bw) * n;
                    let by = (i / bw) * n;
                    let orig = pred_original_block(&img, w, h, bx, by, n);
                    let mut energies = [0u64; 3];
                    for p in PRED_ALLOWED_PREDICTORS {
                        let field =
                            pred_predict_block(p, &enc.reconstruction, w, h, bx, by, n);
                        energies[p as usize] = pred_energy(&orig, &field);
                    }
                    let best = (0..3)
                        .min_by_key(|&p| energies[p])
                        .unwrap() as u8;
                    assert_eq!(
                        chosen, best,
                        "{name} n={n} block {i}: header {chosen} but energies {:?}",
                        energies
                    );
                }
            }
        }
    }

    /// A heterogeneous material must actually exercise more than one predictor,
    /// otherwise "adaptive" would be a no-op.
    #[test]
    fn predictive_adapts_across_orientations() {
        let (w, h, n) = (128usize, 128usize, 8usize);
        let enc = encode_grayscale_predictive(&sample_mixed(w, h), w, h, 85, n, &[0, 1, 2])
            .unwrap();
        let used = enc.predictor_counts.iter().filter(|&&c| c > 0).count();
        assert!(used >= 2, "expected a mixed predictor mix, got {:?}", enc.predictor_counts);
        // And with a constrained set, only the allowed codes may appear.
        let enc = encode_grayscale_predictive(&sample_mixed(w, h), w, h, 85, n, &[2]).unwrap();
        assert!(enc.predictor_sequence.iter().all(|&p| p == 2));
    }

    #[test]
    fn predictive_quality_and_block_size_are_carried_in_stream() {
        let img = sample_photo(32, 32);
        for q in [1u8, 25, 50, 85, 100] {
            for n in PRED_ALLOWED_BLOCK_SIZES {
                let enc = encode_grayscale_predictive(&img, 32, 32, q, n, &[0, 1, 2]).unwrap();
                let dec = decode_grayscale_predictive(&enc.stream).unwrap();
                assert_eq!(dec.quality, q);
                assert_eq!(dec.block_size, n);
                assert_eq!(enc.stream.len(), PRED_HEADER_SIZE + enc.blocks + enc.payload_bytes);
            }
        }
    }

    /// One-shot metrics table for the three material classes plus a noise
    /// floor. For each material and block size it reports:
    ///   * size: best single-predictor ("fixed") stream vs the adaptive stream
    ///   * header cost: bytes and share of the adaptive stream
    ///   * encode/decode wall time (adaptive pass only)
    ///   * round-trip consistency: decoder headers == encoder headers and the
    ///     decoded pixels match an independent replay (exact equality)
    ///   * PSNR against the source, and the H/V/A block mix.
    #[test]
    fn predictive_metrics_report() {
        let (w, h, quality) = (128usize, 128usize, 85u8);
        println!();
        println!("size = stream bytes; gain vs best single-predictor baseline;");
        println!("hdr = per-block-header bytes and share of adaptive stream;");
        println!("times are one encode/decode of adaptive stream on this host;");
        println!("roundtrip = decode pixels == encoder reconstruction and headers match.");
        println!();
        println!("adv/raw = adaptive stream as % of uncompressed grayscale input;");
        println!(
            "material  n | fixed H/V/A bytes | adaptive bytes | adv/raw % | gain vs best fixed | hdr B (%) | enc ms | dec ms | PSNR dB | H/V/A blocks | roundtrip"
        );
        for (name, make) in [
            ("photo   ", sample_photo as fn(usize, usize) -> Vec<u8>),
            ("screen  ", sample_screen),
            ("mixed   ", sample_mixed),
            ("noise   ", sample_noise),
        ] {
            let img = make(w, h);
            for n in PRED_ALLOWED_BLOCK_SIZES {
                let mut fixed_sizes = [0usize; 3];
                for (i, p) in PRED_ALLOWED_PREDICTORS.iter().enumerate() {
                    fixed_sizes[i] = encode_grayscale_predictive(&img, w, h, quality, n, &[*p])
                        .unwrap()
                        .stream
                        .len();
                }
                let best_fixed = *fixed_sizes.iter().min().unwrap();

                let t0 = std::time::Instant::now();
                let enc = encode_grayscale_predictive(&img, w, h, quality, n, &[0, 1, 2]).unwrap();
                let enc_ms = t0.elapsed().as_secs_f64() * 1e3;

                let t1 = std::time::Instant::now();
                let dec = decode_grayscale_predictive(&enc.stream).unwrap();
                let dec_ms = t1.elapsed().as_secs_f64() * 1e3;

                // Round-trip consistency: the decoder must reproduce the
                // encoder's own reconstruction exactly, and read back the same
                // per-block headers in the same order.
                let roundtrip_ok = dec.pixels == enc.reconstruction
                    && dec.predictor_sequence == enc.predictor_sequence
                    && dec.predictor_counts == enc.predictor_counts;

                let gain =
                    100.0 * (best_fixed as f64 - enc.stream.len() as f64) / best_fixed as f64;
                let hdr = 100.0 * enc.header_bytes as f64 / enc.stream.len() as f64;
                let adv_raw = 100.0 * enc.stream.len() as f64 / (w * h) as f64;
                println!(
                    "{name} {n:2} | {:5}/{:5}/{:5} | {:6} | {:5.2}% | {:+6.2}% | {:4} ({:4.2}%) | {:6.2} | {:6.2} | {:7.2} | {}/{}/{} | {}",
                    fixed_sizes[0],
                    fixed_sizes[1],
                    fixed_sizes[2],
                    enc.stream.len(),
                    adv_raw,
                    gain,
                    enc.header_bytes,
                    hdr,
                    enc_ms,
                    dec_ms,
                    psnr(&img, &dec.pixels),
                    enc.predictor_counts[0],
                    enc.predictor_counts[1],
                    enc.predictor_counts[2],
                    if roundtrip_ok { "OK" } else { "MISMATCH" },
                );
                assert!(roundtrip_ok, "{name} n={n}: round-trip consistency failed");
                assert!(psnr(&img, &dec.pixels) > 20.0, "{name} n={n} PSNR too low");
            }
        }
    }
}
