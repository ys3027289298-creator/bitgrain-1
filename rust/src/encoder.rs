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
    match rd_config_from_env() {
        Some(Err(msg)) => panic!("bitgrain: {msg}"),
        Some(Ok(cfg)) => {
            let encode_at = |q: u8, buf: &mut [u8]| -> usize {
                let mut p: i32 = 0;
                encode_grayscale_inner(image, width, height, q, buf, &mut p);
                p as usize
            };
            encode_with_rate_control(
                width.saturating_mul(height),
                quality,
                &cfg,
                out,
                pos,
                encode_at,
                |stream| psnr_of_stream(image, stream),
            );
        }
        None => encode_grayscale_inner(image, width, height, quality, out, pos),
    }
}

fn encode_grayscale_inner(
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
    match rd_config_from_env() {
        Some(Err(msg)) => panic!("bitgrain: {msg}"),
        Some(Ok(cfg)) => {
            let encode_at = |q: u8, buf: &mut [u8]| -> usize {
                let mut p: i32 = 0;
                encode_rgb_ycbcr(image, width, height, q, buf, &mut p, icc);
                p as usize
            };
            encode_with_rate_control(
                width.saturating_mul(height),
                quality,
                &cfg,
                out,
                pos,
                encode_at,
                |stream| psnr_of_stream(image, stream),
            );
        }
        // Default to the better YCbCr path; callers that need legacy RLE use encode_rgb_rle
        None => encode_rgb_ycbcr(image, width, height, quality, out, pos, icc),
    }
}

pub fn encode_rgba(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
    match rd_config_from_env() {
        Some(Err(msg)) => panic!("bitgrain: {msg}"),
        Some(Ok(cfg)) => {
            let encode_at = |q: u8, buf: &mut [u8]| -> usize {
                let mut p: i32 = 0;
                encode_rgba_ycbcr(image, width, height, q, buf, &mut p, icc);
                p as usize
            };
            encode_with_rate_control(
                width.saturating_mul(height),
                quality,
                &cfg,
                out,
                pos,
                encode_at,
                |stream| psnr_of_stream(image, stream),
            );
        }
        None => encode_rgba_ycbcr(image, width, height, quality, out, pos, icc),
    }
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
// Optional rate-distortion control (opt-in via environment; default unchanged)
// ---------------------------------------------------------------------------
//
// When BITGRAIN_TARGET_BPP is set to a target bits-per-pixel value, the public
// encode entry points treat the quality argument as an initial guess and
// iteratively adjust the quantization step (bisection over the 1..=100 quality
// grid) until the produced stream lands within BITGRAIN_RD_TOLERANCE bpp of
// the target (default 0.05) or the BITGRAIN_RD_MAX_ITER budget (default 12)
// is exhausted. If neither happens (budget spent, or no grid point lands
// within tolerance), the encoder falls back to the best-effort stream
// (closest bpp, preferring under-target) and reports
// status=fallback-best-effort. An invalid target (non-numeric, non-finite,
// <= 0, or > 64 bpp) aborts the encode with an error. When the env var is
// unset the encode path is byte-identical to the historical behavior.
//
// bpp is measured against source pixels (w*h), i.e. stream_bytes*8/(w*h),
// covering all planes (luma + chroma [+ alpha]) and the .bg header.

const RD_TARGET_BPP_ENV: &str = "BITGRAIN_TARGET_BPP";
const RD_MAX_ITER_ENV: &str = "BITGRAIN_RD_MAX_ITER";
const RD_TOLERANCE_ENV: &str = "BITGRAIN_RD_TOLERANCE";
const RD_DEFAULT_MAX_ITER: u32 = 12;
const RD_DEFAULT_TOLERANCE_BPP: f64 = 0.05;
const RD_MAX_TARGET_BPP: f64 = 64.0;

#[derive(Clone, Copy, Debug, PartialEq)]
struct RdConfig {
    target_bpp: f64,
    max_iterations: u32,
    tolerance_bpp: f64,
}

fn parse_rd_config(
    target: Option<String>,
    max_iter: Option<String>,
    tolerance: Option<String>,
) -> Option<Result<RdConfig, String>> {
    let raw = target?;
    let target_bpp: f64 = match raw.trim().parse() {
        Ok(v) => v,
        Err(_) => {
            return Some(Err(format!(
                "invalid {RD_TARGET_BPP_ENV}='{raw}': expected a positive finite number (bits per pixel)"
            )));
        }
    };
    if !target_bpp.is_finite() || target_bpp <= 0.0 || target_bpp > RD_MAX_TARGET_BPP {
        return Some(Err(format!(
            "invalid {RD_TARGET_BPP_ENV}={target_bpp}: must be in (0, {RD_MAX_TARGET_BPP}] bits/pixel"
        )));
    }
    let max_iterations = match max_iter {
        Some(v) => match v.trim().parse::<u32>() {
            Ok(n) if n >= 1 => n,
            _ => {
                return Some(Err(format!(
                    "invalid {RD_MAX_ITER_ENV}='{v}': expected an integer >= 1"
                )));
            }
        },
        None => RD_DEFAULT_MAX_ITER,
    };
    let tolerance_bpp = match tolerance {
        Some(v) => match v.trim().parse::<f64>() {
            Ok(t) if t.is_finite() && t > 0.0 => t,
            _ => {
                return Some(Err(format!(
                    "invalid {RD_TOLERANCE_ENV}='{v}': expected a positive finite number (bpp)"
                )));
            }
        },
        None => RD_DEFAULT_TOLERANCE_BPP,
    };
    Some(Ok(RdConfig { target_bpp, max_iterations, tolerance_bpp }))
}

fn rd_config_from_env() -> Option<Result<RdConfig, String>> {
    use std::sync::OnceLock;
    static CONFIG: OnceLock<Option<Result<RdConfig, String>>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            parse_rd_config(
                std::env::var(RD_TARGET_BPP_ENV).ok(),
                std::env::var(RD_MAX_ITER_ENV).ok(),
                std::env::var(RD_TOLERANCE_ENV).ok(),
            )
        })
        .clone()
}

struct RdOutcome {
    quality: u8,
    stream: Vec<u8>,
    bpp: f64,
    iterations: u32,
    converged: bool,
    grid_exhausted: bool,
}

/// Bisection over the quality grid toward `cfg.target_bpp`.
/// `encode_at(q, buf) -> bytes written` must be deterministic for a given q.
fn rd_search<F>(
    capacity: usize,
    pixel_count: usize,
    initial_quality: u8,
    cfg: &RdConfig,
    encode_at: F,
) -> RdOutcome
where
    F: Fn(u8, &mut [u8]) -> usize,
{
    let pixels = pixel_count.max(1) as f64;
    let mut lo: i32 = 1;
    let mut hi: i32 = 100;
    let mut q: i32 = initial_quality.clamp(1, 100) as i32;
    let mut best: Option<(u8, Vec<u8>, f64)> = None;
    let mut iterations: u32 = 0;
    let mut converged = false;
    let mut grid_exhausted = false;

    while iterations < cfg.max_iterations {
        iterations += 1;
        let mut scratch = vec![0u8; capacity];
        let trial = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            encode_at(q as u8, &mut scratch)
        }));
        // A panicking or over-capacity trial means the stream did not fit:
        // treat it as oversize and keep searching at lower qualities.
        let (written, fits) = match trial {
            Ok(n) if n <= capacity => (n, true),
            _ => (0usize, false),
        };
        let bpp = if fits { (written as f64 * 8.0) / pixels } else { f64::INFINITY };
        let err = bpp - cfg.target_bpp;
        if fits {
            scratch.truncate(written);
            eprintln!("[bitgrain-rd] iter={iterations} quality={q} bytes={written} bpp={bpp:.4}");
            let better = match &best {
                None => true,
                Some((_, _, best_bpp)) => {
                    let best_err = (*best_bpp - cfg.target_bpp).abs();
                    let new_err = err.abs();
                    new_err < best_err
                        || (new_err == best_err
                            && bpp <= cfg.target_bpp
                            && *best_bpp > cfg.target_bpp)
                }
            };
            if better {
                best = Some((q as u8, scratch, bpp));
            }
        } else {
            eprintln!("[bitgrain-rd] iter={iterations} quality={q} bytes=overflow bpp=inf");
        }
        if err.abs() <= cfg.tolerance_bpp {
            converged = true;
            break;
        }
        if bpp > cfg.target_bpp {
            hi = q - 1;
        } else {
            lo = q + 1;
        }
        if lo > hi {
            grid_exhausted = true;
            break;
        }
        q = (lo + hi) / 2;
    }

    let (quality, stream, bpp) = match best {
        Some(b) => b,
        None => {
            // No trial fit the buffer; force one minimal-quality encode.
            // If even that overflows, the panic propagates as an encode error.
            let mut scratch = vec![0u8; capacity];
            let written = encode_at(1, &mut scratch);
            scratch.truncate(written);
            (1u8, scratch, (written as f64 * 8.0) / pixels)
        }
    };

    RdOutcome { quality, stream, bpp, iterations, converged, grid_exhausted }
}

/// PSNR (dB) between the original pixels and the decoded stream.
fn psnr_of_stream(original: &[u8], stream: &[u8]) -> f64 {
    let mut w = 0u32;
    let mut h = 0u32;
    let mut ch = 0u32;
    let mut decoded = vec![0u8; original.len()];
    if !crate::decoder::decode(stream, &mut decoded, &mut w, &mut h, &mut ch, None) {
        return f64::NAN;
    }
    let n = original.len().min(decoded.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sum_sq = 0f64;
    for i in 0..n {
        let d = original[i] as f64 - decoded[i] as f64;
        sum_sq += d * d;
    }
    let mse = sum_sq / n as f64;
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

fn encode_with_rate_control<F, G>(
    pixel_count: usize,
    quality: u8,
    cfg: &RdConfig,
    out: &mut [u8],
    pos: &mut i32,
    encode_at: F,
    psnr_db: G,
) where
    F: Fn(u8, &mut [u8]) -> usize,
    G: FnOnce(&[u8]) -> f64,
{
    let result = rd_search(out.len(), pixel_count, quality, cfg, encode_at);
    if result.stream.len() > out.len() {
        panic!("bitgrain: encode buffer too small for rate-controlled stream");
    }
    let psnr = psnr_db(&result.stream);
    let status = if result.converged { "converged" } else { "fallback-best-effort" };
    if !result.converged && result.grid_exhausted {
        eprintln!(
            "[bitgrain-rd] warning: no quality grid point within tolerance of target; falling back to best-effort quality={}",
            result.quality
        );
    } else if !result.converged {
        eprintln!(
            "[bitgrain-rd] warning: target bpp not reached within {} iteration(s); falling back to best-effort quality={}",
            cfg.max_iterations, result.quality
        );
    }
    eprintln!(
        "[bitgrain-rd] target_bpp={:.4} actual_bpp={:.4} bytes={} psnr_db={:.2} iterations={} status={}",
        cfg.target_bpp, result.bpp, result.stream.len(), psnr, result.iterations, status
    );
    out[..result.stream.len()].copy_from_slice(&result.stream);
    *pos = result.stream.len() as i32;
}

#[cfg(test)]
mod rd_tests {
    use super::*;

    fn cfg(target_bpp: f64, max_iterations: u32, tolerance_bpp: f64) -> RdConfig {
        RdConfig { target_bpp, max_iterations, tolerance_bpp }
    }

    #[test]
    fn parse_rejects_invalid_target() {
        assert!(parse_rd_config(None, None, None).is_none());
        assert!(parse_rd_config(Some("abc".into()), None, None).unwrap().is_err());
        assert!(parse_rd_config(Some("0".into()), None, None).unwrap().is_err());
        assert!(parse_rd_config(Some("-1.5".into()), None, None).unwrap().is_err());
        assert!(parse_rd_config(Some("inf".into()), None, None).unwrap().is_err());
        assert!(parse_rd_config(Some("65".into()), None, None).unwrap().is_err());
        assert!(parse_rd_config(Some("1.5".into()), None, None).unwrap().is_ok());
        assert!(parse_rd_config(Some("1.5".into()), Some("0".into()), None).unwrap().is_err());
        assert!(parse_rd_config(Some("1.5".into()), Some("x".into()), None).unwrap().is_err());
        assert!(parse_rd_config(Some("1.5".into()), None, Some("-0.1".into())).unwrap().is_err());
        let ok = parse_rd_config(Some("1.5".into()), Some("8".into()), Some("0.05".into()))
            .unwrap()
            .unwrap();
        assert_eq!(ok.max_iterations, 8);
        assert!((ok.tolerance_bpp - 0.05).abs() < 1e-12);
        assert!((ok.target_bpp - 1.5).abs() < 1e-12);
    }

    #[test]
    fn search_converges_within_tolerance() {
        // Synthetic deterministic encoder: size grows strictly with quality.
        let pixels = 10_000usize;
        let encode_at = |q: u8, buf: &mut [u8]| -> usize {
            let n = 100 + 10 * q as usize;
            assert!(n <= buf.len());
            n
        };
        let c = cfg(0.5, 12, 0.02);
        let r = rd_search(4096, pixels, 85, &c, encode_at);
        assert!(r.converged);
        assert!(r.iterations <= 12);
        assert!((r.bpp - 0.5).abs() <= 0.02);
    }

    #[test]
    fn search_falls_back_when_budget_exhausted() {
        let pixels = 10_000usize;
        let encode_at = |q: u8, _buf: &mut [u8]| -> usize { 100 + 10 * q as usize };
        // Unreachable tolerance + tiny budget: must fall back to best effort.
        let c = cfg(0.9, 2, 1e-9);
        let r = rd_search(4096, pixels, 1, &c, encode_at);
        assert!(!r.converged);
        assert_eq!(r.iterations, 2);
        // Visited q=1 (bpp 0.088) and q=51 (bpp 0.488); 0.9 target -> best is q=51.
        assert_eq!(r.quality, 51);
        assert!((r.bpp - 0.488).abs() < 1e-9);
    }

    #[test]
    fn search_prefers_under_target_on_tie() {
        let pixels = 8_000usize;
        // size(q) = q bytes exactly -> bpp = q/1000; target 0.05 bpp = 50 bytes.
        let encode_at = |q: u8, _buf: &mut [u8]| -> usize { q as usize };
        let c = cfg(0.05, 12, 1e-12);
        let r = rd_search(4096, pixels, 50, &c, encode_at);
        // q=50 gives exactly 0.05 bpp: converged immediately.
        assert!(r.converged);
        assert_eq!(r.quality, 50);
    }

}
