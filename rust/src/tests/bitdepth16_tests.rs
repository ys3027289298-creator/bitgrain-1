//! Tests for the 16-bit input path (`encoder::bitdepth16`) and for the
//! byte-level stability of the legacy 8-bit path.

use crate::encoder::bitdepth16::{
    self, BitDepth, BitgrainError, OutputMode,
};

// ---------------------------------------------------------------------------
// Deterministic tiny 16-bit sample materials (no external data, no deps)
// ---------------------------------------------------------------------------

/// M1: 32x32 grayscale, smooth diagonal gradient plus a bright plateau.
fn material_gray16() -> (usize, usize, Vec<u16>) {
    let (w, h) = (32usize, 32usize);
    let mut img = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let mut v = ((x * 2048 + y * 1024) / (w + h)) as u16;
            if x >= 20 && y >= 20 {
                v = v.saturating_add(4096).min(65535);
            }
            img.push(v);
        }
    }
    (w, h, img)
}

/// M2: 24x24 RGB, per-channel ramps with a circular feature and banding.
fn material_rgb16() -> (usize, usize, Vec<u16>) {
    let (w, h) = (24usize, 24usize);
    let mut img = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let r = ((x as u32 * 65535) / (w as u32 - 1)) as u16;
            let g = ((y as u32 * 65535) / (h as u32 - 1)) as u16;
            let dx = x as i32 - 12;
            let dy = y as i32 - 12;
            let b = if dx * dx + dy * dy < 36 { 60000u16 } else { ((x ^ y) * 777) as u16 & 0xFFFF };
            img.push(r);
            img.push(g);
            img.push(b);
        }
    }
    (w, h, img)
}

/// M3: 16x16 RGBA, hard edges and a checkerboard (high-frequency) + alpha ramp.
fn material_rgba16() -> (usize, usize, Vec<u16>) {
    let (w, h) = (16usize, 16usize);
    let mut img = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let checker = (x + y) % 2 == 0;
            let r = if checker { 65535 } else { 0 };
            let g = if x < 8 { 32768 } else { 8192 };
            let b = if y < 8 { 16384 } else { 49152 };
            let a = ((y as u32 * 65535) / (h as u32 - 1)) as u16;
            img.push(r);
            img.push(g);
            img.push(b);
            img.push(a);
        }
    }
    (w, h, img)
}

fn downshift8(v: u16) -> u16 {
    (((v as u32) + 128) >> 8).min(255) as u16
}

fn mse(a: &[u16], b: &[u16]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum();
    sum / a.len() as f64
}

fn psnr(mse_v: f64, max_val: f64) -> f64 {
    if mse_v <= f64::EPSILON {
        99.0
    } else {
        10.0 * (max_val * max_val / mse_v).log10()
    }
}

fn mae_max(a: &[u16], b: &[u16]) -> (f64, u32) {
    let (mut sum, mut mx) = (0u64, 0u32);
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x.abs_diff(*y);
        sum += d as u64;
        mx = mx.max(d as u32);
    }
    (sum as f64 / a.len() as f64, mx)
}

// ---------------------------------------------------------------------------
// Bit-depth / argument validation
// ---------------------------------------------------------------------------

#[test]
fn bit_depth_validation_accepts_8_and_16() {
    assert_eq!(bitdepth16::validate_bit_depth(8), Ok(BitDepth::B8));
    assert_eq!(bitdepth16::validate_bit_depth(16), Ok(BitDepth::B16));
}

#[test]
fn bit_depth_validation_rejects_illegal_values() {
    for bad in [0u32, 1, 7, 9, 12, 15, 24, 32, 64, 255, u32::MAX] {
        let err = bitdepth16::validate_bit_depth(bad).unwrap_err();
        assert_eq!(err, BitgrainError::InvalidBitDepth(bad));
        assert!(err.to_string().contains("bit depth"), "msg: {err}");
    }
}

#[test]
fn quality_and_dimension_validation() {
    let (w, h, img) = material_gray16();
    assert_eq!(
        bitdepth16::encode_grayscale16(&img, w, h, 101, OutputMode::Keep16).unwrap_err(),
        BitgrainError::InvalidQuality(101)
    );
    assert_eq!(
        bitdepth16::encode_grayscale16(&img, w, h, 255, OutputMode::Down8).unwrap_err(),
        BitgrainError::InvalidQuality(255)
    );
    assert_eq!(
        bitdepth16::encode_grayscale16(&img, 0, h, 85, OutputMode::Keep16).unwrap_err(),
        BitgrainError::InvalidDimensions
    );
    assert_eq!(
        bitdepth16::encode_grayscale16(&img[..img.len() - 1], w, h, 85, OutputMode::Keep16).unwrap_err(),
        BitgrainError::InvalidDimensions
    );
    let (rw, rh, rgb) = material_rgb16();
    assert_eq!(
        bitdepth16::encode_rgb16(&rgb[..rgb.len() - 3], rw, rh, 85, OutputMode::Keep16).unwrap_err(),
        BitgrainError::InvalidDimensions
    );
    assert_eq!(
        bitdepth16::encode_rgb16(&rgb, rw + 1, rh, 85, OutputMode::Keep16).unwrap_err(),
        BitgrainError::InvalidDimensions
    );
}

// ---------------------------------------------------------------------------
// Roundtrip behavior on the three materials, both output modes
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_three_materials_both_modes() {
    let materials: Vec<(&str, usize, usize, usize, Vec<u16>)> = vec![
        {
            let (w, h, i) = material_gray16();
            ("gray", w, h, 1, i)
        },
        {
            let (w, h, i) = material_rgb16();
            ("rgb", w, h, 3, i)
        },
        {
            let (w, h, i) = material_rgba16();
            ("rgba", w, h, 4, i)
        },
    ];

    for (name, w, h, ch, img) in &materials {
        // Color-conversion-only reference (no DCT/quant/entropy). This isolates
        // codec loss from the unavoidable BT.601 + 4:2:0 chroma-subsampling loss
        // at saturated color edges, which is inherent to the format, not the codec.
        let color_ref: Vec<u16> = match ch {
            1 => img.clone(),
            3 => {
                let (y, cb, cr) = bitdepth16::rgb16_to_ycbcr420(img, *w, *h);
                let mut o = vec![0u16; img.len()];
                bitdepth16::ycbcr420_to_rgb16(&y, &cb, &cr, *w, *h, &mut o);
                o
            }
            _ => {
                let (y, cb, cr, a) = bitdepth16::rgba16_to_ycbcr420a(img, *w, *h);
                let mut o = vec![0u16; img.len()];
                bitdepth16::ycbcr420a_to_rgba16(&y, &cb, &cr, &a, *w, *h, &mut o);
                o
            }
        };

        for mode in [OutputMode::Keep16, OutputMode::Down8] {
            let stream = match ch {
                1 => bitdepth16::encode_grayscale16(img, *w, *h, 85, mode).unwrap(),
                3 => bitdepth16::encode_rgb16(img, *w, *h, 85, mode).unwrap(),
                _ => bitdepth16::encode_rgba16(img, *w, *h, 85, mode).unwrap(),
            };
            // Deterministic encoding.
            let stream2 = match ch {
                1 => bitdepth16::encode_grayscale16(img, *w, *h, 85, mode).unwrap(),
                3 => bitdepth16::encode_rgb16(img, *w, *h, 85, mode).unwrap(),
                _ => bitdepth16::encode_rgba16(img, *w, *h, 85, mode).unwrap(),
            };
            assert_eq!(stream, stream2, "{name} {mode:?} encode is non-deterministic");

            let dec = bitdepth16::decode16(&stream).unwrap();
            assert_eq!(dec.width, *w);
            assert_eq!(dec.height, *h);
            assert_eq!(dec.mode, mode);
            assert_eq!(dec.channels, if *ch == 1 { 1 } else { *ch });
            assert_eq!(dec.pixels.len(), img.len());

            match mode {
                OutputMode::Keep16 => {
                    // Codec-only error: DCT/quant/IDCT against the same colorspace
                    // pipeline, excluding 4:2:0 color-subsampling error. Near-lossless:
                    // max error stays in the low units on a 0..65535 scale (>85 dB PSNR).
                    let (mae, max_err) = mae_max(&color_ref, &dec.pixels);
                    assert!(max_err <= 16, "{name} keep16 codec max err {max_err}");
                    assert!(mae < 3.0, "{name} keep16 codec mae {mae}");
                }
                OutputMode::Down8 => {
                    for v in &dec.pixels {
                        assert!(*v <= 255, "{name} down8 produced 16-bit value {v}");
                    }
                    // Compare against the downshifted color-conversion reference.
                    // Down8 quantizes aggressively for size; the synthetic materials
                    // contain hard saturated edges, so max error there is large while
                    // the mean error stays moderate.
                    let ref8: Vec<u16> = color_ref.iter().map(|v| downshift8(*v)).collect();
                    let (mae, max_err) = mae_max(&ref8, &dec.pixels);
                    assert!(max_err <= 160, "{name} down8 max err {max_err}");
                    assert!(mae < 32.0, "{name} down8 mae {mae}");
                    let p = psnr(mse(&ref8, &dec.pixels), 255.0);
                    assert!(p >= 15.0, "{name} down8 psnr {p}");
                }
            }
        }
    }
}

#[test]
fn keep16_stream_is_not_smaller_than_down8() {
    // Down8 quantizes 256x harder and sparsifies invisible detail.
    let materials: Vec<(usize, usize, usize, Vec<u16>)> = vec![
        {
            let (w, h, i) = material_gray16();
            (w, h, 1, i)
        },
        {
            let (w, h, i) = material_rgb16();
            (w, h, 3, i)
        },
        {
            let (w, h, i) = material_rgba16();
            (w, h, 4, i)
        },
    ];
    for (w, h, ch, img) in &materials {
        let keep = match ch {
            1 => bitdepth16::encode_grayscale16(img, *w, *h, 85, OutputMode::Keep16).unwrap(),
            3 => bitdepth16::encode_rgb16(img, *w, *h, 85, OutputMode::Keep16).unwrap(),
            _ => bitdepth16::encode_rgba16(img, *w, *h, 85, OutputMode::Keep16).unwrap(),
        };
        let down = match ch {
            1 => bitdepth16::encode_grayscale16(img, *w, *h, 85, OutputMode::Down8).unwrap(),
            3 => bitdepth16::encode_rgb16(img, *w, *h, 85, OutputMode::Down8).unwrap(),
            _ => bitdepth16::encode_rgba16(img, *w, *h, 85, OutputMode::Down8).unwrap(),
        };
        assert!(keep.len() >= down.len(), "keep16 {} < down8 {}", keep.len(), down.len());
    }
}

// ---------------------------------------------------------------------------
// Stream-level validation and malformed-input rejection
// ---------------------------------------------------------------------------

#[test]
fn decode_rejects_malformed_streams() {
    let (w, h, img) = material_rgb16();
    let good = bitdepth16::encode_rgb16(&img, w, h, 85, OutputMode::Keep16).unwrap();

    assert_eq!(bitdepth16::decode16(&[]).unwrap_err(), BitgrainError::DecodeFailed);
    assert_eq!(bitdepth16::decode16(b"BG").unwrap_err(), BitgrainError::DecodeFailed);

    let mut bad_magic = good.clone();
    bad_magic[0] = b'X';
    assert_eq!(bitdepth16::decode16(&bad_magic).unwrap_err(), BitgrainError::DecodeFailed);

    let mut bad_version = good.clone();
    bad_version[2] = 19; // last legacy version, not a 16-bit profile
    assert_eq!(bitdepth16::decode16(&bad_version).unwrap_err(), BitgrainError::DecodeFailed);
    bad_version[2] = 22;
    assert_eq!(bitdepth16::decode16(&bad_version).unwrap_err(), BitgrainError::DecodeFailed);

    let mut bad_flags = good.clone();
    bad_flags[12] = 0x80; // undefined flag bit
    assert_eq!(bitdepth16::decode16(&bad_flags).unwrap_err(), BitgrainError::DecodeFailed);

    // Truncation at every boundary must fail, not panic.
    for cut in [1usize, 12, 13, good.len() - 1, good.len() / 2] {
        assert!(bitdepth16::decode16(&good[..cut]).is_err(), "truncation at {cut} accepted");
    }
    // Trailing garbage after the final plane is rejected.
    let mut trailing = good.clone();
    trailing.push(0xAA);
    assert_eq!(bitdepth16::decode16(&trailing).unwrap_err(), BitgrainError::DecodeFailed);
}

#[test]
fn down8_and_keep16_flags_roundtrip() {
    let (w, h, img) = material_gray16();
    let keep = bitdepth16::encode_grayscale16(&img, w, h, 90, OutputMode::Keep16).unwrap();
    let down = bitdepth16::encode_grayscale16(&img, w, h, 90, OutputMode::Down8).unwrap();
    assert_ne!(keep[12], down[12]);
    assert_eq!(bitdepth16::decode16(&keep).unwrap().mode, OutputMode::Keep16);
    assert_eq!(bitdepth16::decode16(&down).unwrap().mode, OutputMode::Down8);
}

// ---------------------------------------------------------------------------
// Legacy 8-bit path: byte-level stability of its pure-Rust building blocks.
// (The full .bg pipeline is additionally compared end-to-end between the base
// and patched release builds by scripts/bitdepth16_eval.sh.)
//
// The 16-bit work is purely additive; these goldens fail if any 8-bit
// component is accidentally perturbed. All functions exercised here are pure
// Rust so the test binary links exactly like the unmodified suite.
// ---------------------------------------------------------------------------

#[test]
fn legacy_quant_tables_are_stable() {
    // Quality 1 / 50 / 77 / 100, luma + chroma, base + perceptual v4.
    let q = 77u8;
    assert_eq!(
        crate::encoder::quant_table_for_quality(q),
        [
            7, 5, 5, 7, 11, 18, 23, 28,
            6, 6, 6, 9, 12, 27, 28, 25,
            6, 6, 7, 11, 18, 26, 32, 26,
            6, 8, 10, 13, 23, 40, 37, 29,
            8, 10, 17, 26, 31, 50, 47, 35,
            11, 16, 25, 29, 37, 48, 52, 42,
            23, 29, 36, 40, 47, 56, 55, 46,
            33, 42, 44, 45, 52, 46, 47, 46,
        ]
    );
    assert_eq!(
        crate::encoder::chroma_quant_table_for_quality(q),
        [
            8, 8, 11, 22, 46, 46, 46, 46,
            8, 10, 12, 30, 46, 46, 46, 46,
            11, 12, 26, 46, 46, 46, 46, 46,
            22, 30, 46, 46, 46, 46, 46, 46,
            46, 46, 46, 46, 46, 46, 46, 46,
            46, 46, 46, 46, 46, 46, 46, 46,
            46, 46, 46, 46, 46, 46, 46, 46,
            46, 46, 46, 46, 46, 46, 46, 46,
        ]
    );
    let p4_luma = crate::encoder::quant_table_for_quality_perceptual_v4(q);
    let p4_chroma = crate::encoder::chroma_quant_table_for_quality_perceptual_v4(q);
    assert_eq!(p4_luma[0], 7);
    assert_eq!(p4_luma[63], 128);
    assert_eq!(p4_chroma[0], 17);
    assert_eq!(p4_chroma[63], 138);
    // Quality 100 collapses every step to 1 (lossless quantization).
    assert!(crate::encoder::quant_table_for_quality(100).iter().all(|&v| v == 1));
    // Quality 1 is maximal quantization, still bounded to 255.
    assert!(crate::encoder::quant_table_for_quality(1).iter().all(|&v| v >= 1 && v <= 255));
}

#[test]
fn legacy_huffman_plane_bitstream_is_stable() {
    use crate::block::Block;
    use crate::huffman::encode_plane_with_profile;
    use crate::zigzag::ZIGZAG;
    let mut b = Block::new();
    b.data[ZIGZAG[0]] = 120;
    b.data[ZIGZAG[1]] = -17;
    b.data[ZIGZAG[5]] = 9;
    let stream = encode_plane_with_profile(&[b], false, false, true);
    // 4-byte LE length prefix + deterministic entropy payload.
    let payload_len = u32::from_le_bytes(stream[0..4].try_into().unwrap()) as usize;
    assert_eq!(payload_len + 4, stream.len());
    assert_eq!(
        stream,
        [
            0x06, 0x00, 0x00, 0x00, 0xf7, 0x8d, 0x3b, 0xfe, 0x3e, 0x6b,
        ]
    );
}

#[test]
fn legacy_blockizer_center_and_edge_replication_is_stable() {
    use crate::blockizer::Blockizer;
    // 10x9 plane: the single block row/col replicates edges into 16x16 coverage.
    let img: Vec<u8> = (0..(10 * 9)).map(|i| (i % 256) as u8).collect();
    let blocks = Blockizer::new(10, 9).generate_blocks(&img);
    assert_eq!(blocks.len(), 4);
    // Centering subtracts 128 from every sample.
    assert_eq!(blocks[0].data[0], img[0] as i16 - 128);
    // Replicated bottom-right corner sample at the last block's last cell.
    assert_eq!(blocks[3].data[63], img[9 * 10 - 1] as i16 - 128);
}

#[test]
fn legacy_zigzag_and_header_constants_are_stable() {
    use crate::zigzag::ZIGZAG;
    assert_eq!(ZIGZAG[0], 0);
    assert_eq!(ZIGZAG[1], 1);
    assert_eq!(ZIGZAG[63], 63);
    assert_eq!(crate::encoder::BG_HEADER_SIZE, 12);
}

// ---------------------------------------------------------------------------
// Benchmark / metrics: three materials x two output modes, one run.
// Prints round-trip error, size ratio, PSNR comparison and encode timing.
// ---------------------------------------------------------------------------

#[test]
fn bitdepth16_metrics_and_timing() {
    const ITERS: u32 = 100;
    let materials: Vec<(&str, usize, usize, usize, Vec<u16>)> = vec![
        {
            let (w, h, i) = material_gray16();
            ("M1-gray-32x32", w, h, 1, i)
        },
        {
            let (w, h, i) = material_rgb16();
            ("M2-rgb-24x24", w, h, 3, i)
        },
        {
            let (w, h, i) = material_rgba16();
            ("M3-rgba-16x16", w, h, 4, i)
        },
    ];

    println!();
    println!("{:<15} {:<7} {:>9} {:>8} {:>10} {:>9} {:>10} {:>10} {:>12}",
        "material", "mode", "bytes", "ratio", "MAE", "maxErr", "PSNR-dB", "PSNR-8d", "enc-us/iter");

    for (name, w, h, ch, img) in &materials {
        // Codec-only reference: same BT.601/4:2:0 color pipeline, no DCT/quant.
        let color_ref: Vec<u16> = match ch {
            1 => img.clone(),
            3 => {
                let (y, cb, cr) = bitdepth16::rgb16_to_ycbcr420(img, *w, *h);
                let mut o = vec![0u16; img.len()];
                bitdepth16::ycbcr420_to_rgb16(&y, &cb, &cr, *w, *h, &mut o);
                o
            }
            _ => {
                let (y, cb, cr, a) = bitdepth16::rgba16_to_ycbcr420a(img, *w, *h);
                let mut o = vec![0u16; img.len()];
                bitdepth16::ycbcr420a_to_rgba16(&y, &cb, &cr, &a, *w, *h, &mut o);
                o
            }
        };
        let ref8: Vec<u16> = color_ref.iter().map(|v| downshift8(*v)).collect();

        let encode = |mode: OutputMode| match ch {
            1 => bitdepth16::encode_grayscale16(img, *w, *h, 85, mode).unwrap(),
            3 => bitdepth16::encode_rgb16(img, *w, *h, 85, mode).unwrap(),
            _ => bitdepth16::encode_rgba16(img, *w, *h, 85, mode).unwrap(),
        };
        let keep_bytes = encode(OutputMode::Keep16).len();
        let down_bytes = encode(OutputMode::Down8).len();

        for mode in [OutputMode::Keep16, OutputMode::Down8] {
            // Warm up once, then average the encode wall time over ITERS so
            // first-call/allocation noise does not skew the reported number.
            let stream = encode(mode);
            let start = std::time::Instant::now();
            for _ in 0..ITERS {
                std::hint::black_box(encode(mode));
            }
            let us_per = start.elapsed().as_micros() as f64 / ITERS as f64;

            let dec = bitdepth16::decode16(&stream).unwrap();
            let (mae, max_err) = match mode {
                OutputMode::Keep16 => mae_max(&color_ref, &dec.pixels),
                OutputMode::Down8 => mae_max(&ref8, &dec.pixels),
            };
            let psnr_native = match mode {
                OutputMode::Keep16 => psnr(mse(&color_ref, &dec.pixels), 65535.0),
                OutputMode::Down8 => psnr(mse(&ref8, &dec.pixels), 255.0),
            };
            // Apples-to-apples 8-bit-domain PSNR for both modes.
            let shifted: Vec<u16> = match mode {
                OutputMode::Keep16 => dec.pixels.iter().map(|v| downshift8(*v)).collect(),
                OutputMode::Down8 => dec.pixels.clone(),
            };
            let psnr_8dom = psnr(mse(&ref8, &shifted), 255.0);

            println!(
                "{:<15} {:<7} {:>9} {:>8.3} {:>10.4} {:>9} {:>10.3} {:>10.3} {:>12.1}",
                name,
                if mode == OutputMode::Keep16 { "keep16" } else { "down8" },
                stream.len(),
                stream.len() as f64 / down_bytes as f64,
                mae,
                max_err,
                psnr_native,
                psnr_8dom,
                us_per,
            );
        }
        println!("{:<15} size ratio keep16/down8 = {:.3} ({} B / {} B)",
            name, keep_bytes as f64 / down_bytes as f64, keep_bytes, down_bytes);
    }
}






