//! Motion-estimation evaluation on four synthetic materials:
//! translation, rotation, zoom, occlusion.
//!
//! Criteria verified here:
//! - residual energy drops vs pure frame differencing,
//! - P-frame bitstream is smaller than both intra and pure-difference coding,
//! - motion-search time stays below 30% of total encode time (release only),
//! - occluded regions emit no (wrong) motion vectors — they go intra,
//! - P-frame roundtrip through encode/decode stays visually faithful.

use crate::colorspace;
use crate::decoder;
use crate::encoder;
use crate::motion::{self, MbMode};
use std::time::Instant;

const Q: u8 = 85;

// ---------------------------------------------------------------------------
// Synthetic material generation
// ---------------------------------------------------------------------------

/// Deterministic textured base frame (RGB): smooth gradients + sinusoidal
/// structure + fine hash noise + distinct rectangles, so block matching has
/// unambiguous texture to lock onto.
fn base_frame(w: usize, h: usize) -> Vec<u8> {
    let mut img = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let fx = x as f64;
            let fy = y as f64;
            let n = ((x as u32)
                .wrapping_mul(2_654_435_761)
                ^ (y as u32).wrapping_mul(2_246_822_519))
                % 9;
            let n = n as f64 - 4.0;
            let lum = 128.0 + 55.0 * (fx * 0.11).sin() * (fy * 0.07).cos()
                + 30.0 * ((fx + fy) * 0.045).sin()
                + n;
            let r = lum + 12.0 * (fx * 0.02).sin();
            let g = lum;
            let b = lum - 12.0 * (fy * 0.017).cos();
            let i = (y * w + x) * 3;
            img[i] = r.clamp(0.0, 255.0) as u8;
            img[i + 1] = g.clamp(0.0, 255.0) as u8;
            img[i + 2] = b.clamp(0.0, 255.0) as u8;
        }
    }
    // A few high-contrast rectangles with their own (non-periodic) texture.
    let rects = [
        (w / 8, h / 8, w / 4, h / 4, 0.31f64, 0.17f64),
        (w * 5 / 8, h / 2, w / 4, h / 4, 0.23, 0.29),
    ];
    for &(rx, ry, rw, rh, fx, fy) in &rects {
        for y in ry..(ry + rh).min(h) {
            for x in rx..(rx + rw).min(w) {
                let v = 128.0
                    + 70.0 * (x as f64 * fx).sin() * (y as f64 * fy).cos()
                    + 30.0 * ((x + y) as f64 * 0.11).sin();
                let i = (y * w + x) * 3;
                img[i] = v.clamp(0.0, 255.0) as u8;
                img[i + 1] = (v + 24.0).clamp(0.0, 255.0) as u8;
                img[i + 2] = (v - 16.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    img
}

#[inline]
fn px(img: &[u8], w: usize, h: usize, x: i64, y: i64, c: usize) -> u8 {
    let cx = x.clamp(0, w as i64 - 1) as usize;
    let cy = y.clamp(0, h as i64 - 1) as usize;
    img[(cy * w + cx) * 3 + c]
}

/// Content shifts by (dx, dy) pixels.
fn translate(img: &[u8], w: usize, h: usize, dx: i64, dy: i64) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            for c in 0..3 {
                out[(y as usize * w + x as usize) * 3 + c] = px(img, w, h, x - dx, y - dy, c);
            }
        }
    }
    out
}

#[inline]
fn bilinear(img: &[u8], w: usize, h: usize, fx: f64, fy: f64, c: usize) -> u8 {
    let x0 = fx.floor() as i64;
    let y0 = fy.floor() as i64;
    let ax = fx - x0 as f64;
    let ay = fy - y0 as f64;
    let p = |dx: i64, dy: i64| px(img, w, h, x0 + dx, y0 + dy, c) as f64;
    let top = p(0, 0) * (1.0 - ax) + p(1, 0) * ax;
    let bot = p(0, 1) * (1.0 - ax) + p(1, 1) * ax;
    (top * (1.0 - ay) + bot * ay).clamp(0.0, 255.0) as u8
}

/// Rotation about the frame center.
fn rotate(img: &[u8], w: usize, h: usize, degrees: f64) -> Vec<u8> {
    let theta = degrees.to_radians();
    let (sin, cos) = theta.sin_cos();
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let sx = cx + dx * cos + dy * sin;
            let sy = cy - dx * sin + dy * cos;
            for c in 0..3 {
                out[(y * w + x) * 3 + c] = bilinear(img, w, h, sx, sy, c);
            }
        }
    }
    out
}

/// Zoom about the frame center (factor > 1 magnifies).
fn zoom(img: &[u8], w: usize, h: usize, factor: f64) -> Vec<u8> {
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let sx = cx + (x as f64 - cx) / factor;
            let sy = cy + (y as f64 - cy) / factor;
            for c in 0..3 {
                out[(y * w + x) * 3 + c] = bilinear(img, w, h, sx, sy, c);
            }
        }
    }
    out
}

/// Background translates while a new object occludes part of the scene.
/// Returns (frame2, occluder_rect) with rect = (x, y, w, h).
fn occlude(img: &[u8], w: usize, h: usize) -> (Vec<u8>, (usize, usize, usize, usize)) {
    let mut out = translate(img, w, h, 5, 3);
    let rect = (w * 3 / 16, h / 4, w / 4 + w / 8, h / 2);
    let (rx, ry, rw, rh) = rect;
    for y in ry..(ry + rh).min(h) {
        for x in rx..(rx + rw).min(w) {
            // Occluder: strong texture unrelated to the background, mean ~128.
            let v = 128.0
                + 75.0 * (x as f64 * 0.21 + 1.7).sin() * (y as f64 * 0.19 + 0.4).cos()
                + 35.0 * ((x as f64 - y as f64) * 0.13).sin();
            let i = (y * w + x) * 3;
            out[i] = v.clamp(0.0, 255.0) as u8;
            out[i + 1] = (v + 20.0).clamp(0.0, 255.0) as u8;
            out[i + 2] = (v - 24.0).clamp(0.0, 255.0) as u8;
        }
    }
    (out, rect)
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

fn luma_of(img: &[u8], w: usize, h: usize) -> Vec<u8> {
    colorspace::rgb_to_ycbcr420(img, w, h).0
}

/// Residual energy of pure frame differencing: Σ(cur − ref)².
fn energy_pure_diff(cur_y: &[u8], ref_y: &[u8]) -> u64 {
    cur_y
        .iter()
        .zip(ref_y.iter())
        .map(|(&c, &r)| {
            let d = c as i64 - r as i64;
            (d * d) as u64
        })
        .sum()
}

/// Residual energy of the ME codec: inter MBs use the compensated reference,
/// intra MBs are coded centered (cur − 128).
fn energy_with_me(cur_y: &[u8], ref_y: &[u8], w: usize, h: usize, modes: &[MbMode]) -> u64 {
    let (mb_w, _) = motion::mb_grid(w, h);
    let mut e: u64 = 0;
    for y in 0..h {
        for x in 0..w {
            let mb = (y / motion::ME_BLOCK) * mb_w + (x / motion::ME_BLOCK);
            let d = match modes[mb] {
                MbMode::Intra => cur_y[y * w + x] as i64 - 128,
                MbMode::Inter(mv) => {
                    let rx = (x as i64 + mv.dx as i64).clamp(0, w as i64 - 1) as usize;
                    let ry = (y as i64 + mv.dy as i64).clamp(0, h as i64 - 1) as usize;
                    cur_y[y * w + x] as i64 - ref_y[ry * w + rx] as i64
                }
            };
            e += (d * d) as u64;
        }
    }
    e
}

fn encode_pframe(cur: &[u8], ref_: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h * 3 * 2 + 4096];
    let mut pos: i32 = 0;
    encoder::encode_pframe_rgb(cur, ref_, w, h, Q, &mut buf, &mut pos, None);
    buf.truncate(pos as usize);
    buf
}

fn encode_intra(img: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h * 3 * 2 + 4096];
    let mut pos: i32 = 0;
    encoder::encode_rgb(img, w, h, Q, &mut buf, &mut pos, None);
    buf.truncate(pos as usize);
    buf
}

/// "帧间只做差分" baseline: code the pixel difference image as an intra frame.
fn encode_pure_diff(cur: &[u8], ref_: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut diff = vec![0u8; w * h * 3];
    for i in 0..w * h * 3 {
        diff[i] = (cur[i] as i16 - ref_[i] as i16 + 128).clamp(0, 255) as u8;
    }
    encode_intra(&diff, w, h)
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return 99.0;
    }
    10.0 * (255.0 * 255.0 / mse).log10()
}

struct MaterialResult {
    name: &'static str,
    energy_diff: u64,
    energy_me: u64,
    size_pframe: usize,
    size_intra: usize,
    size_diff: usize,
}

fn evaluate(name: &'static str, f1: &[u8], f2: &[u8], w: usize, h: usize) -> MaterialResult {
    let y1 = luma_of(f1, w, h);
    let y2 = luma_of(f2, w, h);
    let modes = motion::estimate_frame(&y2, &y1, w, h, Q);

    let energy_diff = energy_pure_diff(&y2, &y1);
    let energy_me = energy_with_me(&y2, &y1, w, h, &modes);

    let size_pframe = encode_pframe(f2, f1, w, h).len();
    let size_intra = encode_intra(f2, w, h).len();
    let size_diff = encode_pure_diff(f2, f1, w, h).len();

    let reduction = 100.0 * (1.0 - energy_me as f64 / energy_diff as f64);
    println!(
        "{name}: residual energy {energy_diff} -> {energy_me} ({reduction:.1}% down), \
         bytes pframe={size_pframe} intra={size_intra} diff={size_diff}"
    );

    // Shared criteria: residual energy down, and P-frame beats both baselines.
    assert!(
        energy_me < energy_diff,
        "{name}: ME residual energy {energy_me} not below pure diff {energy_diff}"
    );
    assert!(
        size_pframe < size_intra,
        "{name}: P-frame {size_pframe} B not smaller than intra {size_intra} B"
    );
    assert!(
        size_pframe < size_diff,
        "{name}: P-frame {size_pframe} B not smaller than pure-diff {size_diff} B"
    );

    MaterialResult {
        name,
        energy_diff,
        energy_me,
        size_pframe,
        size_intra,
        size_diff,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Diamond search recovers a known displacement.
#[test]
fn diamond_search_finds_true_mv() {
    let w = 64;
    let h = 64;
    let f1 = base_frame(w, h);
    let y1 = luma_of(&f1, w, h);
    let f2 = translate(&f1, w, h, 6, -3);
    let y2 = luma_of(&f2, w, h);
    // Block well inside the frame: true MV is (-6, 3).
    let (mv, _sad) = motion::diamond_search(&y2, &y1, w, h, 32, 32, motion::lambda_for_quality(Q));
    assert_eq!((mv.dx, mv.dy), (-6, 3));
}

/// MV section roundtrip.
#[test]
fn mv_section_roundtrip() {
    let modes = vec![
        MbMode::Intra,
        MbMode::Inter(motion::MotionVector { dx: -7, dy: 4 }),
        MbMode::Inter(motion::MotionVector { dx: 0, dy: 0 }),
        MbMode::Inter(motion::MotionVector { dx: 16, dy: -16 }),
    ];
    let sec = motion::encode_mv_section(&modes);
    let (decoded, pos) = motion::decode_mv_section(&sec, 0, modes.len()).unwrap();
    assert_eq!(pos, sec.len());
    assert_eq!(decoded, modes);
}

/// Translation: large uniform displacement — biggest residual-energy win.
#[test]
fn material_translation() {
    let (w, h) = (160, 128);
    let f1 = base_frame(w, h);
    let f2 = translate(&f1, w, h, 7, -4);
    let r = evaluate("translation", &f1, &f2, w, h);
    let reduction = 1.0 - r.energy_me as f64 / r.energy_diff as f64;
    assert!(
        reduction >= 0.70,
        "translation: residual energy reduction {:.1}% < 70%",
        reduction * 100.0
    );

    // Most interior MBs should lock onto the true MV (-7, 4).
    let y1 = luma_of(&f1, w, h);
    let y2 = luma_of(&f2, w, h);
    let modes = motion::estimate_frame(&y2, &y1, w, h, Q);
    let (mb_w, mb_h) = motion::mb_grid(w, h);
    let mut correct = 0usize;
    let mut total = 0usize;
    for mby in 1..mb_h - 1 {
        for mbx in 1..mb_w - 1 {
            if let MbMode::Inter(mv) = modes[mby * mb_w + mbx] {
                total += 1;
                if (mv.dx, mv.dy) == (-7, 4) {
                    correct += 1;
                }
            }
        }
    }
    assert!(
        correct * 5 >= total * 4,
        "translation: only {correct}/{total} interior MBs have the true MV"
    );
}

/// Rotation: spatially varying displacement field.
#[test]
fn material_rotation() {
    let (w, h) = (160, 128);
    let f1 = base_frame(w, h);
    let f2 = rotate(&f1, w, h, 6.0);
    let r = evaluate("rotation", &f1, &f2, w, h);
    let reduction = 1.0 - r.energy_me as f64 / r.energy_diff as f64;
    assert!(
        reduction >= 0.40,
        "rotation: residual energy reduction {:.1}% < 40%",
        reduction * 100.0
    );
}

/// Zoom: radial displacement field.
#[test]
fn material_zoom() {
    let (w, h) = (160, 128);
    let f1 = base_frame(w, h);
    let f2 = zoom(&f1, w, h, 1.06);
    let r = evaluate("zoom", &f1, &f2, w, h);
    let reduction = 1.0 - r.energy_me as f64 / r.energy_diff as f64;
    assert!(
        reduction >= 0.40,
        "zoom: residual energy reduction {:.1}% < 40%",
        reduction * 100.0
    );
}

/// Occlusion: occluded MBs must go intra — no wrong motion vectors.
#[test]
fn material_occlusion() {
    let (w, h) = (160, 128);
    let f1 = base_frame(w, h);
    let (f2, (rx, ry, rw, rh)) = occlude(&f1, w, h);
    let r = evaluate("occlusion", &f1, &f2, w, h);
    let reduction = 1.0 - r.energy_me as f64 / r.energy_diff as f64;
    assert!(
        reduction >= 0.20,
        "occlusion: residual energy reduction {:.1}% < 20%",
        reduction * 100.0
    );

    let y1 = luma_of(&f1, w, h);
    let y2 = luma_of(&f2, w, h);
    let modes = motion::estimate_frame(&y2, &y1, w, h, Q);
    let (mb_w, _) = motion::mb_grid(w, h);
    let mb = motion::ME_BLOCK;
    let mut occluded_mbs = 0usize;
    // Every MB fully covered by the occluder must be intra.
    for mby in 0..h / mb {
        for mbx in 0..w / mb {
            let (bx, by) = (mbx * mb, mby * mb);
            let fully_inside = bx >= rx && by >= ry && bx + mb <= rx + rw && by + mb <= ry + rh;
            if fully_inside {
                occluded_mbs += 1;
                assert_eq!(
                    modes[mby * mb_w + mbx],
                    MbMode::Intra,
                    "occluded MB at ({bx},{by}) emitted a motion vector"
                );
            }
        }
    }
    assert!(occluded_mbs >= 4, "test occluder too small to be meaningful");
    println!("occlusion: {occluded_mbs} occluded MBs all intra (no wrong MVs)");
}

/// P-frame roundtrip: decode against the reference reproduces the frame.
#[test]
fn pframe_roundtrip_all_materials() {
    let (w, h) = (160, 128);
    let f1 = base_frame(w, h);
    let materials: Vec<(&str, Vec<u8>)> = vec![
        ("translation", translate(&f1, w, h, 7, -4)),
        ("rotation", rotate(&f1, w, h, 6.0)),
        ("zoom", zoom(&f1, w, h, 1.06)),
        ("occlusion", occlude(&f1, w, h).0),
    ];
    for (name, f2) in materials {
        let stream = encode_pframe(&f2, &f1, w, h);
        let mut dec = vec![0u8; w * h * 3];
        let (mut ow, mut oh, mut och) = (0u32, 0u32, 0u32);
        assert!(
            decoder::decode_pframe(&stream, &f1, &mut dec, &mut ow, &mut oh, &mut och),
            "{name}: P-frame decode failed"
        );
        assert_eq!((ow, oh, och), (w as u32, h as u32, 3));
        let p = psnr(&f2, &dec);
        println!("{name}: roundtrip PSNR {p:.2} dB");
        assert!(p >= 30.0, "{name}: roundtrip PSNR {p:.2} dB < 30 dB");
    }
}

/// Odd dimensions (partial macroblocks/blocks at the edges) roundtrip.
#[test]
fn pframe_roundtrip_odd_size() {
    let (w, h) = (150, 102);
    let f1 = base_frame(w, h);
    let f2 = translate(&f1, w, h, -5, 6);
    let stream = encode_pframe(&f2, &f1, w, h);
    let mut dec = vec![0u8; w * h * 3];
    let (mut ow, mut oh, mut och) = (0u32, 0u32, 0u32);
    assert!(decoder::decode_pframe(&stream, &f1, &mut dec, &mut ow, &mut oh, &mut och));
    assert_eq!((ow, oh, och), (w as u32, h as u32, 3));
    let p = psnr(&f2, &dec);
    assert!(p >= 30.0, "odd-size roundtrip PSNR {p:.2} dB < 30 dB");
}

/// Motion-search overhead must stay below 30% of total encode time.
/// (Timing assertion only in release builds; debug builds just report.)
#[test]
fn motion_search_overhead_below_30_percent() {
    let (w, h) = (640, 360);
    let f1 = base_frame(w, h);
    let f2 = translate(&f1, w, h, 9, 5);
    let y1 = luma_of(&f1, w, h);
    let y2 = luma_of(&f2, w, h);

    let mut me_best = u128::MAX;
    let mut total_best = u128::MAX;
    for _ in 0..5 {
        let t0 = Instant::now();
        let modes = motion::estimate_frame(&y2, &y1, w, h, Q);
        me_best = me_best.min(t0.elapsed().as_nanos());
        std::hint::black_box(&modes);

        let t1 = Instant::now();
        let stream = encode_pframe(&f2, &f1, w, h);
        total_best = total_best.min(t1.elapsed().as_nanos());
        std::hint::black_box(&stream);
    }

    let ratio = me_best as f64 / total_best as f64;
    println!(
        "motion search: {:.2} ms, total encode: {:.2} ms, overhead {:.1}%",
        me_best as f64 / 1e6,
        total_best as f64 / 1e6,
        ratio * 100.0
    );
    #[cfg(not(debug_assertions))]
    assert!(
        ratio < 0.30,
        "motion search overhead {:.1}% >= 30% of encode time",
        ratio * 100.0
    );
}


