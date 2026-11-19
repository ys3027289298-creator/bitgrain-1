//! Validation harness for in-loop SAO (run: cargo run --release --example sao_validate).
//!
//! Generates the four target material classes at 1080p, encodes each with and
//! without SAO, and reports:
//!   * PSNR against the original
//!   * ringing energy: MSE measured in the flat-region ring around strong
//!     edges (the pixels where DCT ringing actually shows)
//!   * wall-clock time of the SAO stage and of full encode+decode
//!
//! Pure std + the bitgrain crate; no external image files needed.

use bitgrain::decoder;
use bitgrain::encoder;
use bitgrain::sao;
use std::time::Instant;

const W: usize = 1920;
const H: usize = 1080;

#[derive(Clone, Copy)]
enum Material {
    Text,
    Gradient,
    Texture,
    Noise,
}

fn gen_material(m: Material) -> Vec<u8> {
    let mut img = vec![0u8; W * H * 3];
    let mut state: u32 = 0x1234_5678;
    let mut rnd = || -> u8 {
        // xorshift deterministic noise
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state & 0xFF) as u8
    };
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) * 3;
            let v = match m {
                Material::Gradient => {
                    // Smooth multi-axis gradient: banding/contouring prone.
                    let g = ((x as u32 * 255 / W as u32) / 4 * 4
                        + (y as u32 * 255 / H as u32) / 8 * 2)
                        as u8;
                    g
                }
                Material::Text => {
                    // High-contrast glyph-like blocks + sharp strokes on flat bg.
                    let in_glyph = (x / 12 + y / 12) % 5 == 0
                        || (x % 60 < 3 && y % 80 < 40)
                        || (y % 60 < 3 && x % 80 < 40);
                    if in_glyph { 20 } else { 235 }
                }
                Material::Texture => {
                    // Repeating fine checker/diagonal texture.
                    if ((x / 3) + (y / 3)) % 2 == 0 { 180 } else { 70 }
                }
                Material::Noise => rnd(),
            };
            img[i] = v;
            img[i + 1] = v;
            img[i + 2] = v; // grayscale-as-RGB keeps metrics channel-aligned
        }
    }
    img
}

fn mse(a: &[u8], b: &[u8]) -> f64 {
    let s: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as i64 - y as i64).pow(2) as u64)
        .sum();
    s as f64 / a.len() as f64
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let m = mse(a, b);
    if m <= 0.0 {
        99.0
    } else {
        10.0 * (255.0 * 255.0 / m).log10()
    }
}

/// Sobel gradient magnitude on the luma plane (R==G==B for these materials).
fn edge_map(img: &[u8]) -> Vec<f32> {
    let mut mag = vec![0f32; W * H];
    let at = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, W as isize - 1) as usize;
        let cy = y.clamp(0, H as isize - 1) as usize;
        img[(cy * W + cx) * 3] as i32
    };
    for y in 0..H {
        for x in 0..W {
            let (ix, iy) = (x as isize, y as isize);
            let gx = -at(ix - 1, iy - 1) - 2 * at(ix - 1, iy) - at(ix - 1, iy + 1)
                + at(ix + 1, iy - 1) + 2 * at(ix + 1, iy) + at(ix + 1, iy + 1);
            let gy = -at(ix - 1, iy - 1) - 2 * at(ix, iy - 1) - at(ix + 1, iy - 1)
                + at(ix - 1, iy + 1) + 2 * at(ix, iy + 1) + at(ix + 1, iy + 1);
            mag[y * W + x] = ((gx * gx + gy * gy) as f32).sqrt();
        }
    }
    mag
}

/// Ringing energy: MSE in a 2-5 px ring around strong edges, excluding the
/// edge core itself (±1 px) so genuine edge sharpness isn't counted.
fn ringing_mse(orig: &[u8], recon: &[u8]) -> f64 {
    let em = edge_map(orig);
    let threshold = 300.0f32; // strong edges only
    let mut near = vec![false; W * H];
    for y in 0..H {
        for x in 0..W {
            if em[y * W + x] > threshold {
                for dy in -5isize..=5 {
                    for dx in -5isize..=5 {
                        let d = (dx.abs().max(dy.abs())) as usize;
                        if d >= 2 && d <= 5 {
                            let nx = x as isize + dx;
                            let ny = y as isize + dy;
                            if nx >= 0 && ny >= 0 && (nx as usize) < W && (ny as usize) < H {
                                near[(ny as usize) * W + nx as usize] = true;
                            }
                        }
                    }
                }
            }
        }
    }
    let (mut sum, mut cnt) = (0u64, 0u64);
    for p in 0..W * H {
        if near[p] {
            for c in 0..3 {
                let d = orig[p * 3 + c] as i64 - recon[p * 3 + c] as i64;
                sum += (d * d) as u64;
                cnt += 1;
            }
        }
    }
    if cnt == 0 {
        0.0
    } else {
        sum as f64 / cnt as f64
    }
}

fn roundtrip(image: &[u8], use_sao: bool) -> (Vec<u8>, f64, f64, f64) {
    let cap = image.len() * 2 + (1 << 22);
    let mut buf = vec![0u8; cap];
    let mut pos = 0i32;
    let quality = 40u8;

    let t0 = Instant::now();
    if use_sao {
        let cfg = encoder::default_sao_config();
        encoder::encode_rgb_with_sao(image, W, H, quality, &mut buf, &mut pos, None, &cfg);
    } else {
        encoder::encode_rgb_ycbcr(image, W, H, quality, &mut buf, &mut pos, None);
    }
    let t_enc = t0.elapsed().as_secs_f64() * 1000.0;

    buf.truncate(pos as usize);
    let mut out = vec![0u8; W * H * 3];
    let (mut ow, mut oh, mut oc) = (0u32, 0u32, 0u32);
    let t1 = Instant::now();
    let ok = decoder::decode(&buf, &mut out, &mut ow, &mut oh, &mut oc, None);
    let t_dec = t1.elapsed().as_secs_f64() * 1000.0;
    assert!(ok && ow as usize == W && oh as usize == H && oc == 3);

    (out, t_enc, t_dec, (pos as f64) / 1024.0)
}

/// Time the SAO stage (estimate + apply on the three planes) on a
/// 1080p-style plane set. The "reconstruction" is the original plus a
/// quantization-like perturbation so both estimation and application run
/// their full classification paths (no all-Off shortcut).
fn time_sao_stage(image: &[u8]) -> (f64, f64) {
    let (y, cb, cr) = bitgrain::colorspace::rgb_to_ycbcr420(image, W, H);
    let cw = (W + 1) / 2;
    let ch = (H + 1) / 2;
    let cfg = sao::SaoConfig::default();

    let perturb = |p: &[u8]| -> Vec<u8> {
        let mut s: u32 = 0xABCDEF01;
        p.iter()
            .map(|&v| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let delta = ((s >> 24) as i32 % 9) - 4; // -4..=4
                (v as i32 + delta).clamp(0, 255) as u8
            })
            .collect()
    };
    let y_recon = perturb(&y);
    let cb_recon = perturb(&cb);
    let cr_recon = perturb(&cr);

    // Warmup.
    let _ = sao::estimate_plane(&y, &y_recon, W, H, &cfg);

    let t0 = Instant::now();
    let yp = sao::estimate_plane(&y, &y_recon, W, H, &cfg);
    let cbp = sao::estimate_plane(&cb, &cb_recon, cw, ch, &cfg);
    let crp = sao::estimate_plane(&cr, &cr_recon, cw, ch, &cfg);
    let est_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut yy = y_recon.clone();
    let mut cbb = cb_recon.clone();
    let mut crr = cr_recon.clone();
    let t1 = Instant::now();
    sao::apply_plane(&mut yy, W, H, &cfg, &yp);
    sao::apply_plane(&mut cbb, cw, ch, &cfg, &cbp);
    sao::apply_plane(&mut crr, cw, ch, &cfg, &crp);
    let apply_ms = t1.elapsed().as_secs_f64() * 1000.0;
    (est_ms, apply_ms)
}

fn main() {
    let materials = [
        (Material::Text, "text"),
        (Material::Gradient, "gradient"),
        (Material::Texture, "texture"),
        (Material::Noise, "noise"),
    ];

    println!(
        "{:<9} {:>7} | {:>9} {:>9} {:>10} | {:>9} {:>9} {:>10} | {:>8} {:>8} {:>8} {:>8} {:>8}",
        "material", "Q",
        "PSNR_no", "PSNR_sao", "ring_no",
        "ring_sao", "ring_%", "enc_ms",
        "dec_ms", "est_ms", "apply_ms", "sao_ms", "sizeKB"
    );
    println!("{}", "-".repeat(134));

    for (mat, name) in materials {
        let image = gen_material(mat);
        let quality = 40;

        // Best-of-3 timing for stable numbers, but metrics from one run.
        let mut best_enc = f64::MAX;
        let mut best_dec = f64::MAX;
        let mut dec_sao = Vec::new();
        let mut size_kb = 0.0;
        for _ in 0..3 {
            let (out, te, td, kb) = roundtrip(&image, true);
            best_enc = best_enc.min(te);
            best_dec = best_dec.min(td);
            dec_sao = out;
            size_kb = kb;
        }
        let (dec_no, _, _, _) = roundtrip(&image, false);

        let p_no = psnr(&image, &dec_no);
        let p_sao = psnr(&image, &dec_sao);
        let r_no = ringing_mse(&image, &dec_no);
        let r_sao = ringing_mse(&image, &dec_sao);
        let ring_pct = if r_no > 0.0 {
            (r_sao - r_no) / r_no * 100.0
        } else {
            0.0
        };
        let (est_ms, apply_ms) = time_sao_stage(&image);

        println!(
            "{:<9} {:>7} | {:>9.3} {:>9.3} {:>10.4} | {:>9.4} {:>9.1}% {:>10.2} | {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.1}",
            name, quality,
            p_no, p_sao, r_no,
            r_sao, ring_pct, best_enc,
            best_dec, est_ms, apply_ms, est_ms + apply_ms, size_kb
        );
    }
    println!("\nring_% negative = ringing reduced. SAO budget for a 1080p frame: < 25 ms.");
}
