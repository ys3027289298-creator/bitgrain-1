//! Progressive decoding tests: spectral previews, exact final frame,
//! preview memory bound, interruption, and malformed-input handling.

use crate::decoder;
use crate::encoder;
use crate::progressive::{
    decode_progressive, preview_working_bytes, probe, ProgressiveFrameInfo, DEFAULT_SCHEDULE,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Small synthetic RGB image: smooth gradients + a sharp edge + mild
/// periodic texture, so both low and high DCT frequencies are exercised
/// while staying image-like (incompressible noise would dominate PSNR).
fn make_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut img = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            // Mid-frequency texture (±16) so mid rounds have detail to add.
            let tex = (((x * x / 3 + y * y / 5) % 9) as i16 - 4) * 2;
            // Bottom-right quadrant: ±10 checkerboard = highest DCT frequency,
            // only the final full-coefficient round can reproduce it.
            let hf = if x >= w / 2 && y >= h / 2 {
                if (x + y) % 2 == 0 { 14 } else { -14 }
            } else {
                0
            };
            let r = (x * 255 / w) as i16 + tex + hf;
            let g = (y * 255 / h) as i16 + tex + hf;
            let b = (if x < w / 2 { 40 } else { 210 }) + tex + hf; // sharp vertical edge
            img[i] = r.clamp(0, 255) as u8;
            img[i + 1] = g.clamp(0, 255) as u8;
            img[i + 2] = b.clamp(0, 255) as u8;
        }
    }
    img
}

fn make_rgba(w: usize, h: usize) -> Vec<u8> {
    let rgb = make_rgb(w, h);
    let mut img = vec![0u8; w * h * 4];
    for p in 0..w * h {
        img[p * 4..p * 4 + 3].copy_from_slice(&rgb[p * 3..p * 3 + 3]);
        img[p * 4 + 3] = (128 + (p % 97)) as u8;
    }
    img
}

fn encode_rgb(img: &[u8], w: usize, h: usize, q: u8) -> Vec<u8> {
    let mut out = vec![0u8; img.len() * 2 + 1024];
    let mut pos: i32 = 0;
    encoder::encode_rgb(img, w, h, q, &mut out, &mut pos, None);
    out.truncate(pos as usize);
    out
}

fn encode_rgba(img: &[u8], w: usize, h: usize, q: u8) -> Vec<u8> {
    let mut out = vec![0u8; img.len() * 2 + 1024];
    let mut pos: i32 = 0;
    encoder::encode_rgba(img, w, h, q, &mut out, &mut pos, None);
    out.truncate(pos as usize);
    out
}

fn decode_one_shot(bg: &[u8], channels: usize) -> (Vec<u8>, u32, u32) {
    let (w, h, _) = probe(bg).expect("probe failed");
    let mut out = vec![0u8; w as usize * h as usize * channels];
    let (mut ow, mut oh, mut oc) = (0u32, 0u32, 0u32);
    assert!(
        decoder::decode(bg, &mut out, &mut ow, &mut oh, &mut oc, None),
        "one-shot decode failed"
    );
    assert_eq!(oc as usize, channels);
    (out, ow, oh)
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
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

fn max_abs_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.abs_diff(y))
        .max()
        .unwrap_or(0)
}

struct RunResult {
    frames: Vec<(ProgressiveFrameInfo, Vec<u8>)>,
    ok: bool,
}

fn run_progressive(bg: &[u8], schedule: Option<&[usize]>, stop_after_round: Option<usize>) -> RunResult {
    let (w, h, c) = probe(bg).expect("probe failed");
    let mut out = vec![0u8; w as usize * h as usize * c as usize];
    let mut frames: Vec<(ProgressiveFrameInfo, Vec<u8>)> = Vec::new();
    let ok = decode_progressive(bg, schedule, &mut out, |info, pixels| {
        frames.push((*info, pixels.to_vec()));
        stop_after_round != Some(info.round)
    });
    RunResult { frames, ok }
}

// ---------------------------------------------------------------------------
// Core guarantees
// ---------------------------------------------------------------------------

#[test]
fn progressive_final_frame_is_bit_exact_rgb() {
    let (w, h) = (48usize, 40usize);
    let img = make_rgb(w, h);
    let bg = encode_rgb(&img, w, h, 85);
    let (reference, _, _) = decode_one_shot(&bg, 3);

    let run = run_progressive(&bg, None, None);
    assert!(run.ok, "progressive decode failed");
    assert_eq!(run.frames.len(), DEFAULT_SCHEDULE.len() + 1);

    let (info, final_frame) = run.frames.last().unwrap();
    assert!(info.is_final);
    assert_eq!(info.coef_budget, 64);
    assert_eq!(
        max_abs_diff(final_frame, &reference),
        0,
        "final progressive frame must be bit-identical to one-shot decode"
    );
}

#[test]
fn progressive_final_frame_is_bit_exact_rgba() {
    let (w, h) = (40usize, 32usize);
    let img = make_rgba(w, h);
    let bg = encode_rgba(&img, w, h, 85);
    let (reference, _, _) = decode_one_shot(&bg, 4);

    let run = run_progressive(&bg, None, None);
    assert!(run.ok);
    let (_, final_frame) = run.frames.last().unwrap();
    assert_eq!(max_abs_diff(final_frame, &reference), 0);
}

#[test]
fn progressive_frames_are_full_size_and_refine() {
    let (w, h) = (48usize, 40usize);
    let img = make_rgb(w, h);
    let bg = encode_rgb(&img, w, h, 85);
    let (reference, _, _) = decode_one_shot(&bg, 3);

    let run = run_progressive(&bg, None, None);
    assert!(run.ok);

    let mut prev_psnr = 0.0f64;
    println!("round | coefs | PSNR vs final (dB)");
    for (info, frame) in &run.frames {
        // Every interruption point yields a complete full-size image.
        assert_eq!(frame.len(), w * h * 3, "frame must be full-size");
        assert_eq!(info.width as usize, w);
        assert_eq!(info.height as usize, h);
        assert_eq!(info.channels, 3);

        let p = psnr(frame, &reference);
        println!("{:5} | {:5} | {:6.2}", info.round, info.coef_budget, p);
        if !info.is_final {
            assert!(p >= prev_psnr, "PSNR must not regress between rounds");
            prev_psnr = p;
        } else {
            assert!(p.is_infinite(), "final round must match exactly");
        }
    }
}

#[test]
fn progressive_preview_memory_within_half_of_final() {
    for &(w, h, c) in &[(48usize, 40usize, 3usize), (40, 32, 4), (640, 480, 3), (641, 479, 4)] {
        let preview = preview_working_bytes(w, h);
        let final_img = w * h * c;
        println!(
            "{w}x{h}x{c}: preview working = {preview} B, final = {final_img} B, ratio = {:.1}%",
            100.0 * preview as f64 / final_img as f64
        );
        assert!(
            preview * 2 <= final_img,
            "preview working memory {preview} exceeds half of final image {final_img}"
        );
    }
}

#[test]
fn progressive_interruption_keeps_full_size_frame() {
    let (w, h) = (48usize, 40usize);
    let img = make_rgb(w, h);
    let bg = encode_rgb(&img, w, h, 85);

    // Stop after round index 1 (second preview round).
    let run = run_progressive(&bg, None, Some(1));
    assert!(run.ok, "interruption is a normal outcome, not an error");
    assert_eq!(run.frames.len(), 2, "must stop right after round 1");
    let (info, frame) = run.frames.last().unwrap();
    assert!(!info.is_final);
    assert_eq!(frame.len(), w * h * 3, "interrupted frame must be full-size");
}

#[test]
fn progressive_alpha_is_opaque_in_previews() {
    let (w, h) = (40usize, 32usize);
    let img = make_rgba(w, h);
    let bg = encode_rgba(&img, w, h, 85);

    let run = run_progressive(&bg, None, None);
    assert!(run.ok);
    for (info, frame) in &run.frames {
        if !info.is_final {
            assert!(
                frame.iter().skip(3).step_by(4).all(|&a| a == 255),
                "preview alpha must be opaque"
            );
        }
    }
}

#[test]
fn progressive_legacy_versions_fall_back_to_single_frame() {
    // v1 grayscale RLE stream.
    let (w, h) = (24usize, 16usize);
    let img: Vec<u8> = (0..w * h).map(|i| (i % 251) as u8).collect();
    let mut out = vec![0u8; img.len() * 4 + 1024];
    let mut pos: i32 = 0;
    encoder::encode_grayscale(&img, w, h, 85, &mut out, &mut pos);
    let bg = &out[..pos as usize];

    let mut pixels = vec![0u8; w * h];
    let mut n_frames = 0usize;
    let ok = decode_progressive(bg, None, &mut pixels, |info, frame| {
        n_frames += 1;
        assert!(info.is_final);
        assert_eq!(info.channels, 1);
        assert_eq!(frame.len(), w * h);
        true
    });
    assert!(ok);
    assert_eq!(n_frames, 1, "legacy streams deliver exactly one frame");
}

// ---------------------------------------------------------------------------
// Failure paths (must fail cleanly, never panic)
// ---------------------------------------------------------------------------

#[test]
fn progressive_rejects_malformed_inputs() {
    let (w, h) = (48usize, 40usize);
    let img = make_rgb(w, h);
    let bg = encode_rgb(&img, w, h, 85);
    let mut out = vec![0u8; w * h * 3];

    // Empty / short buffers.
    assert!(!decode_progressive(&[], None, &mut out, |_, _| true));
    assert!(!decode_progressive(&bg[..5], None, &mut out, |_, _| true));
    // Bad magic.
    let mut bad = bg.clone();
    bad[0] = b'X';
    assert!(!decode_progressive(&bad, None, &mut out, |_, _| true));
    // Reserved versions.
    let mut bad = bg.clone();
    bad[2] = 0;
    assert!(!decode_progressive(&bad, None, &mut out, |_, _| true));
    let mut bad = bg.clone();
    bad[2] = 20;
    assert!(!decode_progressive(&bad, None, &mut out, |_, _| true));
    // Zero dimensions.
    let mut bad = bg.clone();
    bad[3..7].copy_from_slice(&0u32.to_le_bytes());
    assert!(!decode_progressive(&bad, None, &mut out, |_, _| true));
    // Output buffer too small.
    let mut tiny = vec![0u8; w * h];
    assert!(!decode_progressive(&bg, None, &mut tiny, |_, _| true));
}

#[test]
fn progressive_rejects_truncated_stream() {
    let (w, h) = (48usize, 40usize);
    let img = make_rgb(w, h);
    let bg = encode_rgb(&img, w, h, 85);
    let mut out = vec![0u8; w * h * 3];

    // Cut the stream at several points inside the entropy-coded planes.
    let cuts = [bg.len() / 4, bg.len() / 2, bg.len() * 3 / 4, bg.len() - 2];
    for &cut in &cuts {
        let truncated = &bg[..cut];
        // Must return false (or stop without panic); never panic, never
        // report a final frame.
        let mut saw_final = false;
        let _ = decode_progressive(truncated, None, &mut out, |info, _| {
            saw_final |= info.is_final;
            true
        });
        assert!(!saw_final, "truncated stream must not yield a final frame");
    }
}

// ---------------------------------------------------------------------------
// Report: PSNR after rounds 1/2/4 + timing + memory (run with --nocapture)
// ---------------------------------------------------------------------------

#[test]
fn progressive_report_psnr_and_memory() {
    let (w, h) = (96usize, 80usize);
    let img = make_rgb(w, h);
    // q95 keeps high-frequency coefficients alive so every round is visible.
    let bg = encode_rgb(&img, w, h, 95);
    let (reference, _, _) = decode_one_shot(&bg, 3);

    // Time one-shot decode (the "before": first image only after full decode).
    let t0 = std::time::Instant::now();
    let (reference2, _, _) = decode_one_shot(&bg, 3);
    let one_shot_us = t0.elapsed().as_micros();
    assert_eq!(reference, reference2);

    // Time progressive decode, capturing when the first frame becomes visible.
    let mut out = vec![0u8; w * h * 3];
    let mut frames: Vec<(ProgressiveFrameInfo, Vec<u8>, u128)> = Vec::new();
    let t1 = std::time::Instant::now();
    let ok = decode_progressive(&bg, None, &mut out, |info, pixels| {
        frames.push((*info, pixels.to_vec(), t1.elapsed().as_micros()));
        true
    });
    let progressive_total_us = t1.elapsed().as_micros();
    assert!(ok);

    println!("\n=== progressive decode report ({}x{} RGB, q95, {} bytes .bg) ===", w, h, bg.len());
    println!("round | coefs | PSNR vs original | PSNR vs final | max diff vs final | frame ready at");
    for (info, frame, at_us) in &frames {
        let p_orig = psnr(frame, &img);
        let p_final = psnr(frame, &reference);
        let md = max_abs_diff(frame, &reference);
        println!(
            "{:5} | {:5} | {:16.2} | {:13.2} | {:17} | {} us",
            info.round, info.coef_budget, p_orig, p_final, md, at_us
        );
    }
    let preview = preview_working_bytes(w, h);
    let final_img = w * h * 3;
    // One-shot decoder internal peak: Y/Cb/Cr planes (1.5·w·h) plus a full
    // i16 block Vec for the plane being decoded (luma: 2·w·h) ≈ 3.5·w·h.
    let one_shot_internal = final_img / 3 * 7 / 2;
    println!(
        "memory: preview working = {} B ({:.1}% of final {} B, limit 50%) | one-shot internal peak ≈ {} B ({:.0}%)",
        preview,
        100.0 * preview as f64 / final_img as f64,
        final_img,
        one_shot_internal,
        100.0 * one_shot_internal as f64 / final_img as f64
    );
    println!(
        "time: first preview frame at {} us | one-shot decode {} us | progressive total {} us",
        frames[0].2, one_shot_us, progressive_total_us
    );

    let (final_info, final_frame, _) = frames.last().unwrap();
    assert!(final_info.is_final);
    assert_eq!(max_abs_diff(final_frame, &reference), 0);
}

#[test]
fn oversized_output_buffer_is_accepted() {
    // Callers may hand in a larger capacity buffer (the repo's own Python
    // binding allocates w*h*4 as an upper bound). Decode must use only the
    // exact w*h*channels prefix and never run past the planes.
    let (w, h) = (48usize, 40usize);
    let img = make_rgb(w, h);
    let bg = encode_rgb(&img, w, h, 85);
    let (reference, _, _) = decode_one_shot(&bg, 3);

    // One-shot decode into an oversized buffer.
    let mut big = vec![0u8; w * h * 4];
    let (mut ow, mut oh, mut oc) = (0u32, 0u32, 0u32);
    assert!(decoder::decode(&bg, &mut big, &mut ow, &mut oh, &mut oc, None));
    assert_eq!(&big[..w * h * 3], &reference[..]);

    // Progressive decode into an oversized buffer.
    let mut big2 = vec![0u8; w * h * 4];
    let ok = decode_progressive(&bg, None, &mut big2, |info, pixels| {
        assert_eq!(pixels.len(), w * h * 3, "frame view must be exact-size");
        let _ = info;
        true
    });
    assert!(ok);
    assert_eq!(&big2[..w * h * 3], &reference[..]);
}
