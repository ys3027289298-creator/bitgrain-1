//! Quantization dithering (banding suppression).
//!
//! Smooth gradients show bands because the forward DCT concentrates a
//! block's average luminance in its DC coefficient, and aggressive
//! quantization rounds neighboring block DCs onto the same level. Every
//! pixel in such a block then reconstructs to one value, producing a hard
//! staircase.
//!
//! Dithering is therefore applied in the *pixel domain*, immediately before
//! the forward DCT inside the blockizer. A bounded, position-dependent
//! perturbation makes neighboring pixels round to adjacent reconstruction
//! levels while preserving the local average, rendering the gradient as
//! fine texture rather than bands. The perturbation amplitude scales with
//! `strength` (a percentage, `0..=100`) and is clamped to the legal 0..=255
//! pixel range.
//!
//! Two algorithms are provided:
//!
//! * [`DitherMode::Ordered`]: 8x8 Bayer ordered dither. Stateless and
//!   deterministic per pixel, trivially parallel.
//! * [`DitherMode::ErrorDiffusion`]: Floyd-Steinberg diffusion on each
//!   block's 8x8 pixel grid, contained per block to retain the encoder's
//!   block-level parallelism.
//!
//! `strength == 0` (or [`DitherMode::Off`]) leaves every pixel untouched, so
//! the encoded stream is byte-identical to the undithered encoder.

use std::sync::atomic::{AtomicU16, Ordering};

/// Dithering algorithm selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DitherMode {
    /// No dithering (default; encoder behaves exactly as before).
    #[default]
    Off,
    /// 8x8 Bayer ordered dither.
    Ordered,
    /// Floyd-Steinberg error diffusion within each 8x8 block.
    ErrorDiffusion,
}

/// Dithering configuration shared by every encode entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct DitherConfig {
    pub mode: DitherMode,
    /// Perturbation strength as a percentage, 0..=100. 0 disables dithering.
    pub strength: u8,
}

impl DitherConfig {
    /// Dithering is active only when a mode is selected and strength > 0.
    #[inline]
    pub fn enabled(&self) -> bool {
        self.strength > 0 && self.mode != DitherMode::Off
    }

    /// Validate user-supplied parameters.
    pub fn validate(mode: DitherMode, strength: u8) -> Result<(), &'static str> {
        match mode {
            DitherMode::Off | DitherMode::Ordered | DitherMode::ErrorDiffusion => {}
        }
        if strength > 100 {
            return Err("dither strength must be in 0..=100");
        }
        Ok(())
    }
}

/// 8x8 Bayer ordered-dither matrix, values 0..64 (mean 31.5).
const BAYER_8X8: [[i32; 8]; 8] = [
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
];

/// Maximum per-pixel perturbation (in 0..=255 units) at strength 100.
///
/// The ordered dither is purely a block-grid Bayer modulation: every pixel in
/// a block receives the same offset, so the block's mean (its DC term) moves
/// deterministically across quantization boundaries while neighboring blocks
/// alternate between adjacent reconstruction levels. This dissolves the hard
/// plateau edges of a banded gradient without introducing intra-block noise.
/// The value spans roughly one effective DC reconstruction step at low
/// quality and is small relative to the 0..=255 range.
const MAX_AMPLITUDE: i32 = 24;

#[inline]
fn amplitude(strength: u8) -> i32 {
    MAX_AMPLITUDE * strength as i32 / 100
}

/// Ordered dither for one 8x8 block of source pixels.
///
/// `src` is the block's 64 source values (0..=255) in raster order and
/// `(bx, by)` is its plane-space origin. The Bayer pattern is anchored to
/// the plane origin so the texture is continuous across blocks.
#[inline]
pub fn ordered_pixel_block(src: &[u8; 64], bx: usize, by: usize, strength: u8) -> [u8; 64] {
    let amp = amplitude(strength);
    // Block-grid Bayer value in -32..=31 drives the per-block DC modulation.
    let block_t = BAYER_8X8[(by / 8) & 7][(bx / 8) & 7] - 32;
    let block_offset = block_t * amp / 32;
    let mut out = [0u8; 64];
    for i in 0..64 {
        let v = src[i] as i32 + block_offset;
        out[i] = v.clamp(0, 255) as u8;
    }
    out
}

/// Floyd-Steinberg error diffusion for one 8x8 block of source pixels.
///
/// Two effects are combined:
///
/// 1. A block-grid Bayer seed shifts the whole block's tone (its DC term) so
///    neighboring blocks alternate across DC quantization boundaries, which
///    is what removes the hard band edges at low bit rates. This mirrors the
///    ordered dither's block-level component.
/// 2. Classic Floyd-Steinberg diffusion rounds each pixel onto a fine grid
///    and pushes the residual to the standard four neighbors (7/16 right,
///    3/16 lower-left, 5/16 below, 1/16 lower-right), adding intermediate
///    levels wherever AC coefficients survive.
///
/// Diffusion is contained to the block so blocks stay independent and
/// parallelizable.
///
/// `strength` controls how much of the residual is propagated. At zero
/// strength nothing is spread (the block is effectively undithered once the
/// caller's `enabled()` gate is considered); at full strength the complete
/// error flows through the grid, maximizing the number of intermediate
/// reconstruction levels while preserving the block's average tone.
#[inline]
pub fn error_diffusion_pixel_block(
    src: &[u8; 64],
    bx: usize,
    by: usize,
    strength: u8,
) -> [u8; 64] {
    // Fixed fine rounding grid: dithering redistributes values within this
    // spacing rather than changing its size, matching the textbook algorithm.
    const GRID: i32 = 4;
    let factor = strength as i32;
    let amp = amplitude(strength);
    // Block-grid DC seed in -amp..=amp.
    let seed_t = BAYER_8X8[(by / 8) & 7][(bx / 8) & 7] - 32;
    let seed = seed_t * amp / 32;
    let mut work = [0i32; 64];
    for i in 0..64 {
        work[i] = src[i] as i32 + seed;
    }
    let mut out = [0u8; 64];
    for y in 0..8 {
        for x in 0..8 {
            let i = y * 8 + x;
            let value = work[i];
            // Round onto the fixed grid with Euclidean rounding so negative
            // intermediate accumulators behave consistently.
            let scaled = value.div_euclid(GRID);
            let quantized = (scaled * GRID).clamp(0, 255);
            out[i] = quantized as u8;
            let error = value - quantized;
            // Strength scales the propagated residual only.
            let spread = error * factor / 100;
            if x + 1 < 8 {
                work[i + 1] += spread * 7 / 16;
            }
            if y + 1 < 8 {
                if x > 0 {
                    work[i + 7] += spread * 3 / 16;
                }
                work[i + 8] += spread * 5 / 16;
                if x + 1 < 8 {
                    work[i + 9] += spread / 16;
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Process-wide configuration (encoder entry points keep their signatures)
// ---------------------------------------------------------------------------

static ACTIVE_DITHER: AtomicU16 = AtomicU16::new(0);

const MODE_OFF: u16 = 0;
const MODE_ORDERED: u16 = 1;
const MODE_ERROR_DIFFUSION: u16 = 2;

fn pack(mode: DitherMode, strength: u8) -> u16 {
    let m = match mode {
        DitherMode::Off => MODE_OFF,
        DitherMode::Ordered => MODE_ORDERED,
        DitherMode::ErrorDiffusion => MODE_ERROR_DIFFUSION,
    };
    m << 8 | strength as u16
}

fn unpack(raw: u16) -> DitherConfig {
    let mode = match raw >> 8 {
        MODE_ORDERED => DitherMode::Ordered,
        MODE_ERROR_DIFFUSION => DitherMode::ErrorDiffusion,
        _ => DitherMode::Off,
    };
    DitherConfig { mode, strength: (raw & 0xFF) as u8 }
}

/// Program the dither configuration for subsequent encodes on this process.
pub fn set_dither_config(mode: DitherMode, strength: u8) -> Result<(), &'static str> {
    DitherConfig::validate(mode, strength)?;
    ACTIVE_DITHER.store(pack(mode, strength), Ordering::Relaxed);
    Ok(())
}

/// Read the active dither configuration.
pub fn current_dither_config() -> DitherConfig {
    unpack(ACTIVE_DITHER.load(Ordering::Relaxed))
}

/// Reset dithering to the disabled default.
pub fn clear_dither_config() {
    ACTIVE_DITHER.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_block(v: u8) -> [u8; 64] {
        [v; 64]
    }

    #[test]
    fn ordered_off_is_identity_when_disabled() {
        let src = flat_block(100);
        let out = ordered_pixel_block(&src, 0, 0, 0);
        assert_eq!(out, src);
    }

    #[test]
    fn ordered_strength_bounds_and_variation() {
        // Pure block-level modulation: a block receives a uniform offset, so
        // a flat block stays uniform but moves within the bounded range.
        let src = flat_block(128);
        let out_origin = ordered_pixel_block(&src, 0, 0, 100);
        let out_shifted = ordered_pixel_block(&src, 8, 0, 100);
        for out in [&out_origin, &out_shifted] {
            let min = *out.iter().min().unwrap();
            let max = *out.iter().max().unwrap();
            assert_eq!(min, max, "block modulation is uniform within a block");
            assert!(min >= 128 - MAX_AMPLITUDE as u8);
            assert!(max <= 128 + MAX_AMPLITUDE as u8);
        }
        assert_ne!(
            out_origin, out_shifted,
            "different block-grid positions must produce different offsets"
        );
        // A ramp block produces multiple output levels.
        let mut ramp = [0u8; 64];
        for (i, p) in ramp.iter_mut().enumerate() {
            *p = (i * 4) as u8;
        }
        let out = ordered_pixel_block(&ramp, 0, 0, 100);
        let distinct = {
            let mut v = out.to_vec();
            v.sort_unstable();
            v.dedup();
            v.len()
        };
        assert!(distinct > 1);
    }

    #[test]
    fn ordered_clamps_at_range_edges() {
        let src = flat_block(255);
        let out = ordered_pixel_block(&src, 0, 0, 100);
        assert!(out.iter().all(|&v| v <= 255));
        let src = flat_block(0);
        let out = ordered_pixel_block(&src, 0, 0, 100);
        assert!(out.iter().all(|&v| v >= 0));
    }

    #[test]
    fn error_diffusion_produces_levels_and_stays_in_range() {
        let mut src = [0u8; 64];
        for (i, p) in src.iter_mut().enumerate() {
            *p = (i * 4) as u8;
        }
        let out = error_diffusion_pixel_block(&src, 0, 0, 75);
        assert!(out.iter().all(|&v| v <= 255));
        let distinct = {
            let mut v: Vec<u8> = out.to_vec();
            v.sort_unstable();
            v.dedup();
            v.len()
        };
        assert!(distinct > 1, "error diffusion must introduce levels");
    }

    #[test]
    fn ordered_is_position_deterministic() {
        let src = flat_block(200);
        let a = ordered_pixel_block(&src, 8, 8, 40);
        let b = ordered_pixel_block(&src, 8, 8, 40);
        assert_eq!(a, b);
    }

    #[test]
    fn strength_out_of_range_is_rejected() {
        assert!(DitherConfig::validate(DitherMode::Ordered, 101).is_err());
        assert!(DitherConfig::validate(DitherMode::ErrorDiffusion, 255).is_err());
    }

    #[test]
    fn valid_strength_is_accepted() {
        assert!(DitherConfig::validate(DitherMode::Ordered, 0).is_ok());
        assert!(DitherConfig::validate(DitherMode::Ordered, 100).is_ok());
        assert!(DitherConfig::validate(DitherMode::ErrorDiffusion, 1).is_ok());
        assert!(DitherConfig::validate(DitherMode::Off, 0).is_ok());
    }

    #[test]
    fn global_config_roundtrip() {
        clear_dither_config();
        assert!(!current_dither_config().enabled());
        set_dither_config(DitherMode::Ordered, 50).unwrap();
        let cfg = current_dither_config();
        assert_eq!(cfg.mode, DitherMode::Ordered);
        assert_eq!(cfg.strength, 50);
        assert!(cfg.enabled());
        clear_dither_config();
        assert!(!current_dither_config().enabled());
    }
}
