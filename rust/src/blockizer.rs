use crate::block::Block;
use crate::dither::{self, DitherConfig, DitherMode};
use rayon::prelude::*;

pub struct Blockizer {
    width: usize,
    height: usize,
}

impl Blockizer {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Apply the active pixel-domain dither to one block's source pixels.
    /// Returns the input untouched when dithering is disabled.
    #[inline]
    fn dither_pixels(src: &[u8; 64], bx: usize, by: usize, cfg: DitherConfig) -> [u8; 64] {
        if !cfg.enabled() {
            return *src;
        }
        match cfg.mode {
            DitherMode::Off => *src,
            DitherMode::Ordered => {
                dither::ordered_pixel_block(src, bx, by, cfg.strength)
            }
            DitherMode::ErrorDiffusion => {
                dither::error_diffusion_pixel_block(src, bx, by, cfg.strength)
            }
        }
    }

    /// Generate 8×8 blocks from a grayscale plane. Parallel via Rayon.
    pub fn generate_blocks(&self, image: &[u8]) -> Vec<Block> {
        let w = self.width;
        let h = self.height;
        let cfg = dither::current_dither_config();
        let blocks_wide = (w + 7) / 8;
        let blocks_high = (h + 7) / 8;
        let num_blocks = blocks_wide * blocks_high;

        (0..num_blocks)
            .into_par_iter()
            .map(|idx| {
                let bx = (idx % blocks_wide) * 8;
                let by = (idx / blocks_wide) * 8;
                let mut src = [0u8; 64];
                for y in 0..8 {
                    let iy = (by + y).min(h.saturating_sub(1));
                    let row_base = iy * w;
                    for x in 0..8 {
                        let ix = (bx + x).min(w.saturating_sub(1));
                        src[y * 8 + x] = image[row_base + ix];
                    }
                }
                let pixels = Self::dither_pixels(&src, bx, by, cfg);
                let mut block = [0i16; 64];
                for i in 0..64 {
                    block[i] = pixels[i] as i16 - 128;
                }
                Block { data: block }
            })
            .collect()
    }

    /// Generate blocks for one channel from interleaved RGB. Parallel via Rayon.
    pub fn generate_blocks_rgb(&self, image: &[u8], channel: usize) -> Vec<Block> {
        self.generate_blocks_interleaved_par(image, 3, channel)
    }

    /// Generate blocks for one channel from interleaved RGBA. Parallel via Rayon.
    pub fn generate_blocks_rgba(&self, image: &[u8], channel: usize) -> Vec<Block> {
        self.generate_blocks_interleaved_par(image, 4, channel)
    }

    fn generate_blocks_interleaved_par(&self, image: &[u8], stride: usize, channel: usize) -> Vec<Block> {
        let w = self.width;
        let h = self.height;
        let cfg = dither::current_dither_config();
        let blocks_wide = (w + 7) / 8;
        let blocks_high = (h + 7) / 8;
        let num_blocks = blocks_wide * blocks_high;

        (0..num_blocks)
            .into_par_iter()
            .map(|idx| {
                let bx = (idx % blocks_wide) * 8;
                let by = (idx / blocks_wide) * 8;
                let mut src = [0u8; 64];
                for y in 0..8 {
                    let iy = (by + y).min(h.saturating_sub(1));
                    let row_base = iy * w;
                    for x in 0..8 {
                        let ix = (bx + x).min(w.saturating_sub(1));
                        src[y * 8 + x] = image[(row_base + ix) * stride + channel];
                    }
                }
                let pixels = Self::dither_pixels(&src, bx, by, cfg);
                let mut block = [0i16; 64];
                for i in 0..64 {
                    block[i] = pixels[i] as i16 - 128;
                }
                Block { data: block }
            })
            .collect()
    }
}
