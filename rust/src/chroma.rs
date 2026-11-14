//! Chroma subsampling modes (4:4:4 / 4:2:2 / 4:2:0) with centroid-aligned
//! sampling positions and matching nearest-neighbor reconstruction.
//!
//! Sampling grid (luma pixel centers at integer coordinates, origin top-left):
//!
//! 4:4:4 (no subsampling):
//!   chroma sample (cx, cy) reads luma pixel (cx, cy) — 1:1, no offset.
//!
//! 4:2:2 (horizontal 2x, vertical 1x):
//!   chroma sample (cx, cy) averages luma pixels (2*cx, cy) and (2*cx+1, cy).
//!   The contributing pixel centers are x = 2*cx and x = 2*cx+1, whose
//!   centroid is x = 2*cx + 0.5; the chroma sample is placed at that same
//!   centroid, so the grid does not shift relative to the luma grid.
//!   Edge column (odd width): the single pixel (2*cx, cy) is replicated.
//!
//! 4:2:0 (horizontal 2x, vertical 2x):
//!   chroma sample (cx, cy) averages the 2x2 luma block at
//!   (2*cx, 2*cy), (2*cx+1, 2*cy), (2*cx, 2*cy+1), (2*cx+1, 2*cy+1).
//!   Contributing centroids: x = 2*cx + 0.5, y = 2*cy + 0.5 — the chroma
//!   sample is placed exactly at the block centroid (co-sited averaging),
//!   no half-pixel grid shift. Edge blocks average only in-bounds pixels.
//!
//! Reconstruction (decoder) is nearest-neighbor: every luma position picks
//! the chroma sample whose centroid cell contains it. Because the encoder
//! averaging cells and decoder selection cells are the identical partition,
//! the round trip is symmetric and introduces no positional offset.

/// Chroma subsampling selector. The default (`H2V1`) reproduces the
/// historical 4:2:0 bitstream byte-for-byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaSubsampling {
    /// Cb/Cr at full resolution (w x h).
    H1V1,
    /// Cb/Cr at (ceil(w/2)) x h.
    H2V1,
    /// Cb/Cr at (ceil(w/2)) x (ceil(h/2)).
    H2V2,
}

impl ChromaSubsampling {
    /// Chroma plane dimensions for a luma plane of size `w` x `h`.
    #[inline]
    pub fn chroma_dims(self, w: usize, h: usize) -> (usize, usize) {
        match self {
            ChromaSubsampling::H1V1 => (w, h),
            ChromaSubsampling::H2V1 => ((w + 1) / 2, h),
            ChromaSubsampling::H2V2 => ((w + 1) / 2, (h + 1) / 2),
        }
    }

    /// Parse the `BITGRAIN_CHROMA` override. Unknown values fall back to the
    /// default so the legacy path and its byte output stay untouched.
    pub fn from_env() -> Self {
        match std::env::var("BITGRAIN_CHROMA") {
            Ok(v) => match v.trim() {
                "444" | "4:4:4" => ChromaSubsampling::H1V1,
                "422" | "4:2:2" => ChromaSubsampling::H2V1,
                _ => ChromaSubsampling::H2V2,
            },
            Err(_) => ChromaSubsampling::H2V2,
        }
    }
}

#[inline]
fn clamp_u8(v: i32) -> u8 {
    if v < 0 {
        0
    } else if v > 255 {
        255
    } else {
        v as u8
    }
}

/// Full-resolution per-pixel Cb/Cr (BT.601 full-range, same fixed point as
/// the 4:2:0 path in `colorspace`).
#[inline]
fn rgb_cb_cr(r: i32, g: i32, b: i32) -> (i32, i32) {
    let cb = ((-43 * r - 85 * g + 128 * b + 128) >> 8) + 128;
    let cr = ((128 * r - 107 * g - 21 * b + 128) >> 8) + 128;
    (cb, cr)
}

/// Downsample interleaved RGB (3 bytes/pixel) into Cb/Cr planes.
pub fn rgb_to_chroma(
    image: &[u8],
    w: usize,
    h: usize,
    mode: ChromaSubsampling,
) -> (Vec<u8>, Vec<u8>) {
    let (cw, ch) = mode.chroma_dims(w, h);
    let mut cb_plane = vec![0u8; cw * ch];
    let mut cr_plane = vec![0u8; cw * ch];

    let step_x = if matches!(mode, ChromaSubsampling::H1V1) { 1 } else { 2 };
    let step_y = if matches!(mode, ChromaSubsampling::H2V2) { 2 } else { 1 };

    for cy in 0..ch {
        for cx in 0..cw {
            let mut sum_cb = 0i32;
            let mut sum_cr = 0i32;
            let mut count = 0i32;
            for dy in 0..step_y {
                for dx in 0..step_x {
                    let px = cx * step_x + dx;
                    let py = cy * step_y + dy;
                    if px < w && py < h {
                        let base = (py * w + px) * 3;
                        let (cb, cr) = rgb_cb_cr(
                            image[base] as i32,
                            image[base + 1] as i32,
                            image[base + 2] as i32,
                        );
                        sum_cb += cb;
                        sum_cr += cr;
                        count += 1;
                    }
                }
            }
            let idx = cy * cw + cx;
            cb_plane[idx] = clamp_u8((sum_cb + (count >> 1)) / count);
            cr_plane[idx] = clamp_u8((sum_cr + (count >> 1)) / count);
        }
    }

    (cb_plane, cr_plane)
}

/// Downsample interleaved RGBA (4 bytes/pixel) into Cb/Cr planes.
pub fn rgba_to_chroma(
    image: &[u8],
    w: usize,
    h: usize,
    mode: ChromaSubsampling,
) -> (Vec<u8>, Vec<u8>) {
    let (cw, ch) = mode.chroma_dims(w, h);
    let mut cb_plane = vec![0u8; cw * ch];
    let mut cr_plane = vec![0u8; cw * ch];

    let step_x = if matches!(mode, ChromaSubsampling::H1V1) { 1 } else { 2 };
    let step_y = if matches!(mode, ChromaSubsampling::H2V2) { 2 } else { 1 };

    for cy in 0..ch {
        for cx in 0..cw {
            let mut sum_cb = 0i32;
            let mut sum_cr = 0i32;
            let mut count = 0i32;
            for dy in 0..step_y {
                for dx in 0..step_x {
                    let px = cx * step_x + dx;
                    let py = cy * step_y + dy;
                    if px < w && py < h {
                        let base = (py * w + px) * 4;
                        let (cb, cr) = rgb_cb_cr(
                            image[base] as i32,
                            image[base + 1] as i32,
                            image[base + 2] as i32,
                        );
                        sum_cb += cb;
                        sum_cr += cr;
                        count += 1;
                    }
                }
            }
            let idx = cy * cw + cx;
            cb_plane[idx] = clamp_u8((sum_cb + (count >> 1)) / count);
            cr_plane[idx] = clamp_u8((sum_cr + (count >> 1)) / count);
        }
    }

    (cb_plane, cr_plane)
}

/// Nearest-neighbor upsample of a Cb/Cr plane back to luma resolution.
///
/// Luma position `(px, py)` selects sample `(px / step_x, py / step_y)`,
/// which is exactly the cell whose centroid the encoder averaged. This
/// keeps the sampling grid aligned — no shifted or interpolated positions.
pub fn upsample_plane(plane: &[u8], cw: usize, ch: usize, w: usize, h: usize,
                      mode: ChromaSubsampling) -> Vec<u8> {
    let step_x = if matches!(mode, ChromaSubsampling::H1V1) { 1 } else { 2 };
    let step_y = if matches!(mode, ChromaSubsampling::H2V2) { 2 } else { 1 };
    debug_assert_eq!(plane.len(), cw * ch);

    let mut out = vec![0u8; w * h];
    for py in 0..h {
        let cy = (py / step_y).min(ch - 1);
        for px in 0..w {
            let cx = (px / step_x).min(cw - 1);
            out[py * w + px] = plane[cy * cw + cx];
        }
    }
    out
}
