//! Block-matching motion estimation for inter (P-frame) coding.
//!
//! - 16×16 luma macroblocks, SAD matching on the luma plane.
//! - Diamond search: large-diamond (9-point) descent, then small-diamond
//!   (5-point) refinement.
//! - Rate-distortion motion-vector cost: J = SAD + λ·bits(MV), with λ
//!   derived from the encoding quality.
//! - Intra fallback with a safety margin: a macroblock only goes inter when
//!   motion compensation clearly beats intra, so occluded or otherwise
//!   unmatchable regions never emit (wrong) motion vectors.
//!
//! MV convention: the vector points from the current block to its prediction
//! in the reference frame, i.e. `pred(x, y) = ref(y + dy, x + dx)` with
//! edge-clamped coordinates. Chroma (4:2:0) reuses the luma MVs at half
//! resolution.

use crate::block::Block;
use rayon::prelude::*;

/// Luma macroblock size for motion estimation.
pub const ME_BLOCK: usize = 16;
/// Maximum |dx|, |dy| explored by the diamond search.
pub const SEARCH_RANGE: i16 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotionVector {
    pub dx: i16,
    pub dy: i16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MbMode {
    /// Coded without prediction (pixel − 128, like an intra frame block).
    Intra,
    /// Coded as motion-compensated residual against the reference frame.
    Inter(MotionVector),
}

// Large diamond search pattern (8 points around the center).
const LDSP: [(i16, i16); 8] = [
    (0, -2), (1, -1), (2, 0), (1, 1),
    (0, 2), (-1, 1), (-2, 0), (-1, -1),
];
// Small diamond search pattern (4 points around the center).
const SDSP: [(i16, i16); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

/// Estimated bit cost of coding one motion vector (mode bit + magnitude
/// categories + signs). Used only for the RD search cost, not the bitstream.
#[inline]
fn mv_bits(mv: MotionVector) -> u32 {
    #[inline]
    fn cat(v: i16) -> u32 {
        let mut a = (v as i32).unsigned_abs();
        let mut n = 0;
        while a > 0 {
            n += 1;
            a >>= 1;
        }
        n
    }
    2 + 2 * (cat(mv.dx) + cat(mv.dy))
}

/// Motion-search λ: how many SAD units one MV bit is worth. Coarser
/// quantization (lower quality) raises the price of motion bits.
#[inline]
pub fn lambda_for_quality(quality: u8) -> u32 {
    let q = quality.clamp(1, 100) as u32;
    ((101 - q) / 2).max(2)
}

/// SAD of the 16×16 block at (bx, by) in `cur` against `ref` shifted by
/// (dx, dy). Reference samples are edge-clamped. Aborts early (returning a
/// value ≥ `abort_above`) once the partial sum cannot beat `abort_above`.
#[inline]
fn sad_block(
    cur: &[u8],
    ref_: &[u8],
    w: usize,
    h: usize,
    bx: usize,
    by: usize,
    dx: i16,
    dy: i16,
    abort_above: u32,
) -> u32 {
    let mut sad: u32 = 0;
    for y in 0..ME_BLOCK {
        let cy = (by + y).min(h - 1);
        let ry = (cy as i32 + dy as i32).clamp(0, h as i32 - 1) as usize;
        let cur_row = &cur[cy * w..cy * w + w];
        let ref_row = &ref_[ry * w..ry * w + w];
        for x in 0..ME_BLOCK {
            let cx = (bx + x).min(w - 1);
            let rx = (cx as i32 + dx as i32).clamp(0, w as i32 - 1) as usize;
            sad += (cur_row[cx] as i32 - ref_row[rx] as i32).unsigned_abs();
        }
        if sad >= abort_above {
            return sad;
        }
    }
    sad
}

#[inline]
fn rd_cost(sad: u32, mv: MotionVector, lambda: u32) -> u32 {
    sad + lambda * mv_bits(mv)
}

/// Diamond search for the best motion vector of the 16×16 block at pixel
/// (bx, by). Returns the winning MV and its SAD.
pub fn diamond_search(
    cur: &[u8],
    ref_: &[u8],
    w: usize,
    h: usize,
    bx: usize,
    by: usize,
    lambda: u32,
) -> (MotionVector, u32) {
    let zero = MotionVector { dx: 0, dy: 0 };
    let zero_sad = sad_block(cur, ref_, w, h, bx, by, 0, 0, u32::MAX);
    // Static-block fast path: perfect match at (0,0) needs no search.
    if zero_sad == 0 {
        return (zero, 0);
    }
    let mut best = zero;
    let mut best_sad = zero_sad;
    let mut best_cost = rd_cost(zero_sad, zero, lambda);

    // Large-diamond descent until the center wins.
    loop {
        let mut improved = false;
        for &(ox, oy) in &LDSP {
            let cand = MotionVector {
                dx: (best.dx + ox).clamp(-SEARCH_RANGE, SEARCH_RANGE),
                dy: (best.dy + oy).clamp(-SEARCH_RANGE, SEARCH_RANGE),
            };
            if cand == best {
                continue;
            }
            let sad = sad_block(cur, ref_, w, h, bx, by, cand.dx, cand.dy, best_cost);
            let cost = rd_cost(sad, cand, lambda);
            if cost < best_cost {
                best_cost = cost;
                best_sad = sad;
                best = cand;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    // Small-diamond refinement.
    loop {
        let mut improved = false;
        for &(ox, oy) in &SDSP {
            let cand = MotionVector {
                dx: (best.dx + ox).clamp(-SEARCH_RANGE, SEARCH_RANGE),
                dy: (best.dy + oy).clamp(-SEARCH_RANGE, SEARCH_RANGE),
            };
            if cand == best {
                continue;
            }
            let sad = sad_block(cur, ref_, w, h, bx, by, cand.dx, cand.dy, best_cost);
            let cost = rd_cost(sad, cand, lambda);
            if cost < best_cost {
                best_cost = cost;
                best_sad = sad;
                best = cand;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    (best, best_sad)
}

/// Intra cost proxy: SAD of the block against its own mean (approximates the
/// energy an intra DCT block would have to code).
fn intra_cost(cur: &[u8], w: usize, h: usize, bx: usize, by: usize) -> u32 {
    let mut sum: u32 = 0;
    for y in 0..ME_BLOCK {
        let cy = (by + y).min(h - 1);
        for x in 0..ME_BLOCK {
            let cx = (bx + x).min(w - 1);
            sum += cur[cy * w + cx] as u32;
        }
    }
    let mean = (sum + 128) / 256;
    let mut cost: u32 = 0;
    for y in 0..ME_BLOCK {
        let cy = (by + y).min(h - 1);
        for x in 0..ME_BLOCK {
            let cx = (bx + x).min(w - 1);
            cost += (cur[cy * w + cx] as i32 - mean as i32).unsigned_abs();
        }
    }
    cost
}

/// Number of macroblocks in a frame of `w`×`h` luma pixels.
#[inline]
pub fn mb_grid(w: usize, h: usize) -> (usize, usize) {
    ((w + ME_BLOCK - 1) / ME_BLOCK, (h + ME_BLOCK - 1) / ME_BLOCK)
}

/// Decide inter vs intra for one macroblock.
///
/// Inter must beat intra decisively (RD cost below half the intra cost):
/// genuine motion matches land far under that bar, while occluded or newly
/// exposed regions — where the "best" match is unrelated content with a cost
/// close to (or above) intra — fall back to intra instead of emitting a
/// wrong motion vector.
fn decide_mb(
    cur: &[u8],
    ref_: &[u8],
    w: usize,
    h: usize,
    mbx: usize,
    mby: usize,
    lambda: u32,
) -> MbMode {
    let bx = mbx * ME_BLOCK;
    let by = mby * ME_BLOCK;
    let (mv, sad) = diamond_search(cur, ref_, w, h, bx, by, lambda);
    let inter_cost = rd_cost(sad, mv, lambda) as u64;
    let intra = intra_cost(cur, w, h, bx, by) as u64;
    if inter_cost * 2 < intra {
        MbMode::Inter(mv)
    } else {
        MbMode::Intra
    }
}

/// Estimate motion for a whole frame (luma planes, same dimensions).
/// Returns one mode per macroblock in raster order. Parallel over MB rows.
pub fn estimate_frame(
    cur_y: &[u8],
    ref_y: &[u8],
    w: usize,
    h: usize,
    quality: u8,
) -> Vec<MbMode> {
    let (mb_w, mb_h) = mb_grid(w, h);
    let lambda = lambda_for_quality(quality);
    (0..mb_h)
        .into_par_iter()
        .flat_map_iter(|mby| {
            (0..mb_w).map(move |mbx| decide_mb(cur_y, ref_y, w, h, mbx, mby, lambda))
        })
        .collect()
}

/// Round-to-nearest (away from zero) halving for luma→chroma MV scaling.
#[inline]
pub fn half_mv(v: i16) -> i16 {
    if v >= 0 {
        (v + 1) / 2
    } else {
        (v - 1) / 2
    }
}

/// Macroblocks per plane row and 8×8 blocks per macroblock for a plane.
/// Luma: 16×16 MBs (2×2 blocks). Chroma 4:2:0: 8×8 MBs (1 block), and
/// ceil(cw/8) == ceil(w/16) so both grids share the same MB count.
#[inline]
fn plane_mb_params(plane_w: usize, chroma: bool) -> (usize, usize) {
    if chroma {
        ((plane_w + 7) / 8, 1)
    } else {
        ((plane_w + ME_BLOCK - 1) / ME_BLOCK, ME_BLOCK / 8)
    }
}

/// Serialize the per-MB mode/MV decisions.
/// Format: [payload_len: u32 LE][payload]
/// Payload per MB in raster order: mode byte (0 = intra, 1 = inter);
/// inter entries are followed by dx, dy as signed bytes.
pub fn encode_mv_section(modes: &[MbMode]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(modes.len() * 3);
    for m in modes {
        match m {
            MbMode::Intra => payload.push(0u8),
            MbMode::Inter(mv) => {
                payload.push(1u8);
                payload.push(mv.dx as i8 as u8);
                payload.push(mv.dy as i8 as u8);
            }
        }
    }
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

/// Inverse of `encode_mv_section`. Expects exactly `n_mb` decisions and the
/// payload to be fully consumed. Returns (modes, position past the section).
pub fn decode_mv_section(buf: &[u8], pos: usize, n_mb: usize) -> Option<(Vec<MbMode>, usize)> {
    if pos + 4 > buf.len() {
        return None;
    }
    let len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    let data_start = pos + 4;
    let data_end = data_start.checked_add(len)?;
    if data_end > buf.len() {
        return None;
    }
    let data = &buf[data_start..data_end];
    let mut modes = Vec::with_capacity(n_mb);
    let mut p = 0usize;
    while p < data.len() && modes.len() < n_mb {
        match data[p] {
            0 => {
                modes.push(MbMode::Intra);
                p += 1;
            }
            1 => {
                if p + 3 > data.len() {
                    return None;
                }
                let dx = data[p + 1] as i8 as i16;
                let dy = data[p + 2] as i8 as i16;
                modes.push(MbMode::Inter(MotionVector { dx, dy }));
                p += 3;
            }
            _ => return None,
        }
    }
    if modes.len() != n_mb || p != data.len() {
        return None;
    }
    Some((modes, data_end))
}

/// Build the 8×8 residual blocks of one plane for a P-frame.
/// Inter MBs: residual = cur − ref(x+dx, y+dy). Intra MBs: cur − 128.
/// `chroma` selects the 4:2:0 mapping (1 block/MB, half-scale MVs).
pub fn generate_residual_blocks(
    plane: &[u8],
    ref_plane: &[u8],
    w: usize,
    h: usize,
    modes: &[MbMode],
    chroma: bool,
) -> Vec<Block> {
    let bw = (w + 7) / 8;
    let bh = (h + 7) / 8;
    let (mb_w, bpm) = plane_mb_params(w, chroma);
    (0..bw * bh)
        .into_par_iter()
        .map(|idx| {
            let bx8 = idx % bw;
            let by8 = idx / bw;
            let mb_idx = (by8 / bpm) * mb_w + (bx8 / bpm);
            let mut block = [0i16; 64];
            match modes[mb_idx] {
                MbMode::Intra => {
                    for y in 0..8 {
                        let py = (by8 * 8 + y).min(h - 1);
                        for x in 0..8 {
                            let px = (bx8 * 8 + x).min(w - 1);
                            block[y * 8 + x] = plane[py * w + px] as i16 - 128;
                        }
                    }
                }
                MbMode::Inter(mv) => {
                    let (dx, dy) = if chroma {
                        (half_mv(mv.dx) as i32, half_mv(mv.dy) as i32)
                    } else {
                        (mv.dx as i32, mv.dy as i32)
                    };
                    for y in 0..8 {
                        let py = (by8 * 8 + y).min(h - 1);
                        let ry = (py as i32 + dy).clamp(0, h as i32 - 1) as usize;
                        for x in 0..8 {
                            let px = (bx8 * 8 + x).min(w - 1);
                            let rx = (px as i32 + dx).clamp(0, w as i32 - 1) as usize;
                            block[y * 8 + x] =
                                plane[py * w + px] as i16 - ref_plane[ry * w + rx] as i16;
                        }
                    }
                }
            }
            Block { data: block }
        })
        .collect()
}

/// Reconstruct one plane from decoded residual blocks (post dequant+IDCT)
/// plus motion-compensated prediction from the reference plane.
pub fn reconstruct_plane(
    resid_blocks: &[Block],
    ref_plane: &[u8],
    w: usize,
    h: usize,
    modes: &[MbMode],
    chroma: bool,
    out: &mut [u8],
) {
    let bw = (w + 7) / 8;
    let (mb_w, bpm) = plane_mb_params(w, chroma);
    for (idx, block) in resid_blocks.iter().enumerate() {
        let bx8 = idx % bw;
        let by8 = idx / bw;
        let mb_idx = (by8 / bpm) * mb_w + (bx8 / bpm);
        match modes[mb_idx] {
            MbMode::Intra => {
                for y in 0..8 {
                    let py = by8 * 8 + y;
                    if py >= h {
                        break;
                    }
                    for x in 0..8 {
                        let px = bx8 * 8 + x;
                        if px < w {
                            out[py * w + px] = (block.data[y * 8 + x] + 128).clamp(0, 255) as u8;
                        }
                    }
                }
            }
            MbMode::Inter(mv) => {
                let (dx, dy) = if chroma {
                    (half_mv(mv.dx) as i32, half_mv(mv.dy) as i32)
                } else {
                    (mv.dx as i32, mv.dy as i32)
                };
                for y in 0..8 {
                    let py = by8 * 8 + y;
                    if py >= h {
                        break;
                    }
                    let ry = (py as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    for x in 0..8 {
                        let px = bx8 * 8 + x;
                        if px < w {
                            let rx = (px as i32 + dx).clamp(0, w as i32 - 1) as usize;
                            let pred = ref_plane[ry * w + rx] as i16;
                            out[py * w + px] = (pred + block.data[y * 8 + x]).clamp(0, 255) as u8;
                        }
                    }
                }
            }
        }
    }
}
