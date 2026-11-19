//! Sample Adaptive Offset (SAO) — in-loop post-reconstruction filter.
//!
//! Applied to reconstructed Y/Cb/Cr planes right after IDCT (before YCbCr→RGB),
//! targeting the ringing and contouring/banding left by 8×8 DCT quantization.
//!
//! Two classification families, both signaled per region and fully reproducible
//! by the decoder from the stream alone:
//!
//! * **Edge Offset (EO)** — pixels are classified by local edge orientation.
//!   Four configurable directions (horizontal, vertical, 135°, 45°); each
//!   direction yields five classes (valley / side-valley / flat / side-peak /
//!   peak). One signed offset is estimated per non-flat class.
//! * **Band Offset (BO)** — the 0..=255 intensity range is split into 32 bands
//!   of 8 levels; four consecutive bands receive signed offsets. This directly
//!   repairs contouring on smooth gradients.
//!
//! The encoder evaluates every enabled candidate per region and keeps the one
//! minimizing SSE against the original plane (a small signaling penalty guards
//! against noise). The decoder only applies the signaled choice, so results are
//! bit-exact between the two sides.

use rayon::prelude::*;

use crate::block::Block;
use crate::dct;
use crate::ffi::dequantize_block;

/// SAO modes. The wire encoding of each variant must stay stable (see
/// [`RegionParams::write`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaoMode {
    /// No offset applied in this region. Wire id 0.
    Off = 0,
    /// Edge offset, direction d (0..=3). Wire id 1..=4.
    Edge0 = 1, // horizontal: neighbors (x-1,y),(x+1,y)
    Edge1 = 2, // vertical:   neighbors (x,y-1),(x,y+1)
    Edge2 = 3, // 135 deg:    neighbors (x-1,y-1),(x+1,y+1)
    Edge3 = 4, // 45 deg:     neighbors (x-1,y+1),(x+1,y-1)
    /// Band offset. Wire id 5.
    Band = 5,
}

impl SaoMode {
    #[inline]
    fn from_id(id: u8) -> Option<SaoMode> {
        Some(match id {
            0 => SaoMode::Off,
            1 => SaoMode::Edge0,
            2 => SaoMode::Edge1,
            3 => SaoMode::Edge2,
            4 => SaoMode::Edge3,
            5 => SaoMode::Band,
            _ => return None,
        })
    }
}

/// Number of edge classes signaled/estimated per EO region.
///
/// HEVC always uses five classes; this codec exposes a configurable subset so
/// the class count can be traded against signaling and estimation cost.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeClasses {
    /// Peak + valley only (2 non-flat classes).
    Two = 2,
    /// Peak, side-peak, side-valley, valley (4 non-flat classes).
    Four = 4,
    /// Full HEVC set including the explicit flat class (5 classes; flat always
    /// carries offset 0 but is counted and reported).
    Five = 5,
}

impl EdgeClasses {
    #[inline]
    pub fn from_count(n: u8) -> Option<EdgeClasses> {
        Some(match n {
            2 => EdgeClasses::Two,
            4 => EdgeClasses::Four,
            5 => EdgeClasses::Five,
            _ => return None,
        })
    }

    /// Number of offset slots reserved on the wire for one EO region.
    #[inline]
    fn slots(self) -> usize {
        match self {
            EdgeClasses::Two => 2,
            EdgeClasses::Four => 4,
            EdgeClasses::Five => 4, // flat slot is implicit zero, never written
        }
    }
}

/// Maximum absolute offset for an 8-bit sample (HEVC uses the same bound).
pub const MAX_EO_OFFSET: i32 = 7;
/// Band offsets may reach ±15 (four-bit signed range).
pub const MAX_BO_OFFSET: i32 = 15;

/// SAO configuration. Thread-safe and shared by every plane of an encode.
/// `#[repr(C)]` keeps the layout identical to `bitgrain_sao_config_t` in the C
/// public header (the FFI layer passes this value by copy).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SaoConfig {
    /// Master switch. When false no SAO chunk is written and old format bytes
    /// are produced.
    pub enabled: bool,
    /// Square region side in pixels (clamped to 8..=256, rounded to a multiple
    /// of 8 internally).
    pub region_size: u16,
    /// Edge classes per EO region (2 / 4 / 5).
    pub edge_classes: EdgeClasses,
    /// Candidate EO directions. Entries are mode ids; valid values 1..=4.
    pub edge_directions: [bool; 4],
    /// Whether band-offset candidates are considered.
    pub enable_band: bool,
}

impl Default for SaoConfig {
    fn default() -> Self {
        SaoConfig {
            enabled: true,
            region_size: 64,
            edge_classes: EdgeClasses::Five,
            edge_directions: [true; 4],
            enable_band: true,
        }
    }
}

impl SaoConfig {
    #[inline]
    pub fn region_side(&self) -> usize {
        (self.region_size as usize).clamp(8, 256) & !0x7
    }

    /// Per-region wire size for one SAO region record. Fixed at 6 bytes
    /// regardless of configuration: mode(1) + 5 payload bytes. EO uses at most
    /// 4 offset slots; BO uses band_start(1) + 4 offsets. Fixed-width records
    /// let the decoder index regions directly without length prefixes.
    #[inline]
    pub fn region_wire_size(&self) -> usize {
        6
    }
}

/// Parameters chosen for one region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionParams {
    pub mode: SaoMode,
    /// EO: indexed offsets. Layout for [`EdgeClasses`]:
    /// - Two:  [valley, peak]
    /// - Four: [valley, side-valley, side-peak, peak]
    /// - Five: same four slots; class 2 (flat) is implicit zero
    /// BO: first four entries hold the four consecutive-band offsets.
    pub offsets: [i8; 4],
    /// BO only: index (0..=31) of the first covered band.
    pub band_start: u8,
}

impl RegionParams {
    pub const fn off() -> Self {
        RegionParams { mode: SaoMode::Off, offsets: [0; 4], band_start: 0 }
    }

    fn write(&self, cfg: &SaoConfig, out: &mut Vec<u8>) {
        // All records are exactly region_wire_size() (6) bytes.
        let mut rec = [0u8; 6];
        rec[0] = self.mode as u8;
        match self.mode {
            SaoMode::Off => {}
            SaoMode::Band => {
                rec[1] = self.band_start;
                for i in 0..4 {
                    rec[2 + i] = self.offsets[i] as u8;
                }
            }
            _ => {
                let slots = cfg.edge_classes.slots();
                for i in 0..slots {
                    rec[1 + i] = self.offsets[i] as u8;
                }
            }
        }
        out.extend_from_slice(&rec);
    }

    fn read(cfg: &SaoConfig, buf: &[u8]) -> Option<RegionParams> {
        if buf.len() < cfg.region_wire_size() {
            return None;
        }
        let mode = SaoMode::from_id(buf[0])?;
        let mut p = RegionParams::off();
        p.mode = mode;
        match mode {
            SaoMode::Off => {}
            SaoMode::Band => {
                if buf[1] > 31 {
                    return None;
                }
                p.band_start = buf[1];
                for i in 0..4 {
                    p.offsets[i] = buf[2 + i] as i8;
                }
            }
            _ => {
                let slots = cfg.edge_classes.slots();
                for i in 0..slots {
                    p.offsets[i] = buf[1 + i] as i8;
                }
            }
        }
        Some(p)
    }
}

/// Chunk magic "BGs" (Bitgrain SAO).
pub const SAO_MAGIC: [u8; 3] = [b'B', b'G', b's'];
/// Chunk header: magic(3) + flags(1) + region_side(1) + n_regions per plane as
/// u32 LE × 3.
const SAO_HEADER_LEN: usize = 3 + 1 + 1 + 3 * 4;

// Flag bits.
const FLAG_ENABLED: u8 = 0x01;
const FLAG_BAND: u8 = 0x02;
const CLASS_MASK: u8 = 0x0C;
const CLASS_SHIFT: u8 = 2;
const DIR_MASK: u8 = 0xF0;
const DIR_SHIFT: u8 = 4;

/// Region grid dimensions for a plane of size `w × h`.
#[inline]
pub fn grid_dims(w: usize, h: usize, side: usize) -> (usize, usize) {
    ((w + side - 1) / side, (h + side - 1) / side)
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// Clamped neighbor fetch used by every EO direction (edge replication).
#[inline]
fn neigh(plane: &[u8], w: usize, h: usize, x: isize, y: isize) -> i32 {
    let cx = x.clamp(0, w as isize - 1) as usize;
    let cy = y.clamp(0, h as isize - 1) as usize;
    plane[cy * w + cx] as i32
}

/// Classify one pixel for EO direction `dir`.
///
/// Returns the HEVC class id:
///   0 = valley (c < both neighbors)
///   1 = side-valley (c == one neighbor and < the other)
///   2 = flat / monotonic (no offset)
///   3 = side-peak (c == one neighbor and > the other)
///   4 = peak (c > both neighbors)
#[inline]
fn edge_class(c: i32, n1: i32, n2: i32) -> u8 {
    if c < n1 && c < n2 {
        0
    } else if (c == n1 && c < n2) || (c == n2 && c < n1) {
        1
    } else if c > n1 && c > n2 {
        4
    } else if (c == n1 && c > n2) || (c == n2 && c > n1) {
        3
    } else {
        2
    }
}

#[inline]
fn neighbors(plane: &[u8], w: usize, h: usize, x: usize, y: usize, dir: u8) -> (i32, i32) {
    let (ix, iy) = (x as isize, y as isize);
    match dir {
        0 => (
            neigh(plane, w, h, ix - 1, iy),
            neigh(plane, w, h, ix + 1, iy),
        ),
        1 => (
            neigh(plane, w, h, ix, iy - 1),
            neigh(plane, w, h, ix, iy + 1),
        ),
        2 => (
            neigh(plane, w, h, ix - 1, iy - 1),
            neigh(plane, w, h, ix + 1, iy + 1),
        ),
        _ => (
            neigh(plane, w, h, ix - 1, iy + 1),
            neigh(plane, w, h, ix + 1, iy - 1),
        ),
    }
}

/// Map a raw 5-way class id to the offset slot under the configured class set.
/// Returns None for flat / disabled classes.
#[inline]
fn class_to_slot(class: u8, classes: EdgeClasses) -> Option<usize> {
    match (class, classes) {
        (0, _) => Some(0),                 // valley
        (1, EdgeClasses::Four | EdgeClasses::Five) => Some(1), // side-valley
        (3, EdgeClasses::Four | EdgeClasses::Five) => Some(2), // side-peak
        (4, _) => {
            // Two-class set: peak is slot 1; otherwise slot 3.
            Some(if classes == EdgeClasses::Two { 1 } else { 3 })
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Estimation (encoder side)
// ---------------------------------------------------------------------------

#[inline]
fn clamp_offset(v: i64, max: i32) -> i8 {
    v.clamp(-max as i64, max as i64) as i8
}

/// Round `sum / n` to the nearest integer with symmetric half-away-from-zero
/// semantics, so positive and negative mean errors share the same magnitude.
#[inline]
fn rounded_mean(sum: i64, n: i64) -> i64 {
    if sum >= 0 {
        (sum + (n >> 1)) / n
    } else {
        -((-sum + (n >> 1)) / n)
    }
}

/// Estimate the best-fitting parameters for a single region.
///
/// `recon` is the post-IDCT plane the decoder can reproduce; `orig` is the
/// corresponding pre-encode plane. All math is integer and deterministic.
fn estimate_region(
    orig: &[u8],
    recon: &[u8],
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    cfg: &SaoConfig,
) -> RegionParams {
    // Fixed-size candidate storage avoids per-region heap allocation; absent
    // directions simply stay zeroed and are skipped during offset evaluation.
    let mut edge_count = [[0u64; 5]; 4];
    let mut edge_sum = [[0i64; 5]; 4];
    let mut band_count = [0u64; 32];
    let mut band_sum = [0i64; 32];

    let mut sse_base: u64 = 0;

    // Fetch the 8 neighbors directly via row pointers. Interior pixels (the
    // common case) skip all boundary clamping; image-edge pixels fall back to
    // the clamped helper.
    for y in y0..y1 {
        let row = y * w;
        let y_up = if y == 0 { 0 } else { y - 1 };
        let y_dn = if y + 1 >= h { h - 1 } else { y + 1 };
        let up = y_up * w;
        let dn = y_dn * w;
        for x in x0..x1 {
            let idx = row + x;
            let c = recon[idx] as i32;
            let d = (orig[idx] as i32 - c) as i64;
            sse_base += (d * d) as u64;

            let x_lf = if x == 0 { 0 } else { x - 1 };
            let x_rt = if x + 1 >= w { w - 1 } else { x + 1 };
            // Direction 0 (horizontal), 1 (vertical), 2 (135°), 3 (45°).
            let h_n1 = recon[row + x_lf] as i32;
            let h_n2 = recon[row + x_rt] as i32;
            let v_n1 = recon[up + x] as i32;
            let v_n2 = recon[dn + x] as i32;
            let d2_n1 = recon[up + x_lf] as i32;
            let d2_n2 = recon[dn + x_rt] as i32;
            let d3_n1 = recon[dn + x_lf] as i32;
            let d3_n2 = recon[up + x_rt] as i32;

            macro_rules! acc_dir {
                ($k:expr, $n1:expr, $n2:expr) => {
                    if cfg.edge_directions[$k] {
                        let class = edge_class(c, $n1, $n2) as usize;
                        edge_count[$k][class] += 1;
                        edge_sum[$k][class] += d;
                    }
                };
            }
            acc_dir!(0, h_n1, h_n2);
            acc_dir!(1, v_n1, v_n2);
            acc_dir!(2, d2_n1, d2_n2);
            acc_dir!(3, d3_n1, d3_n2);

            if cfg.enable_band {
                let band = (c >> 3) as usize;
                band_count[band] += 1;
                band_sum[band] += d;
            }
        }
    }

    let mut best = RegionParams::off();
    // Off mode never improves SSE; its cost is the baseline itself.
    let mut best_sse = sse_base;

    // ---- EO candidates ----
    for k in 0..4usize {
        if !cfg.edge_directions[k] {
            continue;
        }
        let mut params = RegionParams::off();
        params.mode = SaoMode::from_id((k + 1) as u8).unwrap();

        // Classes that carry an offset under the configured class set.
        let raw_classes: &[u8] = match cfg.edge_classes {
            EdgeClasses::Two => &[0u8, 4],
            EdgeClasses::Four | EdgeClasses::Five => &[0, 1, 3, 4],
        };

        let mut sse = sse_base as i64;
        for &class in raw_classes {
            let ci = class as usize;
            let n = edge_count[k][ci] as i64;
            if n == 0 {
                continue;
            }
            // Rounded mean error = best constant offset in the SSE sense.
            let off = clamp_offset(rounded_mean(edge_sum[k][ci], n), MAX_EO_OFFSET);
            if off == 0 {
                continue;
            }
            let slot = class_to_slot(class, cfg.edge_classes).unwrap();
            params.offsets[slot] = off;
            // ΔSSE = sum((e-off)^2 - e^2) = n*off^2 - 2*off*sum_err
            sse += n * (off as i64) * (off as i64)
                - 2 * (off as i64) * edge_sum[k][ci];
        }

        if sse < best_sse as i64 {
            best_sse = sse as u64;
            best = params;
        }
    }

    // ---- BO candidate: choose the best 4-consecutive-band window ----
    if cfg.enable_band {
        // Precompute per-band best offset and its SSE delta.
        let mut band_off = [0i8; 32];
        let mut band_delta = [0i64; 32];
        for b in 0..32 {
            let n = band_count[b] as i64;
            if n == 0 {
                continue; // empty band: no samples, offset stays zero
            }
            let off = clamp_offset(rounded_mean(band_sum[b], n), MAX_BO_OFFSET);
            band_off[b] = off;
            band_delta[b] =
                n * (off as i64) * (off as i64) - 2 * (off as i64) * band_sum[b];
        }
        let mut best_start = 0usize;
        let mut best_win_delta = 0i64;
        for start in 0..29 {
            let delta: i64 = (start..start + 4).map(|b| band_delta[b]).sum();
            if delta < best_win_delta {
                best_win_delta = delta;
                best_start = start;
            }
        }
        let sse = sse_base as i64 + best_win_delta;
        if best_win_delta < 0 && sse < best_sse as i64 {
            let mut params = RegionParams::off();
            params.mode = SaoMode::Band;
            params.band_start = best_start as u8;
            for (i, b) in (best_start..best_start + 4).enumerate() {
                params.offsets[i] = band_off[b];
            }
            best_sse = sse as u64;
            best = params;
        }
    }

    // Signaling guard: require a small absolute SSE improvement, otherwise
    // leave the region unfiltered (also keeps flat/noise regions untouched and
    // avoids spending a record on zero/near-zero offsets).
    let threshold: u64 = 16;
    if sse_base.saturating_sub(best_sse) < threshold {
        return RegionParams::off();
    }
    best
}

// ---------------------------------------------------------------------------
// Application (shared by encoder and decoder — the decoder's only path)
// ---------------------------------------------------------------------------

/// Apply one region's parameters. Reads from `src`, writes corrected samples
/// to `dst` (both full-plane slices, same geometry). Pure integer arithmetic,
/// identical on encoder and decoder.
fn apply_region(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    cfg: &SaoConfig,
    params: &RegionParams,
) {
    match params.mode {
        SaoMode::Off => {}
        SaoMode::Band => {
            let start = params.band_start as usize;
            for y in y0..y1 {
                let row = y * w;
                for x in x0..x1 {
                    let idx = row + x;
                    let c = src[idx] as i32;
                    let band = (c >> 3) as usize;
                    if band >= start && band < start + 4 {
                        let off = params.offsets[band - start] as i32;
                        dst[idx] = (c + off).clamp(0, 255) as u8;
                    }
                }
            }
        }
        mode => {
            let dir = (mode as u8) - 1;
            for y in y0..y1 {
                let row = y * w;
                for x in x0..x1 {
                    let idx = row + x;
                    let c = src[idx] as i32;
                    let (n1, n2) = neighbors(src, w, h, x, y, dir);
                    let class = edge_class(c, n1, n2);
                    if let Some(slot) = class_to_slot(class, cfg.edge_classes) {
                        let off = params.offsets[slot] as i32;
                        if off != 0 {
                            dst[idx] = (c + off).clamp(0, 255) as u8;
                        }
                    }
                }
            }
        }
    }
}

/// Estimate SAO parameters for a whole plane (encoder side).
///
/// Regions are independent and processed in parallel; output region order is
/// row-major, matching [`apply_plane`] and the wire layout.
pub fn estimate_plane(
    orig: &[u8],
    recon: &[u8],
    w: usize,
    h: usize,
    cfg: &SaoConfig,
) -> Vec<RegionParams> {
    let side = cfg.region_side();
    let (gw, gh) = grid_dims(w, h, side);
    let n = gw * gh;
    let mut params = vec![RegionParams::off(); n];

    if n == 1 || w * h < 262_144 {
        for ry in 0..gh {
            for rx in 0..gw {
                let (x0, y0, x1, y1) = region_bounds(w, h, side, rx, ry);
                params[ry * gw + rx] =
                    estimate_region(orig, recon, w, h, x0, y0, x1, y1, cfg);
            }
        }
    } else {
        params
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, slot)| {
                let rx = i % gw;
                let ry = i / gw;
                let (x0, y0, x1, y1) = region_bounds(w, h, side, rx, ry);
                *slot = estimate_region(orig, recon, w, h, x0, y0, x1, y1, cfg);
            });
    }
    params
}

/// Apply signaled SAO parameters to a plane in place (both sides).
///
/// Internally double-buffered: every classification reads the unmodified
/// reconstruction, so serial and parallel execution produce identical bytes.
pub fn apply_plane(plane: &mut [u8], w: usize, h: usize, cfg: &SaoConfig, params: &[RegionParams]) {
    let side = cfg.region_side();
    let (gw, gh) = grid_dims(w, h, side);
    debug_assert_eq!(params.len(), gw * gh);

    let src = plane.to_vec();
    // Start the destination from the source so Off / flat samples pass through.
    plane.copy_from_slice(&src);
    let dst = plane;

    if params.len() == 1 || w * h < 262_144 {
        for ry in 0..gh {
            for rx in 0..gw {
                let (x0, y0, x1, y1) = region_bounds(w, h, side, rx, ry);
                apply_region(&src, dst, w, h, x0, y0, x1, y1, cfg, &params[ry * gw + rx]);
            }
        }
    } else {
        // Process region rows into independent band buffers, then splice back.
        // Every classification reads the shared immutable source with full-plane
        // clamped coordinates, so serial and parallel outputs match exactly.
        let bands: Vec<Vec<u8>> = (0..gh)
            .into_par_iter()
            .map(|ry| {
                let y0 = (ry * side).min(h);
                let y1 = ((ry + 1) * side).min(h);
                // Start from the source so Off / flat regions pass through.
                let mut band = src[y0 * w..y1 * w].to_vec();
                for rx in 0..gw {
                    let x0 = rx * side;
                    let x1 = ((rx + 1) * side).min(w);
                    apply_band(&src, &mut band, w, h, y0, y1, x0, x1, cfg, &params[ry * gw + rx]);
                }
                band
            })
            .collect();
        let mut off = 0usize;
        for band in bands {
            dst[off..off + band.len()].copy_from_slice(&band);
            off += band.len();
        }
    }
}

/// Apply one region into a band buffer covering rows `y0..y1` of the plane.
/// Source indexing stays full-plane; destination indexing is band-relative.
fn apply_band(
    src: &[u8],
    band: &mut [u8],
    w: usize,
    h: usize,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
    cfg: &SaoConfig,
    params: &RegionParams,
) {
    match params.mode {
        SaoMode::Off => {}
        SaoMode::Band => {
            let start = params.band_start as usize;
            for (y, brow) in (y0..y1).enumerate() {
                let srow = brow * w;
                let drow = y * w;
                for x in x0..x1 {
                    let c = src[srow + x] as i32;
                    let band_id = (c >> 3) as usize;
                    if band_id >= start && band_id < start + 4 {
                        let off = params.offsets[band_id - start] as i32;
                        band[drow + x] = (c + off).clamp(0, 255) as u8;
                    }
                }
            }
        }
        mode => {
            let dir = (mode as u8) - 1;
            for (y, brow) in (y0..y1).enumerate() {
                let srow = brow * w;
                let drow = y * w;
                for x in x0..x1 {
                    let c = src[srow + x] as i32;
                    let (n1, n2) = neighbors(src, w, h, x, brow, dir);
                    let class = edge_class(c, n1, n2);
                    if let Some(slot) = class_to_slot(class, cfg.edge_classes) {
                        let off = params.offsets[slot] as i32;
                        if off != 0 {
                            band[drow + x] = (c + off).clamp(0, 255) as u8;
                        }
                    }
                }
            }
        }
    }
}

#[inline]
fn region_bounds(w: usize, h: usize, side: usize, rx: usize, ry: usize) -> (usize, usize, usize, usize) {
    let x0 = rx * side;
    let y0 = ry * side;
    let x1 = ((rx + 1) * side).min(w);
    let y1 = ((ry + 1) * side).min(h);
    (x0, y0, x1, y1)
}

// ---------------------------------------------------------------------------
// Encoder-side reconstruction
// ---------------------------------------------------------------------------

/// Rebuild the post-IDCT plane the decoder will produce, directly from the
/// quantized coefficient blocks (no bitstream round trip). Deterministic and
/// identical to `decode_plane_huffman` output, including edge replication.
///
/// `blocks` are the same blocks fed to the entropy coder; they are cloned
/// before dequantization so the encoded (still-quantized) coefficients are not
/// mutated — the decoder dequantizes its own copy.
pub fn reconstruct_plane(
    blocks: &[Block],
    quant: &[i16; 64],
    w: usize,
    h: usize,
) -> Vec<u8> {
    let bw = (w + 7) / 8;
    let n = blocks.len();

    let mut plane = vec![0u8; w * h];
    let mut reconstruct_one = |idx: usize, block: &Block| {
        // Per-block clone keeps the entropy-coded coefficients untouched.
        let mut work = *block;
        unsafe { dequantize_block(work.data.as_mut_ptr(), quant.as_ptr()) };
        dct::idct(&mut work);
        let by = (idx / bw) * 8;
        let bx = (idx % bw) * 8;
        for y in 0..8 {
            for x in 0..8 {
                let py = by + y;
                let px = bx + x;
                if py < h && px < w {
                    plane[py * w + px] =
                        (work.data[y * 8 + x] + 128).clamp(0, 255) as u8;
                }
            }
        }
    };

    if n >= 384 && w * h >= 262_144 {
        // Reconstruct each block into its own 8×8 pixel tile, then splice the
        // tiles into the plane. Each iteration only reads the shared blocks.
        let tiles: Vec<(usize, [u8; 64])> = (0..n)
            .into_par_iter()
            .map(|idx| {
                let mut work = blocks[idx];
                unsafe { dequantize_block(work.data.as_mut_ptr(), quant.as_ptr()) };
                dct::idct(&mut work);
                let mut tile = [0u8; 64];
                for y in 0..8 {
                    for x in 0..8 {
                        tile[y * 8 + x] =
                            (work.data[y * 8 + x] + 128).clamp(0, 255) as u8;
                    }
                }
                (idx, tile)
            })
            .collect();
        for (idx, tile) in tiles {
            let by = (idx / bw) * 8;
            let bx = (idx % bw) * 8;
            for y in 0..8 {
                for x in 0..8 {
                    let py = by + y;
                    let px = bx + x;
                    if py < h && px < w {
                        plane[py * w + px] = tile[y * 8 + x];
                    }
                }
            }
        }
    } else {
        for (idx, block) in blocks.iter().enumerate() {
            reconstruct_one(idx, block);
        }
    }
    plane
}

// ---------------------------------------------------------------------------
// Chunk serialization
// ---------------------------------------------------------------------------

fn encode_flags(cfg: &SaoConfig) -> u8 {
    let mut flags = 0u8;
    if cfg.enabled {
        flags |= FLAG_ENABLED;
    }
    if cfg.enable_band {
        flags |= FLAG_BAND;
    }
    let class_bits = match cfg.edge_classes {
        EdgeClasses::Two => 0u8,
        EdgeClasses::Four => 1,
        EdgeClasses::Five => 2,
    };
    flags |= (class_bits << CLASS_SHIFT) & CLASS_MASK;
    let mut dirs = 0u8;
    for (i, &on) in cfg.edge_directions.iter().enumerate() {
        if on {
            dirs |= 1 << i;
        }
    }
    flags |= (dirs << DIR_SHIFT) & DIR_MASK;
    flags
}

fn decode_flags(flags: u8) -> Option<SaoConfig> {
    if flags & FLAG_ENABLED == 0 {
        return None;
    }
    let class_bits = (flags & CLASS_MASK) >> CLASS_SHIFT;
    let edge_classes = match class_bits {
        0 => EdgeClasses::Two,
        1 => EdgeClasses::Four,
        2 => EdgeClasses::Five,
        _ => return None,
    };
    let dir_bits = (flags & DIR_MASK) >> DIR_SHIFT;
    let mut edge_directions = [false; 4];
    let mut any_dir = false;
    for i in 0..4 {
        edge_directions[i] = dir_bits & (1 << i) != 0;
        any_dir |= edge_directions[i];
    }
    if !any_dir && flags & FLAG_BAND == 0 {
        return None;
    }
    Some(SaoConfig {
        enabled: true,
        region_size: 64, // filled from dedicated byte by caller
        edge_classes,
        edge_directions,
        enable_band: flags & FLAG_BAND != 0,
    })
}

/// Serialize the SAO chunk for planes `(y, cb, cr)`. Returns an empty vec when
/// SAO is disabled. Region counts are derived by the decoder from plane sizes
/// and the region side, but are stored explicitly for fast validation.
pub fn write_chunk(
    cfg: &SaoConfig,
    planes: [&[RegionParams]; 3],
    region_counts: [usize; 3],
) -> Vec<u8> {
    if !cfg.enabled {
        return Vec::new();
    }
    let side = cfg.region_side();
    let rec_size = cfg.region_wire_size();
    let payload: usize = region_counts.iter().sum::<usize>() * rec_size;
    let mut out = Vec::with_capacity(SAO_HEADER_LEN + 4 + payload);
    out.extend_from_slice(&SAO_MAGIC);
    out.push(encode_flags(cfg));
    out.push(side as u8);
    for &n in &region_counts {
        out.extend_from_slice(&(n as u32).to_le_bytes());
    }
    out.extend_from_slice(&(payload as u32).to_le_bytes());
    for plane_params in planes {
        for p in plane_params.iter() {
            p.write(cfg, &mut out);
        }
    }
    out
}

/// Parsed SAO chunk.
pub struct SaoChunk {
    pub cfg: SaoConfig,
    pub regions: [Vec<RegionParams>; 3],
}

/// Parse an SAO chunk starting at `pos`. Returns the chunk and the byte
/// position immediately after it, or None on any malformed data.
pub fn read_chunk(buf: &[u8], pos: usize) -> Option<(SaoChunk, usize)> {
    if pos + SAO_HEADER_LEN + 4 > buf.len() {
        return None;
    }
    if buf[pos..pos + 3] != SAO_MAGIC {
        return None;
    }
    let mut cfg = decode_flags(buf[pos + 3])?;
    let side = buf[pos + 4] as usize;
    if !(8..=256).contains(&side) || side & 0x7 != 0 {
        return None;
    }
    cfg.region_size = side as u16;

    let mut counts = [0usize; 3];
    for i in 0..3 {
        let off = pos + 5 + i * 4;
        counts[i] = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?) as usize;
    }
    let payload_len =
        u32::from_le_bytes(buf[pos + 17..pos + 21].try_into().ok()?) as usize;
    let rec_size = cfg.region_wire_size();
    let expected: usize = counts.iter().sum::<usize>() * rec_size;
    if payload_len != expected {
        return None;
    }
    let data_start = pos + SAO_HEADER_LEN + 4;
    if data_start + payload_len > buf.len() {
        return None;
    }

    let mut regions: [Vec<RegionParams>; 3] =
        [Vec::new(), Vec::new(), Vec::new()];
    let mut cursor = data_start;
    for i in 0..3 {
        let mut v = Vec::with_capacity(counts[i]);
        for _ in 0..counts[i] {
            let p = RegionParams::read(&cfg, &buf[cursor..cursor + rec_size])?;
            v.push(p);
            cursor += rec_size;
        }
        regions[i] = v;
    }
    Some((SaoChunk { cfg, regions }, cursor))
}

/// True when the bytes at `pos` begin an SAO chunk.
#[inline]
pub fn is_chunk_at(buf: &[u8], pos: usize) -> bool {
    pos + 3 <= buf.len() && buf[pos..pos + 3] == SAO_MAGIC
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> SaoConfig {
        SaoConfig::default()
    }

    #[test]
    fn edge_classification_peak_valley() {
        // valley: center lower than both
        assert_eq!(edge_class(5, 10, 10), 0);
        // peak
        assert_eq!(edge_class(20, 10, 10), 4);
        // side-valley / side-peak
        assert_eq!(edge_class(10, 10, 20), 1);
        assert_eq!(edge_class(20, 20, 10), 3);
        // flat / monotonic
        assert_eq!(edge_class(10, 10, 10), 2);
        assert_eq!(edge_class(15, 10, 20), 2);
    }

    #[test]
    fn band_offset_repairs_lifted_pixels() {
        // Reconstruction uniformly 3 levels above the original within one band.
        // Symmetric rounded mean-error offset moves every sample back exactly.
        let w = 16usize;
        let h = 16usize;
        let orig = vec![100u8; w * h];
        let mut recon = vec![103u8; w * h]; // band 12 (104..111 is band 13)
        let cfg = make_config();
        let params = estimate_plane(&orig, &recon, w, h, &cfg);
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].mode, SaoMode::Band);
        assert!(params[0].offsets.contains(&-3));
        apply_plane(&mut recon, w, h, &cfg, &params);
        assert!(recon.iter().all(|&v| v == 100));
    }

    #[test]
    fn edge_offset_repairs_valley_ringing() {
        let w = 32usize;
        let h = 8usize;
        let mut orig = vec![128u8; w * h];
        // A vertical bright edge in the original.
        for y in 0..h {
            for x in 0..w {
                orig[y * w + x] = if x >= 16 { 200 } else { 60 };
            }
        }
        // Reconstruction dips one pixel on the dark side of the edge (ringing).
        let mut recon = orig.clone();
        for y in 0..h {
            recon[y * w + 15] = 40;
        }
        let cfg = make_config();
        let params = estimate_plane(&orig, &recon, w, h, &cfg);
        apply_plane(&mut recon, w, h, &cfg, &params);
        let sse_before: u64 = orig
            .iter()
            .zip(recon.iter())
            .map(|(&a, &b)| (a as i64 - b as i64).pow(2) as u64)
            .sum();
        // Ringing pixels must be fully or partially corrected.
        assert!(sse_before < (20u64 * h as u64 * 20));
    }

    #[test]
    fn off_region_on_identical_planes() {
        let w = 32usize;
        let h = 32usize;
        let plane: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
        let cfg = make_config();
        let params = estimate_plane(&plane, &plane, w, h, &cfg);
        assert!(params.iter().all(|p| p.mode == SaoMode::Off));
    }

    #[test]
    fn chunk_roundtrip_preserves_params() {
        let cfg = make_config();
        let p = RegionParams {
            mode: SaoMode::Edge2,
            offsets: [-3, 1, -1, 5],
            band_start: 0,
        };
        let counts = [4usize, 1, 1];
        let chunk = write_chunk(
            &cfg,
            [
                &vec![p; counts[0]],
                &vec![RegionParams::off(); counts[1]],
                &vec![RegionParams {
                    mode: SaoMode::Band,
                    offsets: [1, -2, 3, -4],
                    band_start: 7,
                }; counts[2]],
            ],
            counts,
        );
        let (parsed, end) = read_chunk(&chunk, 0).expect("chunk must parse");
        assert_eq!(end, chunk.len());
        assert_eq!(parsed.cfg.edge_classes, cfg.edge_classes);
        assert_eq!(parsed.cfg.region_side(), cfg.region_side());
        assert_eq!(parsed.regions[0].len(), 4);
        assert_eq!(parsed.regions[0][0], p);
        assert_eq!(parsed.regions[1][0].mode, SaoMode::Off);
        assert_eq!(parsed.regions[2][0].band_start, 7);
        assert_eq!(parsed.regions[2][0].offsets, [1, -2, 3, -4]);
    }

    #[test]
    fn malformed_chunk_rejected() {
        assert!(read_chunk(&[0u8; 4], 0).is_none());
        let cfg = make_config();
        let mut chunk = write_chunk(&cfg, [&vec![RegionParams::off()], &[], &[]], [1, 0, 0]);
        // Corrupt payload length.
        let len_pos = SAO_HEADER_LEN;
        chunk[len_pos] = 0xFF;
        assert!(read_chunk(&chunk, 0).is_none());
    }

    #[test]
    fn deterministic_encode_decode_apply() {
        // Reproducibility property: identical input plus identical signaled
        // parameters must produce identical output on every invocation.
        let w = 48usize;
        let h = 48usize;
        let orig: Vec<u8> = (0..w * h).map(|i| ((i * 7) % 256) as u8).collect();
        let recon0: Vec<u8> = orig.iter().map(|&v| v.saturating_sub(3)).collect();
        let cfg = make_config();
        let params = estimate_plane(&orig, &recon0, w, h, &cfg);

        let mut run_a = recon0.clone();
        apply_plane(&mut run_a, w, h, &cfg, &params);
        let mut run_b = recon0.clone();
        apply_plane(&mut run_b, w, h, &cfg, &params);
        assert_eq!(run_a, run_b);

        // Filtering must not increase SSE against the original.
        let sse = |a: &[u8], b: &[u8]| -> u64 {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| (x as i64 - y as i64).pow(2) as u64)
                .sum()
        };
        assert!(sse(&orig, &run_a) <= sse(&orig, &recon0));
    }
}
