//! Bidirectional inter-frame prediction (B-frames) layered on the intra codec.
//!
//! A sequence container ("BGS1") holds I-frames (payloads are byte-identical to
//! standalone [`encoder::encode_grayscale`] output) and B-frames. For each 8x8
//! block of a B-frame the encoder searches a configurable window in the
//! previous/next *decoded* reference frames, solves for the pair of weights
//! that minimizes the block error, quantizes them to `weight_bits` precision
//! and streams them; the residual (current - weighted prediction) is coded
//! with the unmodified intra encoder. The decoder rebuilds the prediction from
//! the streamed weights and adds the decoded residual.
//!
//! Reference search range and weight precision are configurable via
//! [`BiPredConfig`]. Missing reference frames and out-of-range weights are
//! hard errors ([`BiPredError`]).

use crate::decoder;
use crate::encoder;
use std::fmt;

/// Sequence container magic.
pub const BIPRED_MAGIC: &[u8; 4] = b"BGS1";
const HEADER_SIZE: usize = 20; // magic(4) w(4) h(4) nframes(4) wbits(1) range(1) quality(1) reserved(1)
const FRAME_INTRA: u8 = 0;
const FRAME_BIPRED: u8 = 1;
const BLOCK: usize = 8;
/// Worst-case RLE bytes per 8x8 block: DC(2) + 63 AC pairs(3) + EOB(3).
const RLE_BLOCK_CAP: usize = 2 + 63 * 3 + 3;
/// Bytes per B-frame block record: dx(i8) dy(i8) w_prev(u16 LE) w_next(u16 LE).
const BLOCK_REC_SIZE: usize = 6;
const MAX_SEARCH_RANGE: i32 = 32;
const MAX_WEIGHT_BITS: u8 = 8;

/// Per-frame coding decision inside a sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameType {
    /// Coded standalone with the unmodified intra encoder.
    Intra,
    /// Predicted from the previous and next intra frames plus residual.
    BiPred,
}

/// Tunables for bidirectional prediction.
#[derive(Clone, Copy, Debug)]
pub struct BiPredConfig {
    /// Half-size of the reference-block search window, in pixels (0..=32).
    pub search_range: i32,
    /// Weight precision in bits (1..=8); weights are integers in `0..=1<<weight_bits`.
    pub weight_bits: u8,
    /// Intra quality (1..=100) for I-frames and B-frame residuals.
    pub quality: u8,
    /// Cap on coordinate-descent passes of the weight search (0 = LS estimate only).
    pub max_weight_iters: usize,
}

impl Default for BiPredConfig {
    fn default() -> Self {
        Self { search_range: 4, weight_bits: 4, quality: 75, max_weight_iters: 16 }
    }
}

impl BiPredConfig {
    pub fn validate(&self) -> Result<(), BiPredError> {
        if self.weight_bits == 0 || self.weight_bits > MAX_WEIGHT_BITS {
            return Err(BiPredError::InvalidConfig(format!(
                "weight_bits must be 1..={MAX_WEIGHT_BITS}, got {}", self.weight_bits
            )));
        }
        if !(0..=MAX_SEARCH_RANGE).contains(&self.search_range) {
            return Err(BiPredError::InvalidConfig(format!(
                "search_range must be 0..={MAX_SEARCH_RANGE}, got {}", self.search_range
            )));
        }
        if self.quality == 0 || self.quality > 100 {
            return Err(BiPredError::InvalidConfig(format!(
                "quality must be 1..=100, got {}", self.quality
            )));
        }
        Ok(())
    }

    #[inline]
    fn scale(&self) -> i32 {
        1i32 << self.weight_bits
    }
}

/// Errors produced by the bipred sequence layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BiPredError {
    InvalidConfig(String),
    DimensionMismatch { expected: usize, got: usize },
    /// B-frame has no usable previous/next reference frame.
    MissingReference { frame: usize },
    /// A streamed weight exceeds the quantized range `0..=scale`.
    WeightOutOfRange { frame: usize, block: usize, w_prev: u32, w_next: u32, scale: u32 },
    /// A streamed motion offset exceeds the configured search range.
    MotionOutOfRange { frame: usize, block: usize, dx: i32, dy: i32, range: i32 },
    BadMagic,
    Truncated,
    IntraCodecFailed { frame: usize },
}

impl fmt::Display for BiPredError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BiPredError::InvalidConfig(m) => write!(f, "invalid bipred config: {m}"),
            BiPredError::DimensionMismatch { expected, got } => {
                write!(f, "frame size mismatch: expected {expected} bytes, got {got}")
            }
            BiPredError::MissingReference { frame } => write!(
                f,
                "frame {frame}: missing reference frame (B-frame needs intra frames on both sides)"
            ),
            BiPredError::WeightOutOfRange { frame, block, w_prev, w_next, scale } => write!(
                f,
                "frame {frame} block {block}: weights ({w_prev}, {w_next}) out of range 0..={scale}"
            ),
            BiPredError::MotionOutOfRange { frame, block, dx, dy, range } => write!(
                f,
                "frame {frame} block {block}: motion ({dx}, {dy}) outside search range ±{range}"
            ),
            BiPredError::BadMagic => write!(f, "not a BGS1 bipred stream"),
            BiPredError::Truncated => write!(f, "truncated or malformed bipred stream"),
            BiPredError::IntraCodecFailed { frame } => write!(f, "frame {frame}: intra codec failed"),
        }
    }
}

impl std::error::Error for BiPredError {}

/// Counters collected while encoding a sequence.
#[derive(Debug, Default, Clone)]
pub struct EncodeStats {
    pub i_frames: usize,
    pub b_frames: usize,
    /// Total coordinate-descent passes performed by the weight search,
    /// summed over every candidate offset of every B-frame block.
    pub weight_iterations: u64,
    pub bytes: usize,
}

/// Result of [`decode_sequence`].
#[derive(Debug)]
pub struct DecodedSequence {
    pub width: usize,
    pub height: usize,
    pub frames: Vec<Vec<u8>>,
}

// ---------------------------------------------------------------------------
// Intra helpers (unmodified codec, byte-identical output)
// ---------------------------------------------------------------------------

fn encode_intra(frame: &[u8], w: usize, h: usize, quality: u8) -> Vec<u8> {
    let nblocks = w.div_ceil(BLOCK) * h.div_ceil(BLOCK);
    let mut buf = vec![0u8; encoder::BG_HEADER_SIZE + nblocks * RLE_BLOCK_CAP + 16];
    let mut pos: i32 = 0;
    encoder::encode_grayscale(frame, w, h, quality, &mut buf, &mut pos);
    buf.truncate(pos as usize);
    buf
}

fn decode_intra(data: &[u8], w: usize, h: usize) -> Option<Vec<u8>> {
    let mut out = vec![0u8; w * h];
    let (mut ow, mut oh) = (0u32, 0u32);
    if !decoder::decode_grayscale(data, &mut out, &mut ow, &mut oh) {
        return None;
    }
    if ow as usize != w || oh as usize != h {
        return None;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Prediction core
// ---------------------------------------------------------------------------

/// Weighted prediction of one pixel, matching the decoder exactly.
#[inline]
fn pred_px(a: i32, b: i32, wp: i32, wn: i32, scale: i32) -> i32 {
    (wp * a + wn * b + scale / 2) / scale
}

/// Current block pixels with edge clamping (mirrors `Blockizer`).
fn cur_block(frame: &[u8], w: usize, h: usize, bx: usize, by: usize) -> [i32; 64] {
    let mut out = [0i32; 64];
    for y in 0..BLOCK {
        let iy = (by + y).min(h - 1);
        for x in 0..BLOCK {
            let ix = (bx + x).min(w - 1);
            out[y * BLOCK + x] = frame[iy * w + ix] as i32;
        }
    }
    out
}

/// Reference 8x8 block at pixel origin (x0, y0); caller guarantees in-bounds.
fn ref_block(plane: &[u8], w: usize, x0: usize, y0: usize) -> [i32; 64] {
    let mut out = [0i32; 64];
    for y in 0..BLOCK {
        let base = (y0 + y) * w + x0;
        for x in 0..BLOCK {
            out[y * BLOCK + x] = plane[base + x] as i32;
        }
    }
    out
}

#[inline]
fn block_sse(cur: &[i32; 64], a: &[i32; 64], b: &[i32; 64], wp: i32, wn: i32, scale: i32) -> i64 {
    let mut s = 0i64;
    for i in 0..64 {
        let d = cur[i] - pred_px(a[i], b[i], wp, wn, scale);
        s += d as i64 * d as i64;
    }
    s
}

/// Minimize `sum(cur - (wp*a + wn*b)/scale)^2` over integer weights in
/// `0..=scale`: least-squares estimate, then coordinate descent on the
/// quantized grid. Each descent pass counts as one weight-search iteration.
fn optimize_weights(
    cur: &[i32; 64],
    a: &[i32; 64],
    b: &[i32; 64],
    scale: i32,
    max_iters: usize,
    iters: &mut u64,
) -> (i32, i32, i64) {
    let (mut aa, mut bb, mut ab, mut ca, mut cb) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..64 {
        let (av, bv, cv) = (a[i] as f64, b[i] as f64, cur[i] as f64);
        aa += av * av;
        bb += bv * bv;
        ab += av * bv;
        ca += cv * av;
        cb += cv * bv;
    }
    let det = aa * bb - ab * ab;
    let (mut wp, mut wn) = if det > 1e-6 {
        let s = scale as f64;
        (
            (s * (ca * bb - cb * ab) / det).round() as i32,
            (s * (cb * aa - ca * ab) / det).round() as i32,
        )
    } else {
        (scale / 2, scale / 2)
    };
    wp = wp.clamp(0, scale);
    wn = wn.clamp(0, scale);

    let mut best = block_sse(cur, a, b, wp, wn, scale);
    for _ in 0..max_iters {
        *iters += 1;
        let mut improved = false;
        for (dp, dn) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
            let (np, nn) = (wp + dp, wn + dn);
            if np < 0 || np > scale || nn < 0 || nn > scale {
                continue;
            }
            let e = block_sse(cur, a, b, np, nn, scale);
            if e < best {
                best = e;
                wp = np;
                wn = nn;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    (wp, wn, best)
}

struct BlockRec {
    dx: i32,
    dy: i32,
    wp: i32,
    wn: i32,
}

/// Encode one B-frame: per-block motion + weights, then the residual plane
/// (clamped to [-128,127] and shifted to u8) with the unmodified intra encoder.
fn encode_bframe(
    cur: &[u8],
    prev: &[u8],
    next: &[u8],
    w: usize,
    h: usize,
    cfg: &BiPredConfig,
    iters: &mut u64,
) -> Result<Vec<u8>, BiPredError> {
    if w < BLOCK || h < BLOCK {
        return Err(BiPredError::InvalidConfig(
            "B-frames need width and height >= 8".into(),
        ));
    }
    let scale = cfg.scale();
    let bw = w.div_ceil(BLOCK);
    let bh = h.div_ceil(BLOCK);
    let nblocks = bw * bh;
    let mut recs: Vec<BlockRec> = Vec::with_capacity(nblocks);
    let mut pred = vec![0i32; w * h];

    for by in 0..bh {
        for bx in 0..bw {
            let (px, py) = (bx * BLOCK, by * BLOCK);
            let cur_b = cur_block(cur, w, h, px, py);
            let mut best: Option<(i64, i32, i32, i32, i32)> = None;
            for dy in -cfg.search_range..=cfg.search_range {
                for dx in -cfg.search_range..=cfg.search_range {
                    let (x0, y0) = (px as i32 + dx, py as i32 + dy);
                    if x0 < 0 || y0 < 0 || x0 + BLOCK as i32 > w as i32 || y0 + BLOCK as i32 > h as i32 {
                        continue;
                    }
                    let a = ref_block(prev, w, x0 as usize, y0 as usize);
                    let b = ref_block(next, w, x0 as usize, y0 as usize);
                    let (wp, wn, sse) =
                        optimize_weights(&cur_b, &a, &b, scale, cfg.max_weight_iters, iters);
                    if best.map_or(true, |(bs, ..)| sse < bs) {
                        best = Some((sse, dx, dy, wp, wn));
                    }
                }
            }
            // Offset (0,0) is always in bounds for w,h >= 8.
            let (_, dx, dy, wp, wn) = best.expect("search window contains (0,0)");
            recs.push(BlockRec { dx, dy, wp, wn });

            let (x0, y0) = ((px as i32 + dx) as usize, (py as i32 + dy) as usize);
            let a = ref_block(prev, w, x0, y0);
            let b = ref_block(next, w, x0, y0);
            for y in 0..BLOCK {
                for x in 0..BLOCK {
                    let (ix, iy) = (px + x, py + y);
                    if ix < w && iy < h {
                        pred[iy * w + ix] = pred_px(a[y * BLOCK + x], b[y * BLOCK + x], wp, wn, scale);
                    }
                }
            }
        }
    }

    let mut residual = vec![0u8; w * h];
    for i in 0..w * h {
        residual[i] = ((cur[i] as i32 - pred[i]).clamp(-128, 127) + 128) as u8;
    }
    let res_bg = encode_intra(&residual, w, h, cfg.quality);

    let mut out = Vec::with_capacity(4 + nblocks * BLOCK_REC_SIZE + 4 + res_bg.len());
    out.extend_from_slice(&(nblocks as u32).to_le_bytes());
    for r in &recs {
        out.push(r.dx as i8 as u8);
        out.push(r.dy as i8 as u8);
        out.extend_from_slice(&(r.wp as u16).to_le_bytes());
        out.extend_from_slice(&(r.wn as u16).to_le_bytes());
    }
    out.extend_from_slice(&(res_bg.len() as u32).to_le_bytes());
    out.extend_from_slice(&res_bg);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Public sequence API
// ---------------------------------------------------------------------------

/// Encode a grayscale frame sequence with explicit per-frame types.
///
/// A [`FrameType::BiPred`] frame requires both neighbours to exist and be
/// [`FrameType::Intra`]; otherwise [`BiPredError::MissingReference`] is
/// returned. I-frame payloads are byte-identical to standalone
/// [`encoder::encode_grayscale`] output.
pub fn encode_sequence(
    frames: &[&[u8]],
    width: usize,
    height: usize,
    frame_types: &[FrameType],
    config: &BiPredConfig,
) -> Result<(Vec<u8>, EncodeStats), BiPredError> {
    config.validate()?;
    let n = frames.len();
    if n == 0 {
        return Err(BiPredError::InvalidConfig("no frames".into()));
    }
    if frame_types.len() != n {
        return Err(BiPredError::InvalidConfig(format!(
            "frame_types len {} != frames len {}",
            frame_types.len(),
            n
        )));
    }
    let expected = width
        .checked_mul(height)
        .filter(|&v| v > 0)
        .ok_or_else(|| BiPredError::InvalidConfig("bad dimensions".into()))?;
    for f in frames {
        if f.len() != expected {
            return Err(BiPredError::DimensionMismatch { expected, got: f.len() });
        }
    }
    for (i, t) in frame_types.iter().enumerate() {
        if *t == FrameType::BiPred {
            let ok = i > 0
                && i + 1 < n
                && frame_types[i - 1] == FrameType::Intra
                && frame_types[i + 1] == FrameType::Intra;
            if !ok {
                return Err(BiPredError::MissingReference { frame: i });
            }
        }
    }

    let mut stats = EncodeStats::default();
    let mut payloads: Vec<Option<Vec<u8>>> = vec![None; n];
    let mut decoded: Vec<Option<Vec<u8>>> = vec![None; n];

    // Pass 1: intra frames; keep the decoded planes as prediction references
    // so encoder and decoder see identical pixels (no drift).
    for i in 0..n {
        if frame_types[i] == FrameType::Intra {
            let payload = encode_intra(frames[i], width, height, config.quality);
            let dec =
                decode_intra(&payload, width, height).ok_or(BiPredError::IntraCodecFailed { frame: i })?;
            payloads[i] = Some(payload);
            decoded[i] = Some(dec);
            stats.i_frames += 1;
        }
    }
    // Pass 2: B frames against the decoded references.
    for i in 0..n {
        if frame_types[i] == FrameType::BiPred {
            let prev = decoded[i - 1].as_ref().ok_or(BiPredError::MissingReference { frame: i })?;
            let next = decoded[i + 1].as_ref().ok_or(BiPredError::MissingReference { frame: i })?;
            let payload = encode_bframe(
                frames[i],
                prev,
                next,
                width,
                height,
                config,
                &mut stats.weight_iterations,
            )?;
            payloads[i] = Some(payload);
            stats.b_frames += 1;
        }
    }

    let mut out = Vec::with_capacity(HEADER_SIZE + n * 5);
    out.extend_from_slice(BIPRED_MAGIC);
    out.extend_from_slice(&(width as u32).to_le_bytes());
    out.extend_from_slice(&(height as u32).to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.push(config.weight_bits);
    out.push(config.search_range as u8);
    out.push(config.quality);
    out.push(0); // reserved
    for (i, p) in payloads.iter().enumerate() {
        let p = p.as_ref().expect("every frame coded");
        out.push(if frame_types[i] == FrameType::Intra { FRAME_INTRA } else { FRAME_BIPRED });
        out.extend_from_slice(&(p.len() as u32).to_le_bytes());
        out.extend_from_slice(p);
    }
    stats.bytes = out.len();
    Ok((out, stats))
}

/// Encode a sequence with an automatic `I B I B ...` pattern (last frame
/// forced to intra so every B-frame has both references).
pub fn encode_sequence_auto(
    frames: &[&[u8]],
    width: usize,
    height: usize,
    config: &BiPredConfig,
) -> Result<(Vec<u8>, EncodeStats), BiPredError> {
    let n = frames.len();
    let mut types = vec![FrameType::Intra; n];
    let mut i = 1;
    while i + 1 < n {
        types[i] = FrameType::BiPred;
        i += 2;
    }
    encode_sequence(frames, width, height, &types, config)
}

/// Decode a `BGS1` sequence container back to grayscale frames.
pub fn decode_sequence(data: &[u8]) -> Result<DecodedSequence, BiPredError> {
    if data.len() < HEADER_SIZE || &data[..4] != BIPRED_MAGIC {
        return Err(BiPredError::BadMagic);
    }
    let w = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let h = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    let weight_bits = data[16];
    let search_range = data[17] as i32;
    if weight_bits == 0 || weight_bits > MAX_WEIGHT_BITS {
        return Err(BiPredError::InvalidConfig(format!("bad weight_bits {weight_bits}")));
    }
    if search_range > MAX_SEARCH_RANGE {
        return Err(BiPredError::InvalidConfig(format!("bad search_range {search_range}")));
    }
    if n == 0 || w.checked_mul(h).filter(|&v| v > 0).is_none() {
        return Err(BiPredError::InvalidConfig("bad dimensions".into()));
    }
    let scale = 1i32 << weight_bits;

    let mut pos = HEADER_SIZE;
    let mut types = Vec::with_capacity(n);
    let mut payloads: Vec<&[u8]> = Vec::with_capacity(n);
    for _ in 0..n {
        if pos + 5 > data.len() {
            return Err(BiPredError::Truncated);
        }
        let t = data[pos];
        if t != FRAME_INTRA && t != FRAME_BIPRED {
            return Err(BiPredError::Truncated);
        }
        let len = u32::from_le_bytes(data[pos + 1..pos + 5].try_into().unwrap()) as usize;
        pos += 5;
        if pos + len > data.len() {
            return Err(BiPredError::Truncated);
        }
        types.push(t);
        payloads.push(&data[pos..pos + len]);
        pos += len;
    }
    if pos != data.len() {
        return Err(BiPredError::Truncated);
    }

    let bw = w.div_ceil(BLOCK);
    let nblocks = bw * h.div_ceil(BLOCK);

    // Pass 1: intra frames.
    let mut frames: Vec<Option<Vec<u8>>> = vec![None; n];
    for i in 0..n {
        if types[i] == FRAME_INTRA {
            frames[i] =
                Some(decode_intra(payloads[i], w, h).ok_or(BiPredError::IntraCodecFailed { frame: i })?);
        }
    }
    // Pass 2: B frames.
    for i in 0..n {
        if types[i] != FRAME_BIPRED {
            continue;
        }
        if i == 0 || i + 1 >= n || types[i - 1] != FRAME_INTRA || types[i + 1] != FRAME_INTRA {
            return Err(BiPredError::MissingReference { frame: i });
        }
        if w < BLOCK || h < BLOCK {
            return Err(BiPredError::InvalidConfig(
                "B-frames need width and height >= 8".into(),
            ));
        }
        let p = payloads[i];
        if p.len() < 4 {
            return Err(BiPredError::Truncated);
        }
        let nb = u32::from_le_bytes(p[0..4].try_into().unwrap()) as usize;
        if nb != nblocks {
            return Err(BiPredError::Truncated);
        }
        let recs_end = 4 + nb * BLOCK_REC_SIZE;
        if p.len() < recs_end + 4 {
            return Err(BiPredError::Truncated);
        }
        let res_len = u32::from_le_bytes(p[recs_end..recs_end + 4].try_into().unwrap()) as usize;
        if recs_end + 4 + res_len != p.len() {
            return Err(BiPredError::Truncated);
        }
        let residual =
            decode_intra(&p[recs_end + 4..], w, h).ok_or(BiPredError::IntraCodecFailed { frame: i })?;
        let prev = frames[i - 1].as_ref().expect("intra neighbour decoded");
        let next = frames[i + 1].as_ref().expect("intra neighbour decoded");
        let mut out = vec![0u8; w * h];

        for bi in 0..nb {
            let (bx, by) = (bi % bw, bi / bw);
            let ro = 4 + bi * BLOCK_REC_SIZE;
            let dx = p[ro] as i8 as i32;
            let dy = p[ro + 1] as i8 as i32;
            let wp = u16::from_le_bytes([p[ro + 2], p[ro + 3]]) as i32;
            let wn = u16::from_le_bytes([p[ro + 4], p[ro + 5]]) as i32;
            if dx.abs() > search_range || dy.abs() > search_range {
                return Err(BiPredError::MotionOutOfRange { frame: i, block: bi, dx, dy, range: search_range });
            }
            if wp > scale || wn > scale {
                return Err(BiPredError::WeightOutOfRange {
                    frame: i,
                    block: bi,
                    w_prev: wp as u32,
                    w_next: wn as u32,
                    scale: scale as u32,
                });
            }
            let (x0, y0) = ((bx * BLOCK) as i32 + dx, (by * BLOCK) as i32 + dy);
            if x0 < 0 || y0 < 0 || x0 + BLOCK as i32 > w as i32 || y0 + BLOCK as i32 > h as i32 {
                return Err(BiPredError::MissingReference { frame: i });
            }
            let (x0, y0) = (x0 as usize, y0 as usize);
            for y in 0..BLOCK {
                for x in 0..BLOCK {
                    let (ix, iy) = (bx * BLOCK + x, by * BLOCK + y);
                    if ix < w && iy < h {
                        let a = prev[(y0 + y) * w + x0 + x] as i32;
                        let b = next[(y0 + y) * w + x0 + x] as i32;
                        let rec = pred_px(a, b, wp, wn, scale) + residual[iy * w + ix] as i32 - 128;
                        out[iy * w + ix] = rec.clamp(0, 255) as u8;
                    }
                }
            }
        }
        frames[i] = Some(out);
    }

    Ok(DecodedSequence {
        width: w,
        height: h,
        frames: frames.into_iter().map(|f| f.expect("every frame decoded")).collect(),
    })
}

// ---------------------------------------------------------------------------
// Test-only stand-ins for the C SIMD quantizer symbols, so `cargo test` links
// without the C library (same scalar semantics as c/quant.c). Release builds
// keep using the real C implementations; this module exists only under
// cfg(test) and never ships in the static/shared libs.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod quant_shim {
    #[no_mangle]
    pub extern "C" fn quantize_block(block: *mut i16, table: *const i16) {
        for i in 0..64 {
            unsafe {
                let b = *block.add(i) as i32;
                let t = (*table.add(i) as i32).max(1);
                *block.add(i) =
                    (b as f32 / t as f32).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn dequantize_block(block: *mut i16, table: *const i16) {
        for i in 0..64 {
            unsafe {
                let v = *block.add(i) as i32 * *table.add(i) as i32;
                *block.add(i) = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn gradient_frame(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = 96.0
                    + 60.0 * ((x as f64 + seed as f64) * 0.15).sin()
                    + 40.0 * (y as f64 * 0.11).cos()
                    + x as f64 * 0.5;
                out[y * w + x] = v.clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut mse = 0f64;
        for i in 0..a.len() {
            let d = a[i] as f64 - b[i] as f64;
            mse += d * d;
        }
        mse /= a.len() as f64;
        if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        }
    }

    fn seq_psnr(orig: &[Vec<u8>], dec: &[Vec<u8>]) -> f64 {
        let flat_o: Vec<u8> = orig.iter().flatten().copied().collect();
        let flat_d: Vec<u8> = dec.iter().flatten().copied().collect();
        psnr(&flat_o, &flat_d)
    }

    #[test]
    fn intra_frames_are_byte_identical_to_standalone_encoder() {
        let (w, h) = (24usize, 16usize);
        let frame = gradient_frame(w, h, 1);
        let cfg = BiPredConfig::default();
        let (container, stats) =
            encode_sequence(&[&frame], w, h, &[FrameType::Intra], &cfg).unwrap();
        assert_eq!(stats.i_frames, 1);
        assert_eq!(stats.b_frames, 0);

        // Standalone intra encode of the same frame with the same quality.
        let nblocks = w.div_ceil(8) * h.div_ceil(8);
        let mut solo = vec![0u8; encoder::BG_HEADER_SIZE + nblocks * RLE_BLOCK_CAP + 16];
        let mut pos = 0i32;
        encoder::encode_grayscale(&frame, w, h, cfg.quality, &mut solo, &mut pos);
        solo.truncate(pos as usize);

        assert_eq!(&container[..4], BIPRED_MAGIC);
        assert_eq!(container[20], FRAME_INTRA);
        let len = u32::from_le_bytes(container[21..25].try_into().unwrap()) as usize;
        assert_eq!(&container[25..25 + len], &solo[..], "I-frame payload must be byte-identical to the intra encoder");
        assert_eq!(25 + len, container.len());

        // Deterministic: same input -> same container bytes.
        let (again, _) = encode_sequence(&[&frame], w, h, &[FrameType::Intra], &cfg).unwrap();
        assert_eq!(container, again);
    }

    #[test]
    fn bipred_roundtrip_reconstructs_sequence() {
        let (w, h) = (32usize, 24usize);
        let frames: Vec<Vec<u8>> = (0..3).map(|s| gradient_frame(w, h, s)).collect();
        let refs: Vec<&[u8]> = frames.iter().map(|f| &f[..]).collect();
        let cfg = BiPredConfig::default();
        let types = [FrameType::Intra, FrameType::BiPred, FrameType::Intra];
        let (data, stats) = encode_sequence(&refs, w, h, &types, &cfg).unwrap();
        assert_eq!(stats.i_frames, 2);
        assert_eq!(stats.b_frames, 1);
        assert!(stats.weight_iterations > 0);

        let dec = decode_sequence(&data).unwrap();
        assert_eq!((dec.width, dec.height, dec.frames.len()), (w, h, 3));
        for (orig, rec) in frames.iter().zip(&dec.frames) {
            let p = psnr(orig, rec);
            assert!(p > 25.0, "roundtrip PSNR too low: {p}");
        }
    }

    #[test]
    fn missing_reference_frames_are_rejected() {
        let (w, h) = (16usize, 16usize);
        let f = gradient_frame(w, h, 0);
        let cfg = BiPredConfig::default();

        // B-frame with no neighbours at all.
        assert!(matches!(
            encode_sequence(&[&f], w, h, &[FrameType::BiPred], &cfg),
            Err(BiPredError::MissingReference { frame: 0 })
        ));
        // B-frame as the last frame: no next reference.
        assert!(matches!(
            encode_sequence(&[&f, &f], w, h, &[FrameType::Intra, FrameType::BiPred], &cfg),
            Err(BiPredError::MissingReference { frame: 1 })
        ));
        // Consecutive B-frames: frame 1's next neighbour is not intra.
        let types = [FrameType::Intra, FrameType::BiPred, FrameType::BiPred, FrameType::Intra];
        assert!(matches!(
            encode_sequence(&[&f, &f, &f, &f], w, h, &types, &cfg),
            Err(BiPredError::MissingReference { frame: 1 })
        ));

        // Decode side: a container whose first frame is (illegally) a B-frame.
        let (mut data, _) = encode_sequence(
            &[&f, &f],
            w,
            h,
            &[FrameType::Intra, FrameType::Intra],
            &cfg,
        )
        .unwrap();
        data[20] = FRAME_BIPRED;
        assert!(matches!(
            decode_sequence(&data),
            Err(BiPredError::MissingReference { frame: 0 })
        ));
    }

    #[test]
    fn decode_rejects_out_of_range_weights() {
        let (w, h) = (16usize, 16usize);
        let frames: Vec<Vec<u8>> = (0..3).map(|s| gradient_frame(w, h, s)).collect();
        let refs: Vec<&[u8]> = frames.iter().map(|f| &f[..]).collect();
        let cfg = BiPredConfig::default();
        let types = [FrameType::Intra, FrameType::BiPred, FrameType::Intra];
        let (mut data, _) = encode_sequence(&refs, w, h, &types, &cfg).unwrap();

        // Locate the B-frame payload: header(20) + frame0(type+len+payload).
        let len0 = u32::from_le_bytes(data[21..25].try_into().unwrap()) as usize;
        let f1 = 25 + len0;
        assert_eq!(data[f1], FRAME_BIPRED);
        let payload = f1 + 5;
        // First block record: nblocks(4) then dx,dy then w_prev(u16) at +6.
        let wp_off = payload + 4 + 2;
        let scale = 1u16 << cfg.weight_bits;
        data[wp_off..wp_off + 2].copy_from_slice(&(scale + 1).to_le_bytes());

        match decode_sequence(&data) {
            Err(BiPredError::WeightOutOfRange { frame, block, w_prev, .. }) => {
                assert_eq!((frame, block), (1, 0));
                assert_eq!(w_prev, scale as u32 + 1);
            }
            other => panic!("expected WeightOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn decode_rejects_out_of_range_motion() {
        let (w, h) = (16usize, 16usize);
        let frames: Vec<Vec<u8>> = (0..3).map(|s| gradient_frame(w, h, s)).collect();
        let refs: Vec<&[u8]> = frames.iter().map(|f| &f[..]).collect();
        let cfg = BiPredConfig::default();
        let types = [FrameType::Intra, FrameType::BiPred, FrameType::Intra];
        let (mut data, _) = encode_sequence(&refs, w, h, &types, &cfg).unwrap();

        let len0 = u32::from_le_bytes(data[21..25].try_into().unwrap()) as usize;
        let payload = 25 + len0 + 5;
        // First block record dx is at payload+4; push it past the search range.
        data[payload + 4] = (cfg.search_range + 1) as u8;

        assert!(matches!(
            decode_sequence(&data),
            Err(BiPredError::MotionOutOfRange { frame: 1, block: 0, .. })
        ));
    }

    #[test]
    fn invalid_configs_are_rejected() {
        let (w, h) = (16usize, 16usize);
        let f = gradient_frame(w, h, 0);
        let bad = [
            BiPredConfig { weight_bits: 0, ..Default::default() },
            BiPredConfig { weight_bits: 9, ..Default::default() },
            BiPredConfig { search_range: -1, ..Default::default() },
            BiPredConfig { search_range: 33, ..Default::default() },
            BiPredConfig { quality: 0, ..Default::default() },
            BiPredConfig { quality: 101, ..Default::default() },
        ];
        for cfg in bad {
            assert!(matches!(
                encode_sequence(&[&f], w, h, &[FrameType::Intra], &cfg),
                Err(BiPredError::InvalidConfig(_))
            ));
        }
        // Frame size mismatch.
        assert!(matches!(
            encode_sequence(&[&f[..f.len() - 1]], w, h, &[FrameType::Intra], &BiPredConfig::default()),
            Err(BiPredError::DimensionMismatch { .. })
        ));
        // Bad magic on decode.
        assert!(matches!(decode_sequence(b"NOPE...."), Err(BiPredError::BadMagic)));
    }

    // -----------------------------------------------------------------------
    // Benchmark report: three synthetic materials, run once.
    // -----------------------------------------------------------------------

    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// Slow horizontal pan: frame t samples the base pattern shifted by t px.
    fn material_pan(w: usize, h: usize, n: usize) -> Vec<Vec<u8>> {
        (0..n)
            .map(|t| {
                let mut f = vec![0u8; w * h];
                for y in 0..h {
                    for x in 0..w {
                        let sx = x + t;
                        let v = 128.0
                            + 70.0 * (sx as f64 * 0.13).sin() * (y as f64 * 0.09).cos()
                            + 40.0 * (sx as f64 * 0.031 + y as f64 * 0.017).sin();
                        f[y * w + x] = v.clamp(0.0, 255.0) as u8;
                    }
                }
                f
            })
            .collect()
    }

    /// Brightness ramp: static scene, global gain decreasing per frame.
    fn material_fade(w: usize, h: usize, n: usize) -> Vec<Vec<u8>> {
        (0..n)
            .map(|t| {
                let gain = 1.0 - 0.09 * t as f64;
                let mut f = vec![0u8; w * h];
                for y in 0..h {
                    for x in 0..w {
                        let base = 130.0
                            + 60.0 * (x as f64 * 0.21).sin() * (y as f64 * 0.13).cos()
                            + (x as f64 + y as f64) * 0.3;
                        f[y * w + x] = (base * gain).clamp(0.0, 255.0) as u8;
                    }
                }
                f
            })
            .collect()
    }

    /// Independent frames: fresh noise every frame (no inter-frame correlation).
    fn material_independent(w: usize, h: usize, n: usize) -> Vec<Vec<u8>> {
        (0..n)
            .map(|t| {
                let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ (t as u64 + 1).wrapping_mul(0x2545_F491_4F6C_DD1D);
                let mut f = vec![0u8; w * h];
                for px in f.iter_mut() {
                    *px = (xorshift(&mut rng) >> 32) as u8;
                }
                f
            })
            .collect()
    }

    fn encode_all_intra(frames: &[Vec<u8>], w: usize, h: usize, quality: u8) -> (Vec<Vec<u8>>, usize) {
        let nblocks = w.div_ceil(8) * h.div_ceil(8);
        let mut total = 0usize;
        let payloads = frames
            .iter()
            .map(|f| {
                let mut buf = vec![0u8; encoder::BG_HEADER_SIZE + nblocks * RLE_BLOCK_CAP + 16];
                let mut pos = 0i32;
                encoder::encode_grayscale(f, w, h, quality, &mut buf, &mut pos);
                buf.truncate(pos as usize);
                total += buf.len();
                buf
            })
            .collect();
        (payloads, total)
    }

    fn decode_all_intra(payloads: &[Vec<u8>], w: usize, h: usize) -> Vec<Vec<u8>> {
        payloads
            .iter()
            .map(|p| decode_intra(p, w, h).expect("intra decode"))
            .collect()
    }

    #[test]
    fn bipred_materials_report() {
        let (w, h, n) = (96usize, 64usize, 7usize);
        let cfg = BiPredConfig::default();
        let materials: Vec<(&str, Vec<Vec<u8>>)> = vec![
            ("slow_pan", material_pan(w, h, n)),
            ("brightness_fade", material_fade(w, h, n)),
            ("independent", material_independent(w, h, n)),
        ];
        let nblocks = w.div_ceil(8) * h.div_ceil(8);

        println!();
        println!(
            "bipred report: {w}x{h} x {n} frames, pattern I B I B I B I, \
             search_range=±{}, weight_bits={} (scale {}), quality={}, max_weight_iters={}",
            cfg.search_range,
            cfg.weight_bits,
            1i32 << cfg.weight_bits,
            cfg.quality,
            cfg.max_weight_iters
        );
        println!(
            "{:<16} | {:>7} | {:>7} | {:>5} | {:>6} | {:>6} | {:>12} | {:>12} | {:>16}",
            "material", "intra B", "bip B", "ratio", "PSNR i", "PSNR b", "enc ms i/b", "dec ms i/b", "w-iters tot/avg"
        );

        for (name, frames) in &materials {
            let refs: Vec<&[u8]> = frames.iter().map(|f| &f[..]).collect();

            let t0 = Instant::now();
            let (intra_payloads, intra_bytes) = encode_all_intra(frames, w, h, cfg.quality);
            let intra_enc_ms = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            let intra_dec = decode_all_intra(&intra_payloads, w, h);
            let intra_dec_ms = t1.elapsed().as_secs_f64() * 1e3;

            let t2 = Instant::now();
            let (data, stats) = encode_sequence_auto(&refs, w, h, &cfg).unwrap();
            let bip_enc_ms = t2.elapsed().as_secs_f64() * 1e3;
            let t3 = Instant::now();
            let dec = decode_sequence(&data).unwrap();
            let bip_dec_ms = t3.elapsed().as_secs_f64() * 1e3;

            let psnr_i = seq_psnr(frames, &intra_dec);
            let psnr_b = seq_psnr(frames, &dec.frames);
            let ratio = data.len() as f64 / intra_bytes as f64;
            let avg_iters = stats.weight_iterations as f64 / (stats.b_frames * nblocks) as f64;

            println!(
                "{name:<16} | {intra_bytes:>7} | {bip:>7} | {ratio:>5.3} | {psnr_i:>6.2} | {psnr_b:>6.2} | {intra_enc_ms:>5.2}/{bip_enc_ms:<6.2} | {intra_dec_ms:>5.2}/{bip_dec_ms:<6.2} | {tot:>7}/{avg_iters:>7.2}",
                bip = data.len(),
                tot = stats.weight_iterations,
            );

            assert!(psnr_b > 20.0, "{name}: bipred PSNR too low ({psnr_b})");
            if *name != "independent" {
                assert!(
                    data.len() < intra_bytes,
                    "{name}: bipred stream ({}) should beat all-intra ({intra_bytes})",
                    data.len()
                );
            }
        }
    }
}
