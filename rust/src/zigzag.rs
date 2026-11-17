//! Zigzag scan order for 8x8 DCT blocks (JPEG order). DC is index 0, AC follow zigzag.
//!
//! This module also provides **adaptive scan-order coding** (`encode_plane_adaptive` /
//! `decode_plane_adaptive`): instead of always scanning quantized coefficients in the
//! fixed zigzag order, the encoder picks — per 8x8 block — the scan order that best
//! matches the block's energy distribution, among:
//!
//!   * [`ZIGZAG`]     — classic JPEG diagonal scan
//!   * [`ROW_MAJOR`]  — row-first scan
//!   * [`COL_MAJOR`]  — column-first scan
//!   * custom         — a caller-supplied permutation ([`ScanPolicy::custom`],
//!                      defaults to [`SERPENTINE`])
//!
//! The selection criterion is configurable via [`CostModel`]. The chosen order is
//! stored as a 2-bit per-block header inside the entropy stream; the custom
//! permutation (when enabled) is stored once per plane together with a checksum, so
//! a corrupted scan-order table is detected at decode time.
//!
//! Adaptive plane layout (all integers little-endian):
//!
//! ```text
//! [len: u32]            bytes after this field
//! [magic: u8 = 0x5A]    adaptive scan plane, format v1
//! [flags: u8]           bit0: custom permutation present
//!                       bit1: per-block 2-bit order ids present (adaptive)
//!                       bit2-3: fixed order id (only when bit1 clear)
//!                       bit4-7: reserved, must be 0
//! [perm: 64 B]          only if bit0: perm[k] = spatial index of k-th coefficient
//! [checksum: u8]        only if bit0: rotate-xor checksum of perm
//! [bitstream]           per block: [order id: 2 bits if bit1] DC + AC (JPEG luma
//!                       Huffman tables, 0xFF byte stuffing, final 1-bit padding)
//! ```
//!
//! The custom permutation is transmitted only when at least one block actually
//! uses the custom order, so fixed zigzag/row/col planes carry no table at all.
//!
//! The classic fixed-zigzag encoder in `huffman.rs` is untouched and produces
//! byte-identical output; this module is an opt-in alternative.

/// Zigzag order: ZIGZAG[k] is the spatial index (0..64) of the k-th coefficient in the stream.
/// So we emit DC = block[ZIGZAG[0]], then AC = block[ZIGZAG[1]], ..., block[ZIGZAG[63]].
#[rustfmt::skip]
pub const ZIGZAG: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

// ---------------------------------------------------------------------------
// Scan-order tables
// ---------------------------------------------------------------------------

const fn build_row_major() -> [usize; 64] {
    let mut t = [0usize; 64];
    let mut k = 0;
    while k < 64 {
        t[k] = k;
        k += 1;
    }
    t
}

const fn build_col_major() -> [usize; 64] {
    let mut t = [0usize; 64];
    let mut k = 0;
    while k < 64 {
        t[k] = (k % 8) * 8 + k / 8;
        k += 1;
    }
    t
}

const fn build_serpentine() -> [usize; 64] {
    let mut t = [0usize; 64];
    let mut k = 0;
    while k < 64 {
        let r = k / 8;
        let c = k % 8;
        t[k] = if r % 2 == 0 { r * 8 + c } else { r * 8 + (7 - c) };
        k += 1;
    }
    t
}

/// Row-first (raster) scan order.
pub const ROW_MAJOR: [usize; 64] = build_row_major();

/// Column-first scan order.
pub const COL_MAJOR: [usize; 64] = build_col_major();

/// Serpentine (boustrophedon) row scan — the default custom permutation.
pub const SERPENTINE: [usize; 64] = build_serpentine();

// ---------------------------------------------------------------------------
// Public configuration types
// ---------------------------------------------------------------------------

/// One of the four scan orders selectable per block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScanOrder {
    Zigzag = 0,
    RowMajor = 1,
    ColMajor = 2,
    Custom = 3,
}

impl ScanOrder {
    fn from_index(i: usize) -> Self {
        match i {
            0 => ScanOrder::Zigzag,
            1 => ScanOrder::RowMajor,
            2 => ScanOrder::ColMajor,
            _ => ScanOrder::Custom,
        }
    }
}

/// Criterion used to pick the per-block scan order (configurable selection basis).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CostModel {
    /// Exact JPEG Huffman bit cost of the AC coefficients under each order
    /// (code lengths + magnitude bits). Picks the true minimum-size order.
    HuffmanBits,
    /// Number of run/size symbols emitted (EOB + ZRL + one per non-zero coef).
    /// Cheaper proxy for HuffmanBits.
    ZeroRuns,
    /// Sum of |coef| * (scan position + 1). Prefers orders that concentrate
    /// block energy at the front of the scan.
    EnergyWeighted,
}

/// Policy controlling adaptive scan-order selection.
///
/// * `candidates` — which orders may be picked (indexed by `ScanOrder as usize`).
///   Exactly one enabled candidate => fixed-order coding, no per-block headers.
/// * `cost` — selection criterion (see [`CostModel`]).
/// * `custom` — permutation used for [`ScanOrder::Custom`]. Must be a bijection
///   of 0..64 with `custom[0] == 0` (DC stays first).
#[derive(Clone)]
pub struct ScanPolicy {
    pub candidates: [bool; 4],
    pub cost: CostModel,
    pub custom: [usize; 64],
}

impl Default for ScanPolicy {
    /// Adaptive over all four orders, exact Huffman bit-cost selection,
    /// serpentine as the custom permutation.
    fn default() -> Self {
        Self { candidates: [true; 4], cost: CostModel::HuffmanBits, custom: SERPENTINE }
    }
}

impl ScanPolicy {
    /// Fixed-order policy (no per-block order ids written).
    pub fn fixed(order: ScanOrder) -> Self {
        let mut candidates = [false; 4];
        candidates[order as usize] = true;
        Self { candidates, cost: CostModel::HuffmanBits, custom: SERPENTINE }
    }

    pub fn with_cost(mut self, cost: CostModel) -> Self {
        self.cost = cost;
        self
    }

    pub fn with_custom(mut self, perm: [usize; 64]) -> Self {
        self.custom = perm;
        self
    }
}

/// Errors returned by the adaptive scan-order codec.
#[derive(Debug, PartialEq, Eq)]
pub enum ScanError {
    /// Policy has no candidate order enabled.
    NoCandidates,
    /// Custom permutation is not a bijection of 0..64 or does not keep DC first.
    InvalidPermutation,
    /// Plane bytes ended before the declared length / header.
    Truncated,
    /// Magic byte mismatch (not an adaptive scan plane v1).
    BadMagic,
    /// Unknown or inconsistent flag bits.
    BadFlags,
    /// Custom permutation checksum mismatch (order table corruption).
    ChecksumMismatch,
    /// A block references the custom order but no table was transmitted.
    MissingCustomTable,
    /// Huffman bitstream could not be decoded.
    Bitstream,
    /// Unconsumed bytes remain after the last block.
    TrailingData,
    /// Final padding bits are not all 1s.
    BadPadding,
}

impl std::fmt::Display for ScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ScanError::NoCandidates => "scan policy has no candidate order",
            ScanError::InvalidPermutation => "invalid custom scan permutation",
            ScanError::Truncated => "truncated adaptive scan plane",
            ScanError::BadMagic => "bad adaptive scan plane magic",
            ScanError::BadFlags => "bad adaptive scan plane flags",
            ScanError::ChecksumMismatch => "scan-order table checksum mismatch",
            ScanError::MissingCustomTable => "custom scan order referenced but table missing",
            ScanError::Bitstream => "adaptive scan bitstream decode failed",
            ScanError::TrailingData => "trailing data after adaptive scan plane",
            ScanError::BadPadding => "bad adaptive scan plane padding",
        };
        f.write_str(s)
    }
}

impl std::error::Error for ScanError {}

// ---------------------------------------------------------------------------
// Format constants + tables
// ---------------------------------------------------------------------------

const ADAPTIVE_MAGIC: u8 = 0x5A;
const FLAG_CUSTOM_TABLE: u8 = 0x01;
const FLAG_ORDER_IDS: u8 = 0x02;
const KNOWN_FLAGS: u8 = FLAG_CUSTOM_TABLE | FLAG_ORDER_IDS | 0x0C;

/// JPEG Annex K luminance DC Huffman table: (code length, code) per category.
const LUMA_DC: [(u8, u16); 12] = [
    (2, 0b00),
    (3, 0b010),
    (3, 0b011),
    (3, 0b100),
    (3, 0b101),
    (3, 0b110),
    (4, 0b1110),
    (5, 0b11110),
    (6, 0b111110),
    (7, 0b1111110),
    (8, 0b11111110),
    (9, 0b111111110),
];

use crate::block::Block;
use crate::huffman::BitWriter;
use crate::jpeg_luma_ac_ht::JPEG_LUMA_AC_HT;

/// JPEG luminance AC encode table: (code length, code) per RS symbol, (0,0) = absent.
fn ac_encode_table() -> [(u8, u16); 256] {
    let mut t = [(0u8, 0u16); 256];
    for (i, e) in t.iter_mut().enumerate() {
        let code = JPEG_LUMA_AC_HT[i][0];
        let len = JPEG_LUMA_AC_HT[i][1];
        if len > 0 {
            *e = (len as u8, code);
        }
    }
    t
}

/// Order-sensitive checksum of a permutation (a plain sum is invariant under
/// permutation and useless here).
fn perm_checksum(perm: &[usize; 64]) -> u8 {
    let mut acc = 0u8;
    for &p in perm.iter() {
        acc = acc.rotate_left(1) ^ (p as u8);
    }
    acc
}

fn validate_permutation(perm: &[usize; 64]) -> Result<(), ScanError> {
    if perm[0] != 0 {
        return Err(ScanError::InvalidPermutation);
    }
    let mut seen = [false; 64];
    for &p in perm.iter() {
        if p >= 64 || seen[p] {
            return Err(ScanError::InvalidPermutation);
        }
        seen[p] = true;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Magnitude helpers (same JPEG category/magnitude coding as huffman.rs)
// ---------------------------------------------------------------------------

#[inline]
fn category(v: i16) -> u8 {
    if v == 0 {
        return 0;
    }
    (16 - v.unsigned_abs().leading_zeros()) as u8
}

#[inline]
fn magnitude_bits(v: i16, cat: u8) -> u16 {
    if v >= 0 {
        v as u16
    } else {
        ((1u16 << cat) - 1).wrapping_add(v as u16)
    }
}

#[inline]
fn magnitude_decode(bits: u16, cat: u8) -> i16 {
    let threshold = 1u16 << (cat - 1);
    if bits >= threshold {
        bits as i16
    } else {
        bits as i16 - (1i16 << cat) + 1
    }
}

// ---------------------------------------------------------------------------
// Cost models (selection criteria)
// ---------------------------------------------------------------------------

const AC_CLAMP: i16 = 1023; // category 10 magnitude limit (JPEG AC table)
const DC_CLAMP: i16 = 2047; // category 11 magnitude limit (JPEG DC table)

fn cost_huffman_bits(block: &Block, table: &[usize; 64], ac_lens: &[u8; 256]) -> u64 {
    let mut last_nz = 63usize;
    while last_nz > 0 && block.data[table[last_nz]] == 0 {
        last_nz -= 1;
    }
    if last_nz == 0 {
        return ac_lens[0x00] as u64; // EOB only
    }
    let mut bits = 0u64;
    let mut run = 0u32;
    for i in 1..=last_nz {
        let val = block.data[table[i]].clamp(-AC_CLAMP, AC_CLAMP);
        if val == 0 {
            run += 1;
            continue;
        }
        bits += (run / 16) as u64 * ac_lens[0xF0] as u64;
        run %= 16;
        let cat = category(val);
        bits += ac_lens[((run << 4) as u8 | cat) as usize] as u64 + cat as u64;
        run = 0;
    }
    bits + ac_lens[0x00] as u64 // trailing EOB
}

fn cost_zero_runs(block: &Block, table: &[usize; 64]) -> u64 {
    let mut last_nz = 63usize;
    while last_nz > 0 && block.data[table[last_nz]] == 0 {
        last_nz -= 1;
    }
    let mut symbols = 1u64; // EOB
    let mut run = 0u32;
    for i in 1..=last_nz {
        let val = block.data[table[i]];
        if val == 0 {
            run += 1;
            continue;
        }
        symbols += (run / 16) as u64 + 1; // ZRLs + this coefficient
        run = 0;
    }
    symbols
}

fn cost_energy_weighted(block: &Block, table: &[usize; 64]) -> u64 {
    let mut cost = 0u64;
    for (k, &idx) in table.iter().enumerate().skip(1) {
        let v = block.data[idx].clamp(-AC_CLAMP, AC_CLAMP) as i64;
        cost += v.unsigned_abs() * (k as u64 + 1);
    }
    cost
}

fn select_order(
    block: &Block,
    policy: &ScanPolicy,
    tables: &[[usize; 64]; 4],
    ac_lens: &[u8; 256],
) -> ScanOrder {
    let mut best = ScanOrder::Zigzag;
    let mut best_cost = u64::MAX;
    for (i, &enabled) in policy.candidates.iter().enumerate() {
        if !enabled {
            continue;
        }
        let cost = match policy.cost {
            CostModel::HuffmanBits => cost_huffman_bits(block, &tables[i], ac_lens),
            CostModel::ZeroRuns => cost_zero_runs(block, &tables[i]),
            CostModel::EnergyWeighted => cost_energy_weighted(block, &tables[i]),
        };
        if cost < best_cost {
            best_cost = cost;
            best = ScanOrder::from_index(i);
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Block-level Huffman coding (bit-compatible with huffman.rs, no DC delta)
// ---------------------------------------------------------------------------

fn write_block_huffman(w: &mut BitWriter, block: &Block, table: &[usize; 64], ac: &[(u8, u16); 256]) {
    // DC
    let dc = block.data[table[0]].clamp(-DC_CLAMP, DC_CLAMP);
    let dc_cat = category(dc);
    let (dl, dcode) = LUMA_DC[dc_cat as usize];
    w.write_bits(dcode, dl);
    if dc_cat > 0 {
        w.write_bits(magnitude_bits(dc, dc_cat), dc_cat);
    }

    // AC: stop at last non-zero, always terminate with EOB (same as huffman.rs).
    let eob = ac[0x00];
    let zrl = ac[0xF0];
    let mut last_nz = 63usize;
    while last_nz > 0 && block.data[table[last_nz]] == 0 {
        last_nz -= 1;
    }
    if last_nz == 0 {
        w.write_bits(eob.1, eob.0);
        return;
    }
    let mut run = 0u8;
    for i in 1..=last_nz {
        let val = block.data[table[i]].clamp(-AC_CLAMP, AC_CLAMP);
        if val == 0 {
            run += 1;
            continue;
        }
        while run >= 16 {
            w.write_bits(zrl.1, zrl.0);
            run -= 16;
        }
        let cat = category(val);
        let sym = ((run << 4) | cat) as usize;
        let (al, code) = ac[sym];
        if al == 0 {
            // Defensive: unreachable after clamping with the full JPEG table.
            w.write_bits(eob.1, eob.0);
            return;
        }
        w.write_bits(code, al);
        w.write_bits(magnitude_bits(val, cat), cat);
        run = 0;
    }
    w.write_bits(eob.1, eob.0);
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Encode a plane of quantized blocks with adaptive (or fixed, per `policy`)
/// scan-order selection. See module docs for the plane layout.
pub fn encode_plane_adaptive(blocks: &[Block], policy: &ScanPolicy) -> Result<Vec<u8>, ScanError> {
    let n_candidates = policy.candidates.iter().filter(|&&c| c).count();
    if n_candidates == 0 {
        return Err(ScanError::NoCandidates);
    }
    let custom_candidate = policy.candidates[ScanOrder::Custom as usize];
    if custom_candidate {
        validate_permutation(&policy.custom)?;
    }

    let tables = [ZIGZAG, ROW_MAJOR, COL_MAJOR, policy.custom];
    let ac = ac_encode_table();
    let mut ac_lens = [0u8; 256];
    for (i, e) in ac.iter().enumerate() {
        ac_lens[i] = e.0;
    }

    let adaptive = n_candidates > 1;
    let mut fixed_order = ScanOrder::Zigzag;
    if !adaptive {
        fixed_order = ScanOrder::from_index(
            policy.candidates.iter().position(|&c| c).unwrap_or(0),
        );
    }

    // Pick an order per block up front: the custom permutation is only
    // transmitted when at least one block actually uses it.
    let orders: Vec<ScanOrder> = blocks
        .iter()
        .map(|b| {
            if adaptive {
                select_order(b, policy, &tables, &ac_lens)
            } else {
                fixed_order
            }
        })
        .collect();
    let custom_present =
        custom_candidate && orders.iter().any(|&o| o == ScanOrder::Custom);

    let mut flags = 0u8;
    if adaptive {
        flags |= FLAG_ORDER_IDS;
    } else {
        flags |= (fixed_order as u8) << 2;
    }
    if custom_present {
        flags |= FLAG_CUSTOM_TABLE;
    }

    let mut w = BitWriter::new();
    for (block, &order) in blocks.iter().zip(orders.iter()) {
        if adaptive {
            w.write_bits(order as u16, 2);
        }
        write_block_huffman(&mut w, block, &tables[order as usize], &ac);
    }
    w.flush();

    let table_bytes = if custom_present { 65 } else { 0 };
    let len = (2 + table_bytes + w.buf.len()) as u32;
    let mut out = Vec::with_capacity(4 + len as usize);
    out.extend_from_slice(&len.to_le_bytes());
    out.push(ADAPTIVE_MAGIC);
    out.push(flags);
    if custom_present {
        for &p in policy.custom.iter() {
            out.push(p as u8);
        }
        out.push(perm_checksum(&policy.custom));
    }
    out.extend_from_slice(&w.buf);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Bit reader with JPEG 0xFF 0x00 unstuffing. Exposes its final state so the
/// decoder can verify the plane was consumed exactly (integrity check).
struct ScanBitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    bit_buf: u64,
    bits_in: u8,
}

impl<'a> ScanBitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0, bit_buf: 0, bits_in: 0 }
    }

    #[inline]
    fn refill(&mut self) {
        while self.bits_in <= 56 && self.pos < self.buf.len() {
            let byte = self.buf[self.pos];
            self.pos += 1;
            if byte == 0xFF && self.pos < self.buf.len() && self.buf[self.pos] == 0x00 {
                self.pos += 1;
            }
            self.bit_buf = (self.bit_buf << 8) | byte as u64;
            self.bits_in += 8;
        }
    }

    #[inline]
    fn read_bit(&mut self) -> Option<u8> {
        if self.bits_in == 0 {
            self.refill();
            if self.bits_in == 0 {
                return None;
            }
        }
        self.bits_in -= 1;
        let bit = ((self.bit_buf >> self.bits_in) & 1) as u8;
        if self.bits_in == 0 {
            self.bit_buf = 0;
        } else {
            self.bit_buf &= (1u64 << self.bits_in) - 1;
        }
        Some(bit)
    }

    #[inline]
    fn read_bits(&mut self, n: u8) -> Option<u16> {
        if n == 0 {
            return Some(0);
        }
        let mut v = 0u16;
        for _ in 0..n {
            v = (v << 1) | self.read_bit()? as u16;
        }
        Some(v)
    }

    /// A valid plane ends exactly at the buffer end with at most 7 pad bits,
    /// all set to 1 (JPEG padding convention).
    fn finish(&self) -> Result<(), ScanError> {
        if self.pos != self.buf.len() || self.bits_in > 7 {
            return Err(ScanError::TrailingData);
        }
        let mask = (1u64 << self.bits_in) - 1;
        if self.bit_buf & mask != mask {
            return Err(ScanError::BadPadding);
        }
        Ok(())
    }
}

/// Canonical Huffman decoder built from an encode table: per code length, the
/// list of (code, symbol) pairs.
struct CanonicalDecoder {
    by_len: [Vec<(u16, u8)>; 17],
}

impl CanonicalDecoder {
    fn new() -> Self {
        Self { by_len: std::array::from_fn(|_| Vec::new()) }
    }

    fn dc_luma() -> Self {
        let mut d = Self::new();
        for (sym, &(len, code)) in LUMA_DC.iter().enumerate() {
            d.by_len[len as usize].push((code, sym as u8));
        }
        d
    }

    fn ac_luma() -> Self {
        let mut d = Self::new();
        for (sym, e) in JPEG_LUMA_AC_HT.iter().enumerate() {
            let (code, len) = (e[0], e[1]);
            if len > 0 {
                d.by_len[len as usize].push((code, sym as u8));
            }
        }
        d
    }

    fn decode(&self, r: &mut ScanBitReader) -> Option<u8> {
        let mut code = 0u16;
        for entries in self.by_len.iter().skip(1) {
            code = (code << 1) | r.read_bit()? as u16;
            for &(c, s) in entries {
                if c == code {
                    return Some(s);
                }
            }
        }
        None
    }
}

fn read_block_huffman(
    r: &mut ScanBitReader,
    table: &[usize; 64],
    dc_dec: &CanonicalDecoder,
    ac_dec: &CanonicalDecoder,
) -> Result<Block, ScanError> {
    let mut block = Block::new();

    // DC
    let dc_cat = dc_dec.decode(r).ok_or(ScanError::Bitstream)?;
    if dc_cat > 11 {
        return Err(ScanError::Bitstream);
    }
    let dc = if dc_cat == 0 {
        0
    } else {
        magnitude_decode(r.read_bits(dc_cat).ok_or(ScanError::Bitstream)?, dc_cat)
    };
    block.data[table[0]] = dc;

    // AC
    let mut idx = 1usize;
    loop {
        let sym = ac_dec.decode(r).ok_or(ScanError::Bitstream)?;
        if sym == 0x00 {
            break; // EOB
        }
        if sym == 0xF0 {
            idx += 16;
            if idx > 64 {
                return Err(ScanError::Bitstream);
            }
            continue;
        }
        let run = (sym >> 4) as usize;
        let cat = sym & 0x0F;
        if cat == 0 || cat > 10 {
            return Err(ScanError::Bitstream);
        }
        idx += run;
        if idx >= 64 {
            return Err(ScanError::Bitstream);
        }
        let bits = r.read_bits(cat).ok_or(ScanError::Bitstream)?;
        block.data[table[idx]] = magnitude_decode(bits, cat);
        idx += 1;
    }
    Ok(block)
}

/// Decode a plane written by [`encode_plane_adaptive`].
/// Returns the blocks and the byte position just past the plane.
pub fn decode_plane_adaptive(
    buf: &[u8],
    start: usize,
    n_blocks: usize,
) -> Result<(Vec<Block>, usize), ScanError> {
    if start + 4 > buf.len() {
        return Err(ScanError::Truncated);
    }
    let plane_len = u32::from_le_bytes(buf[start..start + 4].try_into().unwrap()) as usize;
    let data_start = start + 4;
    let data_end = data_start.checked_add(plane_len).ok_or(ScanError::Truncated)?;
    if data_end > buf.len() {
        return Err(ScanError::Truncated);
    }
    let data = &buf[data_start..data_end];
    if data.len() < 2 {
        return Err(ScanError::Truncated);
    }
    if data[0] != ADAPTIVE_MAGIC {
        return Err(ScanError::BadMagic);
    }
    let flags = data[1];
    if flags & !KNOWN_FLAGS != 0 {
        return Err(ScanError::BadFlags);
    }
    let has_ids = flags & FLAG_ORDER_IDS != 0;
    let has_custom = flags & FLAG_CUSTOM_TABLE != 0;
    if has_ids && flags & 0x0C != 0 {
        return Err(ScanError::BadFlags);
    }

    let mut pos = 2usize;
    let custom = if has_custom {
        if pos + 65 > data.len() {
            return Err(ScanError::Truncated);
        }
        let mut perm = [0usize; 64];
        for (i, p) in perm.iter_mut().enumerate() {
            *p = data[pos + i] as usize;
        }
        let checksum = data[pos + 64];
        pos += 65;
        if checksum != perm_checksum(&perm) {
            return Err(ScanError::ChecksumMismatch);
        }
        validate_permutation(&perm)?;
        perm
    } else {
        SERPENTINE
    };
    let tables = [ZIGZAG, ROW_MAJOR, COL_MAJOR, custom];

    let fixed = ((flags >> 2) & 3) as usize;
    if !has_ids && fixed == ScanOrder::Custom as usize && !has_custom {
        return Err(ScanError::MissingCustomTable);
    }

    let dc_dec = CanonicalDecoder::dc_luma();
    let ac_dec = CanonicalDecoder::ac_luma();
    let mut reader = ScanBitReader::new(&data[pos..]);
    let mut blocks = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        let order = if has_ids {
            let id = reader.read_bits(2).ok_or(ScanError::Bitstream)? as usize;
            if id == ScanOrder::Custom as usize && !has_custom {
                return Err(ScanError::MissingCustomTable);
            }
            id
        } else {
            fixed
        };
        blocks.push(read_block_huffman(&mut reader, &tables[order], &dc_dec, &ac_dec)?);
    }
    reader.finish()?;
    Ok((blocks, data_end))
}

// ---------------------------------------------------------------------------
// Tests + benchmark report
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::huffman;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }
    }

    fn random_blocks(n: usize, seed: u64, range: i16) -> Vec<Block> {
        let mut l = Lcg(seed);
        (0..n)
            .map(|_| {
                let mut b = Block::new();
                for v in b.data.iter_mut() {
                    *v = (l.next() % (2 * range as u64 + 1)) as i16 - range;
                }
                b
            })
            .collect()
    }

    fn roundtrip(policy: &ScanPolicy, blocks: &[Block]) {
        let enc = encode_plane_adaptive(blocks, policy).expect("encode failed");
        let (dec, end) = decode_plane_adaptive(&enc, 0, blocks.len()).expect("decode failed");
        assert_eq!(end, enc.len());
        assert_eq!(dec.len(), blocks.len());
        for (i, (a, b)) in blocks.iter().zip(dec.iter()).enumerate() {
            assert_eq!(a.data, b.data, "block {i} roundtrip mismatch");
        }
    }

    #[test]
    fn scan_tables_are_valid_permutations() {
        for t in [&ROW_MAJOR, &COL_MAJOR, &SERPENTINE] {
            validate_permutation(t).expect("built-in table must be a valid permutation");
        }
    }

    #[test]
    fn roundtrip_each_fixed_order() {
        let blocks = random_blocks(64, 0xC0FFEE, 200);
        for order in [
            ScanOrder::Zigzag,
            ScanOrder::RowMajor,
            ScanOrder::ColMajor,
            ScanOrder::Custom,
        ] {
            roundtrip(&ScanPolicy::fixed(order), &blocks);
        }
    }

    #[test]
    fn roundtrip_adaptive_all_cost_models() {
        let blocks = random_blocks(64, 0xBEEF, 1023);
        for cost in [CostModel::HuffmanBits, CostModel::ZeroRuns, CostModel::EnergyWeighted] {
            roundtrip(&ScanPolicy::default().with_cost(cost), &blocks);
        }
    }

    #[test]
    fn roundtrip_empty_plane() {
        roundtrip(&ScanPolicy::default(), &[]);
        roundtrip(&ScanPolicy::fixed(ScanOrder::Zigzag), &[]);
    }

    #[test]
    fn roundtrip_custom_permutation() {
        // Reverse scan (DC stays first): perm[k] = 64 - k for k >= 1.
        let mut perm = [0usize; 64];
        for (k, p) in perm.iter_mut().enumerate().skip(1) {
            *p = 64 - k;
        }
        let blocks = random_blocks(32, 0xABCD, 500);
        roundtrip(&ScanPolicy::fixed(ScanOrder::Custom).with_custom(perm), &blocks);
        roundtrip(&ScanPolicy::default().with_custom(perm), &blocks);
    }

    #[test]
    fn invalid_policy_rejected() {
        let blocks = random_blocks(4, 1, 10);
        let no_candidates = ScanPolicy {
            candidates: [false; 4],
            cost: CostModel::HuffmanBits,
            custom: SERPENTINE,
        };
        assert_eq!(
            encode_plane_adaptive(&blocks, &no_candidates),
            Err(ScanError::NoCandidates)
        );
        // Not a bijection (duplicate 1, missing 2).
        let mut bad = SERPENTINE;
        bad[2] = 1;
        let p = ScanPolicy::fixed(ScanOrder::Custom).with_custom(bad);
        assert_eq!(
            encode_plane_adaptive(&blocks, &p),
            Err(ScanError::InvalidPermutation)
        );
        // Bijection but DC not first.
        let mut shifted = SERPENTINE;
        shifted.swap(0, 1);
        let p = ScanPolicy::fixed(ScanOrder::Custom).with_custom(shifted);
        assert_eq!(
            encode_plane_adaptive(&blocks, &p),
            Err(ScanError::InvalidPermutation)
        );
    }

    #[test]
    fn selection_prefers_matching_orientation() {
        let policy = ScanPolicy::default();
        let tables = [ZIGZAG, ROW_MAJOR, COL_MAJOR, SERPENTINE];
        let ac = ac_encode_table();
        let mut lens = [0u8; 256];
        for (i, e) in ac.iter().enumerate() {
            lens[i] = e.0;
        }

        // Energy along row 0 -> row-first scan wins (ties against serpentine
        // row 0 break toward the lower order id).
        let mut row_block = Block::new();
        row_block.data[1] = 40;
        row_block.data[2] = 30;
        row_block.data[3] = 20;
        assert_eq!(
            select_order(&row_block, &policy, &tables, &lens),
            ScanOrder::RowMajor
        );

        // Energy along column 0 -> column-first scan wins.
        let mut col_block = Block::new();
        col_block.data[8] = 40;
        col_block.data[16] = 30;
        col_block.data[24] = 20;
        assert_eq!(
            select_order(&col_block, &policy, &tables, &lens),
            ScanOrder::ColMajor
        );
    }

    #[test]
    fn energy_weighted_model_is_configurable() {
        let policy = ScanPolicy::default().with_cost(CostModel::EnergyWeighted);
        let tables = [ZIGZAG, ROW_MAJOR, COL_MAJOR, SERPENTINE];
        let lens = [0u8; 256];
        // Single coefficient at spatial 8: scan positions are zigzag=2, row=8,
        // col=1, serpentine=15 -> column-first front-loads it best.
        let mut b = Block::new();
        b.data[8] = 10;
        assert_eq!(
            select_order(&b, &policy, &tables, &lens),
            ScanOrder::ColMajor
        );
    }

    #[test]
    fn corrupted_custom_table_detected() {
        // Fixed custom policy: the permutation table is always transmitted.
        let blocks = random_blocks(8, 0xF00D, 100);
        let enc = encode_plane_adaptive(&blocks, &ScanPolicy::fixed(ScanOrder::Custom)).unwrap();
        // Layout: [len 0..4][magic 4][flags 5][perm 6..70][checksum 70][stream].
        // Flip one permutation byte -> checksum no longer matches.
        let mut bad = enc.clone();
        bad[10] ^= 0x01;
        assert_eq!(
            decode_plane_adaptive(&bad, 0, blocks.len()).err(),
            Some(ScanError::ChecksumMismatch)
        );
        // Corrupt the checksum byte itself.
        let mut bad = enc.clone();
        bad[70] ^= 0xFF;
        assert_eq!(
            decode_plane_adaptive(&bad, 0, blocks.len()).err(),
            Some(ScanError::ChecksumMismatch)
        );
        // Checksum repaired but permutation no longer a bijection.
        let mut bad = enc.clone();
        bad[7] = bad[6]; // duplicate first entry
        let mut perm = [0usize; 64];
        for (i, p) in perm.iter_mut().enumerate() {
            *p = bad[6 + i] as usize;
        }
        bad[70] = perm_checksum(&perm);
        assert_eq!(
            decode_plane_adaptive(&bad, 0, blocks.len()).err(),
            Some(ScanError::InvalidPermutation)
        );
    }

    #[test]
    fn corrupted_header_detected() {
        let blocks = random_blocks(8, 0x1234, 100);
        let enc = encode_plane_adaptive(&blocks, &ScanPolicy::default()).unwrap();

        let mut bad = enc.clone();
        bad[4] ^= 0xFF; // magic
        assert_eq!(
            decode_plane_adaptive(&bad, 0, blocks.len()).err(),
            Some(ScanError::BadMagic)
        );

        let mut bad = enc.clone();
        bad[5] |= 0x80; // reserved flag bit
        assert_eq!(
            decode_plane_adaptive(&bad, 0, blocks.len()).err(),
            Some(ScanError::BadFlags)
        );

        // Truncated plane (declared length exceeds available bytes).
        let truncated = &enc[..enc.len() - 2];
        assert_eq!(
            decode_plane_adaptive(truncated, 0, blocks.len()).err(),
            Some(ScanError::Truncated)
        );

        // Extra trailing bytes inside the declared plane length.
        let mut bad = enc.clone();
        let len = bad.len() as u32 - 4 + 3;
        bad[0..4].copy_from_slice(&len.to_le_bytes());
        bad.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        assert_eq!(
            decode_plane_adaptive(&bad, 0, blocks.len()).err(),
            Some(ScanError::TrailingData)
        );
    }

    #[test]
    fn zigzag_only_bitstream_matches_huffman_encoder() {
        // With a zigzag-only policy the adaptive plane carries no order ids and
        // no custom table; its bitstream section must equal the classic
        // huffman.rs output bit-for-bit.
        let blocks = random_blocks(16, 0x777, 300);
        let adaptive = encode_plane_adaptive(&blocks, &ScanPolicy::fixed(ScanOrder::Zigzag)).unwrap();
        let classic = huffman::encode_plane_with_profile(&blocks, false, false, false);
        assert_eq!(&adaptive[6..], &classic[4..]);
    }

    /// Deterministic sample blocks for the byte-identity golden test.
    fn golden_blocks() -> Vec<Block> {
        let mut blocks = Vec::new();
        let mut l = Lcg(0x1234_5678_9ABC_DEF0);
        for _ in 0..8 {
            let mut b = Block::new();
            for v in b.data.iter_mut() {
                *v = (l.next() % 41) as i16 - 20;
            }
            blocks.push(b);
        }
        blocks
    }

    #[test]
    fn original_fixed_order_encoding_byte_identical() {
        // The classic fixed-zigzag encoder (huffman.rs) is untouched by the
        // adaptive scan-order work; pin its exact output bytes.
        let enc = huffman::encode_plane_with_profile(&golden_blocks(), false, false, true);
        assert_eq!(enc, GOLDEN_FIXED_ZIGZAG);
    }

    // Golden bytes generated from huffman::encode_plane_with_profile(
    //   golden_blocks(), is_chroma=false, use_chroma_ac=false, use_dc_delta=true)
    // on the unmodified 2.0.0 encoder.
    const GOLDEN_FIXED_ZIGZAG: &[u8] = &[
        0xdc, 0x01, 0x00, 0x00, 0xd2, 0x9b, 0x4f, 0xd3, 0xee, 0x65, 0x82, 0xe6, 0xfe, 0xd7, 0x4e,
        0xb0, 0xb3, 0xbe, 0x5b, 0x0d, 0x46, 0xf3, 0x51, 0xb8, 0xba, 0xb1, 0xb7, 0x8e, 0x4b, 0xeb,
        0x6d, 0x43, 0x52, 0xb4, 0xfb, 0x5e, 0x9f, 0x1b, 0xea, 0x89, 0x70, 0xda, 0x95, 0x83, 0xa5,
        0xfe, 0x97, 0xa5, 0xc7, 0xa8, 0x41, 0x6f, 0xfd, 0x99, 0x6b, 0x77, 0x6b, 0x67, 0xaa, 0x45,
        0xa6, 0x4d, 0x02, 0xdf, 0x5e, 0xd3, 0xaf, 0xad, 0x2d, 0x2d, 0x61, 0x6b, 0x1f, 0xb2, 0x5c,
        0xb5, 0x95, 0xf6, 0xa7, 0x7b, 0x7b, 0x7d, 0x36, 0x97, 0x63, 0x7f, 0xa6, 0x5a, 0x5f, 0x5f,
        0x69, 0xd7, 0x9a, 0xa4, 0x82, 0x2d, 0x3b, 0x4f, 0xb8, 0x4b, 0x1b, 0xcd, 0x3e, 0xd2, 0xf6,
        0x2d, 0x2e, 0xd2, 0xee, 0xe2, 0xc7, 0x52, 0xd4, 0x74, 0xfb, 0x2b, 0xbd, 0x45, 0x09, 0x90,
        0x35, 0x8e, 0xa7, 0x7d, 0x36, 0xa3, 0x4d, 0xd4, 0xef, 0xc6, 0x97, 0x69, 0x3e, 0x9d, 0xa7,
        0xae, 0x99, 0x76, 0x97, 0x36, 0x1a, 0x75, 0xa5, 0xd6, 0xa7, 0x61, 0xba, 0xee, 0xcc, 0x5b,
        0x25, 0xbe, 0x99, 0x65, 0xa6, 0x36, 0x9f, 0x61, 0x79, 0x61, 0xa5, 0xea, 0x9a, 0x94, 0x21,
        0xee, 0xec, 0xbc, 0xd4, 0xd4, 0x74, 0xcb, 0x19, 0x6d, 0xf5, 0x37, 0xbd, 0xf2, 0x8a, 0x69,
        0xc1, 0x67, 0xb7, 0xaa, 0x57, 0xf6, 0xfa, 0x6e, 0xa6, 0x2c, 0xa0, 0xb0, 0xd2, 0xe6, 0x87,
        0x4e, 0xb1, 0xd2, 0xe3, 0x4d, 0x4a, 0xe7, 0x51, 0x82, 0x48, 0x3e, 0xdf, 0x33, 0xdf, 0x69,
        0xba, 0x74, 0x96, 0x77, 0xba, 0x8c, 0x76, 0xda, 0xa1, 0xd2, 0xee, 0x65, 0xbf, 0x9f, 0x53,
        0xd2, 0xf5, 0x27, 0xd5, 0x2e, 0x5e, 0x5d, 0x4c, 0xdc, 0x26, 0xa9, 0x7b, 0xa8, 0x45, 0xa9,
        0x5e, 0xcf, 0x73, 0x67, 0x50, 0x49, 0xa6, 0x44, 0xcc, 0x35, 0x4b, 0xcb, 0x9b, 0x1b, 0xcb,
        0x2b, 0xcb, 0x2d, 0x8d, 0x3b, 0x26, 0xa5, 0xfd, 0x9a, 0xf7, 0x1f, 0xda, 0x13, 0x7f, 0x68,
        0x5b, 0x4d, 0x63, 0x16, 0xa7, 0x63, 0x7f, 0x79, 0x32, 0x6a, 0x7a, 0xa5, 0xcc, 0xf7, 0xfa,
        0x5d, 0x8d, 0x95, 0xad, 0xb4, 0x7a, 0x94, 0x70, 0xe9, 0x9a, 0x7e, 0xa3, 0xa7, 0x18, 0xec,
        0xec, 0x2c, 0x6a, 0x95, 0xdb, 0x49, 0x61, 0x3d, 0xf0, 0xb4, 0x86, 0xce, 0xce, 0x4b, 0xcd,
        0x46, 0xff, 0x00, 0x52, 0xbc, 0xb4, 0xb9, 0xbb, 0xd4, 0xcd, 0xfc, 0x92, 0x5a, 0x65, 0x2d,
        0x83, 0xdb, 0x5f, 0x69, 0xba, 0x8e, 0x9f, 0x2d, 0xdd, 0xda, 0xcd, 0x7c, 0xb3, 0x59, 0x69,
        0x76, 0xb6, 0x76, 0x97, 0x1a, 0x8c, 0x56, 0xf7, 0x71, 0xc2, 0x0d, 0x8d, 0xde, 0x9f, 0xa5,
        0xd3, 0x2c, 0xb4, 0xd9, 0x35, 0x49, 0xe2, 0xbf, 0x89, 0xaf, 0xef, 0xaf, 0x05, 0x81, 0x97,
        0x4f, 0x1a, 0x70, 0xbb, 0xd3, 0xef, 0xec, 0x2e, 0x35, 0x48, 0xf4, 0xe8, 0xb5, 0x27, 0xd4,
        0x6c, 0x2c, 0x64, 0xd3, 0x6f, 0xaf, 0xa0, 0xb1, 0x3a, 0x86, 0xa1, 0x24, 0x5a, 0x6e, 0xa1,
        0x63, 0x63, 0x1d, 0xdd, 0xee, 0xa3, 0x1d, 0xea, 0x5f, 0x5c, 0x5c, 0x46, 0x2f, 0x2c, 0x6b,
        0x7a, 0xdf, 0x50, 0xbb, 0xb6, 0x9a, 0xe3, 0x53, 0x1a, 0x95, 0xad, 0xcd, 0xcd, 0xae, 0x5e,
        0xde, 0xe8, 0x5e, 0xcf, 0x63, 0xa7, 0x5b, 0x41, 0x6f, 0x62, 0xfa, 0x9d, 0x82, 0x58, 0xda,
        0xc5, 0x61, 0x24, 0xb6, 0xb6, 0xd6, 0x17, 0x33, 0x69, 0xb6, 0xfa, 0x5c, 0x16, 0x76, 0x52,
        0xda, 0x5a, 0xe9, 0x7a, 0x6e, 0x9b, 0xa6, 0xb4, 0x2d, 0x6b, 0xa9, 0xd8, 0x5f, 0x41, 0x5f,
    ];

    // -----------------------------------------------------------------------
    // Benchmark report: four synthetic materials x five scan-order policies.
    // Run with: cargo test --release scan_order_bench_report -- --nocapture
    // -----------------------------------------------------------------------

    fn gen_natural(w: usize, h: usize) -> Vec<u8> {
        // Heterogeneous synthetic "natural" scene:
        //   left 2/3        — smooth bilinear-upsampled value noise (sky/texture)
        //   right top       — horizontal stripes (energy in column 0)
        //   right bottom    — vertical stripes (energy in row 0)
        let mut l = Lcg(0xA7A0_0001);
        let gw = 16usize;
        let gh = 16usize;
        let grid: Vec<u8> = (0..gw * gh).map(|_| (l.next() % 256) as u8).collect();
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = if x >= 2 * w / 3 {
                    let noise = (l.next() % 9) as i32 - 4;
                    if y < h / 2 {
                        // horizontal square-wave stripes: energy in column 0
                        // (mid vertical frequencies + harmonics)
                        let s = if (y / 2) % 2 == 0 { 210 } else { 50 };
                        s + noise
                    } else {
                        // vertical square-wave stripes: energy in row 0
                        let s = if (x / 2) % 2 == 0 { 210 } else { 50 };
                        s + noise
                    }
                } else {
                    let fx = x as f64 * (gw - 1) as f64 / (w - 1) as f64;
                    let fy = y as f64 * (gh - 1) as f64 / (h - 1) as f64;
                    let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
                    let (x1, y1) = ((x0 + 1).min(gw - 1), (y0 + 1).min(gh - 1));
                    let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
                    let g = |gx: usize, gy: usize| grid[gy * gw + gx] as f64;
                    let s = g(x0, y0) * (1.0 - tx) * (1.0 - ty)
                        + g(x1, y0) * tx * (1.0 - ty)
                        + g(x0, y1) * (1.0 - tx) * ty
                        + g(x1, y1) * tx * ty;
                    s as i32 + (l.next() % 13) as i32 - 6
                };
                out[y * w + x] = v.clamp(0, 255) as u8;
            }
        }
        out
    }

    fn gen_text(w: usize, h: usize) -> Vec<u8> {
        // White page with deterministic pseudo-glyph black strokes.
        let mut l = Lcg(0x7E57_0002);
        let mut out = vec![235u8; w * h];
        let mut cy = 3;
        while cy + 10 < h {
            let mut cx = 4;
            while cx + 7 < w {
                if l.next() % 8 != 0 {
                    // glyph cell 6x9
                    for py in 0..9 {
                        for px in 0..6 {
                            if l.next() % 5 < 2 {
                                out[(cy + py) * w + cx + px] = 25;
                            }
                        }
                    }
                }
                cx += 8;
            }
            cy += 13;
        }
        out
    }

    fn gen_gradient(w: usize, h: usize) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                out[y * w + x] = ((x as u32 * 2 + y as u32) % 256) as u8;
            }
        }
        out
    }

    fn gen_noise(w: usize, h: usize) -> Vec<u8> {
        let mut l = Lcg(0xDEAD_0004);
        (0..w * h).map(|_| (l.next() % 256) as u8).collect()
    }

    fn blocks_from_plane(plane: &[u8], w: usize, h: usize) -> Vec<Block> {
        let mut blocks = Vec::new();
        for by in (0..h).step_by(8) {
            for bx in (0..w).step_by(8) {
                let mut b = Block::new();
                for y in 0..8 {
                    for x in 0..8 {
                        b.data[y * 8 + x] = plane[(by + y) * w + bx + x] as i16 - 128;
                    }
                }
                blocks.push(b);
            }
        }
        blocks
    }

    fn quantize_rs(data: &mut [i16; 64], table: &[i16; 64]) {
        for i in 0..64 {
            let q = table[i] as i32;
            let v = data[i] as i32;
            data[i] = ((v.abs() + q / 2) / q * v.signum()) as i16;
        }
    }

    /// Zero-order entropy (bits per block) of the emitted DC-category and
    /// run/size symbol stream under the given policy.
    fn symbol_entropy_bits_per_block(blocks: &[Block], policy: &ScanPolicy) -> f64 {
        let tables = [ZIGZAG, ROW_MAJOR, COL_MAJOR, policy.custom];
        let ac = ac_encode_table();
        let mut lens = [0u8; 256];
        for (i, e) in ac.iter().enumerate() {
            lens[i] = e.0;
        }
        let mut dc_hist = [0u64; 12];
        let mut ac_hist = [0u64; 256];
        for b in blocks {
            let order = select_order(b, policy, &tables, &lens);
            let table = &tables[order as usize];
            dc_hist[category(b.data[table[0]].clamp(-DC_CLAMP, DC_CLAMP)) as usize] += 1;
            let mut last_nz = 63usize;
            while last_nz > 0 && b.data[table[last_nz]] == 0 {
                last_nz -= 1;
            }
            ac_hist[0x00] += 1; // EOB
            let mut run = 0u32;
            for i in 1..=last_nz {
                let val = b.data[table[i]].clamp(-AC_CLAMP, AC_CLAMP);
                if val == 0 {
                    run += 1;
                    continue;
                }
                while run >= 16 {
                    ac_hist[0xF0] += 1;
                    run -= 16;
                }
                ac_hist[((run << 4) as u8 | category(val)) as usize] += 1;
                run = 0;
            }
        }
        fn ideal_bits(hist: &[u64]) -> f64 {
            let n: u64 = hist.iter().sum();
            if n == 0 {
                return 0.0;
            }
            let n = n as f64;
            hist.iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let c = c as f64;
                    -c * (c / n).log2()
                })
                .sum()
        }
        (ideal_bits(&dc_hist) + ideal_bits(&ac_hist)) / blocks.len() as f64
    }

    /// Header overhead in bytes, computed from the encoded plane's flags:
    /// magic + flags, custom table (if present), per-block order ids (if adaptive).
    fn header_overhead_bytes(enc: &[u8], n_blocks: usize) -> usize {
        let flags = enc[5];
        let mut bytes = 2; // magic + flags
        if flags & FLAG_CUSTOM_TABLE != 0 {
            bytes += 65; // permutation + checksum
        }
        if flags & FLAG_ORDER_IDS != 0 {
            bytes += (2 * n_blocks + 7) / 8; // 2-bit order id per block
        }
        bytes
    }

    fn time_encode_us(blocks: &[Block], policy: &ScanPolicy, repeats: usize) -> f64 {
        let t = std::time::Instant::now();
        for _ in 0..repeats {
            let v = encode_plane_adaptive(blocks, policy).unwrap();
            std::hint::black_box(&v);
        }
        t.elapsed().as_micros() as f64 / repeats as f64
    }

    #[test]
    fn scan_order_bench_report() {
        const W: usize = 128;
        const H: usize = 128;
        const N_BLOCKS: usize = (W / 8) * (H / 8);
        let repeats = if cfg!(debug_assertions) { 3 } else { 50 };
        let qtable = crate::encoder::quant_table_for_quality(75);

        let materials: [(&str, Vec<u8>); 4] = [
            ("natural", gen_natural(W, H)),
            ("text", gen_text(W, H)),
            ("gradient", gen_gradient(W, H)),
            ("noise", gen_noise(W, H)),
        ];

        println!("\n# Adaptive scan-order report: {W}x{H} luma plane ({N_BLOCKS} blocks), quality 75");
        println!("# entropy = zero-order entropy of DC-category + run/size symbols (bits/block)");
        println!("# stream  = full plane bytes incl. 4-byte length prefix");
        println!("# header  = magic+flags + custom table + per-block 2-bit order ids (bytes)");

        for (name, plane) in &materials {
            let mut blocks = blocks_from_plane(plane, W, H);
            for b in blocks.iter_mut() {
                crate::dct::dct(b);
                quantize_rs(&mut b.data, &qtable);
                crate::huffman::clamp_block_jpeg_coeffs(b);
            }

            println!("\n=== material: {name} ===");
            println!(
                "{:<9} {:>15} {:>13} {:>10} {:>12}",
                "order", "entropy(b/blk)", "stream(bytes)", "header(B)", "encode(us)"
            );

            let policies: [(&str, ScanPolicy); 5] = [
                ("zigzag", ScanPolicy::fixed(ScanOrder::Zigzag)),
                ("row", ScanPolicy::fixed(ScanOrder::RowMajor)),
                ("col", ScanPolicy::fixed(ScanOrder::ColMajor)),
                ("custom", ScanPolicy::fixed(ScanOrder::Custom)),
                ("adaptive", ScanPolicy::default()),
            ];
            for (pname, policy) in &policies {
                let enc = encode_plane_adaptive(&blocks, policy).expect("encode failed");
                let (dec, _) = decode_plane_adaptive(&enc, 0, blocks.len()).expect("decode failed");
                for (a, b) in blocks.iter().zip(dec.iter()) {
                    assert_eq!(a.data, b.data, "bench roundtrip mismatch");
                }
                let entropy = symbol_entropy_bits_per_block(&blocks, policy);
                let header = header_overhead_bytes(&enc, blocks.len());
                let us = time_encode_us(&blocks, policy, repeats);
                println!(
                    "{pname:<9} {:>15.2} {:>13} {:>10} {:>12.1}",
                    entropy,
                    enc.len(),
                    header,
                    us
                );
            }

            // Baseline: the existing fixed-zigzag encoder (huffman.rs, untouched).
            let base = huffman::encode_plane_with_profile(&blocks, false, false, false);
            let t = std::time::Instant::now();
            for _ in 0..repeats {
                let v = huffman::encode_plane_with_profile(&blocks, false, false, false);
                std::hint::black_box(&v);
            }
            let base_us = t.elapsed().as_micros() as f64 / repeats as f64;
            println!(
                "{:<9} {:>15} {:>13} {:>10} {:>12.1}",
                "baseline*", "-",
                base.len(),
                0,
                base_us
            );
            println!("(* baseline = existing fixed-zigzag huffman.rs path, no block headers)");
        }
    }
}
