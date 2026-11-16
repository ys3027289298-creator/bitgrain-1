//! Entropy coding for DCT coefficient planes.
//!
//! Two backends coexist in this module:
//!
//! * [`legacy`] — the original static JPEG Huffman bitstream (ISO 10918-1 Annex K
//!   tables, MSB-first, 0xFF byte-stuffing). It is retained byte-for-byte unchanged so
//!   every pre-existing .bg v4–v19 stream decodes identically and the fixed-bit/Huffman
//!   output remains reproducible.
//! * [`rangecode`] — an adaptive range (arithmetic-style) entropy coder. Symbols are
//!   collected first, a cumulative-frequency table is built, the interval is subdivided
//!   per symbol on encode and reverse-looked-up on decode. The frequency table is
//!   rebuilt per plane (per block group); a single global table over the whole stream
//!   is supported as well ([`rangecode::TableScope`]).
//!
//! The public plane entry points keep their historical signatures. They encode with the
//! range coder and, on decode, auto-detect the backend from the plane marker word, so
//! legacy streams still take the legacy path.

use crate::block::Block;

// Re-exports so existing callers keep working through `crate::huffman::*`.
pub(crate) use legacy::clamp_block_jpeg_coeffs;
pub use legacy::{BitReader, BitWriter};

// The `legacy` and `rangecode` modules are defined inline at the bottom of this file
// so the entire entropy-layer swap lives in a single source file.

// ---------------------------------------------------------------------------
// Plane-level entry points (signatures unchanged).
// Encode: range coder. Decode: marker-based dispatch to either backend.
// ---------------------------------------------------------------------------

/// Marker prefix identifying a range-coded plane payload. Low 3 bits carry the
/// scope id (bits 1..2) and the DC-delta flag (bit 0); all other bits must match.
const RC_MARKER: u32 = rangecode::RC_MARKER;
const RC_MARKER_MASK: u32 = rangecode::RC_MARKER_MASK;

/// Encode all blocks of a plane into a Vec<u8> prefixed with its byte length (u32 LE).
/// Format: [len: u32 LE][plane payload]
/// The length prefix allows the decoder to skip exactly to the next plane boundary.
pub fn encode_plane(blocks: &[Block], is_chroma: bool) -> Vec<u8> {
    encode_plane_with_profile(blocks, is_chroma, false, false)
}

pub fn encode_plane_with_ac(blocks: &[Block], is_chroma: bool, _use_chroma_ac: bool) -> Vec<u8> {
    encode_plane_with_profile(blocks, is_chroma, false, false)
}

pub fn encode_plane_with_profile(
    blocks: &[Block],
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
) -> Vec<u8> {
    let _ = (is_chroma, use_chroma_ac); // profile flags only affected Huffman table choice
    // The table-free adaptive model is the production default: it carries no serialized
    // frequency table, adapts to local statistics, and wins on the small/smooth planes
    // that dominate real images. The per-plane/per-block table modes remain available via
    // rangecode::encode_plane_range for explicit, reproducible frequency-table experiments.
    rangecode::encode_plane_range(blocks, rangecode::TableScope::Global, use_dc_delta)
}

/// Decode a plane of `n_blocks` blocks from `buf[start..]`.
/// Reads the 4-byte length prefix, then decodes exactly that many bytes.
/// Returns (blocks, new_byte_position) or None on error.
pub fn decode_plane(buf: &[u8], start: usize, n_blocks: usize, is_chroma: bool) -> Option<(Vec<Block>, usize)> {
    decode_plane_with_profile(buf, start, n_blocks, is_chroma, false, false)
}

pub fn decode_plane_with_ac(
    buf: &[u8],
    start: usize,
    n_blocks: usize,
    is_chroma: bool,
    use_chroma_ac: bool,
) -> Option<(Vec<Block>, usize)> {
    decode_plane_with_profile(buf, start, n_blocks, is_chroma, use_chroma_ac, false)
}

pub fn decode_plane_with_profile(
    buf: &[u8],
    start: usize,
    n_blocks: usize,
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
) -> Option<(Vec<Block>, usize)> {
    if start + 4 > buf.len() {
        return None;
    }
    let plane_len = u32::from_le_bytes(buf[start..start + 4].try_into().unwrap()) as usize;
    let data_start = start + 4;
    let data_end = data_start.checked_add(plane_len)?;
    if data_end > buf.len() {
        return None;
    }

    if plane_len >= 4 && (u32::from_le_bytes(buf[data_start..data_start + 4].try_into().unwrap())
        & RC_MARKER_MASK)
        == RC_MARKER
    {
        return rangecode::decode_plane_range(buf, start, n_blocks);
    }

    // Legacy fixed-bit/Huffman stream: delegate to the untouched backend.
    legacy::decode_plane_with_profile(buf, start, n_blocks, is_chroma, use_chroma_ac, use_dc_delta)
}

// ===========================================================================
// Legacy fixed-bit/Huffman backend (preserved byte-for-byte)
// ===========================================================================

pub mod legacy {
    //! Static Huffman coding for DCT coefficients.
    //!
    //! Uses the standard JPEG Huffman tables (ISO 10918-1 Annex K):
    //!   - Luma DC + full luminance AC for Y (and A); chroma DC for Cb/Cr; **same luminance AC** for
    //!     all AC planes so .bg v4/v5 matches historical bitgrain streams.
    //!
    //! The bitstream is packed MSB-first, continuous across all blocks in a plane.
    //! 0xFF bytes are stuffed as 0xFF 0x00 (JPEG convention).
    //! A single flush (pad with 1s) is written at the end of each plane.
    //!
    //! This module is the original fixed-bit/Huffman backend, preserved unchanged so legacy
    //! .bg files keep decoding byte-identically.

    use crate::block::Block;
    use crate::zigzag::ZIGZAG;

    // ---------------------------------------------------------------------------
    // Standard JPEG Huffman tables (ISO 10918-1 Annex K)
    // Each entry: (code_length_bits, code_value)
    // ---------------------------------------------------------------------------

    static LUMA_DC_TABLE: &[(u8, u16)] = &[
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

    static CHROMA_DC_TABLE: &[(u8, u16)] = &[
        (2, 0b00),
        (2, 0b01),
        (2, 0b10),
        (3, 0b110),
        (4, 0b1110),
        (5, 0b11110),
        (6, 0b111110),
        (7, 0b1111110),
        (8, 0b11111110),
        (9, 0b111111110),
        (10,0b1111111110),
        (11,0b11111111110),
    ];

    /// AC table indexed by symbol byte: high nibble = run (0–15), low nibble = category (1–10).
    /// Symbol 0x00 = EOB, 0xF0 = ZRL. Entry (0,0) means "not in table".
    type AcTable = [(u8, u16); 256];

    use crate::jpeg_luma_ac_ht::JPEG_LUMA_AC_HT;
    use std::sync::OnceLock;
    const FAST_BITS: u8 = 14;
    type FastEntry = (i16, u8); // (sym, bits_len), sym<0 means miss

    /// stb_image_write / ISO 10918-1 Annex K luminance AC Huffman (full 256-entry encode table).
    fn ac_table_from_stb_ht(ht: &[[u16; 2]; 256]) -> AcTable {
        let mut t = [(0u8, 0u16); 256];
        for i in 0..256 {
            let code = ht[i][0];
            let len = ht[i][1];
            if len > 0 {
                t[i] = (len as u8, code);
            }
        }
        t
    }

    /// Full JPEG luminance AC table. Used for **every** plane in .bg v4/v5 (Y, Cb, Cr, A) so the
    /// on-wire format stays compatible with earlier bitgrain, which always applied the luma AC spec
    /// to chroma too (not the separate chrominance AC table).
    static JPEG_AC_TABLE: OnceLock<AcTable> = OnceLock::new();
    static JPEG_CHROMA_AC_TABLE: OnceLock<AcTable> = OnceLock::new();

    fn jpeg_ac_table() -> &'static AcTable {
        JPEG_AC_TABLE.get_or_init(|| ac_table_from_stb_ht(&JPEG_LUMA_AC_HT))
    }

    fn ac_table_from_canonical(nrcodes: &[u8; 17], values: &[u8]) -> AcTable {
        let mut t = [(0u8, 0u16); 256];
        let mut code: u16 = 0;
        let mut k = 0usize;
        for len in 1..=16usize {
            let n = nrcodes[len] as usize;
            for _ in 0..n {
                let sym = values[k] as usize;
                t[sym] = (len as u8, code);
                code = code.wrapping_add(1);
                k += 1;
            }
            code <<= 1;
        }
        t
    }

    fn jpeg_chroma_ac_table() -> &'static AcTable {
        JPEG_CHROMA_AC_TABLE.get_or_init(|| {
            // JPEG Annex K / stb_image_write chrominance AC Huffman table.
            const NR_CODES: [u8; 17] = [0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
            const VALUES: [u8; 162] = [
                0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
                0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
                0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
                0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
                0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
                0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
                0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
                0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
                0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
                0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
                0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
                0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
                0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
                0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
                0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
                0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
                0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
                0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
                0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
                0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
                0xf9, 0xfa,
            ];
            ac_table_from_canonical(&NR_CODES, &VALUES)
        })
    }

    #[derive(Clone, Copy)]
    struct DecodeNode {
        child: [i16; 2],
        sym: i16,
    }

    impl DecodeNode {
        #[inline]
        fn new() -> Self {
            Self { child: [-1, -1], sym: -1 }
        }
    }

    struct DecodeTree {
        nodes: Vec<DecodeNode>,
        fast: [FastEntry; 1 << FAST_BITS],
    }

    impl DecodeTree {
        fn with_root() -> Self {
            Self {
                nodes: vec![DecodeNode::new()],
                fast: [(-1, 0); 1 << FAST_BITS],
            }
        }

        fn insert(&mut self, code: u16, len: u8, sym: u8) -> bool {
            if len == 0 || len > 16 {
                return false;
            }
            let mut idx = 0usize;
            for shift in (0..len).rev() {
                // A leaf cannot be extended.
                if self.nodes[idx].sym >= 0 {
                    return false;
                }
                let bit = ((code >> shift) & 1) as usize;
                let next = self.nodes[idx].child[bit];
                let next_idx = if next >= 0 {
                    next as usize
                } else {
                    self.nodes.push(DecodeNode::new());
                    let ni = (self.nodes.len() - 1) as i16;
                    self.nodes[idx].child[bit] = ni;
                    ni as usize
                };
                idx = next_idx;
            }
            // A node with children cannot become a leaf; an existing leaf cannot be overwritten.
            if self.nodes[idx].sym >= 0 || self.nodes[idx].child[0] >= 0 || self.nodes[idx].child[1] >= 0 {
                return false;
            }
            self.nodes[idx].sym = sym as i16;

            // Build a small prefix LUT for fast symbol decode.
            if len <= FAST_BITS {
                let shift = (FAST_BITS - len) as usize;
                let base = (code as usize) << shift;
                let reps = 1usize << shift;
                for i in 0..reps {
                    self.fast[base + i] = (sym as i16, len);
                }
            }
            true
        }
    }

    static LUMA_DC_TREE: OnceLock<DecodeTree> = OnceLock::new();
    static CHROMA_DC_TREE: OnceLock<DecodeTree> = OnceLock::new();
    static AC_TREE_LUMA: OnceLock<DecodeTree> = OnceLock::new();
    static AC_TREE_CHROMA: OnceLock<DecodeTree> = OnceLock::new();

    fn build_dc_tree(table: &[(u8, u16)]) -> DecodeTree {
        let mut t = DecodeTree::with_root();
        for (sym, &(len, code)) in table.iter().enumerate() {
            assert!(t.insert(code, len, sym as u8), "invalid DC Huffman table");
        }
        t
    }

    fn build_ac_tree(table: &AcTable) -> DecodeTree {
        let mut t = DecodeTree::with_root();
        for (sym, &(len, code)) in table.iter().enumerate() {
            if len > 0 {
                assert!(t.insert(code, len, sym as u8), "invalid AC Huffman table");
            }
        }
        t
    }

    #[inline]
    fn luma_dc_tree() -> &'static DecodeTree {
        LUMA_DC_TREE.get_or_init(|| build_dc_tree(LUMA_DC_TABLE))
    }

    #[inline]
    fn chroma_dc_tree() -> &'static DecodeTree {
        CHROMA_DC_TREE.get_or_init(|| build_dc_tree(CHROMA_DC_TABLE))
    }

    #[inline]
    fn ac_tree(use_chroma_ac: bool) -> &'static DecodeTree {
        if use_chroma_ac {
            AC_TREE_CHROMA.get_or_init(|| build_ac_tree(jpeg_chroma_ac_table()))
        } else {
            AC_TREE_LUMA.get_or_init(|| build_ac_tree(jpeg_ac_table()))
        }
    }

    // ---------------------------------------------------------------------------
    // Magnitude helpers
    // ---------------------------------------------------------------------------

    #[inline]
    fn category(v: i16) -> u8 {
        if v == 0 { return 0; }
        (16 - v.unsigned_abs().leading_zeros()) as u8
    }

    #[inline]
    fn magnitude_bits(v: i16, cat: u8) -> u16 {
        if v >= 0 { v as u16 } else { ((1u16 << cat) - 1).wrapping_add(v as u16) }
    }

    #[inline]
    fn magnitude_decode(bits: u16, cat: u8) -> i16 {
        let threshold = 1u16 << (cat - 1);
        if bits >= threshold { bits as i16 } else { bits as i16 - (1i16 << cat) + 1 }
    }

    /// JPEG-style DC/AC Huffman tables only cover DC category ≤11 and AC category ≤10.
    /// Without this clamp, `encode_plane` can hit the `al == 0` fallback and emit a bitstream
    /// the decoder cannot parse (decode returns `None` / CLI reports corrupt .bg).
    pub(crate) fn clamp_block_jpeg_coeffs(block: &mut Block) {
        const DC_MAX: i16 = 2047; // category 11 magnitude
        const AC_MAX: i16 = 1023; // category 10 magnitude
        block.data[ZIGZAG[0]] = block.data[ZIGZAG[0]].clamp(-DC_MAX, DC_MAX);
        for i in 1..64 {
            let z = ZIGZAG[i];
            block.data[z] = block.data[z].clamp(-AC_MAX, AC_MAX);
        }
    }

    // ---------------------------------------------------------------------------
    // Bit writer — continuous across blocks, flush once per plane
    // ---------------------------------------------------------------------------

    pub struct BitWriter {
        pub buf:     Vec<u8>,
        bit_buf:     u64,
        bits_in:     u8,
    }

    impl BitWriter {
        pub fn new() -> Self {
            Self { buf: Vec::with_capacity(4096), bit_buf: 0, bits_in: 0 }
        }

        #[inline]
        fn push_entropy_byte(&mut self, byte: u8) {
            self.buf.push(byte);
            if byte == 0xFF {
                self.buf.push(0x00);
            }
        }

        #[inline]
        pub fn write_bits(&mut self, code: u16, n: u8) {
            if n == 0 { return; }
            let mask = (1u64 << n) - 1;
            self.bit_buf = (self.bit_buf << n) | ((code as u64) & mask);
            self.bits_in += n;

            while self.bits_in >= 8 {
                let shift = self.bits_in - 8;
                let byte = (self.bit_buf >> shift) as u8;
                self.push_entropy_byte(byte);
                self.bits_in -= 8;
                if self.bits_in == 0 {
                    self.bit_buf = 0;
                } else {
                    self.bit_buf &= (1u64 << self.bits_in) - 1;
                }
            }
        }

        /// Flush remaining bits, padding with 1s (JPEG convention). Call once per plane.
        pub fn flush(&mut self) {
            if self.bits_in > 0 {
                let pad = 8 - self.bits_in;
                self.write_bits((1u16 << pad) - 1, pad);
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Bit reader — continuous across blocks
    // ---------------------------------------------------------------------------

    pub struct BitReader<'a> {
        buf:     &'a [u8],
        pos:     usize,
        bit_buf: u64,
        bits_in: u8,
    }

    impl<'a> BitReader<'a> {
        pub fn new(buf: &'a [u8], start: usize) -> Self {
            Self { buf, pos: start, bit_buf: 0, bits_in: 0 }
        }

        #[inline]
        fn refill(&mut self) {
            while self.bits_in <= 56 && self.pos < self.buf.len() {
                let byte = self.buf[self.pos];
                self.pos += 1;
                // JPEG byte-unstuffing
                if byte == 0xFF && self.pos < self.buf.len() && self.buf[self.pos] == 0x00 {
                    self.pos += 1;
                }
                self.bit_buf = (self.bit_buf << 8) | byte as u64;
                self.bits_in += 8;
            }
        }

        #[inline]
        fn ensure_bits(&mut self, n: u8) -> bool {
            if self.bits_in < n {
                self.refill();
            }
            self.bits_in >= n
        }

        #[inline]
        pub fn peek_bits(&mut self, n: u8) -> Option<u16> {
            if n == 0 {
                return Some(0);
            }
            if !self.ensure_bits(n) {
                return None;
            }
            let shift = self.bits_in - n;
            Some(((self.bit_buf >> shift) & ((1u64 << n) - 1)) as u16)
        }

        #[inline]
        pub fn drop_bits(&mut self, n: u8) -> bool {
            if n == 0 {
                return true;
            }
            if !self.ensure_bits(n) {
                return false;
            }
            self.bits_in -= n;
            if self.bits_in == 0 {
                self.bit_buf = 0;
            } else {
                self.bit_buf &= (1u64 << self.bits_in) - 1;
            }
            true
        }

        #[inline]
        pub fn read_bits(&mut self, n: u8) -> Option<u16> {
            if n == 0 { return Some(0); }
            if n == 1 {
                return self.read_one_bit().map(|b| b as u16);
            }
            if !self.ensure_bits(n) {
                return None;
            }
            let shift = self.bits_in - n;
            let out = ((self.bit_buf >> shift) & ((1u64 << n) - 1)) as u16;
            self.bits_in -= n;
            if self.bits_in == 0 {
                self.bit_buf = 0;
            } else {
                self.bit_buf &= (1u64 << self.bits_in) - 1;
            }
            Some(out)
        }

        #[inline]
        pub fn read_one_bit(&mut self) -> Option<u8> {
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
            }
            Some(bit)
        }

        pub fn byte_position(&self) -> usize { self.pos }
    }

    // ---------------------------------------------------------------------------
    // Huffman symbol decode via prebuilt binary tree
    // ---------------------------------------------------------------------------

    fn decode_sym(reader: &mut BitReader, tree: &DecodeTree) -> Option<u8> {
        if reader.ensure_bits(FAST_BITS) {
            let shift = reader.bits_in - FAST_BITS;
            let prefix = ((reader.bit_buf >> shift) & ((1u64 << FAST_BITS) - 1)) as usize;
            let (sym, len) = tree.fast[prefix as usize];
            if sym >= 0 {
                reader.bits_in -= len;
                if reader.bits_in == 0 {
                    reader.bit_buf = 0;
                } else {
                    reader.bit_buf &= (1u64 << reader.bits_in) - 1;
                }
                return Some(sym as u8);
            }
        }

        // Branch-reduced tree walk when we have enough buffered bits.
        if reader.ensure_bits(16) {
            let bits = reader.bit_buf;
            let mut idx = 0usize;
            for depth in 1u8..=16 {
                let bit_shift = reader.bits_in - depth;
                let bit = ((bits >> bit_shift) & 1) as usize;
                let next = tree.nodes[idx].child[bit];
                if next < 0 {
                    return None;
                }
                idx = next as usize;
                let sym = tree.nodes[idx].sym;
                if sym >= 0 {
                    reader.bits_in -= depth;
                    if reader.bits_in == 0 {
                        reader.bit_buf = 0;
                    } else {
                        reader.bit_buf &= (1u64 << reader.bits_in) - 1;
                    }
                    return Some(sym as u8);
                }
            }
            return None;
        }

        let mut idx = 0usize;
        for _ in 0..16 {
            let bit = reader.read_one_bit()? as usize;
            let next = tree.nodes[idx].child[bit];
            if next < 0 {
                return None;
            }
            idx = next as usize;
            let sym = tree.nodes[idx].sym;
            if sym >= 0 {
                return Some(sym as u8);
            }
        }
        None
    }

    // ---------------------------------------------------------------------------
    // Encode / decode a full plane of blocks
    // ---------------------------------------------------------------------------

    /// Encode all blocks of a plane into a Vec<u8> prefixed with its byte length (u32 LE).
    /// Format: [len: u32 LE][bitstream bytes]
    /// The length prefix allows the decoder to skip exactly to the next plane boundary.
    pub fn encode_plane(blocks: &[Block], is_chroma: bool) -> Vec<u8> {
        encode_plane_with_profile(blocks, is_chroma, false, false)
    }

    pub fn encode_plane_with_ac(blocks: &[Block], is_chroma: bool, use_chroma_ac: bool) -> Vec<u8> {
        encode_plane_with_profile(blocks, is_chroma, use_chroma_ac, false)
    }

    pub fn encode_plane_with_profile(
        blocks: &[Block],
        is_chroma: bool,
        use_chroma_ac: bool,
        use_dc_delta: bool,
    ) -> Vec<u8> {
        let dc_table = if is_chroma { CHROMA_DC_TABLE } else { LUMA_DC_TABLE };
        let ac_table = if use_chroma_ac { jpeg_chroma_ac_table() } else { jpeg_ac_table() };
        let eob = ac_table[0x00];
        let zrl = ac_table[0xF0];
        let mut w = BitWriter::new();
        let mut prev_dc: i16 = 0;

        for block in blocks {
            // DC
            let dc_val = block.data[ZIGZAG[0]];
            let dc_emit = if use_dc_delta {
                let d = dc_val.wrapping_sub(prev_dc);
                prev_dc = dc_val;
                d
            } else {
                dc_val
            };
            let dc_cat = category(dc_emit);
            let (dc_len, dc_code) = dc_table[dc_cat as usize];
            w.write_bits(dc_code, dc_len);
            if dc_cat > 0 { w.write_bits(magnitude_bits(dc_emit, dc_cat), dc_cat); }

            // AC (JPEG-style optimization): stop at last non-zero and emit EOB only when needed.
            let mut last_nz = 63usize;
            while last_nz > 0 && block.data[ZIGZAG[last_nz]] == 0 {
                last_nz -= 1;
            }
            if last_nz == 0 {
                w.write_bits(eob.1, eob.0);
                continue;
            }

            let mut zero_run: u8 = 0;
            for i in 1..=last_nz {
                let val = block.data[ZIGZAG[i]];
                if val == 0 {
                    zero_run += 1;
                    continue;
                }
                while zero_run >= 16 {
                    w.write_bits(zrl.1, zrl.0);
                    zero_run -= 16;
                }
                let cat = category(val);
                let sym = (zero_run << 4) | cat;
                let (al, ac) = ac_table[sym as usize];
                debug_assert!(al > 0, "missing AC Huffman for RS {sym:#04x} (run={zero_run} cat={cat})");
                if al == 0 {
                    // Defensive fallback for unexpected table/profile mismatch.
                    let (el, ec) = ac_table[0x00];
                    w.write_bits(ec, el);
                    break;
                }
                w.write_bits(ac, al);
                w.write_bits(magnitude_bits(val, cat), cat);
                zero_run = 0;
            }
            w.write_bits(eob.1, eob.0);
        }
        w.flush();

        // Prepend 4-byte length so decoder can skip exactly to next plane
        let data = w.buf;
        let len = data.len() as u32;
        let mut out = Vec::with_capacity(4 + data.len());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&data);
        out
    }

    /// Decode a plane of `n_blocks` blocks from `buf[start..]`.
    /// Reads the 4-byte length prefix, then decodes exactly that many bytes.
    /// Returns (blocks, new_byte_position) or None on error.
    pub fn decode_plane(buf: &[u8], start: usize, n_blocks: usize, is_chroma: bool) -> Option<(Vec<Block>, usize)> {
        decode_plane_with_profile(buf, start, n_blocks, is_chroma, false, false)
    }

    pub fn decode_plane_with_ac(
        buf: &[u8],
        start: usize,
        n_blocks: usize,
        is_chroma: bool,
        use_chroma_ac: bool,
    ) -> Option<(Vec<Block>, usize)> {
        decode_plane_with_profile(buf, start, n_blocks, is_chroma, use_chroma_ac, false)
    }

    pub fn decode_plane_with_profile(
        buf: &[u8],
        start: usize,
        n_blocks: usize,
        is_chroma: bool,
        use_chroma_ac: bool,
        use_dc_delta: bool,
    ) -> Option<(Vec<Block>, usize)> {
        if start + 4 > buf.len() { return None; }
        let plane_len = u32::from_le_bytes(buf[start..start+4].try_into().unwrap()) as usize;
        let data_start = start + 4;
        let data_end   = data_start + plane_len;
        if data_end > buf.len() {
            return None;
        }

        let dc_tree = if is_chroma { chroma_dc_tree() } else { luma_dc_tree() };
        let ac_tree = ac_tree(use_chroma_ac);
        let data = &buf[data_start..data_end];

        let mut reader = BitReader::new(data, 0);
        let blocks = decode_plane_blocks(&mut reader, n_blocks, dc_tree, ac_tree, use_dc_delta)?;

        // Return data_end as the next byte position (exact plane boundary)
        Some((blocks, data_end))
    }

    fn decode_plane_blocks(
        reader: &mut BitReader,
        n_blocks: usize,
        dc_tree: &DecodeTree,
        ac_tree: &DecodeTree,
        use_dc_delta: bool,
    ) -> Option<Vec<Block>> {
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut prev_dc: i16 = 0;
        for _bi in 0..n_blocks {
            let mut block = Block::new();

            // DC
            let dc_cat = match decode_sym(reader, dc_tree) {
                Some(c) => c,
                None => return None,
            };
            let dc_diff = if dc_cat == 0 {
                0i16
            } else {
                let bits = match reader.read_bits(dc_cat) {
                    Some(b) => b,
                    None => return None,
                };
                magnitude_decode(bits, dc_cat)
            };
            let dc_val = if use_dc_delta {
                let v = prev_dc.wrapping_add(dc_diff);
                prev_dc = v;
                v
            } else {
                dc_diff
            };
            block.data[ZIGZAG[0]] = dc_val;

            // AC
            let mut ac_idx = 1usize;
            loop {
                let sym = match decode_sym(reader, ac_tree) {
                    Some(s) => s,
                    None => return None,
                };
                if sym == 0x00 { break; } // EOB
                if sym == 0xF0 {
                    // ZRL: exactly 16 consecutive zeros.
                    if ac_idx + 16 > 64 { return None; }
                    ac_idx += 16;
                    continue;
                }
                let run = (sym >> 4) as usize;
                let cat = sym & 0x0F;
                // Need room for run zeros plus one non-zero coefficient.
                if ac_idx + run >= 64 { return None; }
                ac_idx += run;
                let bits = match reader.read_bits(cat) {
                    Some(b) => b,
                    None => return None,
                };
                block.data[ZIGZAG[ac_idx]] = magnitude_decode(bits, cat);
                ac_idx += 1;
                if ac_idx > 64 { return None; }
            }

            blocks.push(block);
        }

        Some(blocks)
    }

}

// ===========================================================================
// Range-coding entropy backend (frequency table + interval subdivision)
// ===========================================================================

pub mod rangecode {
    //! Range (arithmetic-style) entropy coder with explicit frequency tables.
    //!
    //! # Symbol alphabet
    //!
    //! Each plane is flattened to a `u32` symbol stream with a fixed alphabet:
    //!
    //! | Symbol range        | Meaning                                              |
    //! |---------------------|------------------------------------------------------|
    //! | `0..=4094`          | DC coefficient value `v + 2047` (`v` in -2047..=2047) |
    //! | `4096`              | EOB (end of block)                                   |
    //! | `4097..=4272`       | AC pair (run 0..=15, category 1..=10)                |
    //! | `4273`              | ZRL (run of 16 zeros)                                |
    //! | `4274..=6321`       | AC magnitude 1..=1023                               |
    //! | `6322`              | AC sign: next magnitude is negative                  |
    //!
    //! # Frequency tables
    //!
    //! Symbol frequencies are counted first, then normalized to a cumulative-frequency
    //! table whose total is at most [`TOTAL_FREQ_LIMIT`]. The encoder subdivides the
    //! current probability interval by `(cum, freq)`; the decoder reverse-looks-up the
    //! interval via binary search over the cumulative table. A symbol with probability 0
    //! can never be encoded or decoded: both directions report [`RangeCoderError`].
    //!
    //! Tables are rebuildable per block ([`TableScope::PerBlock`]), once per plane
    //! ([`TableScope::PerPlane`], the production default), or once for the whole symbol
    //! stream ([`TableScope::Global`]).
    //!
    //! # Wire format (inside the plane payload, after the 4-byte plane length prefix)
    //!
    //! ```text
    //! marker        : u32 LE = RC_MARKER | scope_id << 1 | dc_delta (bit 0)
    //! PerPlane/Global:
    //!   total_symbols : u32 LE
    //!   n_used        : u32 LE
    //!   entries[n]    : (symbol: u32 LE, freq: u16 LE)  -- sorted by symbol, freq > 0
    //!   coded bytes   : range-coder stream (Subbotin carry coder)
    //! PerBlock:
    //!   for each block: seg_len: u32 LE, then the same layout as above for one block
    //! ```

    use crate::block::Block;
    use crate::zigzag::ZIGZAG;

    // ---------------------------------------------------------------------------
    // Event alphabet (symbols are u32). Only structural events are arithmetic-coded;
    // coefficient magnitudes are carried as raw, unbiased bits through the same coder.
    // ---------------------------------------------------------------------------

    /// DC magnitude category 0..=16 (0 means a zero DC/delta). Delta coding can produce
    /// any i16 value, so the full 16-bit category range is needed, not just JPEG's 0..=11.
    const DC_CAT_BASE: u32 = 0;
    const DC_CAT_COUNT: u32 = 17;
    /// AC (run, category) events: run 0..=15, category 1..=10 (160 slots).
    const AC_PAIR_BASE: u32 = DC_CAT_BASE + DC_CAT_COUNT; // 17
    const AC_PAIR_COUNT: u32 = 16 * 10;
    const ZRL_SYMBOL: u32 = AC_PAIR_BASE + AC_PAIR_COUNT; // 177
    const EOB_SYMBOL: u32 = ZRL_SYMBOL + 1; // 178
    pub const ALPHABET_SIZE: usize = EOB_SYMBOL as usize + 1; // 179

    /// Cumulative frequency budget. Kept inside u16 so serialized frequencies stay 2 bytes
    /// and the normalization invariant `range >= total` always holds (range >= 1<<24).
    pub const TOTAL_FREQ_LIMIT: u32 = 65535;

    /// Marker prefix; low 3 bits carry scope id (bits 1..2) and the DC-delta flag (bit 0).
    /// 0x52434F50 in LE byte order is 52 43 4F 50 = "RCOP"; its low 3 bits are zero,
    /// leaving them free for the scope id (bits 1..2) and delta flag (bit 0).
    pub const RC_MARKER: u32 = 0x5243_4F50;
    pub const RC_MARKER_MASK: u32 = 0xFFFF_FFF8;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum TableScope {
        /// One frequency table rebuilt for every block.
        PerBlock = 2,
        /// One frequency table per plane (default; best table-overhead/compression trade-off).
        PerPlane = 0,
        /// One frequency table for the whole symbol stream.
        Global = 1,
    }

    /// Errors reported by the range-coding layer. Empty tables and zero-probability symbols
    /// are always rejected instead of silently producing an undecodable stream.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum RangeCoderError {
        /// No symbols were supplied: a frequency table cannot be built.
        EmptySymbolStream,
        /// The requested symbol has frequency 0 (never observed in the training stream).
        ZeroProbabilitySymbol(u32),
        /// A serialized/normalized frequency table is malformed (empty, zero entry, bad total).
        InvalidFrequencyTable,
        /// The coded stream ended before all symbols could be decoded.
        TruncatedStream,
        /// A decoded symbol fell outside the valid alphabet.
        SymbolOutOfRange,
    }

    impl core::fmt::Display for RangeCoderError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self {
                RangeCoderError::EmptySymbolStream => {
                    write!(f, "range coder: cannot build frequency table from an empty symbol stream")
                }
                RangeCoderError::ZeroProbabilitySymbol(s) => {
                    write!(f, "range coder: symbol {s} has probability 0")
                }
                RangeCoderError::InvalidFrequencyTable => {
                    write!(f, "range coder: invalid frequency table (empty or zero-probability entry)")
                }
                RangeCoderError::TruncatedStream => write!(f, "range coder: truncated coded stream"),
                RangeCoderError::SymbolOutOfRange => write!(f, "range coder: symbol out of alphabet range"),
            }
        }
    }

    impl std::error::Error for RangeCoderError {}

    // ---------------------------------------------------------------------------
    // Event mapping + JPEG-style magnitude bits
    // ---------------------------------------------------------------------------

    #[inline]
    fn category(v: i16) -> u8 {
        16 - v.unsigned_abs().leading_zeros() as u8
    }

    /// Event symbol for a DC category (0..=11).
    #[inline]
    fn dc_cat_symbol(cat: u8) -> u32 {
        DC_CAT_BASE + cat as u32
    }

    /// Event symbol for an AC (run, category) pair.
    #[inline]
    fn ac_pair_symbol(run: usize, cat: u8) -> u32 {
        AC_PAIR_BASE + (run as u32) * 10 + (cat as u32 - 1)
    }

    /// JPEG magnitude bits for a nonzero value (sign folded into the bit pattern).
    #[inline]
    fn magnitude_bits(v: i16, cat: u8) -> u32 {
        if v >= 0 {
            v as u32
        } else {
            (1u32 << cat) - 1 + (v as u32)
        }
    }

    /// Inverse of [`magnitude_bits`].
    #[inline]
    fn magnitude_decode(bits: u32, cat: u8) -> i16 {
        let threshold = 1u32 << (cat - 1);
        if bits >= threshold {
            bits as i16
        } else {
            bits as i16 - (1i16 << cat) + 1
        }
    }

    // ---------------------------------------------------------------------------
    // Frequency model
    // ---------------------------------------------------------------------------

    /// Cumulative-frequency model built from observed symbol counts.
    #[derive(Debug)]
    pub struct FreqModel {
        /// Cumulative start of every alphabet slot (length [`ALPHABET_SIZE`]).
        cum: Vec<u32>,
        /// Normalized frequency of every alphabet slot (0 for unseen symbols).
        freq: Vec<u32>,
        /// Symbols with freq > 0, sorted.
        used: Vec<u32>,
        /// Sum of all frequencies, 1..=TOTAL_FREQ_LIMIT.
        pub total: u32,
    }

    impl FreqModel {
        /// Count symbols and build the normalized cumulative-frequency table.
        ///
        /// Errors with [`RangeCoderError::EmptySymbolStream`] when `symbols` is empty,
        /// i.e. a frequency table can never be built from nothing.
        pub fn from_symbols(symbols: &[u32]) -> Result<Self, RangeCoderError> {
            if symbols.is_empty() {
                return Err(RangeCoderError::EmptySymbolStream);
            }
            let mut counts = vec![0u32; ALPHABET_SIZE];
            for &s in symbols {
                if (s as usize) >= ALPHABET_SIZE {
                    return Err(RangeCoderError::SymbolOutOfRange);
                }
                counts[s as usize] = counts[s as usize].saturating_add(1);
            }
            Self::from_counts(&counts)
        }

        /// Build from a dense per-symbol count array. Every positive entry must survive
        /// normalization with frequency >= 1 (zero-probability stays zero and is rejected
        /// at encode/decode time).
        pub fn from_counts(counts: &[u32]) -> Result<Self, RangeCoderError> {
            if counts.len() != ALPHABET_SIZE {
                return Err(RangeCoderError::InvalidFrequencyTable);
            }
            let raw_total: u64 = counts.iter().map(|&c| c as u64).sum();
            if raw_total == 0 {
                return Err(RangeCoderError::EmptySymbolStream);
            }

            let mut freq = vec![0u32; ALPHABET_SIZE];
            let mut used = Vec::new();
            // Always renormalize onto the full TOTAL_FREQ_LIMIT budget. Keeping the
            // raw (smaller) total would starve the interval arithmetic of precision:
            // the coder wastes roughly TOTAL_FREQ_LIMIT/raw_total bits per symbol, so a
            // short plane would expand instead of compress. Every observed symbol keeps
            // freq >= 1; rounding drift is corrected against the budget afterward.
            for (s, &c) in counts.iter().enumerate() {
                if c > 0 {
                    let scaled = ((c as u64 * TOTAL_FREQ_LIMIT as u64) / raw_total) as u32;
                    freq[s] = scaled.max(1);
                    used.push(s as u32);
                }
            }
            let sum: u32 = used.iter().map(|&s| freq[s as usize]).sum();
            if sum > TOTAL_FREQ_LIMIT {
                let mut excess = sum - TOTAL_FREQ_LIMIT;
                while excess > 0 {
                    // Pick the current largest-frequency entry that can still give up a unit.
                    let mut best = used[0] as usize;
                    for &s in used.iter().skip(1) {
                        let i = s as usize;
                        if freq[i] > freq[best] && freq[i] > 1 {
                            best = i;
                        }
                    }
                    if freq[best] <= 1 {
                        break; // cannot reduce further without zeroing a symbol
                    }
                    let take = excess.min(freq[best] - 1);
                    freq[best] -= take;
                    excess -= take;
                }
            } else if sum < TOTAL_FREQ_LIMIT {
                let mut deficit = TOTAL_FREQ_LIMIT - sum;
                // Distribute the rounding slack over the largest entries proportionally.
                while deficit > 0 {
                    let best = used
                        .iter()
                        .copied()
                        .max_by_key(|&s| freq[s as usize])
                        .unwrap() as usize;
                    freq[best] += 1;
                    deficit -= 1;
                }
            }

            let total: u32 = used.iter().map(|&s| freq[s as usize]).sum();
            if total == 0 || total > TOTAL_FREQ_LIMIT {
                return Err(RangeCoderError::InvalidFrequencyTable);
            }

            let mut cum = vec![0u32; ALPHABET_SIZE];
            let mut running = 0u32;
            for &s in &used {
                cum[s as usize] = running;
                running += freq[s as usize];
            }
            debug_assert_eq!(running, total);
            Ok(Self { cum, freq, used, total })
        }

        /// Serialize the table: `n_used: u32` then `(symbol: u32, freq: u16)` entries sorted
        /// by symbol. Six bytes per used symbol; unseen (zero-probability) symbols are omitted.
        pub fn serialize(&self) -> Vec<u8> {
            let mut out = Vec::with_capacity(4 + self.used.len() * 6);
            out.extend_from_slice(&(self.used.len() as u32).to_le_bytes());
            for &s in &self.used {
                out.extend_from_slice(&s.to_le_bytes());
                let f = self.freq[s as usize];
                debug_assert!(f > 0 && f <= TOTAL_FREQ_LIMIT);
                out.extend_from_slice(&(f as u16).to_le_bytes());
            }
            out
        }

        /// Rebuild a table from serialized entries. Rejects empty tables, duplicate/unsorted
        /// symbols, out-of-range symbols and zero frequencies.
        pub fn deserialize(data: &[u8]) -> Result<(Self, usize), RangeCoderError> {
            if data.len() < 4 {
                return Err(RangeCoderError::TruncatedStream);
            }
            let n_used = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
            if n_used == 0 {
                return Err(RangeCoderError::InvalidFrequencyTable);
            }
            let need = 4 + n_used.checked_mul(6).ok_or(RangeCoderError::InvalidFrequencyTable)?;
            if data.len() < need {
                return Err(RangeCoderError::TruncatedStream);
            }
            let mut freq = vec![0u32; ALPHABET_SIZE];
            let mut used = Vec::with_capacity(n_used);
            let mut prev: Option<u32> = None;
            for k in 0..n_used {
                let p = 4 + k * 6;
                let sym = u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]);
                let f = u16::from_le_bytes([data[p + 4], data[p + 5]]) as u32;
                if (sym as usize) >= ALPHABET_SIZE || f == 0 {
                    return Err(RangeCoderError::InvalidFrequencyTable);
                }
                if let Some(pr) = prev {
                    if sym <= pr {
                        return Err(RangeCoderError::InvalidFrequencyTable);
                    }
                }
                prev = Some(sym);
                freq[sym as usize] = f;
                used.push(sym);
            }
            let total: u32 = used.iter().map(|&s| freq[s as usize]).sum();
            if total == 0 || total > TOTAL_FREQ_LIMIT {
                return Err(RangeCoderError::InvalidFrequencyTable);
            }
            let mut cum = vec![0u32; ALPHABET_SIZE];
            let mut running = 0u32;
            for &s in &used {
                cum[s as usize] = running;
                running += freq[s as usize];
            }
            Ok((Self { cum, freq, used, total }, need))
        }

        #[inline]
        pub fn cumulative(&self, symbol: u32) -> Result<(u32, u32, u32), RangeCoderError> {
            let i = symbol as usize;
            if i >= ALPHABET_SIZE {
                return Err(RangeCoderError::SymbolOutOfRange);
            }
            if self.freq[i] == 0 {
                return Err(RangeCoderError::ZeroProbabilitySymbol(symbol));
            }
            Ok((self.cum[i], self.freq[i], self.total))
        }

        /// Inverse lookup: find the symbol whose cumulative interval contains `threshold`.
        pub fn decode_symbol_at(&self, threshold: u32) -> Result<u32, RangeCoderError> {
            if threshold >= self.total {
                return Err(RangeCoderError::InvalidFrequencyTable);
            }
            // Binary search for the last used symbol with cum <= threshold.
            let mut lo = 0usize;
            let mut hi = self.used.len();
            while lo < hi {
                let mid = (lo + hi) / 2;
                if self.cum[self.used[mid] as usize] <= threshold {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if lo == 0 {
                return Err(RangeCoderError::InvalidFrequencyTable);
            }
            let sym = self.used[lo - 1];
            let i = sym as usize;
            if threshold >= self.cum[i] + self.freq[i] {
                return Err(RangeCoderError::InvalidFrequencyTable);
            }
            Ok(sym)
        }

        pub fn used_count(&self) -> usize {
            self.used.len()
        }
    }

    // ---------------------------------------------------------------------------
    // Subbotin range encoder (carry-propagating, LZMA-SDK style)
    // ---------------------------------------------------------------------------

    const TOP: u32 = 1 << 24;

    pub struct RangeEncoder {
        low: u64,
        range: u32,
        cache: u8,
        cache_size: u64,
        out: Vec<u8>,
    }

    impl Default for RangeEncoder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl RangeEncoder {
        pub fn new() -> Self {
            Self { low: 0, range: 0xFFFF_FFFF, cache: 0, cache_size: 1, out: Vec::new() }
        }

        fn shift_low(&mut self) {
            let carry = (self.low >> 32) as u32;
            if carry != 0 || (self.low as u32) < 0xFF00_0000 {
                let mut temp = self.cache;
                loop {
                    self.out.push(temp.wrapping_add(carry as u8));
                    if self.cache_size <= 1 {
                        break;
                    }
                    self.cache_size -= 1;
                    temp = 0xFF;
                }
                self.cache = ((self.low >> 24) & 0xFF) as u8;
                self.cache_size = 1;
            } else {
                self.cache_size += 1;
            }
            self.low = ((self.low as u32) << 8) as u64;
        }

        /// Encode one symbol given its cumulative interval. Rejects zero-probability symbols.
        pub fn encode_symbol(
            &mut self,
            model: &FreqModel,
            symbol: u32,
        ) -> Result<(), RangeCoderError> {
            let (cum, freq, total) = model.cumulative(symbol)?;
            let r = self.range / total;
            self.low += cum as u64 * r as u64;
            self.range = r * freq;
            while self.range < TOP {
                self.range <<= 8;
                self.shift_low();
            }
            Ok(())
        }

        /// Write one unbiased raw bit by splitting the current interval in half.
        /// Magnitude bits are uniformly distributed, so a binary 50/50 subdivision is
        /// optimal and keeps them out of the frequency table entirely.
        #[inline]
        pub fn encode_bit(&mut self, bit: u32) {
            let half = self.range >> 1;
            if bit != 0 {
                self.low += half as u64;
                self.range -= half;
            } else {
                self.range = half;
            }
            while self.range < TOP {
                self.range <<= 8;
                self.shift_low();
            }
        }

        /// Write the low `n` bits of `value`, most-significant first (JPEG convention).
        #[inline]
        pub fn encode_bits(&mut self, value: u32, n: u8) {
            for shift in (0..n).rev() {
                self.encode_bit((value >> shift) & 1);
            }
        }

        /// Flush pending bytes. The five ShiftLow calls follow the LZMA-SDK convention:
        /// the first emits the dummy initialization byte the decoder's 5-byte preamble
        /// consumes, the rest encode the final low value and pad the tail.
        pub fn finish(mut self) -> Vec<u8> {
            for _ in 0..5 {
                self.shift_low();
            }
            debug_assert!(self.out.len() >= 5, "range encoder must flush a full decoder preamble");
            self.out
        }
    }

    // ---------------------------------------------------------------------------
    // Subbotin range decoder
    // ---------------------------------------------------------------------------

    pub struct RangeDecoder<'a> {
        code: u32,
        range: u32,
        buf: &'a [u8],
        pos: usize,
    }

    impl<'a> RangeDecoder<'a> {
        /// Reads the 5-byte preamble (first byte is the encoder's dummy initialization byte).
        pub fn new(buf: &'a [u8]) -> Result<Self, RangeCoderError> {
            if buf.len() < 5 {
                return Err(RangeCoderError::TruncatedStream);
            }
            let mut d = Self { code: 0, range: 0xFFFF_FFFF, buf, pos: 0 };
            for _ in 0..5 {
                d.code = d.code.wrapping_shl(8).wrapping_add(d.read_byte()? as u32);
            }
            Ok(d)
        }

        fn read_byte(&mut self) -> Result<u8, RangeCoderError> {
            if self.pos >= self.buf.len() {
                return Err(RangeCoderError::TruncatedStream);
            }
            let b = self.buf[self.pos];
            self.pos += 1;
            Ok(b)
        }

        fn normalize(&mut self) -> Result<(), RangeCoderError> {
            while self.range < TOP {
                self.range <<= 8;
                self.code = self
                    .code
                    .wrapping_shl(8)
                    .wrapping_add(self.read_byte()? as u32);
            }
            Ok(())
        }

        /// Decode one symbol and advance the interval.
        pub fn decode_symbol(&mut self, model: &FreqModel) -> Result<u32, RangeCoderError> {
            let r = self.range / model.total;
            let threshold = self.code / r;
            let sym = model.decode_symbol_at(threshold)?;
            let i = sym as usize;
            self.code = self.code.wrapping_sub(self.cum_of(model, i) * r);
            self.range = r * model.freq[i];
            self.normalize()?;
            Ok(sym)
        }

        #[inline]
        fn cum_of(&self, model: &FreqModel, idx: usize) -> u32 {
            model.cum[idx]
        }

        /// Read one unbiased raw bit (inverse of [`RangeEncoder::encode_bit`]).
        #[inline]
        pub fn decode_bit(&mut self) -> Result<u32, RangeCoderError> {
            let half = self.range >> 1;
            let bit = if self.code < half {
                self.range = half;
                0
            } else {
                self.code = self.code.wrapping_sub(half);
                self.range = self.range.wrapping_sub(half);
                1
            };
            self.normalize()?;
            Ok(bit)
        }

        /// Read `n` raw bits, most-significant first.
        #[inline]
        pub fn decode_bits(&mut self, n: u8) -> Result<u32, RangeCoderError> {
            let mut value = 0u32;
            for _ in 0..n {
                value = (value << 1) | self.decode_bit()?;
            }
            Ok(value)
        }

        pub fn bytes_consumed(&self) -> usize {
            self.pos
        }
    }

    // ---------------------------------------------------------------------------
    // Block <-> event stream (structural events only; magnitudes are raw bits)
    // ---------------------------------------------------------------------------

    /// Collect the arithmetic-coded structural events of one block, in wire order.
    /// Magnitude bits are emitted separately during the streaming coding pass.
    fn block_to_events(block: &Block, use_dc_delta: bool, prev_dc: &mut i16, events: &mut Vec<u32>) {
        // DC (optionally delta-coded, JPEG-style): one category event per block.
        let dc_val = block.data[ZIGZAG[0]];
        let dc_emit = if use_dc_delta {
            let d = dc_val.wrapping_sub(*prev_dc);
            *prev_dc = dc_val;
            d
        } else {
            dc_val
        };
        events.push(dc_cat_symbol(category(dc_emit)));

        // AC: stop at the last non-zero coefficient, mirroring the Huffman path.
        let mut last_nz = 63usize;
        while last_nz > 0 && block.data[ZIGZAG[last_nz]] == 0 {
            last_nz -= 1;
        }
        if last_nz == 0 {
            events.push(EOB_SYMBOL);
            return;
        }

        let mut zero_run: usize = 0;
        for i in 1..=last_nz {
            let val = block.data[ZIGZAG[i]];
            if val == 0 {
                zero_run += 1;
                continue;
            }
            while zero_run >= 16 {
                events.push(ZRL_SYMBOL);
                zero_run -= 16;
            }
            let cat = category(val);
            events.push(ac_pair_symbol(zero_run, cat));
            zero_run = 0;
        }
        events.push(EOB_SYMBOL);
    }

    /// Streaming encode of one block: events go through the frequency model, while each
    /// coefficient's magnitude bits are written raw immediately after its category event.
    fn encode_block_stream(
        block: &Block,
        use_dc_delta: bool,
        prev_dc: &mut i16,
        model: &FreqModel,
        enc: &mut RangeEncoder,
    ) -> Result<(), RangeCoderError> {
        // DC
        let dc_val = block.data[ZIGZAG[0]];
        let dc_emit = if use_dc_delta {
            let d = dc_val.wrapping_sub(*prev_dc);
            *prev_dc = dc_val;
            d
        } else {
            dc_val
        };
        let dc_cat = category(dc_emit);
        enc.encode_symbol(model, dc_cat_symbol(dc_cat))?;
        if dc_cat > 0 {
            enc.encode_bits(magnitude_bits(dc_emit, dc_cat), dc_cat);
        }

        // AC
        let mut last_nz = 63usize;
        while last_nz > 0 && block.data[ZIGZAG[last_nz]] == 0 {
            last_nz -= 1;
        }
        if last_nz == 0 {
            enc.encode_symbol(model, EOB_SYMBOL)?;
            return Ok(());
        }
        let mut zero_run: usize = 0;
        for i in 1..=last_nz {
            let val = block.data[ZIGZAG[i]];
            if val == 0 {
                zero_run += 1;
                continue;
            }
            while zero_run >= 16 {
                enc.encode_symbol(model, ZRL_SYMBOL)?;
                zero_run -= 16;
            }
            let cat = category(val);
            enc.encode_symbol(model, ac_pair_symbol(zero_run, cat))?;
            enc.encode_bits(magnitude_bits(val, cat), cat);
            zero_run = 0;
        }
        enc.encode_symbol(model, EOB_SYMBOL)?;
        Ok(())
    }

    /// Streaming decode of one block (inverse of [`encode_block_stream`]).
    fn decode_block_stream(
        dec: &mut RangeDecoder<'_>,
        model: &FreqModel,
        use_dc_delta: bool,
        prev_dc: &mut i16,
    ) -> Result<Block, RangeCoderError> {
        let mut block = Block::new();

        // DC category event + raw magnitude bits.
        let dc_ev = dec.decode_symbol(model)?;
        if dc_ev >= AC_PAIR_BASE {
            return Err(RangeCoderError::SymbolOutOfRange);
        }
        let dc_cat = dc_ev as u8;
        let dc_diff = if dc_cat == 0 {
            0i16
        } else {
            magnitude_decode(dec.decode_bits(dc_cat)?, dc_cat)
        };
        let dc_val = if use_dc_delta {
            let v = prev_dc.wrapping_add(dc_diff);
            *prev_dc = v;
            v
        } else {
            dc_diff
        };
        block.data[ZIGZAG[0]] = dc_val;

        // AC events until EOB.
        let mut ac_idx = 1usize;
        loop {
            let sym = dec.decode_symbol(model)?;
            if sym == EOB_SYMBOL {
                break;
            }
            if sym == ZRL_SYMBOL {
                if ac_idx + 16 > 64 {
                    return Err(RangeCoderError::SymbolOutOfRange);
                }
                ac_idx += 16;
                continue;
            }
            if (AC_PAIR_BASE..ZRL_SYMBOL).contains(&sym) {
                let rel = sym - AC_PAIR_BASE;
                let run = (rel / 10) as usize;
                let cat = (rel % 10) as u8 + 1;
                if ac_idx + run >= 64 {
                    return Err(RangeCoderError::SymbolOutOfRange);
                }
                ac_idx += run;
                let val = magnitude_decode(dec.decode_bits(cat)?, cat);
                block.data[ZIGZAG[ac_idx]] = val;
                ac_idx += 1;
                if ac_idx > 64 {
                    return Err(RangeCoderError::SymbolOutOfRange);
                }
                continue;
            }
            return Err(RangeCoderError::SymbolOutOfRange);
        }
        Ok(block)
    }

    // ---------------------------------------------------------------------------
    // Segment encode/decode (one event table over a block group)
    // ---------------------------------------------------------------------------

    struct Segment {
        table_bytes: Vec<u8>,
        coded_bytes: Vec<u8>,
    }

    // -----------------------------------------------------------------------
    // Adaptive, table-free model (Global scope). Both sides start from the same
    // uniform prior and update frequencies in lockstep after every event, so no
    // frequency table has to be stored in the stream at all.
    // -----------------------------------------------------------------------

    /// Re-normalization half-life for the adaptive model: when the running total reaches
    /// this many counts, every frequency is halved (floor, minimum 1), halving the total.
    const ADAPT_MAX_TOTAL: u32 = 1 << 14;

    struct AdaptiveModel {
        freq: Vec<u32>,
        cum: Vec<u32>,
        total: u32,
    }

    impl AdaptiveModel {
        /// Uniform prior: every alphabet event starts with one count, hence a nonzero
        /// probability — an event can never hit a zero-probability error in this mode.
        fn new() -> Self {
            let freq = vec![1u32; ALPHABET_SIZE];
            let total = ALPHABET_SIZE as u32;
            let mut cum = vec![0u32; ALPHABET_SIZE];
            let mut running = 0u32;
            for s in 0..ALPHABET_SIZE {
                cum[s] = running;
                running += 1;
            }
            Self { freq, cum, total }
        }

        #[inline]
        fn cumulative(&self, symbol: u32) -> (u32, u32, u32) {
            let i = symbol as usize;
            (self.cum[i], self.freq[i], self.total)
        }

        #[inline]
        fn decode_symbol_at(&self, threshold: u32) -> u32 {
            // Binary search for the last symbol with cum <= threshold.
            let mut lo = 0usize;
            let mut hi = ALPHABET_SIZE;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if self.cum[mid] <= threshold {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            (lo - 1) as u32
        }

        /// Record one occurrence and rebuild the cumulative table. When the running total
        /// grows past [`ADAPT_MAX_TOTAL`], halve all frequencies first (rescaling window).
        fn update(&mut self, symbol: u32) {
            if self.total >= ADAPT_MAX_TOTAL {
                let mut running = 0u32;
                for s in 0..ALPHABET_SIZE {
                    self.cum[s] = running;
                    self.freq[s] = (self.freq[s] >> 1).max(1);
                    running += self.freq[s];
                }
                self.total = running;
            }
            self.freq[symbol as usize] += 1;
            self.total += 1;
            // Rebuild cumulative table (alphabet is small: 179 entries).
            let mut running = 0u32;
            for s in 0..ALPHABET_SIZE {
                self.cum[s] = running;
                running += self.freq[s];
            }
        }
    }

    /// Encode one event through the adaptive model, then update it.
    #[inline]
    fn encode_adaptive_event(
        enc: &mut RangeEncoder,
        model: &mut AdaptiveModel,
        symbol: u32,
    ) {
        let (cum, freq, total) = model.cumulative(symbol);
        let r = enc.range / total;
        enc.low += cum as u64 * r as u64;
        enc.range = r * freq;
        while enc.range < TOP {
            enc.range <<= 8;
            enc.shift_low();
        }
        model.update(symbol);
    }

    /// Decode one event through the adaptive model, then update it.
    #[inline]
    fn decode_adaptive_event(
        dec: &mut RangeDecoder<'_>,
        model: &mut AdaptiveModel,
    ) -> Result<u32, RangeCoderError> {
        let r = dec.range / model.total;
        let threshold = dec.code / r;
        let sym = model.decode_symbol_at(threshold);
        let i = sym as usize;
        dec.code = dec.code.wrapping_sub(model.cum[i] * r);
        dec.range = r * model.freq[i];
        dec.normalize()?;
        model.update(sym);
        Ok(sym)
    }

    /// Streaming encode of one block against the adaptive model (events + raw mag bits).
    fn encode_block_adaptive(
        block: &Block,
        use_dc_delta: bool,
        prev_dc: &mut i16,
        model: &mut AdaptiveModel,
        enc: &mut RangeEncoder,
    ) {
        let dc_val = block.data[ZIGZAG[0]];
        let dc_emit = if use_dc_delta {
            let d = dc_val.wrapping_sub(*prev_dc);
            *prev_dc = dc_val;
            d
        } else {
            dc_val
        };
        let dc_cat = category(dc_emit);
        encode_adaptive_event(enc, model, dc_cat_symbol(dc_cat));
        if dc_cat > 0 {
            enc.encode_bits(magnitude_bits(dc_emit, dc_cat), dc_cat);
        }

        let mut last_nz = 63usize;
        while last_nz > 0 && block.data[ZIGZAG[last_nz]] == 0 {
            last_nz -= 1;
        }
        if last_nz == 0 {
            encode_adaptive_event(enc, model, EOB_SYMBOL);
            return;
        }
        let mut zero_run: usize = 0;
        for i in 1..=last_nz {
            let val = block.data[ZIGZAG[i]];
            if val == 0 {
                zero_run += 1;
                continue;
            }
            while zero_run >= 16 {
                encode_adaptive_event(enc, model, ZRL_SYMBOL);
                zero_run -= 16;
            }
            let cat = category(val);
            encode_adaptive_event(enc, model, ac_pair_symbol(zero_run, cat));
            enc.encode_bits(magnitude_bits(val, cat), cat);
            zero_run = 0;
        }
        encode_adaptive_event(enc, model, EOB_SYMBOL);
    }

    /// Streaming decode of one block against the adaptive model.
    fn decode_block_adaptive(
        dec: &mut RangeDecoder<'_>,
        model: &mut AdaptiveModel,
        use_dc_delta: bool,
        prev_dc: &mut i16,
    ) -> Result<Block, RangeCoderError> {
        let mut block = Block::new();
        let dc_ev = decode_adaptive_event(dec, model)?;
        if dc_ev >= AC_PAIR_BASE {
            return Err(RangeCoderError::SymbolOutOfRange);
        }
        let dc_cat = dc_ev as u8;
        let dc_diff = if dc_cat == 0 {
            0i16
        } else {
            magnitude_decode(dec.decode_bits(dc_cat)?, dc_cat)
        };
        let dc_val = if use_dc_delta {
            let v = prev_dc.wrapping_add(dc_diff);
            *prev_dc = v;
            v
        } else {
            dc_diff
        };
        block.data[ZIGZAG[0]] = dc_val;

        let mut ac_idx = 1usize;
        loop {
            let sym = decode_adaptive_event(dec, model)?;
            if sym == EOB_SYMBOL {
                break;
            }
            if sym == ZRL_SYMBOL {
                if ac_idx + 16 > 64 {
                    return Err(RangeCoderError::SymbolOutOfRange);
                }
                ac_idx += 16;
                continue;
            }
            if (AC_PAIR_BASE..ZRL_SYMBOL).contains(&sym) {
                let rel = sym - AC_PAIR_BASE;
                let run = (rel / 10) as usize;
                let cat = (rel % 10) as u8 + 1;
                if ac_idx + run >= 64 {
                    return Err(RangeCoderError::SymbolOutOfRange);
                }
                ac_idx += run;
                block.data[ZIGZAG[ac_idx]] = magnitude_decode(dec.decode_bits(cat)?, cat);
                ac_idx += 1;
                if ac_idx > 64 {
                    return Err(RangeCoderError::SymbolOutOfRange);
                }
                continue;
            }
            return Err(RangeCoderError::SymbolOutOfRange);
        }
        Ok(block)
    }

    /// Build the event table from all blocks in the group, then run the streaming coder.
    fn encode_segment(
        blocks: &[Block],
        use_dc_delta: bool,
    ) -> Result<Segment, RangeCoderError> {
        let mut events = Vec::new();
        let mut prev_dc = 0i16;
        for block in blocks {
            block_to_events(block, use_dc_delta, &mut prev_dc, &mut events);
        }
        let model = FreqModel::from_symbols(&events)?;
        let mut enc = RangeEncoder::new();
        let mut prev_dc = 0i16;
        for block in blocks {
            encode_block_stream(block, use_dc_delta, &mut prev_dc, &model, &mut enc)?;
        }
        Ok(Segment { table_bytes: model.serialize(), coded_bytes: enc.finish() })
    }

    fn segment_layout_size(seg: &Segment) -> usize {
        seg.table_bytes.len() + seg.coded_bytes.len()
    }

    fn write_segment(out: &mut Vec<u8>, seg: &Segment) {
        out.extend_from_slice(&seg.table_bytes);
        out.extend_from_slice(&seg.coded_bytes);
    }

    /// Decode the block group of a segment starting at `data[pos..]`.
    fn decode_segment(
        data: &[u8],
        mut pos: usize,
        n_blocks: usize,
        use_dc_delta: bool,
    ) -> Result<(Vec<Block>, usize), RangeCoderError> {
        let (model, table_len) = FreqModel::deserialize(&data[pos..])?;
        pos += table_len;
        if pos >= data.len() {
            return Err(RangeCoderError::TruncatedStream);
        }
        let mut dec = RangeDecoder::new(&data[pos..])?;
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut prev_dc = 0i16;
        for _ in 0..n_blocks {
            blocks.push(decode_block_stream(&mut dec, &model, use_dc_delta, &mut prev_dc)?);
        }
        Ok((blocks, pos + dec.bytes_consumed()))
    }

    // ---------------------------------------------------------------------------
    // Plane-level API
    // ---------------------------------------------------------------------------

    fn marker_word(scope: TableScope, use_dc_delta: bool) -> u32 {
        RC_MARKER | ((scope as u32) << 1) | (use_dc_delta as u32)
    }

    fn marker_delta_flag(word: u32) -> bool {
        word & 1 == 1
    }

    fn marker_scope(word: u32) -> Option<TableScope> {
        // Scope id occupies bits 1..2 (2 bits), delta is bit 0.
        match (word >> 1) & 0x3 {
            0 => Some(TableScope::PerPlane),
            1 => Some(TableScope::Global),
            2 => Some(TableScope::PerBlock),
            _ => None,
        }
    }

    /// Encode a plane with range coding. Mirrors the legacy plane framing:
    /// `[len: u32 LE][payload]` so planes remain independently skippable.
    pub fn encode_plane_range(
        blocks: &[Block],
        scope: TableScope,
        use_dc_delta: bool,
    ) -> Vec<u8> {
        let payload = encode_plane_payload(blocks, scope, use_dc_delta)
            .expect("range-coded plane encoding failed: invalid symbol stream");
        let mut out = Vec::with_capacity(4 + payload.len());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }

    fn encode_plane_payload(
        blocks: &[Block],
        scope: TableScope,
        use_dc_delta: bool,
    ) -> Result<Vec<u8>, RangeCoderError> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&marker_word(scope, use_dc_delta).to_le_bytes());

        match scope {
            TableScope::PerPlane => {
                // One serialized event table for the whole plane.
                let seg = encode_segment(blocks, use_dc_delta)?;
                write_segment(&mut payload, &seg);
            }
            TableScope::Global => {
                // Table-free adaptive model: only the coded bytes follow the marker.
                let mut model = AdaptiveModel::new();
                let mut enc = RangeEncoder::new();
                let mut prev_dc = 0i16;
                for block in blocks {
                    encode_block_adaptive(block, use_dc_delta, &mut prev_dc, &mut model, &mut enc);
                }
                payload.extend_from_slice(&enc.finish());
            }
            TableScope::PerBlock => {
                // A fresh table rebuilt for every block; each segment is length-framed.
                for block in blocks {
                    let seg = encode_segment(std::slice::from_ref(block), use_dc_delta)?;
                    payload.extend_from_slice(&(segment_layout_size(&seg) as u32).to_le_bytes());
                    write_segment(&mut payload, &seg);
                }
            }
        }
        Ok(payload)
    }

    /// Decode a range-coded plane from `buf[start..]`. Returns (blocks, exact plane end).
    pub fn decode_plane_range(
        buf: &[u8],
        start: usize,
        n_blocks: usize,
    ) -> Option<(Vec<Block>, usize)> {
        decode_plane_range_result(buf, start, n_blocks).ok()
    }

    fn decode_plane_range_result(
        buf: &[u8],
        start: usize,
        n_blocks: usize,
    ) -> Result<(Vec<Block>, usize), RangeCoderError> {
        if start + 4 > buf.len() {
            return Err(RangeCoderError::TruncatedStream);
        }
        let plane_len = u32::from_le_bytes(
            buf[start..start + 4].try_into().map_err(|_| RangeCoderError::TruncatedStream)?,
        ) as usize;
        let data_start = start + 4;
        let data_end = data_start
            .checked_add(plane_len)
            .ok_or(RangeCoderError::TruncatedStream)?;
        if data_end > buf.len() || plane_len < 4 {
            return Err(RangeCoderError::TruncatedStream);
        }
        let data = &buf[data_start..data_end];
        let word = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if word & RC_MARKER_MASK != RC_MARKER {
            return Err(RangeCoderError::InvalidFrequencyTable);
        }
        let scope = marker_scope(word).ok_or(RangeCoderError::InvalidFrequencyTable)?;
        let use_dc_delta = marker_delta_flag(word);

        let mut blocks = Vec::with_capacity(n_blocks);

        match scope {
            TableScope::PerPlane => {
                let (decoded, _consumed) = decode_segment(data, 4, n_blocks, use_dc_delta)?;
                blocks.extend(decoded);
            }
            TableScope::Global => {
                // Table-free adaptive stream: coded bytes start right after the marker.
                let mut dec = RangeDecoder::new(&data[4..])?;
                let mut model = AdaptiveModel::new();
                let mut prev_dc = 0i16;
                for _ in 0..n_blocks {
                    blocks.push(decode_block_adaptive(
                        &mut dec, &mut model, use_dc_delta, &mut prev_dc,
                    )?);
                }
            }
            TableScope::PerBlock => {
                let mut pos = 4usize;
                for _ in 0..n_blocks {
                    if pos + 4 > data.len() {
                        return Err(RangeCoderError::TruncatedStream);
                    }
                    let seg_len =
                        u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                            as usize;
                    pos += 4;
                    if seg_len == 0 || pos + seg_len > data.len() {
                        return Err(RangeCoderError::TruncatedStream);
                    }
                    let (decoded, consumed) = decode_segment(data, pos, 1, use_dc_delta)?;
                    if consumed != pos + seg_len {
                        return Err(RangeCoderError::InvalidFrequencyTable);
                    }
                    blocks.extend(decoded);
                    pos += seg_len;
                }
            }
        }

        if blocks.len() != n_blocks {
            return Err(RangeCoderError::InvalidFrequencyTable);
        }
        Ok((blocks, data_end))
    }

    // ---------------------------------------------------------------------------
    // Size accounting (used by benchmarks/reporting)
    // ---------------------------------------------------------------------------

    /// Payload size breakdown for a plane: total payload bytes, frequency-table bytes
    /// (marker plus serialized event tables), and pure coded bytes.
    pub fn plane_size_breakdown(
        blocks: &[Block],
        scope: TableScope,
        use_dc_delta: bool,
    ) -> (usize, usize, usize, usize) {
        let payload = encode_plane_payload(blocks, scope, use_dc_delta)
            .expect("range-coded plane encoding failed: invalid symbol stream");
        let word = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
        let sc = marker_scope(word).unwrap();
        let mut table_bytes = 4usize; // marker
        let mut coded_bytes = 0usize;
        match sc {
            TableScope::PerPlane => {
                // After marker: n_used(4) + n_used*6 table, then coded bytes.
                let n_used = u32::from_le_bytes([
                    payload[4], payload[5], payload[6], payload[7],
                ]) as usize;
                let tbl = 4 + n_used * 6;
                table_bytes += tbl;
                coded_bytes = payload.len() - table_bytes;
            }
            TableScope::Global => {
                // Table-free adaptive model: no serialized table, only coded bytes.
                coded_bytes = payload.len() - table_bytes;
            }
            TableScope::PerBlock => {
                let mut pos = 4usize;
                while pos < payload.len() {
                    let seg_len = u32::from_le_bytes([
                        payload[pos],
                        payload[pos + 1],
                        payload[pos + 2],
                        payload[pos + 3],
                    ]) as usize;
                    pos += 4;
                    table_bytes += 4; // segment length frame
                    // Inside segment: n_used(4) + n_used*6 table, then coded bytes.
                    let n_used = u32::from_le_bytes([
                        payload[pos], payload[pos + 1], payload[pos + 2], payload[pos + 3],
                    ]) as usize;
                    let tbl = 4 + n_used * 6;
                    table_bytes += tbl;
                    coded_bytes += seg_len - tbl;
                    pos += seg_len;
                }
            }
        }
        (payload.len(), table_bytes, coded_bytes, payload.len() - 4 /* excludes plane len prefix */)
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use super::*;

        fn lcg(state: &mut u64) -> u32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*state >> 32) as u32
        }

        #[test]
        fn range_coder_symbol_level_roundtrip() {
            let mut seed = 0xABCD_1234u64;
            let mut stream = Vec::new();
            for _ in 0..10_000 {
                // Skewed distribution: symbol 5 dominates, like EOB in real planes.
                let s = if lcg(&mut seed) % 10 < 7 {
                    5u32
                } else {
                    lcg(&mut seed) % (ALPHABET_SIZE as u32)
                };
                stream.push(s);
            }
            let model = FreqModel::from_symbols(&stream).unwrap();
            let mut enc = RangeEncoder::new();
            for &s in &stream {
                enc.encode_symbol(&model, s).unwrap();
            }
            let bytes = enc.finish();

            let mut dec = RangeDecoder::new(&bytes).unwrap();
            for &want in &stream {
                let got = dec.decode_symbol(&model).unwrap();
                assert_eq!(got, want);
            }
        }

        #[test]
        fn raw_bits_roundtrip_through_range_state() {
            let mut seed = 0x0BAD_F00Du64;
            let mut bits = Vec::new();
            for _ in 0..5_000 {
                let n = (lcg(&mut seed) % 11) as u8; // 0..=10
                let v = if n == 0 { 0 } else { lcg(&mut seed) & ((1u32 << n) - 1) };
                bits.push((v, n));
            }
            let mut enc = RangeEncoder::new();
            for &(v, n) in &bits {
                enc.encode_bits(v, n);
            }
            let bytes = enc.finish();
            let mut dec = RangeDecoder::new(&bytes).unwrap();
            for &(want, n) in &bits {
                assert_eq!(dec.decode_bits(n).unwrap(), want);
            }
        }

        #[test]
        fn empty_symbol_stream_is_rejected() {
            let err = FreqModel::from_symbols(&[]).unwrap_err();
            assert_eq!(err, RangeCoderError::EmptySymbolStream);
            let counts = vec![0u32; ALPHABET_SIZE];
            let err = FreqModel::from_counts(&counts).unwrap_err();
            assert_eq!(err, RangeCoderError::EmptySymbolStream);
        }

        #[test]
        fn zero_probability_symbol_is_rejected() {
            let model = FreqModel::from_symbols(&[1u32, 1, 2]).unwrap();
            let mut enc = RangeEncoder::new();
            let err = enc.encode_symbol(&model, 7).unwrap_err();
            assert_eq!(err, RangeCoderError::ZeroProbabilitySymbol(7));
            // Decoder-side: threshold outside the table is an error too.
            assert_eq!(
                model.decode_symbol_at(model.total).unwrap_err(),
                RangeCoderError::InvalidFrequencyTable
            );
        }

        #[test]
        fn serialized_table_rejects_zero_and_duplicate_entries() {
            let model = FreqModel::from_symbols(&[3u32, 3, 9]).unwrap();
            let mut bytes = model.serialize();
            // Corrupt the first frequency to zero. Layout: n_used(4) + sym(4) + freq(2);
            // the first entry's freq is at bytes 8..10.
            bytes[8] = 0;
            bytes[9] = 0;
            assert_eq!(
                FreqModel::deserialize(&bytes).unwrap_err(),
                RangeCoderError::InvalidFrequencyTable
            );
            // Empty table (n_used = 0)
            assert_eq!(
                FreqModel::deserialize(&[0, 0, 0, 0]).unwrap_err(),
                RangeCoderError::InvalidFrequencyTable
            );
        }

        #[test]
        fn normalization_keeps_every_observed_symbol_positive() {
            let mut seed = 0x9999u64;
            let mut stream = Vec::new();
            for _ in 0..200_000 {
                stream.push(lcg(&mut seed) % (ALPHABET_SIZE as u32));
            }
            let model = FreqModel::from_symbols(&stream).unwrap();
            assert!(model.total <= TOTAL_FREQ_LIMIT);
            for &s in &model.used {
                assert!(model.freq[s as usize] > 0);
            }
        }

        #[test]
        fn plane_roundtrip_per_plane_scope() {
            let blocks = random_blocks(300, 0x1111);
            let encoded = encode_plane_range(&blocks, TableScope::PerPlane, true);
            let (decoded, _) = decode_plane_range(&encoded, 0, blocks.len()).unwrap();
            assert_blocks_eq(&blocks, &decoded);
        }

        #[test]
        fn plane_roundtrip_global_scope() {
            let blocks = random_blocks(300, 0x2222);
            let encoded = encode_plane_range(&blocks, TableScope::Global, false);
            let (decoded, _) = decode_plane_range(&encoded, 0, blocks.len()).unwrap();
            assert_blocks_eq(&blocks, &decoded);
        }

        #[test]
        fn plane_roundtrip_per_block_scope() {
            let blocks = random_blocks(128, 0x3333);
            let encoded = encode_plane_range(&blocks, TableScope::PerBlock, true);
            let (decoded, end) = decode_plane_range(&encoded, 0, blocks.len()).unwrap();
            assert_eq!(end, encoded.len());
            assert_blocks_eq(&blocks, &decoded);
        }

        #[test]
        fn all_zero_and_single_coefficient_planes() {
            let zeros = vec![Block::new(); 16];
            let enc = encode_plane_range(&zeros, TableScope::PerPlane, false);
            let (dec, _) = decode_plane_range(&enc, 0, zeros.len()).unwrap();
            assert_blocks_eq(&zeros, &dec);

            let mut b = Block::new();
            b.data[ZIGZAG[63]] = -7;
            let one = vec![b];
            let enc = encode_plane_range(&one, TableScope::PerBlock, false);
            let (dec, _) = decode_plane_range(&enc, 0, 1).unwrap();
            assert_blocks_eq(&one, &dec);
        }

        #[test]
        fn truncated_stream_is_detected() {
            let blocks = random_blocks(8, 0x4444);
            let enc = encode_plane_range(&blocks, TableScope::PerPlane, false);
            let cut = enc.len() - 3;
            assert!(decode_plane_range(&enc[..cut], 0, blocks.len()).is_none());
        }

        fn random_blocks(n: usize, seed: u64) -> Vec<Block> {
            let mut state = seed;
            let mut blocks = Vec::with_capacity(n);
            for _ in 0..n {
                let mut b = Block::new();
                // Stay inside the production alphabet: DC is clamped to +-2047, AC to +-1023
                // by clamp_block_jpeg_coeffs before entropy coding.
                b.data[ZIGZAG[0]] = (lcg(&mut state) % 4095) as i16 - 2047;
                for zi in 1..64 {
                    let r = lcg(&mut state) % 100;
                    if r >= 55 {
                        let mag = (lcg(&mut state) % 1023) as i16 + 1;
                        b.data[ZIGZAG[zi]] = if lcg(&mut state) & 1 == 0 { mag } else { -mag };
                    }
                }
                blocks.push(b);
            }
            blocks
        }

        fn assert_blocks_eq(a: &[Block], b: &[Block]) {
            assert_eq!(a.len(), b.len());
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(x.data, y.data, "block {i} mismatch");
            }
        }
    }

}
