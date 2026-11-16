use std::panic::{catch_unwind, AssertUnwindSafe};
use std::cell::{Cell, RefCell};
use std::ffi::CString;
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};

extern "C" {
    pub fn quantize_block(
        block: *mut i16,
        table: *const i16,
    );
    pub fn dequantize_block(
        block: *mut i16,
        table: *const i16,
    );
    pub fn bitgrain_dct_block(block: *mut i16);
    pub fn bitgrain_idct_block(block: *mut i16);
}

static RAYON_THREADS_CONFIGURED: AtomicUsize = AtomicUsize::new(0);

const BITGRAIN_OK: i32 = 0;
const BITGRAIN_ERR_INVALID_ARG: i32 = 1;
const BITGRAIN_ERR_DECODE_FAILED: i32 = 2;
const BITGRAIN_ERR_THREAD_INIT: i32 = 3;
const BITGRAIN_ERR_BUFFER_TOO_SMALL: i32 = 4;
const BITGRAIN_ERR_PANIC: i32 = 100;

thread_local! {
    static LAST_ERROR_CODE: Cell<i32> = const { Cell::new(BITGRAIN_OK) };
    static LAST_ERROR_MSG: RefCell<CString> = RefCell::new(CString::new("ok").expect("static cstring"));
}

#[inline]
fn set_last_error(code: i32, msg: &'static str) {
    LAST_ERROR_CODE.with(|c| c.set(code));
    LAST_ERROR_MSG.with(|m| {
        *m.borrow_mut() = CString::new(msg).expect("static cstring");
    });
}

#[inline]
fn clear_last_error() {
    set_last_error(BITGRAIN_OK, "ok");
}

#[inline]
fn fail(code: i32, msg: &'static str) -> i32 {
    set_last_error(code, msg);
    -1
}

#[inline]
fn ffi_guard<F>(f: F) -> i32
where
    F: FnOnce() -> i32,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(code) => code,
        Err(_) => fail(BITGRAIN_ERR_PANIC, "panic in codec internals"),
    }
}

#[no_mangle]
pub extern "C" fn bitgrain_last_error_code() -> i32 {
    LAST_ERROR_CODE.with(|c| c.get())
}

#[no_mangle]
pub extern "C" fn bitgrain_last_error_message() -> *const i8 {
    LAST_ERROR_MSG.with(|m| m.borrow().as_c_str().as_ptr())
}

#[no_mangle]
pub extern "C" fn bitgrain_clear_error() {
    clear_last_error();
}

/// Configure global Rayon thread pool size.
/// Must be called before any parallel codec operation starts.
/// Returns 0 on success, -1 on invalid value or late/failed init.
#[no_mangle]
pub extern "C" fn bitgrain_set_threads(threads: i32) -> i32 {
    clear_last_error();
    if threads <= 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "threads must be > 0");
    }
    let t = threads as usize;
    let current = RAYON_THREADS_CONFIGURED.load(Ordering::Relaxed);
    if current == t {
        return 0;
    }
    if current != 0 && current != t {
        return fail(BITGRAIN_ERR_THREAD_INIT, "thread pool already configured with another value");
    }
    match rayon::ThreadPoolBuilder::new().num_threads(t).build_global() {
        Ok(()) => {
            RAYON_THREADS_CONFIGURED.store(t, Ordering::Relaxed);
            0
        }
        Err(_) => fail(BITGRAIN_ERR_THREAD_INIT, "failed to initialize global thread pool"),
    }
}

// ---------------------------------------------------------------------------
// Optional block-level CRC layer ("BGC1" container).
//
// Disabled by default: with the switch off, bitgrain_encode_* / bitgrain_decode*
// run the exact original code paths and produce byte-for-byte identical output.
// When enabled, bitgrain_encode_* splits the image into fixed-size pixel tiles
// ("blocks"), encodes every tile with the unmodified codec, and stores each
// tile payload with a CRC-32. bitgrain_decode* auto-detects the container,
// verifies every block, marks corrupt ones, and conceals them from the
// previous frame (if given) or from neighbouring good blocks. A verification
// report (total/bad blocks, bad positions) is kept per thread and can be
// fetched with bitgrain_last_verify_report_text().
//
// Switch sources (encode side), in priority order:
//   1. bitgrain_set_checksum_enabled(0/1)  — programmatic
//   2. BITGRAIN_CHECKSUMS=1|0              — environment (CLI use)
//   3. default: off
// Block size: BITGRAIN_CRC_TILE env (pixels, 8..=1024, default 32).
// Set BITGRAIN_CRC_REPORT=1 to print the verification report to stderr.
// ---------------------------------------------------------------------------

// Container layout (32-byte header, then per-block records):
//   0..2   "BG"                       (legacy magic — CLI sniffers accept it)
//   2      version: 1 (gray) / 18 (RGB) / 19 (RGBA)  (channel-correct)
//   3..7   width u32 LE               (real image dimensions)
//   7..11  height u32 LE
//   11     quality u8
//   12..16 "BGS1" container marker    (library detects this)
//   16..20 header CRC-32 over bytes 0..16 ++ 20..32
//   20..22 tile (block) size u16 LE, in pixels
//   22     channels (1/3/4)
//   23     flags (0)
//   24..32 reserved (0)
//   32..   per block: crc32 u32 LE, payload_len u32 LE, payload (a complete
//          .bg stream of that block, produced by the unmodified encoder)
const CONTAINER_MARKER: &[u8; 4] = b"BGS1";
const CONTAINER_MARKER_OFF: usize = 12;
const CONTAINER_HEADER_SIZE: usize = 32;
const DEFAULT_CRC_TILE: usize = 32;

const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

static CRC32_TABLE: [u32; 256] = crc32_table();

fn crc32(data: &[u8]) -> u32 {
    crc32_parts(&[data])
}

fn crc32_parts(parts: &[&[u8]]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for part in parts {
        for &b in *part {
            c = CRC32_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
        }
    }
    c ^ 0xFFFF_FFFF
}

fn container_header_crc(header: &[u8]) -> u32 {
    debug_assert!(header.len() >= CONTAINER_HEADER_SIZE);
    crc32_parts(&[&header[0..16], &header[20..CONTAINER_HEADER_SIZE]])
}

// 0 = follow env/default, 1 = forced on, 2 = forced off.
static CHECKSUM_SWITCH: AtomicUsize = AtomicUsize::new(0);

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn checksums_on() -> bool {
    match CHECKSUM_SWITCH.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => env_flag("BITGRAIN_CHECKSUMS"),
    }
}

fn crc_tile_size() -> usize {
    std::env::var("BITGRAIN_CRC_TILE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&t| (8..=1024).contains(&t))
        .unwrap_or(DEFAULT_CRC_TILE)
}

/// Enable/disable block-level CRC checksums for subsequent encodes.
/// Overrides the BITGRAIN_CHECKSUMS environment variable. Returns 0.
#[no_mangle]
pub extern "C" fn bitgrain_set_checksum_enabled(enabled: i32) -> i32 {
    CHECKSUM_SWITCH.store(if enabled != 0 { 1 } else { 2 }, Ordering::Relaxed);
    0
}

/// Returns 1 if block-level checksums are currently enabled, else 0.
#[no_mangle]
pub extern "C" fn bitgrain_checksum_enabled() -> i32 {
    if checksums_on() { 1 } else { 0 }
}

struct VerifyReport {
    protected: bool,
    header_ok: bool,
    total_blocks: u32,
    bad_blocks: u32,
    tiles_x: u32,
    tiles_y: u32,
    tile_size: u32,
    bad_positions: Vec<u32>, // block indices, scan order
    recovery: String,
    note: String,
}

impl VerifyReport {
    fn plain() -> Self {
        VerifyReport {
            protected: false,
            header_ok: true,
            total_blocks: 0,
            bad_blocks: 0,
            tiles_x: 0,
            tiles_y: 0,
            tile_size: 0,
            bad_positions: Vec::new(),
            recovery: String::new(),
            note: String::from("plain stream (checksums off)"),
        }
    }

    fn to_text(&self) -> String {
        if !self.protected {
            return format!("bitgrain-verify: protected=no {}", self.note);
        }
        let mut s = format!(
            "bitgrain-verify: protected=yes header={} blocks={} bad={}",
            if self.header_ok { "ok" } else { "CORRUPT" },
            self.total_blocks,
            self.bad_blocks
        );
        if self.header_ok {
            s.push_str(&format!(
                " grid={}x{} tile={}px",
                self.tiles_x, self.tiles_y, self.tile_size
            ));
        }
        if !self.bad_positions.is_empty() {
            let pos: Vec<String> = self
                .bad_positions
                .iter()
                .map(|&i| {
                    if self.tiles_x > 0 {
                        format!("{}(x={},y={})", i, i % self.tiles_x, i / self.tiles_x)
                    } else {
                        format!("{}", i)
                    }
                })
                .collect();
            s.push_str(&format!(" bad_positions=[{}]", pos.join(",")));
        }
        if !self.recovery.is_empty() {
            s.push_str(&format!(" recovery={}", self.recovery));
        }
        if !self.note.is_empty() {
            s.push_str(&format!(" note={}", self.note));
        }
        s
    }
}

thread_local! {
    static LAST_REPORT: RefCell<VerifyReport> = RefCell::new(VerifyReport::plain());
}

fn set_report(report: VerifyReport) {
    LAST_REPORT.with(|r| *r.borrow_mut() = report);
}

fn maybe_print_report() {
    if env_flag("BITGRAIN_CRC_REPORT") {
        LAST_REPORT.with(|r| eprintln!("{}", r.borrow().to_text()));
    }
}

/// Copy the last decode's verification report as NUL-terminated text.
/// Returns the text length (excluding NUL). If buf is NULL or cap is 0,
/// nothing is written and the required capacity (length + 1) is returned.
#[no_mangle]
pub extern "C" fn bitgrain_last_verify_report_text(buf: *mut u8, cap: u32) -> i32 {
    let text = LAST_REPORT.with(|r| r.borrow().to_text());
    let bytes = text.as_bytes();
    if buf.is_null() || cap == 0 {
        return bytes.len() as i32 + 1;
    }
    let n = bytes.len().min(cap as usize - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n);
        *buf.add(n) = 0;
    }
    n as i32
}

/// Encode `image` (channels = 1, 3 or 4) as a BGC1 checksummed container.
fn encode_container(
    image: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    quality: u8,
    out: &mut [u8],
    out_len: &mut i32,
) -> i32 {
    let tile = crc_tile_size();
    let tiles_x = width.div_ceil(tile);
    let tiles_y = height.div_ceil(tile);

    let mut container: Vec<u8> = Vec::with_capacity(image.len() / 2 + 4096);
    // Legacy-compatible 12-byte prefix so existing CLI sniffers parse the
    // real dimensions and channel count; the BGS1 marker follows at 12.
    let version: u8 = match channels {
        1 => 1,
        3 => 18,
        4 => 19,
        _ => return fail(BITGRAIN_ERR_INVALID_ARG, "unsupported channel count"),
    };
    container.extend_from_slice(b"BG");
    container.push(version);
    container.extend_from_slice(&(width as u32).to_le_bytes());
    container.extend_from_slice(&(height as u32).to_le_bytes());
    container.push(quality);
    container.extend_from_slice(CONTAINER_MARKER);
    container.extend_from_slice(&0u32.to_le_bytes()); // header CRC placeholder
    container.extend_from_slice(&(tile as u16).to_le_bytes());
    container.push(channels as u8);
    container.push(0); // flags
    container.extend_from_slice(&0u64.to_le_bytes()); // reserved
    debug_assert_eq!(container.len(), CONTAINER_HEADER_SIZE);
    let hcrc = container_header_crc(&container);
    container[16..20].copy_from_slice(&hcrc.to_le_bytes());

    let mut tile_px: Vec<u8> = Vec::new();
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x0 = tx * tile;
            let y0 = ty * tile;
            let tw = (width - x0).min(tile);
            let th = (height - y0).min(tile);
            tile_px.clear();
            tile_px.reserve(tw * th * channels);
            for row in 0..th {
                let start = ((y0 + row) * width + x0) * channels;
                tile_px.extend_from_slice(&image[start..start + tw * channels]);
            }
            // Same sizing rule the C CLI uses (raw*2 + margin), per tile.
            let mut buf = vec![0u8; tw * th * channels * 2 + 4096];
            let mut pos: i32 = 0;
            match channels {
                1 => crate::encoder::encode_grayscale(&tile_px, tw, th, quality, &mut buf, &mut pos),
                3 => crate::encoder::encode_rgb(&tile_px, tw, th, quality, &mut buf, &mut pos, None),
                4 => crate::encoder::encode_rgba(&tile_px, tw, th, quality, &mut buf, &mut pos, None),
                _ => return fail(BITGRAIN_ERR_INVALID_ARG, "unsupported channel count"),
            }
            let payload = &buf[..pos as usize];
            container.extend_from_slice(&crc32(payload).to_le_bytes());
            container.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            container.extend_from_slice(payload);
        }
    }

    if container.len() > out.len() {
        return fail(
            BITGRAIN_ERR_BUFFER_TOO_SMALL,
            "output buffer too small for checksummed stream",
        );
    }
    out[..container.len()].copy_from_slice(&container);
    *out_len = container.len() as i32;
    0
}

/// Copy tile `idx`'s pixel rect from `src` (a full w x h x ch image) into `dst`.
fn blit_tile(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    tile: usize,
    idx: usize,
) {
    let tiles_x = width.div_ceil(tile);
    let x0 = (idx % tiles_x) * tile;
    let y0 = (idx / tiles_x) * tile;
    let tw = (width - x0).min(tile);
    let th = (height - y0).min(tile);
    for row in 0..th {
        let off = ((y0 + row) * width + x0) * channels;
        let n = tw * channels;
        dst[off..off + n].copy_from_slice(&src[off..off + n]);
    }
}

/// Decode a BGC1 container: verify each block's CRC-32, decode good blocks,
/// conceal bad ones from `prev` (previous frame, w*h*channels pixels, may be
/// null) or neighbouring good blocks.
#[allow(clippy::too_many_arguments)]
fn decode_container(
    buf: &[u8],
    out: &mut [u8],
    out_width: &mut u32,
    out_height: &mut u32,
    out_channels: &mut u32,
    prev_pixels: *const u8,
) -> i32 {
    let mut report = VerifyReport {
        protected: true,
        ..VerifyReport::plain()
    };
    report.note.clear();

    let header_bad = |report: &mut VerifyReport| {
        report.header_ok = false;
    };

    if buf.len() < CONTAINER_HEADER_SIZE {
        header_bad(&mut report);
        report.note = String::from("truncated container header");
        set_report(report);
        maybe_print_report();
        return fail(BITGRAIN_ERR_DECODE_FAILED, "protected stream header truncated");
    }
    let stored_hcrc = u32::from_le_bytes(buf[16..20].try_into().unwrap());
    if container_header_crc(buf) != stored_hcrc {
        header_bad(&mut report);
        report.note = String::from("header CRC mismatch; block table not trustworthy");
        set_report(report);
        maybe_print_report();
        return fail(BITGRAIN_ERR_DECODE_FAILED, "protected stream header corrupt");
    }

    let width = u32::from_le_bytes(buf[3..7].try_into().unwrap()) as usize;
    let height = u32::from_le_bytes(buf[7..11].try_into().unwrap()) as usize;
    let channels = buf[22] as usize;
    let tile = u16::from_le_bytes(buf[20..22].try_into().unwrap()) as usize;
    if width == 0
        || height == 0
        || width > 65536
        || height > 65536
        || ![1usize, 3, 4].contains(&channels)
        || !(8..=1024).contains(&tile)
    {
        header_bad(&mut report);
        report.note = String::from("implausible header fields");
        set_report(report);
        maybe_print_report();
        return fail(BITGRAIN_ERR_DECODE_FAILED, "protected stream header invalid");
    }

    let tiles_x = width.div_ceil(tile);
    let tiles_y = height.div_ceil(tile);
    let total = tiles_x * tiles_y;
    report.total_blocks = total as u32;
    report.tiles_x = tiles_x as u32;
    report.tiles_y = tiles_y as u32;
    report.tile_size = tile as u32;

    if out.len() < width * height * channels {
        set_report(report);
        maybe_print_report();
        return fail(BITGRAIN_ERR_INVALID_ARG, "output buffer too small for image");
    }

    let mut good = vec![true; total];
    let mut pos = CONTAINER_HEADER_SIZE;
    let mut truncated = false;
    for idx in 0..total {
        let tx = idx % tiles_x;
        let ty = idx / tiles_x;
        let tw = (width - tx * tile).min(tile);
        let th = (height - ty * tile).min(tile);

        if pos + 8 > buf.len() {
            truncated = true;
        }
        let (stored_crc, payload) = if !truncated {
            let crc = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            let len = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap()) as usize;
            if pos + 8 + len > buf.len() {
                truncated = true;
                (0, &[][..])
            } else {
                let p = &buf[pos + 8..pos + 8 + len];
                pos += 8 + len;
                (crc, p)
            }
        } else {
            (0, &[][..])
        };

        let mut ok = !truncated && crc32(payload) == stored_crc;
        if ok {
            let mut tile_buf = vec![0u8; tw * th * channels];
            let (mut dw, mut dh, mut dch) = (0u32, 0u32, 0u32);
            ok = crate::decoder::decode(payload, &mut tile_buf, &mut dw, &mut dh, &mut dch, None)
                && dw as usize == tw
                && dh as usize == th
                && dch as usize == channels;
            if ok {
                for row in 0..th {
                    let dst_off = ((ty * tile + row) * width + tx * tile) * channels;
                    let src_off = row * tw * channels;
                    let n = tw * channels;
                    out[dst_off..dst_off + n].copy_from_slice(&tile_buf[src_off..src_off + n]);
                }
            }
        }
        if !ok {
            good[idx] = false;
            report.bad_positions.push(idx as u32);
        }
    }
    if truncated {
        report.note = String::from("stream truncated; missing blocks concealed");
    }

    // Conceal bad blocks: previous frame first, then nearest good neighbour
    // (up, left, right, down), then neutral gray.
    report.bad_blocks = report.bad_positions.len() as u32;
    if !report.bad_positions.is_empty() {
        // Contract: prev_pixels, when non-null, points to a decoded frame of
        // the same width/height/channels (w*h*channels bytes).
        let prev = if prev_pixels.is_null() {
            None
        } else {
            Some(unsafe { slice::from_raw_parts(prev_pixels, width * height * channels) })
        };
        let mut used_prev = false;
        let mut used_neighbor = false;
        let mut used_gray = false;
        let bad: Vec<usize> = report.bad_positions.iter().map(|&i| i as usize).collect();
        let mut rowbuf = vec![0u8; tile * channels];
        for &idx in &bad {
            if let Some(p) = prev {
                blit_tile(out, p, width, height, channels, tile, idx);
                used_prev = true;
                continue;
            }
            let tx = idx % tiles_x;
            let ty = idx / tiles_x;
            let mut donor = None;
            for (dx, dy) in [(0isize, -1isize), (-1, 0), (1, 0), (0, 1)] {
                let nx = tx as isize + dx;
                let ny = ty as isize + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < tiles_x && (ny as usize) < tiles_y {
                    let nidx = ny as usize * tiles_x + nx as usize;
                    if good[nidx] {
                        donor = Some(nidx);
                        break;
                    }
                }
            }
            match donor {
                Some(d) => {
                    // Conceal by copying the donor block's pixels into this
                    // block's position (edge-clamped, tail replicated).
                    let dst_x0 = tx * tile;
                    let dst_y0 = ty * tile;
                    let src_x0 = (d % tiles_x) * tile;
                    let src_y0 = (d / tiles_x) * tile;
                    let tw = (width - dst_x0).min(tile);
                    let th = (height - dst_y0).min(tile);
                    let n = tw * channels;
                    for row in 0..th {
                        let srow = (src_y0 + row).min(height - 1);
                        let s_off = (srow * width + src_x0.min(width - 1)) * channels;
                        let got = (s_off + n).min(out.len()) - s_off;
                        rowbuf[..got].copy_from_slice(&out[s_off..s_off + got]);
                        for k in got..n {
                            rowbuf[k] = rowbuf[got - 1];
                        }
                        let d_off = ((dst_y0 + row) * width + dst_x0) * channels;
                        out[d_off..d_off + n].copy_from_slice(&rowbuf[..n]);
                    }
                    used_neighbor = true;
                }
                None => {
                    let x0 = tx * tile;
                    let y0 = ty * tile;
                    let tw = (width - x0).min(tile);
                    let th = (height - y0).min(tile);
                    for row in 0..th {
                        let off = ((y0 + row) * width + x0) * channels;
                        out[off..off + tw * channels].fill(128);
                    }
                    used_gray = true;
                }
            }
        }
        let mut kinds = Vec::new();
        if used_prev {
            kinds.push("previous-frame");
        }
        if used_neighbor {
            kinds.push("neighbor-fill");
        }
        if used_gray {
            kinds.push("gray-fill");
        }
        report.recovery = kinds.join("+");
    }

    *out_width = width as u32;
    *out_height = height as u32;
    *out_channels = channels as u32;
    set_report(report);
    maybe_print_report();
    0
}

fn is_container(buf: &[u8]) -> bool {
    buf.len() >= CONTAINER_MARKER_OFF + 4
        && buf[0] == b'B'
        && buf[1] == b'G'
        && &buf[CONTAINER_MARKER_OFF..CONTAINER_MARKER_OFF + 4] == CONTAINER_MARKER
}

/// Encode grayscale image.
/// quality: 1–100 (higher = less quantization), 0 = default 85.
#[no_mangle]
pub extern "C" fn bitgrain_encode_grayscale(
    image: *const u8,
    width: u32,
    height: u32,
    out_buffer: *mut u8,
    out_capacity: u32,
    out_len: *mut i32,
    quality: u8,
) -> i32 {
    clear_last_error();
    if image.is_null() || out_buffer.is_null() || out_len.is_null() || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid encode_grayscale arguments");
    }
    ffi_guard(|| {
        let q = if quality == 0 { 85 } else { quality };
        let size = (width as usize).saturating_mul(height as usize);
        let image_slice = unsafe { slice::from_raw_parts(image, size) };
        let buffer_slice = unsafe { slice::from_raw_parts_mut(out_buffer, out_capacity as usize) };
        if checksums_on() {
            return encode_container(
                image_slice,
                width as usize,
                height as usize,
                1,
                q,
                buffer_slice,
                unsafe { &mut *out_len },
            );
        }
        let mut pos: i32 = 0;
        crate::encoder::encode_grayscale(
            image_slice,
            width as usize,
            height as usize,
            q,
            buffer_slice,
            &mut pos,
        );
        unsafe { *out_len = pos };
        0
    })
}

/// Encode an RGB image (24 bpp, R G B per pixel) to .bg stream.
/// quality: 1–100, 0 = default 85.
#[no_mangle]
pub extern "C" fn bitgrain_encode_rgb(
    image: *const u8,
    width: u32,
    height: u32,
    out_buffer: *mut u8,
    out_capacity: u32,
    out_len: *mut i32,
    quality: u8,
) -> i32 {
    clear_last_error();
    if image.is_null() || out_buffer.is_null() || out_len.is_null() || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid encode_rgb arguments");
    }
    ffi_guard(|| {
        let q = if quality == 0 { 85 } else { quality };
        let size = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(3);
        let image_slice = unsafe { slice::from_raw_parts(image, size) };
        let buffer_slice = unsafe { slice::from_raw_parts_mut(out_buffer, out_capacity as usize) };
        if checksums_on() {
            return encode_container(
                image_slice,
                width as usize,
                height as usize,
                3,
                q,
                buffer_slice,
                unsafe { &mut *out_len },
            );
        }
        let mut pos: i32 = 0;
        crate::encoder::encode_rgb(
            image_slice,
            width as usize,
            height as usize,
            q,
            buffer_slice,
            &mut pos,
            None,
        );
        unsafe { *out_len = pos };
        0
    })
}

/// Encode an RGBA image (32 bpp, R G B A per pixel) to .bg stream.
/// quality: 1–100, 0 = default 85.
#[no_mangle]
pub extern "C" fn bitgrain_encode_rgba(
    image: *const u8,
    width: u32,
    height: u32,
    out_buffer: *mut u8,
    out_capacity: u32,
    out_len: *mut i32,
    quality: u8,
) -> i32 {
    clear_last_error();
    if image.is_null() || out_buffer.is_null() || out_len.is_null() || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid encode_rgba arguments");
    }
    ffi_guard(|| {
        let q = if quality == 0 { 85 } else { quality };
        let size = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        let image_slice = unsafe { slice::from_raw_parts(image, size) };
        let buffer_slice = unsafe { slice::from_raw_parts_mut(out_buffer, out_capacity as usize) };
        if checksums_on() {
            return encode_container(
                image_slice,
                width as usize,
                height as usize,
                4,
                q,
                buffer_slice,
                unsafe { &mut *out_len },
            );
        }
        let mut pos: i32 = 0;
        crate::encoder::encode_rgba(
            image_slice,
            width as usize,
            height as usize,
            q,
            buffer_slice,
            &mut pos,
            None,
        );
        unsafe { *out_len = pos };
        0
    })
}

/// Decode a .bg stream into pixels (grayscale or RGB per header).
/// out_channels: output, 1 = grayscale (out_pixels = w*h), 3 = RGB (out_pixels = w*h*3).
/// out_capacity debe ser >= width*height*out_channels.
#[no_mangle]
pub extern "C" fn bitgrain_decode(
    buffer: *const u8,
    size: i32,
    out_pixels: *mut u8,
    out_capacity: u32,
    out_width: *mut u32,
    out_height: *mut u32,
    out_channels: *mut u32,
) -> i32 {
    clear_last_error();
    if buffer.is_null() || out_pixels.is_null() || out_width.is_null()
        || out_height.is_null() || out_channels.is_null() {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid decode arguments");
    }
    if size <= 0 || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid decode buffer size/capacity");
    }
    ffi_guard(|| {
        let buf_slice = unsafe { slice::from_raw_parts(buffer, size as usize) };
        let out_slice = unsafe { slice::from_raw_parts_mut(out_pixels, out_capacity as usize) };
        if is_container(buf_slice) {
            return decode_container(
                buf_slice,
                out_slice,
                unsafe { &mut *out_width },
                unsafe { &mut *out_height },
                unsafe { &mut *out_channels },
                std::ptr::null(),
            );
        }
        let ok = crate::decoder::decode(
            buf_slice,
            out_slice,
            unsafe { &mut *out_width },
            unsafe { &mut *out_height },
            unsafe { &mut *out_channels },
            None,
        );
        set_report(VerifyReport::plain());
        maybe_print_report();
        if ok {
            0
        } else {
            fail(BITGRAIN_ERR_DECODE_FAILED, "decode failed")
        }
    })
}

/// Decode a .bg stream with an optional previous frame for corrupt-block
/// concealment. Identical to bitgrain_decode, except that when the stream is
/// a checksummed (BGC1) container and prev_pixels is non-NULL, blocks that
/// fail CRC verification are restored from prev_pixels (must hold a frame of
/// the same dimensions and channel count) instead of neighbouring blocks.
/// The verification report is available via bitgrain_last_verify_report_text.
#[no_mangle]
pub extern "C" fn bitgrain_decode_checked(
    buffer: *const u8,
    size: i32,
    out_pixels: *mut u8,
    out_capacity: u32,
    out_width: *mut u32,
    out_height: *mut u32,
    out_channels: *mut u32,
    prev_pixels: *const u8,
) -> i32 {
    clear_last_error();
    if buffer.is_null() || out_pixels.is_null() || out_width.is_null()
        || out_height.is_null() || out_channels.is_null() {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid decode_checked arguments");
    }
    if size <= 0 || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid decode_checked buffer size/capacity");
    }
    ffi_guard(|| {
        let buf_slice = unsafe { slice::from_raw_parts(buffer, size as usize) };
        let out_slice = unsafe { slice::from_raw_parts_mut(out_pixels, out_capacity as usize) };
        if is_container(buf_slice) {
            return decode_container(
                buf_slice,
                out_slice,
                unsafe { &mut *out_width },
                unsafe { &mut *out_height },
                unsafe { &mut *out_channels },
                prev_pixels,
            );
        }
        let ok = crate::decoder::decode(
            buf_slice,
            out_slice,
            unsafe { &mut *out_width },
            unsafe { &mut *out_height },
            unsafe { &mut *out_channels },
            None,
        );
        set_report(VerifyReport::plain());
        maybe_print_report();
        if ok {
            0
        } else {
            fail(BITGRAIN_ERR_DECODE_FAILED, "decode failed")
        }
    })
}

/// Decode a .bg stream to grayscale (version 1 .bg only).
#[no_mangle]
pub extern "C" fn bitgrain_decode_grayscale(
    buffer: *const u8,
    size: i32,
    out_pixels: *mut u8,
    out_capacity: u32,
    out_width: *mut u32,
    out_height: *mut u32,
) -> i32 {
    clear_last_error();
    if buffer.is_null() || out_pixels.is_null() || out_width.is_null() || out_height.is_null() {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid decode_grayscale arguments");
    }
    if size <= 0 || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid decode_grayscale buffer size/capacity");
    }
    ffi_guard(|| {
        let mut ch = 0u32;
        let buf_slice = unsafe { slice::from_raw_parts(buffer, size as usize) };
        let out_slice = unsafe { slice::from_raw_parts_mut(out_pixels, out_capacity as usize) };
        if is_container(buf_slice) {
            let rc = decode_container(
                buf_slice,
                out_slice,
                unsafe { &mut *out_width },
                unsafe { &mut *out_height },
                &mut ch,
                std::ptr::null(),
            );
            if rc == 0 && ch == 1 {
                return 0;
            }
            if rc != 0 {
                return rc;
            }
            return fail(
                BITGRAIN_ERR_DECODE_FAILED,
                "decode_grayscale failed or output is not grayscale",
            );
        }
        let ok = crate::decoder::decode(
            buf_slice,
            out_slice,
            unsafe { &mut *out_width },
            unsafe { &mut *out_height },
            &mut ch,
            None,
        );
        set_report(VerifyReport::plain());
        maybe_print_report();
        if ok && ch == 1 {
            0
        } else {
            fail(BITGRAIN_ERR_DECODE_FAILED, "decode_grayscale failed or output is not grayscale")
        }
    })
}

/// Encode RGB with optional ICC profile.
#[no_mangle]
pub extern "C" fn bitgrain_encode_rgb_icc(
    image: *const u8,
    width: u32,
    height: u32,
    out_buffer: *mut u8,
    out_capacity: u32,
    out_len: *mut i32,
    quality: u8,
    icc: *const u8,
    icc_len: u32,
) -> i32 {
    clear_last_error();
    if image.is_null() || out_buffer.is_null() || out_len.is_null() || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid encode_rgb_icc arguments");
    }
    ffi_guard(|| {
        let q = if quality == 0 { 85 } else { quality };
        let size = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(3);
        let image_slice = unsafe { slice::from_raw_parts(image, size) };
        let buffer_slice = unsafe { slice::from_raw_parts_mut(out_buffer, out_capacity as usize) };
        let mut pos: i32 = 0;
        let icc_opt = if !icc.is_null() && icc_len > 0 {
            Some(unsafe { slice::from_raw_parts(icc, icc_len as usize) })
        } else {
            None
        };
        crate::encoder::encode_rgb(
            image_slice,
            width as usize,
            height as usize,
            q,
            buffer_slice,
            &mut pos,
            icc_opt,
        );
        unsafe { *out_len = pos };
        0
    })
}

/// Encode RGBA with optional ICC profile.
#[no_mangle]
pub extern "C" fn bitgrain_encode_rgba_icc(
    image: *const u8,
    width: u32,
    height: u32,
    out_buffer: *mut u8,
    out_capacity: u32,
    out_len: *mut i32,
    quality: u8,
    icc: *const u8,
    icc_len: u32,
) -> i32 {
    clear_last_error();
    if image.is_null() || out_buffer.is_null() || out_len.is_null() || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid encode_rgba_icc arguments");
    }
    ffi_guard(|| {
        let q = if quality == 0 { 85 } else { quality };
        let size = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        let image_slice = unsafe { slice::from_raw_parts(image, size) };
        let buffer_slice = unsafe { slice::from_raw_parts_mut(out_buffer, out_capacity as usize) };
        let mut pos: i32 = 0;
        let icc_opt = if !icc.is_null() && icc_len > 0 {
            Some(unsafe { slice::from_raw_parts(icc, icc_len as usize) })
        } else {
            None
        };
        crate::encoder::encode_rgba(
            image_slice,
            width as usize,
            height as usize,
            q,
            buffer_slice,
            &mut pos,
            icc_opt,
        );
        unsafe { *out_len = pos };
        0
    })
}

/// Decode .bg and optionally return embedded ICC. Caller must free ICC with bitgrain_free_icc.
#[no_mangle]
pub extern "C" fn bitgrain_decode_icc(
    buffer: *const u8,
    size: i32,
    out_pixels: *mut u8,
    out_capacity: u32,
    out_width: *mut u32,
    out_height: *mut u32,
    out_channels: *mut u32,
    out_icc: *mut *mut u8,
    out_icc_len: *mut u32,
) -> i32 {
    clear_last_error();
    if buffer.is_null() || out_pixels.is_null() || out_width.is_null()
        || out_height.is_null() || out_channels.is_null() {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid decode_icc arguments");
    }
    if size <= 0 || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid decode_icc buffer size/capacity");
    }
    ffi_guard(|| {
        let buf_slice = unsafe { slice::from_raw_parts(buffer, size as usize) };
        let out_slice = unsafe { slice::from_raw_parts_mut(out_pixels, out_capacity as usize) };
        let mut icc_vec = Vec::new();
        let ok = if is_container(buf_slice) {
            decode_container(
                buf_slice,
                out_slice,
                unsafe { &mut *out_width },
                unsafe { &mut *out_height },
                unsafe { &mut *out_channels },
                std::ptr::null(),
            ) == 0
        } else {
            let ok = crate::decoder::decode(
                buf_slice,
                out_slice,
                unsafe { &mut *out_width },
                unsafe { &mut *out_height },
                unsafe { &mut *out_channels },
                Some(&mut icc_vec),
            );
            set_report(VerifyReport::plain());
            maybe_print_report();
            ok
        };
        if !ok {
            return fail(BITGRAIN_ERR_DECODE_FAILED, "decode_icc failed");
        }
        if !out_icc.is_null() && !out_icc_len.is_null() {
            if icc_vec.is_empty() {
                unsafe {
                    *out_icc = std::ptr::null_mut();
                    *out_icc_len = 0;
                }
            } else {
                let len = icc_vec.len();
                let ptr = icc_vec.as_mut_ptr();
                std::mem::forget(icc_vec);
                unsafe {
                    *out_icc = ptr;
                    *out_icc_len = len as u32;
                }
            }
        }
        0
    })
}

/// Free ICC buffer returned by bitgrain_decode_icc.
#[no_mangle]
pub extern "C" fn bitgrain_free_icc(ptr: *mut u8, len: u32) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let _ = unsafe { Vec::from_raw_parts(ptr, len as usize, len as usize) };
}

#[cfg(test)]
mod crc_tests {
    use super::*;
    use std::sync::Mutex;

    // The checksum switch is process-global; serialize tests that touch it.
    static SWITCH_LOCK: Mutex<()> = Mutex::new(());

    fn sample_rgb(w: usize, h: usize) -> Vec<u8> {
        let mut v = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                v[i] = ((x * 5 + y * 3) & 0xFF) as u8;
                v[i + 1] = ((x * 2 + y * 7 + 40) & 0xFF) as u8;
                v[i + 2] = ((x * 11 + y + 90) & 0xFF) as u8;
            }
        }
        v
    }

    fn encode_rgb_checked(img: &[u8], w: u32, h: u32) -> Vec<u8> {
        let cap = (w * h * 3 * 2) as usize + (1 << 20);
        let mut out = vec![0u8; cap];
        let mut len: i32 = 0;
        let rc = bitgrain_encode_rgb(img.as_ptr(), w, h, out.as_mut_ptr(), cap as u32, &mut len, 90);
        assert_eq!(rc, 0, "encode failed");
        out.truncate(len as usize);
        out
    }

    // NOTE: the plain (non-container) decode path requires out_capacity to
    // be exactly w*h*channels — the upsampler iterates the whole out slice.
    // (Pre-existing codec contract; the C CLI sizes it exactly.)
    fn decode_any(stream: &[u8], prev: Option<&[u8]>, cap: usize) -> (i32, Vec<u8>, u32, u32, u32) {
        let mut out = vec![0u8; cap];
        let (mut w, mut h, mut ch) = (0u32, 0u32, 0u32);
        let rc = match prev {
            Some(p) => bitgrain_decode_checked(
                stream.as_ptr(),
                stream.len() as i32,
                out.as_mut_ptr(),
                out.len() as u32,
                &mut w,
                &mut h,
                &mut ch,
                p.as_ptr(),
            ),
            None => bitgrain_decode(
                stream.as_ptr(),
                stream.len() as i32,
                out.as_mut_ptr(),
                out.len() as u32,
                &mut w,
                &mut h,
                &mut ch,
            ),
        };
        (rc, out, w, h, ch)
    }

    fn report() -> String {
        LAST_REPORT.with(|r| r.borrow().to_text())
    }

    /// Offset of the first tile payload inside a BGC1 container.
    fn first_payload_offset(stream: &[u8]) -> usize {
        assert!(is_container(stream));
        CONTAINER_HEADER_SIZE + 8
    }

    #[test]
    fn crc32_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn checksums_off_produces_plain_stream() {
        let _g = SWITCH_LOCK.lock().unwrap();
        bitgrain_set_checksum_enabled(0);
        let img = sample_rgb(24, 16);
        let stream = encode_rgb_checked(&img, 24, 16);
        assert!(!is_container(&stream), "must stay a plain BG stream");
        assert_eq!(&stream[..2], b"BG");
        let (rc, _, w, h, ch) = decode_any(&stream, None, 24 * 16 * 3);
        assert_eq!((rc, w, h, ch), (0, 24, 16, 3));
        assert!(report().contains("protected=no"));
        bitgrain_set_checksum_enabled(0);
    }

    #[test]
    fn container_roundtrip_clean() {
        let _g = SWITCH_LOCK.lock().unwrap();
        bitgrain_set_checksum_enabled(1);
        let img = sample_rgb(40, 24);
        let stream = encode_rgb_checked(&img, 40, 24);
        assert!(is_container(&stream));
        let (rc, _, w, h, ch) = decode_any(&stream, None, 40 * 24 * 3);
        assert_eq!((rc, w, h, ch), (0, 40, 24, 3));
        let r = report();
        assert!(r.contains("protected=yes"), "{}", r);
        assert!(r.contains("header=ok"), "{}", r);
        assert!(r.contains("bad=0"), "{}", r);
        bitgrain_set_checksum_enabled(0);
    }

    #[test]
    fn single_bad_block_is_marked_and_concealed() {
        let _g = SWITCH_LOCK.lock().unwrap();
        bitgrain_set_checksum_enabled(1);
        let img = sample_rgb(40, 24);
        let mut stream = encode_rgb_checked(&img, 40, 24);
        let off = first_payload_offset(&stream);
        for i in 0..16 {
            stream[off + i] ^= 0xA5;
        }
        let (rc, out, w, h, ch) = decode_any(&stream, None, 40 * 24 * 3);
        assert_eq!((w, h, ch), (40, 24, 3));
        assert_eq!(rc, 0, "decode must succeed with concealment");
        let r = report();
        assert!(r.contains("bad=1"), "{}", r);
        assert!(r.contains("bad_positions=[0(x=0,y=0)]"), "{}", r);
        assert!(r.contains("neighbor-fill"), "{}", r);
        // Concealed block must not be the gray fallback everywhere.
        assert!(out[..40 * 24 * 3].iter().any(|&b| b != 128));
        bitgrain_set_checksum_enabled(0);
    }

    #[test]
    fn corrupt_header_is_reported_and_fails_gracefully() {
        let _g = SWITCH_LOCK.lock().unwrap();
        bitgrain_set_checksum_enabled(1);
        let img = sample_rgb(24, 16);
        let mut stream = encode_rgb_checked(&img, 24, 16);
        stream[9] ^= 0xFF; // width field inside the CRC-protected header
        let (rc, _, _, _, _) = decode_any(&stream, None, 24 * 16 * 3);
        assert_ne!(rc, 0, "corrupt header must not decode silently");
        let r = report();
        assert!(r.contains("header=CORRUPT"), "{}", r);
        bitgrain_set_checksum_enabled(0);
    }

    #[test]
    fn previous_frame_is_used_when_provided() {
        let _g = SWITCH_LOCK.lock().unwrap();
        bitgrain_set_checksum_enabled(1);
        let img = sample_rgb(24, 16);
        let mut stream = encode_rgb_checked(&img, 24, 16);
        let off = first_payload_offset(&stream);
        for i in 0..16 {
            stream[off + i] ^= 0x5A;
        }
        let prev = vec![7u8; 24 * 16 * 3];
        let (rc, out, _, _, _) = decode_any(&stream, Some(&prev), 24 * 16 * 3);
        assert_eq!(rc, 0);
        let r = report();
        assert!(r.contains("bad=1"), "{}", r);
        assert!(r.contains("previous-frame"), "{}", r);
        // First block (tile 0 at x=0,y=0) must come from the previous frame.
        assert!(out[..24 * 3].iter().all(|&b| b == 7));
        bitgrain_set_checksum_enabled(0);
    }
}
