# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- Progressive (spectral-selection) decoding for Huffman .bg streams (v4-v19),
  in the Rust codec layer (`rust/src/progressive.rs`):
  - `decode_progressive`: preview rounds keep the first K zigzag DCT
    coefficients per 8x8 block (default schedule 1/6/15/28) and render a
    full-size frame each round; the final round delegates to `decode`, so the
    end result is bit-identical to one-shot decoding.
  - Any interruption point (callback cancels) leaves a complete full-size
    image in the caller's buffer.
  - Preview working memory is Y+Cb+Cr u8 planes only (1.5*w*h bytes = 50% of
    the final RGB image, 37.5% for RGBA; alpha is opaque until the final
    round). Coefficients are streamed block-by-block, no per-plane block Vec.
  - `probe` (header parse) and `preview_working_bytes` helpers.
  - Legacy RLE streams (v1-v3) fall back to a single one-shot frame.
  - No bitstream or public C API changes.

### Fixed
- `ycbcr420_to_rgb` / `ycbcr420a_to_rgba` iterated over the full output
  buffer length instead of the image height, panicking when the caller's
  output capacity exceeded w*h*channels (e.g. the Python binding's w*h*4
  upper-bound allocation for RGB streams). Both now bound the conversion to
  exactly h rows.

## [2.0.0] - 2026-04-26

### Added
- Scheduler-friendly parallel execution with adaptive thresholds for encode/decode.
- Extended SIMD coverage (SSE2/AVX2/NEON) in colorspace/dequant hot paths.
- Installable shared libraries with ABI symlinks:
  - `libbitgrain.so.<ABI_MAJOR>`
  - `libbitgrain-simd.so.<ABI_MAJOR>`
- `pkg-config` metadata: `bitgrain.pc`.
- CMake consumer config: `BitgrainConfig.cmake`.
- Environment-based thread controls:
  - `BITGRAIN_THREADS`
  - `BITGRAIN_THREADS_CAP`

### Changed
- Improved library installation layout for external consumers.
- Expanded API contract docs for error-message lifetime and decode failure semantics.
- Updated Rust crate metadata repository URL.

### Notes
- The public C API remains in `includes/encoder.h`.
- Semantic versioning policy:
  - Patch/minor: source-compatible C API changes
  - Major: potential ABI break
