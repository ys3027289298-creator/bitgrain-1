# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- Inter-frame (P-frame) coding with block-matching motion estimation:
  - New bitstream versions 20 (RGB) and 21 (RGBA).
  - Diamond search (large + small diamond) over 16×16 luma macroblocks.
  - Rate-distortion motion-vector cost: `J = SAD + λ·bits(MV)`, λ from quality.
  - MVs stored in a dedicated MV section right after the header; the decoder
    applies motion compensation against the previously decoded frame.
  - Occlusion safety: macroblocks whose best match is not decisively better
    than intra fall back to intra and emit no motion vector.
  - Chroma (4:2:0) reuses luma MVs at half resolution.
- New C API: `bitgrain_encode_pframe_rgb`, `bitgrain_encode_pframe_rgba`,
  `bitgrain_decode_pframe`.
- Evaluation suite (`rust/src/tests/motion_tests.rs`) covering translation,
  rotation, zoom and occlusion materials: residual-energy reduction, bitrate
  vs intra and pure-difference baselines, search-overhead budget (< 30% of
  encode time) and occlusion MV correctness.

### Fixed
- Makefile: tolerate newer cargo emitting the staticlib next to
  `libbitgrain.so` instead of under `deps/`.

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
