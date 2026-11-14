# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- Quantization dithering to suppress gradient banding at low bit rates:
  - Ordered (8x8 Bayer) and Floyd-Steinberg error-diffusion algorithms.
  - Configurable perturbation strength (`0..=100`); out-of-range values and
    unknown modes are rejected with `BITGRAIN_ERR_INVALID_ARG`.
  - New public C API functions `bitgrain_set_dither()` and
    `bitgrain_clear_dither()` (no existing entry-point signature changed).
  - CLI flags `--dither <off|ordered|error-diffusion>` and
    `--dither-strength <0-100>` for `encode` and `roundtrip`.

### Notes
- Dithering runs in the codec/processing layer (pixel-domain, before the
  forward DCT) and defaults to disabled; with strength `0` (or mode `off`)
  encoded output is byte-identical to the undithered encoder.
- Verification: `bash tests/dither_test.sh` (self-contained, uses a generated
  gradient and the public C API; no network or new third-party dependencies).

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
