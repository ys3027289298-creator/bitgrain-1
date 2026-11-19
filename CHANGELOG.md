# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- In-loop **Sample Adaptive Offset (SAO)** filter targeting post-deblocking
  ringing and contouring, in new stream versions **v20 (RGB)** and **v21
  (RGBA)**:
  - Edge Offset (EO): pixels classified by local edge orientation across four
    configurable directions (horizontal, vertical, 135°, 45°), with a
    configurable class count (2, 4, or the full 5-class HEVC set). One signed
    offset per class (±7).
  - Band Offset (BO): the 0–255 range split into 32 intensity bands with four
    consecutive-band offsets (±15) to repair gradient banding.
  - Per-region mode selection minimizes SSE against the original; a signaling
    guard leaves flat/noise regions unfiltered.
  - Classifications and offsets are integer-only and deterministic, so the
    decoder reproduces the encoder's filtered samples byte-for-byte from the
    signaled parameters alone.
  - SAO parameters are stored in a self-describing `BGs` chunk after the plane
    payloads (documented in `FORMAT.md`); region grid and class counts are
    fully derived from the stream.
- New C API: `bitgrain_sao_config_new`, `bitgrain_encode_rgb_sao`,
  `bitgrain_encode_rgba_sao` (the existing encode entry points keep producing
  the legacy v18/v19 streams, so all current consumers are unchanged).
- New CLI flags: `--no-sao`, `--sao-classes`, `--sao-region`,
  `--sao-no-band`, `--sao-directions`.
- `rust/build.rs` so `cargo test` links the C SIMD kernels self-contained;
  rlib added to the crate types for in-tree tests/examples.

### Performance
- 1080p SAO stage (estimate + apply over Y/Cb/Cr) runs within the ~25 ms
  budget on the reference 16-thread host (~13–21 ms across four material
  classes), parallelized per region with a double-buffered apply path that
  guarantees identical serial/parallel output.

### Validation
- Measured PSNR improvement on 1080p text/gradient/texture/noise material and
  reduced edge-ringing energy on all four; 27 Rust unit/end-to-end tests
  covering classification, chunk round-trip, malformed-input rejection,
  deterministic decode, configurable class counts, and encode/decode
  reproducibility.

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
