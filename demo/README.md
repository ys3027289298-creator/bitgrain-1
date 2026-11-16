# Three-stage preprocessing demo

Window/demo around the preprocessing implementation added to `c/metrics.c`:

    median (salt-and-pepper) -> bilateral (Gaussian edge-preserving) -> unsharp (detail)

Each stage has its own on/off switch and its own validated parameters.

## Run

    bash demo/run.sh

Builds the demo against the repo's Rust static library, prints the four
measurements, then runs every existing test suite. One shot, no network.

Requires the same libwebp dev headers the project already needs
(`pkg-config libwebp`), and `cargo` for the Rust suite.

## Files

- `preprocess_api.h` — types/prototypes the demo and library agree on.
- `main.c` — synthetic image, noise, comparison table, parameter rejection.
- `Makefile` — demo-only build; the project Makefile is untouched.

## Measurements

1. PSNR before/after denoising (against the clean reference)
2. edge-gradient retention: Sobel energy on reference edges only
3. wall time per stage and for the full chain (median of 9 runs)
4. encoded `.bg` size change versus the noisy input

Plus a side-by-side of the full chain against each stage run alone.
