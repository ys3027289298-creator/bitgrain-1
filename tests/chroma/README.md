# Chroma subsampling verification (4:4:4 / 4:2:2 / 4:2:0)

Self-contained, one-shot verification using the repo CLI plus Python stdlib
only (no third-party packages, no network).

## Inputs

`make_samples.py` generates two tiny 32×32 RGB PNGs:

- `smooth32.png` — low-frequency gradients (typical photo-like content)
- `checker32.png` — high-frequency 1px checker (worst case for chroma)

## Run

```sh
make bitgrain
cp bitgrain /tmp/bitgrain-baseline            # pre-change binary, for the byte check
BITGRAIN_BASE=/tmp/bitgrain-baseline ./tests/chroma/run_verify.sh
```

The script encodes/decodes every mode, asserts the stream versions
(v18/v20/v22), verifies that a default (no `--chroma`) encode is
byte-identical to the baseline, and prints a size + PSNR table.

Codec-level invariants (odd dimensions, RGBA v19/v21/v23, centroid cell
mapping, env override parsing) are covered by the Rust unit tests in
`rust/src/tests/chroma_tests.rs` (`cargo test --release`).
