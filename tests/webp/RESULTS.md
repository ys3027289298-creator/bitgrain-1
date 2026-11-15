# WebP reader — verification results

Run everything (build, sample generation, harness, CLI checks) in one shot:

```bash
./tests/webp/run.sh
```

No network access and no new third-party libraries are used. Samples are tiny
(32×32 and 64×64) and regenerated on every run.

## What is exercised

- Lossy VP8 bitstream, RGB and RGBA (`*_rgb.webp`, `*_rgba.webp`).
- Lossless VP8L bitstream, RGB and RGBA.
- Alpha plane preservation through the RGBA loader and the full `.bg` pipeline.
- Bounded decode error: loader output vs. a direct libwebp reference decode,
  plus vs. the original pre-compression pixels.
- Failure paths: truncated container, corrupt RIFF/WEBP, empty buffer,
  stdin (memory loader) variants.

## Before / after comparison (32×32 synthetic samples)

| Sample | Before: `.bg` version / channels | After: `.bg` version / channels |
|--------|----------------------------------|----------------------------------|
| `lossy_rgb.webp`     | v18 / 3 (RGB) | v18 / 3 (RGB), unchanged |
| `lossless_rgb.webp`  | v18 / 3 (RGB) | v18 / 3 (RGB), unchanged |
| `lossy_rgba.webp`    | v18 / **3 — alpha silently dropped** | v19 / **4 — alpha retained** |
| `lossless_rgba.webp` | v18 / **3 — alpha silently dropped** | v19 / **4 — alpha retained** |
| WebP on stdin        | failed: `could not load '-'` | works (magic-based memory routing) |

Alpha error measured through the full encode→`.bg`→decode→PNG pipeline
(source alpha vs. output alpha, 32×32):

| Build | PNG channels | alpha maxAbs | alpha MAE |
|-------|--------------|-------------:|----------:|
| Before | 3 (no alpha) | 255 | 113.250 |
| After  | 4 (RGBA)     | 2   | 0.281 |

Non-WebP inputs (PNG/JPG/BMP/TGA/PGM, file and stdin) produce byte-identical
`.bg` output before vs. after — verified with `cmp`.

## Decode error vs. reference / source

Loader output vs. an independent libwebp reference call on the same bytes:
maxAbs = 0 on every sample (the loader returns the exact decoded pixels),
alpha maxAbs = 0 on every alpha sample.

Against the original pre-compression source (64×64, visible pixels only):

| Sample | RGB maxAbs | RGB MAE | alpha maxAbs |
|--------|-----------:|--------:|-------------:|
| lossless64 RGBA | 0 | 0.0000 | 0 (bit-exact) |
| lossless64 RGB  | 0 | 0.0000 | n/a |
| lossy64 RGBA (q75) | 202 | 8.9610 | 0 |
| lossy64 RGB (q75)  | 141 | 5.8877 | n/a |

The large lossy RGB deltas are intrinsic to lossy VP8 quantization at q75;
the loader itself introduces zero additional error (maxAbs 0 vs. reference).
Lossless WebP decodes bit-exactly. At fully transparent pixels, WebP is free
to change RGB values, so error is only meaningful where alpha > 0.

## Decode timing (representative run, 64×64 RGBA, 3000 iterations)

| Sample | p50 decode | min decode |
|--------|-----------:|-----------:|
| lossy64_rgba.webp (VP8 + alpha)  | 0.0446 ms | 0.0397 ms |
| lossless64_rgba.webp (VP8L)      | 0.0311 ms | 0.0285 ms |

Timing is dominated by libwebp and varies with machine load; rerun the harness
(`tests/webp/run.sh`) for current numbers. Absolute times are only meaningful
for same-machine comparisons.

## Failure path reproduction

```bash
# truncated bitstream (half of a valid WebP file)
./bitgrain encode tests/webp/samples/truncated.webp -o /tmp/x.bg -y
# -> Error: ... WebP: truncated WebP container (not enough data); exit 1

# corrupt RIFF/WEBP container
./bitgrain encode tests/webp/samples/garbage.webp -o /tmp/x.bg -y
# -> Error: ... WebP: corrupt WebP bitstream; exit 1

# same via stdin (memory loader path)
cat tests/webp/samples/truncated.webp | ./bitgrain encode - -o /tmp/x.bg -y
# -> Error: could not load '-': WebP: truncated WebP container (not enough data); exit 1
```
