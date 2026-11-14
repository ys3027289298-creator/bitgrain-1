#!/bin/bash
# Quantization-dithering end-to-end verification.
#
# Run from the repo root:
#   bash tests/dither_test.sh
#
# Covers:
#   1. byte-identity of every "dither inactive" path vs. a plain encode
#      (off mode, strength 0 under either algorithm) — must be byte-for-byte
#      identical,
#   2. banding metrics (distinct gradient levels, widest band run) and .bg
#      size for both algorithms at every supported strength, via the public
#      C API,
#   3. out-of-range strength / unknown-mode failure paths (CLI + C API).
#
# Uses only the repository's own binary and public C API plus a self-generated
# 256x8 grayscale PGM. No network or extra third-party libraries.

set -e
cd "$(dirname "$0")/.."

BIN=./bitgrain
OUT=tests/out-dither
VERIFY="$OUT/dither_verify"
WIDTH=256
HEIGHT=8
RAMP=32
QUALITY=5

mkdir -p "$OUT"

if [ ! -x "$BIN" ]; then
    echo "Build $BIN first (make bitgrain)" >&2
    exit 1
fi

# --- Self-made tiny banding-prone gradient (256x8 grayscale PGM) --------
# One input step every RAMP pixels so lossy coding turns it into wide bands.
PGM="$OUT/gradient.pgm"
{
    printf 'P5\n%u %u\n255\n' "$WIDTH" "$HEIGHT"
    for y in $(seq 0 $((HEIGHT - 1))); do
        for x in $(seq 0 $((WIDTH - 1))); do
            v=$(( (x / RAMP) * 255 / ((WIDTH - 1) / RAMP) ))
            printf "\\$(printf '%03o' "$v")"
        done
    done
} > "$PGM"

echo "=== 1. Byte-identity when dithering is disabled ==="
"$BIN" encode "$PGM" -o "$OUT/ref.bg"     --quality "$QUALITY" --overwrite
"$BIN" encode "$PGM" -o "$OUT/off_0.bg"   --quality "$QUALITY" \
    --dither off --dither-strength 0   --overwrite
"$BIN" encode "$PGM" -o "$OUT/off_50.bg"  --quality "$QUALITY" \
    --dither off --dither-strength 50  --overwrite
"$BIN" encode "$PGM" -o "$OUT/ord_0.bg"   --quality "$QUALITY" \
    --dither ordered --dither-strength 0 --overwrite
"$BIN" encode "$PGM" -o "$OUT/ed_0.bg"    --quality "$QUALITY" \
    --dither error-diffusion --dither-strength 0 --overwrite

for f in off_0.bg off_50.bg ord_0.bg ed_0.bg; do
    if ! cmp -s "$OUT/ref.bg" "$OUT/$f"; then
        echo "FAIL: $f differs from reference while dithering is inactive" >&2
        exit 1
    fi
done
echo "OK: ref.bg == off_0.bg == off_50.bg == ord_0.bg == ed_0.bg"

echo
echo "=== 2. Dithered encodes are decodable and sized ==="
for s in 25 50 75 100; do
    "$BIN" encode "$PGM" -o "$OUT/ord_$s.bg" --quality "$QUALITY" \
        --dither ordered --dither-strength "$s" --overwrite >/dev/null
    "$BIN" encode "$PGM" -o "$OUT/ed_$s.bg"  --quality "$QUALITY" \
        --dither error-diffusion --dither-strength "$s" --overwrite >/dev/null
    "$BIN" decode "$OUT/ord_$s.bg" -o "$OUT/ord_$s.bmp" --overwrite >/dev/null
    "$BIN" decode "$OUT/ed_$s.bg"  -o "$OUT/ed_$s.bmp"  --overwrite >/dev/null
done
echo "OK: all dithered .bg files encoded and decoded"

echo
echo "=== 3. C API harness: levels, widest band, sizes, error codes ==="
gcc -std=c11 -Wall -Wextra -Iincludes tests/dither_verify.c \
    rust/target/release/deps/libbitgrain.a \
    c/dct.o c/quant.o \
    -lpthread -ldl -lm \
    -o "$VERIFY"
"$VERIFY"

echo
echo "=== 4. Failure paths (CLI) ==="
set +e
"$BIN" encode "$PGM" -o "$OUT/bad.bg" --quality "$QUALITY" \
    --dither ordered --dither-strength 101 --overwrite 2>"$OUT/err_strength.txt"
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
    echo "FAIL: strength 101 was accepted" >&2
    exit 1
fi
grep -q "0..=100" "$OUT/err_strength.txt"
echo "OK: strength 101 rejected ($(cat "$OUT/err_strength.txt"))"

set +e
"$BIN" encode "$PGM" -o "$OUT/bad2.bg" --quality "$QUALITY" \
    --dither bogus --overwrite 2>"$OUT/err_mode.txt"
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
    echo "FAIL: unknown dither mode was accepted" >&2
    exit 1
fi
grep -q "off, ordered, error-diffusion" "$OUT/err_mode.txt"
echo "OK: unknown mode rejected ($(cat "$OUT/err_mode.txt"))"

echo
echo "=== All dithering tests passed ==="
