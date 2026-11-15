#!/bin/bash
# One-shot WebP reader verification:
#   1. generate tiny synthetic samples (lossy/lossless x RGB/RGBA + failure cases)
#   2. build bitgrain against the bundled libwebp dependency
#   3. run the C verification harness (error bounds, alpha, timing, failure paths)
#   4. exercise the CLI incl. stdin and alpha-preserving round-trip
# No network access and no new third-party libraries are used.
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SAMPLES="$ROOT/tests/webp/samples"
OUT="$ROOT/tests/webp/out"
mkdir -p "$SAMPLES" "$OUT"

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "from PIL import Image" 2>/dev/null; then
    for cand in /home/ys/local/venv/bin/python ./venv/bin/python; do
        if [ -x "$cand" ] && "$cand" -c "from PIL import Image" 2>/dev/null; then
            PYTHON="$cand"; break
        fi
    done
fi
"$PYTHON" - "$SAMPLES" <<'EOF'
import os, sys
from PIL import Image
d = sys.argv[1]
os.makedirs(d, exist_ok=True)
W = H = 32
img = Image.new('RGBA', (W, H))
px = img.load()
for y in range(H):
    for x in range(W):
        r = (x * 8) & 255
        g = (y * 8) & 255
        b = ((x + y) * 4) & 255
        if x < W // 4:
            a = 255
        elif x < W // 2:
            a = (x * 16) & 255
        elif x < 3 * W // 4:
            a = 128
        else:
            a = 0
        px[x, y] = (r, g, b, a)
img.save(os.path.join(d, 'src_rgba.png'))
img.convert('RGB').save(os.path.join(d, 'lossy_rgb.webp'),
                        'WEBP', lossless=False, quality=80, method=6)
img.save(os.path.join(d, 'lossy_rgba.webp'),
         'WEBP', lossless=False, quality=80, method=6)
img.convert('RGB').save(os.path.join(d, 'lossless_rgb.webp'),
                        'WEBP', lossless=True, quality=100, method=6)
img.save(os.path.join(d, 'lossless_rgba.webp'),
         'WEBP', lossless=True, quality=100, method=6)
# Slightly larger (64x64) samples for stable decode timing.
import math
W64 = H64 = 64
img64 = Image.new('RGBA', (W64, H64))
p64 = img64.load()
for y in range(H64):
    for x in range(W64):
        r = int(128 + 127 * math.sin(x * 0.35))
        g = int(128 + 127 * math.cos(y * 0.27))
        b = (x * 3 + y * 5) & 255
        a = 0 if (x // 8 + y // 8) % 3 == 0 else (255 if x % 2 == 0 else 128)
        p64[x, y] = (r, g, b, a)
img64.convert('RGB').save(os.path.join(d, 'lossy64_rgb.webp'),
                           'WEBP', lossless=False, quality=75, method=6)
img64.save(os.path.join(d, 'lossy64_rgba.webp'),
           'WEBP', lossless=False, quality=75, method=6)
img64.convert('RGB').save(os.path.join(d, 'lossless64_rgb.webp'),
                           'WEBP', lossless=True, quality=100, method=6)
img64.save(os.path.join(d, 'lossless64_rgba.webp'),
           'WEBP', lossless=True, quality=100, method=6)
data = open(os.path.join(d, 'lossy_rgb.webp'), 'rb').read()
open(os.path.join(d, 'truncated.webp'), 'wb').write(data[:len(data) // 2])
open(os.path.join(d, 'garbage.webp'), 'wb').write(
    b'RIFF' + b'\x00\x00\x00\x00' + b'WEBP' + b'not-a-real-vp8-chunk-at-all')
open(os.path.join(d, 'nowebp.bin'), 'wb').write(bytes(range(256)) * 4)
print('samples generated in', d)
EOF

# Locate the libwebp development headers provided by the environment/setup.
WEBP_INC="${WEBP_INC:-}"
if [ -z "$WEBP_INC" ] && [ -f /tmp/local/usr/include/webp/decode.h ]; then
    WEBP_INC="/tmp/local/usr/include"
fi
if [ -z "$WEBP_INC" ]; then
    WEBP_INC="$(dirname "$(dirname "$(find /tmp /usr/include /usr/local/include -name decode.h -path '*webp*' 2>/dev/null | head -1)")")"
fi
if [ -z "$WEBP_INC" ] || [ ! -f "$WEBP_INC/webp/decode.h" ]; then
    echo "libwebp headers not found; run ./setup.sh (BITGRAIN_USE_WEBP)" >&2
    exit 1
fi

# Link against the runtime soname so no new artifacts are introduced.
WEBP_LIB="${WEBP_LIB:--l:libwebp.so.7}"

echo "=== build bitgrain ==="
make bitgrain CFLAGS_NATIVE= RUSTFLAGS_NATIVE= \
     WEBP_CFLAGS="-I$WEBP_INC" WEBP_LIBS="$WEBP_LIB" >/dev/null

echo "=== build verification harness ==="
gcc -std=c11 -Wall -Wextra -O2 -Ic -I"$WEBP_INC" \
    tests/webp/webp_verify.c c/webp_io.c \
    -DBITGRAIN_USE_WEBP \
    -o "$OUT/webp_verify" $WEBP_LIB -lpng16 -lm

echo "=== harness: error bounds, alpha, timing, failures ==="
"$OUT/webp_verify" "$SAMPLES"

echo "=== CLI: channel selection (0x12=RGB v18, 0x13=RGBA v19) ==="
for f in lossy_rgb lossless_rgb; do
    ./bitgrain encode "$SAMPLES/$f.webp" -o "$OUT/$f.bg" -y >/dev/null
    printf '  %-14s header=%s (expect 3ch ...12)\n' "$f" "$(xxd -p -l 3 "$OUT/$f.bg")"
done
for f in lossy_rgba lossless_rgba; do
    ./bitgrain encode "$SAMPLES/$f.webp" -o "$OUT/$f.bg" -y >/dev/null
    printf '  %-14s header=%s (expect 4ch ...13)\n' "$f" "$(xxd -p -l 3 "$OUT/$f.bg")"
done

echo "=== CLI: stdin WebP (memory loader) ==="
cat "$SAMPLES/lossless_rgba.webp" | ./bitgrain encode - -o "$OUT/stdin_rgba.bg" -y >/dev/null
printf '  stdin RGBA header=%s\n' "$(xxd -p -l 3 "$OUT/stdin_rgba.bg")"

echo "=== CLI: alpha round-trip to PNG and back ==="
./bitgrain roundtrip "$SAMPLES/lossless_rgba.webp" -o "$OUT/rt_rgba.png" -y -m 2>&1 | sed 's/^/  /'

echo "=== CLI: failure messages ==="
set +e
for f in truncated.webp garbage.webp; do
    msg=$(./bitgrain encode "$SAMPLES/$f" -o "$OUT/bad.bg" -y 2>&1)
    rc=$?
    printf '  %-14s rc=%d msg=%s\n' "$f" "$rc" "$msg"
    [ "$rc" -eq 1 ] || exit 1
done
set -e

echo "ALL WEBP TESTS PASSED"
