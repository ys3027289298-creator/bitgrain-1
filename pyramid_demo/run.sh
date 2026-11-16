#!/bin/bash
# Build and run the Gaussian/Laplacian pyramid demo in one shot.
# No third-party libraries, no network: compiles the library source that
# hosts the pyramid (c/image_writer.c) directly, links only libc + libm.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
OUT_DIR="pyramid_demo/build"
mkdir -p "$OUT_DIR"

CC="${CC:-gcc}"
CFLAGS="-std=c11 -Wall -Wextra -O3 -Iincludes -Ic"

echo "[1/2] Compiling pyramid demo (library source + demo)..."
# image_writer.c pulls in stb_image_write; the pyramid itself only needs
# libc/libm. No BITGRAIN_USE_WEBP / PNG flags -> no external image deps.
"$CC" $CFLAGS -c c/image_writer.c -o "$OUT_DIR/image_writer.o"
"$CC" $CFLAGS -c pyramid_demo/pyramid_demo.c -o "$OUT_DIR/pyramid_demo.o"
"$CC" "$OUT_DIR/image_writer.o" "$OUT_DIR/pyramid_demo.o" -o "$OUT_DIR/pyramid_demo" -lm

echo "[2/2] Running demo..."
echo
"$OUT_DIR/pyramid_demo"
