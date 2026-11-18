#!/bin/bash
# Blue-noise dithering validation + comparison harness.
# Self-contained: compiles scripts/test_bluenoise.c against c/bluenoise.h
# with the system C compiler and libm only. No new third-party dependency,
# no network access, no changes to the build system.
set -e
cd "$(dirname "$0")/.."

CC="${CC:-gcc}"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/bitgrain-bn-XXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT

CFLAGS="-std=c11 -Wall -Wextra -Werror -O2 -D_POSIX_C_SOURCE=199309L -Ic"
echo "Compiling test driver into $BUILD_DIR ..."
"$CC" $CFLAGS scripts/test_bluenoise.c -o "$BUILD_DIR/test_bluenoise" -lm
"$BUILD_DIR/test_bluenoise"
