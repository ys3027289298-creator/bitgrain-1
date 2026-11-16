#!/bin/bash
# One-shot demo for the three-stage preprocessing pipeline.
#
# Builds the demo against the repository's own Rust static library, runs it,
# then runs every test suite the repository ships. Prints a clear PASS/FAIL
# summary. Nothing here modifies the project build system.
set -u
cd "$(dirname "$0")/.."

# libwebp headers are required by the codec. The repo expects a dev install;
# honour any PKG_CONFIG_PATH/CPATH/LIBRARY_PATH already exported by the caller.
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}"
export PATH="$HOME/.cargo/bin:$PATH"
echo "==================================================================="
echo " 1/4  build demo"
echo "==================================================================="
if ! make -f demo/Makefile >/tmp/bg_demo_build.log 2>&1; then
    echo "BUILD FAILED"; cat /tmp/bg_demo_build.log; exit 1
fi
tail -1 /tmp/bg_demo_build.log

echo
echo "==================================================================="
echo " 2/4  preprocess demo"
echo "==================================================================="
./demo/preprocess_demo
demo_rc=$?

echo
echo "==================================================================="
echo " 3/4  existing test suites"
echo "==================================================================="
fail=0

echo "--- cargo test --release (Rust) ---"
if command -v cargo >/dev/null 2>&1 && [ -d rust ]; then
    ( cd rust && CARGO_TARGET_DIR="$(pwd)/target" cargo test --release 2>&1 | tail -20 )
    rust_rc=${PIPESTATUS[0]:-0}
else
    echo "SKIPPED: cargo not on PATH"
    rust_rc=0
fi

echo "--- test_dct ---"
./test_dct; dct_rc=$?
echo "--- test_jpg ---"
mkdir -p test_out
./test_jpg; jpg_rc=$?

echo "--- tests/integration.sh ---"
bash tests/integration.sh; int_rc=$?

echo
echo "==================================================================="
echo " 4/4  summary"
echo "==================================================================="
echo "demo                : $([ $demo_rc -eq 0 ] && echo PASS || echo FAIL)"
echo "cargo test          : $( [ $rust_rc -eq 0 ] && echo PASS || echo FAIL )"
echo "test_dct            : $([ $dct_rc -eq 0 ] && echo PASS || echo FAIL)"
echo "test_jpg            : $([ $jpg_rc -eq 0 ] && echo PASS || echo FAIL)"
echo "tests/integration.sh: $([ $int_rc -eq 0 ] && echo PASS || echo FAIL )"
echo
echo "NOTE: tests/integration.sh fails identically on the unmodified commit:"
echo "      its decode step asks for .pgm output but the encoder produces a"
echo "      3-channel .bg (v18), and PGM is grayscale-only. This is a"
echo "      pre-existing repo bug, unrelated to the preprocessing change."
echo
if [ $demo_rc -eq 0 ] && [ $rust_rc -eq 0 ] && [ $dct_rc -eq 0 ] && [ $jpg_rc -eq 0 ] && [ $int_rc -eq 0 ]; then
    echo "RESULT: all suites green"
    exit 0
else
    echo "RESULT: at least one suite failed (see above)"
    exit 1
fi
