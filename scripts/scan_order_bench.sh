#!/bin/bash
# Adaptive scan-order benchmark.
# Generates four small synthetic materials (natural / text / gradient / noise),
# encodes each with every scan order (zigzag, row-major, column-major, custom,
# adaptive) and prints symbol entropy, stream size, encode time and block-header
# overhead. Single run, no external inputs.
#
# Usage: scripts/scan_order_bench.sh
set -e
cd "$(dirname "$0")/.."

if ! command -v cargo >/dev/null 2>&1; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

cargo test --release --manifest-path rust/Cargo.toml \
    scan_order_bench_report -- --nocapture --test-threads=1
