#!/usr/bin/env bash
# Partial re-encode evaluation driver.
#
# Runs the full Rust unit-test suite (which includes the block-index /
# partial-reencode tests and prints the four-scenario metrics scorecard) using
# only dependencies already vendored in the repository. No network access, no
# new third-party crates, no changes to the build system.
#
# Usage:  scripts/partial_reencode_eval.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUST_DIR="$REPO_ROOT/rust"

# Locate a cargo toolchain without mutating the user's global PATH.
if command -v cargo >/dev/null 2>&1; then
    CARGO=cargo
elif [ -x "$HOME/.cargo/bin/cargo" ]; then
    CARGO="$HOME/.cargo/bin/cargo"
else
    echo "cargo not found" >&2
    exit 1
fi

cd "$RUST_DIR"
export CARGO_TARGET_DIR="$RUST_DIR/target"
# --offline: use only the cached/vendored crates already present on this machine.
"$CARGO" test --offline -- --nocapture --test-threads=1
