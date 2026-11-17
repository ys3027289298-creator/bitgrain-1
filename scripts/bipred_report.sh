#!/bin/bash
# Bidirectional-prediction benchmark report.
# Generates three small synthetic materials (slow pan, brightness fade,
# independent frames), encodes each as all-intra vs bipred sequence, and
# prints size ratio, PSNR, encode/decode times and weight-search iterations.
set -e
cd "$(dirname "$0")/../rust"
cargo test --release bipred_materials_report -- --nocapture
