#!/bin/bash
# Motion-estimation evaluation: runs the P-frame test suite (translation,
# rotation, zoom, occlusion) with metrics printed:
#   - residual energy vs pure frame differencing
#   - P-frame size vs intra and pure-difference baselines
#   - motion-search overhead as % of total encode time (asserted < 30%)
#   - occlusion macroblocks verified MV-free (all intra)
# Run from repo root. Release mode: the 30% overhead budget is only
# asserted in optimized builds.
set -e
cd "$(dirname "$0")/.."
cd rust && cargo test --release motion -- --nocapture
