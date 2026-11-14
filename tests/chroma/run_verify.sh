#!/bin/bash
# One-shot verification for chroma subsampling (4:4:4 / 4:2:2 / 4:2:0).
# Repo CLI + stdlib python only. No network, no extra dependencies.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="${BITGRAIN_BIN:-$HERE/../../bitgrain}"
BASE="${BITGRAIN_BASE:-/tmp/bitgrain-baseline}"
cd "$HERE"

if [ ! -x "$BIN" ]; then
    echo "build the CLI first: make bitgrain (or set BITGRAIN_BIN)" >&2
    exit 2
fi
if [ ! -x "$BASE" ]; then
    echo "baseline binary required for the byte-identical check (set BITGRAIN_BASE)" >&2
    exit 2
fi

echo "== build samples =="
python3 make_samples.py

for img in smooth32 checker32; do
  echo
  echo "== $img: encode/decode each chroma mode =="
  for m in 420 422 444; do
      "$BIN" encode "$img.png" -o "$img-$m.bg" --chroma "$m" -y >/dev/null
      "$BIN" decode "$img-$m.bg" -o "$img-$m.bmp" -y >/dev/null
  done
done

echo
echo "== byte-identical guarantee (no --chroma => historical 4:2:0 path) =="
for img in smooth32 checker32; do
  "$BASE" encode "$img.png" -o "base-$img.bg" -y >/dev/null
  "$BIN"  encode "$img.png" -o "new-$img.bg" -y >/dev/null
  cmp "base-$img.bg" "new-$img.bg" && echo "$img default stream: BYTE-IDENTICAL"
done

echo
echo "== size + PSNR table =="
python3 psnr_report.py "$HERE"
