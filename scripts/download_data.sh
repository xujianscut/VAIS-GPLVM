#!/usr/bin/env bash
# Download the 3-phase oil flow dataset (Bishop & James, 1993) used in the paper.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)/data"
mkdir -p "$DIR"; cd "$DIR"
BASE="https://github.com/lawrennd/datasets_mirror/raw/main/three_phase_oil_flow"
for f in DataTrn.txt DataTrnLbls.txt DataTst.txt DataTstLbls.txt; do
  echo "downloading $f ..."
  curl -sL "$BASE/$f" -o "$f"
done
echo "done -> $DIR"
