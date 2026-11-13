#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Materialize Hypersim's HDF5 segmentation channels into shared_seg/hypersim_seg.tar,
the archive the semantic- and instance-segmentation pool members read.

The raw release (download_hypersim.sh) stores HDR colour and semantic /
instance channels as per-frame HDF5, which neither converter reads. This runs
dataset_prep/hypersim/prepare_hypersim_segmentation.py once, with the settings
used for the paper pools: every frame with all four channels, tonemapped colour,
uint16 panoptic PNGs (class*1000 + instance), a 2% scene-level validation split.

Usage:
  scripts/dataset_prep/prepare_hypersim_segmentation.sh

Important environment variables:
  ROOT                   Repo root.
  USE_CONTAINER          1 to use SIF; 0 for the active environment. Default: 0.
  PYTHON                 Python interpreter with h5py. Default: python.
  DATA_ROOT              Persistent data root.
  HYPERSIM_ROOT          Persistent Hypersim root (download_hypersim.sh).
  RAW_ROOT               Extracted Hypersim scenes. Default: $HYPERSIM_ROOT/work/raw_extracted.
  OUT_ROOT               Materialized tree. Default: $HYPERSIM_ROOT/work/hypersim_seg.
  HYPERSIM_SEG_TAR       Output archive. Default: $DATA_ROOT/shared_seg/hypersim_seg.tar.
  NUM_WORKERS            Default: 32.
  FORCE                  1 to rebuild an existing archive.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

ROOT=${ROOT:-$DEFAULT_ROOT}
source "$ROOT/scripts/lib/runtime.sh"

DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT for this dataset}
HYPERSIM_ROOT=${HYPERSIM_ROOT:-$DATA_ROOT/depth_surfn_data/hypersim}
RAW_ROOT=${RAW_ROOT:-$HYPERSIM_ROOT/work/raw_extracted}
OUT_ROOT=${OUT_ROOT:-$HYPERSIM_ROOT/work/hypersim_seg}
HYPERSIM_SEG_TAR=${HYPERSIM_SEG_TAR:-$DATA_ROOT/shared_seg/hypersim_seg.tar}
NUM_WORKERS=${NUM_WORKERS:-32}
FORCE=${FORCE:-0}

if [[ "$FORCE" != 1 && -s "$HYPERSIM_SEG_TAR" ]]; then
  echo "Using existing $HYPERSIM_SEG_TAR (FORCE=1 rebuilds)"
  exit 0
fi
[[ -n "$(find "$RAW_ROOT" -maxdepth 3 -type d -name 'scene_cam_*_final_hdf5' -print -quit 2> /dev/null)" ]] || {
  echo "ERROR: no extracted Hypersim scenes under RAW_ROOT=$RAW_ROOT" >&2
  echo "  Download them with DOWNLOAD_MODE=full scripts/dataset_prep/download_hypersim.sh." >&2
  exit 1
}
mkdir -p "$(dirname -- "$HYPERSIM_SEG_TAR")"

cd "$ROOT"
ovb_run_python -m dataset_prep.hypersim.prepare_hypersim_segmentation \
  --raw-root "$RAW_ROOT" \
  --out-root "$OUT_ROOT" \
  --all-frames \
  --val-ratio 0.02 \
  --min-labeled-fraction 0.10 \
  --max-side 0 \
  --num-workers "$NUM_WORKERS" \
  --seed 42 \
  --overwrite \
  --prepared-tar "$HYPERSIM_SEG_TAR"
echo "Hypersim segmentation archive: $HYPERSIM_SEG_TAR"
