#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Prepare Synscapes RGB-to-surface-normal training data from existing RGB/depth
EXR files.

The output is a canonical Open Vision Banana prepared dataset:

  metadata.jsonl
  conditioning_images/
  target_images/
  masks/

Usage:
  scripts/dataset_prep/prepare_synscapes_surface_normals.sh

Important environment variables:
  ROOT                    Repo root. Defaults to the current checkout.
  USE_CONTAINER           1 to use SIF; 0 for the active environment. Default: 0.
  SIF                     Container image when USE_CONTAINER=1.
  PYTHON                  Python interpreter. Default: python.
  DATA_ROOT               Persistent dataset root.
  SYNSCAPES_ROOT          Optional extracted Synscapes root or parent.
  SYNSCAPES_TAR           Synscapes source tar when SYNSCAPES_ROOT is unset.
  OUT_ROOT                Generated prepared dataset folder.
  PREPARED_TAR            Optional generated tar archive.
  FORCE                   1 to rebuild OUT_ROOT/PREPARED_TAR.

Surface-normal controls:
  RGB_SUBDIR              rgb or rgb-2k. Default: rgb.
  DEPTH_CHANNEL           first, last, mean, red, green, blue. Default: first.
  DEPTH_KIND              z or ray. Default: z.
  MIN_DEPTH, MAX_DEPTH    Valid z-depth range in meters.
  GRADIENT_RADIUS         Pixel radius for central-difference normals.
  MASK_EROSION_RADIUS     Erode valid-normal mask by this many pixels.
  MAX_DEPTH_DELTA         Optional normal-validity depth jump filter. 0 disables.
  MIN_VALID_FRACTION      Skip samples below this valid-mask fraction.
  MAX_SIDE                Optional resize cap. 0 preserves native size.
  WRITE_RAW_NORMALS       1 to include raw_normals/*.npz sidecars.
  LIMIT                   Optional sample limit.
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
SYNSCAPES_ROOT=${SYNSCAPES_ROOT:-}
SYNSCAPES_TAR=${SYNSCAPES_TAR:-$DATA_ROOT/depth_est/synscapes/synscapes.tar}
PREPARED_ROOT=${PREPARED_ROOT:-$DATA_ROOT/open_visionbanana_prepared}
OUT_ROOT=${OUT_ROOT:-$PREPARED_ROOT/surface_normals_synscapes}
PREPARED_TAR=${PREPARED_TAR:-$PREPARED_ROOT/surface_normals_synscapes.tar}

FORCE=${FORCE:-0}
LIMIT=${LIMIT:-}
SEED=${SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-1}
MAX_SIDE=${MAX_SIDE:-0}
RGB_SUBDIR=${RGB_SUBDIR:-rgb}
DEPTH_CHANNEL=${DEPTH_CHANNEL:-first}
DEPTH_KIND=${DEPTH_KIND:-z}
MIN_DEPTH=${MIN_DEPTH:-0.1}
MAX_DEPTH=${MAX_DEPTH:-250.0}
GRADIENT_RADIUS=${GRADIENT_RADIUS:-2}
MASK_EROSION_RADIUS=${MASK_EROSION_RADIUS:-1}
MAX_DEPTH_DELTA=${MAX_DEPTH_DELTA:-0.0}
MIN_VALID_FRACTION=${MIN_VALID_FRACTION:-0.05}
INVALID_NORMAL_RGB=${INVALID_NORMAL_RGB:-128,128,128}
NORMAL_ENCODING=${NORMAL_ENCODING:-vision_banana_x_left_red}
WRITE_RAW_NORMALS=${WRITE_RAW_NORMALS:-0}
RAW_NORMAL_DTYPE=${RAW_NORMAL_DTYPE:-float16}
ORIENT_Z_POSITIVE=${ORIENT_Z_POSITIVE:-1}
STRICT=${STRICT:-0}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -s "$OUT_ROOT/metadata.jsonl" && -s "$PREPARED_TAR" && "$FORCE" != "1" ]]; then
  echo "Using existing Synscapes surface-normal prepared dataset: $OUT_ROOT"
  echo "Using existing Synscapes surface-normal tar: $PREPARED_TAR"
  echo "Set FORCE=1 to rebuild."
  exit 0
fi

if [[ "$FORCE" == "1" ]]; then
  rm -rf "$OUT_ROOT"
  rm -f "$PREPARED_TAR" "$PREPARED_TAR.partial"
fi

args=(
  dataset_prep/surface_normals/prepare_synscapes_surface_normals.py
  --out-root "$OUT_ROOT"
  --prepared-tar "$PREPARED_TAR"
  --seed "$SEED"
  --num-workers "$NUM_WORKERS"
  --max-side "$MAX_SIDE"
  --rgb-subdir "$RGB_SUBDIR"
  --depth-channel "$DEPTH_CHANNEL"
  --depth-kind "$DEPTH_KIND"
  --min-depth "$MIN_DEPTH"
  --max-depth "$MAX_DEPTH"
  --gradient-radius "$GRADIENT_RADIUS"
  --mask-erosion-radius "$MASK_EROSION_RADIUS"
  --max-depth-delta "$MAX_DEPTH_DELTA"
  --min-valid-fraction "$MIN_VALID_FRACTION"
  --invalid-normal-rgb "$INVALID_NORMAL_RGB"
  --normal-encoding "$NORMAL_ENCODING"
  --raw-normal-dtype "$RAW_NORMAL_DTYPE"
  --overwrite
)
if [[ -n "$SYNSCAPES_ROOT" && -d "$SYNSCAPES_ROOT" ]]; then
  args+=(--synscapes-root "$SYNSCAPES_ROOT")
elif [[ -f "$SYNSCAPES_TAR" ]]; then
  args+=(--synscapes-tar "$SYNSCAPES_TAR")
else
  echo "ERROR: Synscapes source is missing. Set SYNSCAPES_ROOT or SYNSCAPES_TAR." >&2
  echo "  SYNSCAPES_ROOT=$SYNSCAPES_ROOT" >&2
  echo "  SYNSCAPES_TAR=$SYNSCAPES_TAR" >&2
  exit 1
fi
if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi
if is_true "$WRITE_RAW_NORMALS"; then
  args+=(--write-raw-normals)
fi
if ! is_true "$ORIENT_Z_POSITIVE"; then
  args+=(--no-orient-z-positive)
fi
if is_true "$STRICT"; then
  args+=(--strict)
fi

OPENCV_IO_ENABLE_OPENEXR=1 ovb_run_python "${args[@]}"

echo "Synscapes surface-normal preparation finished."
echo "  out root:     $OUT_ROOT"
echo "  prepared tar: $PREPARED_TAR"
