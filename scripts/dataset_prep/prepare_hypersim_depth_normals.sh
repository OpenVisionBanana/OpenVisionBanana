#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Prepare Hypersim RGB-to-depth and RGB-to-surface-normal training data.

The output is two canonical Open Vision Banana prepared datasets plus reusable
tar archives:

  hypersim_depth/
    metadata.jsonl
    conditioning_images/
    target_images/
    depth_raw/
    valid_masks/

  hypersim_surface_normals/
    metadata.jsonl
    conditioning_images/
    target_images/
    masks/

Usage:
  scripts/dataset_prep/prepare_hypersim_depth_normals.sh

Important environment variables:
  ROOT                         Repo root.
  USE_CONTAINER                1 to use SIF; 0 for the active environment. Default: 0.
  SIF                          Container image when USE_CONTAINER=1.
  PYTHON                       Python interpreter. Default: python.
  DATA_ROOT                    Persistent data root.
  HYPERSIM_ROOT                Persistent Hypersim root.
  RAW_ROOT                     Extracted Hypersim scene root.
  DEPTH_OUT_ROOT               Prepared depth folder.
  NORMALS_OUT_ROOT             Prepared normals folder.
  DEPTH_PREPARED_TAR           Prepared depth tar.
  NORMALS_PREPARED_TAR         Prepared normals tar.
  TASKS                        "depth normals", "depth", or "normals".
  FORCE                        1 to rebuild outputs.

Sampling and conversion:
  SAMPLES_PER_SCENE            Default: 25.
  TARGET_SAMPLES               Optional exact-ish scene-balanced total.
  SEED                         Default: 42.
  VAL_RATIO                    Scene-level validation holdout. Default: 0 (training-only).
  MIN_DEPTH, MAX_DEPTH         Valid metric depth range. Default: 0.1, 20.0.
  MIN_VALID_FRACTION           Skip samples below this valid ratio. Default: 0.80.
  MAX_SIDE                     Optional resize cap. 0 preserves native size.
  NUM_WORKERS                  Worker threads for extracted-folder input.
  WRITE_RAW_NORMALS            1 to include raw_normals/*.npz sidecars.
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
RAW_ROOT=${RAW_ROOT:-${TMPDIR:-$HYPERSIM_ROOT/work}/raw_extracted}
PREPARED_ROOT=${PREPARED_ROOT:-$HYPERSIM_ROOT/prepared}
DEPTH_OUT_ROOT=${DEPTH_OUT_ROOT:-$PREPARED_ROOT/hypersim_depth}
NORMALS_OUT_ROOT=${NORMALS_OUT_ROOT:-$PREPARED_ROOT/hypersim_surface_normals}
DEPTH_PREPARED_TAR=${DEPTH_PREPARED_TAR:-$HYPERSIM_ROOT/hypersim_depth.tar}
NORMALS_PREPARED_TAR=${NORMALS_PREPARED_TAR:-$HYPERSIM_ROOT/hypersim_surface_normals.tar}

TASKS=${TASKS:-depth normals}
FORCE=${FORCE:-0}
SAMPLES_PER_SCENE=${SAMPLES_PER_SCENE:-25}
TARGET_SAMPLES=${TARGET_SAMPLES:-}
SEED=${SEED:-42}
VAL_RATIO=${VAL_RATIO:-0}
MIN_DEPTH=${MIN_DEPTH:-0.1}
MAX_DEPTH=${MAX_DEPTH:-20.0}
MIN_VALID_FRACTION=${MIN_VALID_FRACTION:-0.80}
MAX_SIDE=${MAX_SIDE:-0}
NUM_WORKERS=${NUM_WORKERS:-1}
DEPTH_ENCODING=${DEPTH_ENCODING:-vision_banana_rgb}
DEPTH_INVALID_RGB=${DEPTH_INVALID_RGB:-0,0,0}
INVALID_NORMAL_RGB=${INVALID_NORMAL_RGB:-128,128,128}
NORMAL_ENCODING=${NORMAL_ENCODING:-vision_banana_x_left_red}
CAMERA_PARAMETERS_CSV=${CAMERA_PARAMETERS_CSV:-$HYPERSIM_ROOT/metadata_camera_parameters.csv}
WRITE_RAW_NORMALS=${WRITE_RAW_NORMALS:-0}
RAW_NORMAL_DTYPE=${RAW_NORMAL_DTYPE:-float16}
INCLUDE_VAL_IN_METADATA=${INCLUDE_VAL_IN_METADATA:-0}
DELETE_RAW_SCENES_AFTER_CONVERT=${DELETE_RAW_SCENES_AFTER_CONVERT:-0}
STRICT=${STRICT:-0}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ ! -d "$RAW_ROOT" ]]; then
  echo "ERROR: RAW_ROOT does not exist: $RAW_ROOT" >&2
  echo "Run scripts/dataset_prep/download_hypersim.sh first, or set RAW_ROOT." >&2
  exit 1
fi
if [[ ! -s "$CAMERA_PARAMETERS_CSV" ]]; then
  echo "ERROR: official Hypersim camera metadata is missing: $CAMERA_PARAMETERS_CSV" >&2
  echo "Set CAMERA_PARAMETERS_CSV to metadata_camera_parameters.csv." >&2
  exit 1
fi

if [[ -s "$DEPTH_OUT_ROOT/metadata.jsonl" && -s "$NORMALS_OUT_ROOT/metadata.jsonl" \
   && -s "$DEPTH_PREPARED_TAR" && -s "$NORMALS_PREPARED_TAR" && "$FORCE" != "1" ]]; then
  echo "Using existing Hypersim prepared datasets and tars."
  echo "  depth:   $DEPTH_OUT_ROOT"
  echo "  normals: $NORMALS_OUT_ROOT"
  echo "Set FORCE=1 to rebuild."
  exit 0
fi

if [[ "$FORCE" == "1" ]]; then
  rm -rf "$DEPTH_OUT_ROOT" "$NORMALS_OUT_ROOT"
  rm -f "$DEPTH_PREPARED_TAR" "$DEPTH_PREPARED_TAR.partial"
  rm -f "$NORMALS_PREPARED_TAR" "$NORMALS_PREPARED_TAR.partial"
fi

mkdir -p "$PREPARED_ROOT" "$(dirname "$DEPTH_PREPARED_TAR")" "$(dirname "$NORMALS_PREPARED_TAR")"

# shellcheck disable=SC2206
task_args=($TASKS)
args=(
  dataset_prep/hypersim/prepare_hypersim_depth_normals.py
  --raw-root "$RAW_ROOT"
  --camera-parameters-csv "$CAMERA_PARAMETERS_CSV"
  --depth-out-root "$DEPTH_OUT_ROOT"
  --normals-out-root "$NORMALS_OUT_ROOT"
  --depth-prepared-tar "$DEPTH_PREPARED_TAR"
  --normals-prepared-tar "$NORMALS_PREPARED_TAR"
  --tasks "${task_args[@]}"
  --seed "$SEED"
  --val-ratio "$VAL_RATIO"
  --min-depth "$MIN_DEPTH"
  --max-depth "$MAX_DEPTH"
  --min-valid-fraction "$MIN_VALID_FRACTION"
  --max-side "$MAX_SIDE"
  --num-workers "$NUM_WORKERS"
  --depth-encoding "$DEPTH_ENCODING"
  --depth-invalid-rgb "$DEPTH_INVALID_RGB"
  --invalid-normal-rgb "$INVALID_NORMAL_RGB"
  --normal-encoding "$NORMAL_ENCODING"
  --raw-normal-dtype "$RAW_NORMAL_DTYPE"
  --overwrite
)
if [[ -n "$TARGET_SAMPLES" ]]; then
  args+=(--target-samples "$TARGET_SAMPLES")
elif [[ -n "$SAMPLES_PER_SCENE" ]]; then
  args+=(--samples-per-scene "$SAMPLES_PER_SCENE")
fi
if is_true "$WRITE_RAW_NORMALS"; then
  args+=(--write-raw-normals)
fi
if is_true "$INCLUDE_VAL_IN_METADATA"; then
  args+=(--include-val-in-metadata)
fi
if is_true "$DELETE_RAW_SCENES_AFTER_CONVERT"; then
  args+=(--delete-raw-scenes-after-convert)
fi
if is_true "$STRICT"; then
  args+=(--strict)
fi

ovb_run_python "${args[@]}"

echo "Hypersim preparation finished."
echo "  raw root:    $RAW_ROOT"
echo "  depth root:  $DEPTH_OUT_ROOT"
echo "  depth tar:   $DEPTH_PREPARED_TAR"
echo "  normal root: $NORMALS_OUT_ROOT"
echo "  normal tar:  $NORMALS_PREPARED_TAR"
