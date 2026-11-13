#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Prepare Taskonomy RGB-to-depth and RGB-to-surface-normal training data.

The converter reads Taskonomy modality tars directly, writes prepared datasets
in scratch, and creates reusable prepared tars for inode-safe training.

Usage:
  scripts/dataset_prep/prepare_taskonomy_depth_normals.sh

Important environment variables:
  ROOT                         Repo root.
  USE_CONTAINER                1 to use SIF; 0 for the active environment. Default: 0.
  SIF                          Container image when USE_CONTAINER=1.
  PYTHON                       Python interpreter. Default: python.
  DATA_ROOT                    Persistent data root.
  TASKONOMY_ROOT               Persistent Taskonomy root.
  RAW_TAR_ROOT                 Directory with *_rgb.tar, *_depth_zbuffer.tar, *_normal.tar.
  SCRATCH_ROOT                 Usually $TMPDIR/taskonomy_prep_$SLURM_JOB_ID.
  PREPARED_ROOT                Scratch prepared-dataset parent.
  DEPTH_OUT_ROOT               Prepared depth folder.
  NORMALS_OUT_ROOT             Prepared normals folder.
  DEPTH_PREPARED_TAR           Final prepared depth tar.
  NORMALS_PREPARED_TAR         Final prepared normals tar.
  TASKS                        "depth normals", "depth", or "normals".
  FORCE                        1 to rebuild outputs.

Sampling and conversion:
  NUM_BUILDINGS                "all" or integer. Default: all.
  BUILDING_LIST                Optional selected building list.
  SAMPLES_PER_BUILDING         Default: 50.
  TARGET_SAMPLES               Optional total sample target.
  SEED                         Default: 42.
  TRAIN_RATIO                  Default: 1.0 (all prepared rows train).
  DEPTH_TYPE                   depth_zbuffer or depth_euclidean.
  DEPTH_ENCODING               Default: vision_banana_rgb.
  MIN_DEPTH, MAX_DEPTH         Valid metric depth range. Default: 0.1, 20.0.
  MIN_VALID_FRACTION           Skip samples below this valid ratio. Default: 0.70.
  MAX_SIDE                     Optional resize cap. 0 preserves native size.
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
TASKONOMY_ROOT=${TASKONOMY_ROOT:-$DATA_ROOT/depth_surfn_data/taskonomy}
RAW_TAR_ROOT=${RAW_TAR_ROOT:-$TASKONOMY_ROOT/raw_tars}
SCRATCH_ROOT=${SCRATCH_ROOT:-${TMPDIR:-/tmp/${USER:-$(id -un)}/taskonomy_prep_${SLURM_JOB_ID:-$$}}}
PREPARED_ROOT=${PREPARED_ROOT:-$SCRATCH_ROOT/prepared}
DEPTH_OUT_ROOT=${DEPTH_OUT_ROOT:-$PREPARED_ROOT/taskonomy_depth}
NORMALS_OUT_ROOT=${NORMALS_OUT_ROOT:-$PREPARED_ROOT/taskonomy_surface_normals}
DEPTH_PREPARED_TAR=${DEPTH_PREPARED_TAR:-$TASKONOMY_ROOT/taskonomy_depth.tar}
NORMALS_PREPARED_TAR=${NORMALS_PREPARED_TAR:-$TASKONOMY_ROOT/taskonomy_surface_normals.tar}

TASKS=${TASKS:-depth normals}
FORCE=${FORCE:-0}
NUM_BUILDINGS=${NUM_BUILDINGS:-all}
BUILDING_LIST=${BUILDING_LIST:-}
SAMPLES_PER_BUILDING=${SAMPLES_PER_BUILDING:-50}
TARGET_SAMPLES=${TARGET_SAMPLES:-}
ALL_FRAMES=${ALL_FRAMES:-0}
SEED=${SEED:-42}
TRAIN_RATIO=${TRAIN_RATIO:-1.0}
DEPTH_TYPE=${DEPTH_TYPE:-depth_zbuffer}
MIN_DEPTH=${MIN_DEPTH:-0.1}
MAX_DEPTH=${MAX_DEPTH:-20.0}
MIN_VALID_FRACTION=${MIN_VALID_FRACTION:-0.70}
MAX_SIDE=${MAX_SIDE:-0}
DEPTH_ENCODING=${DEPTH_ENCODING:-vision_banana_rgb}
DEPTH_INVALID_RGB=${DEPTH_INVALID_RGB:-0,0,0}
INVALID_NORMAL_RGB=${INVALID_NORMAL_RGB:-128,128,128}
NORMAL_ENCODING=${NORMAL_ENCODING:-vision_banana_x_left_red}
WRITE_RAW_NORMALS=${WRITE_RAW_NORMALS:-0}
RAW_NORMAL_DTYPE=${RAW_NORMAL_DTYPE:-float16}
INCLUDE_VAL_IN_METADATA=${INCLUDE_VAL_IN_METADATA:-0}
STRICT=${STRICT:-0}
CLEAN_PREPARED_DIR_ON_EXIT=${CLEAN_PREPARED_DIR_ON_EXIT:-1}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

task_enabled() {
  local wanted=$1
  local task
  # shellcheck disable=SC2206
  local tasks=($TASKS)
  for task in "${tasks[@]}"; do
    [[ "$task" == "$wanted" ]] && return 0
  done
  return 1
}

if [[ ! -d "$RAW_TAR_ROOT" ]]; then
  echo "ERROR: RAW_TAR_ROOT does not exist: $RAW_TAR_ROOT" >&2
  echo "Run scripts/dataset_prep/download_taskonomy.sh first, or set RAW_TAR_ROOT." >&2
  exit 1
fi

if [[ "$FORCE" != "1" ]]; then
  reuse_ok=1
  if task_enabled depth && [[ ! -s "$DEPTH_OUT_ROOT/metadata.jsonl" || ! -s "$DEPTH_PREPARED_TAR" ]]; then
    reuse_ok=0
  fi
  if task_enabled normals && [[ ! -s "$NORMALS_OUT_ROOT/metadata.jsonl" || ! -s "$NORMALS_PREPARED_TAR" ]]; then
    reuse_ok=0
  fi
  if [[ "$reuse_ok" == "1" ]]; then
    echo "Using existing Taskonomy prepared datasets and tars for TASKS=$TASKS."
    task_enabled depth && echo "  depth:   $DEPTH_OUT_ROOT"
    task_enabled normals && echo "  normals: $NORMALS_OUT_ROOT"
    echo "Set FORCE=1 to rebuild."
    exit 0
  fi
fi

if [[ "$FORCE" == "1" ]]; then
  if task_enabled depth; then
    rm -rf "$DEPTH_OUT_ROOT"
    rm -f "$DEPTH_PREPARED_TAR" "$DEPTH_PREPARED_TAR.partial"
  fi
  if task_enabled normals; then
    rm -rf "$NORMALS_OUT_ROOT"
    rm -f "$NORMALS_PREPARED_TAR" "$NORMALS_PREPARED_TAR.partial"
  fi
fi

cleanup() {
  if [[ "$CLEAN_PREPARED_DIR_ON_EXIT" == "1" && "$PREPARED_ROOT" == "$SCRATCH_ROOT"/* ]]; then
    rm -rf "$PREPARED_ROOT"
  fi
}
trap cleanup EXIT

mkdir -p "$TASKONOMY_ROOT" "$SCRATCH_ROOT" "$PREPARED_ROOT" "$(dirname "$DEPTH_PREPARED_TAR")" "$(dirname "$NORMALS_PREPARED_TAR")"

# shellcheck disable=SC2206
task_args=($TASKS)
args=(
  dataset_prep/taskonomy/prepare_taskonomy_depth_normals.py
  --raw-tar-root "$RAW_TAR_ROOT"
  --depth-out-root "$DEPTH_OUT_ROOT"
  --normals-out-root "$NORMALS_OUT_ROOT"
  --depth-prepared-tar "$DEPTH_PREPARED_TAR"
  --normals-prepared-tar "$NORMALS_PREPARED_TAR"
  --tasks "${task_args[@]}"
  --depth-type "$DEPTH_TYPE"
  --num-buildings "$NUM_BUILDINGS"
  --seed "$SEED"
  --train-ratio "$TRAIN_RATIO"
  --min-depth "$MIN_DEPTH"
  --max-depth "$MAX_DEPTH"
  --min-valid-fraction "$MIN_VALID_FRACTION"
  --max-side "$MAX_SIDE"
  --depth-encoding "$DEPTH_ENCODING"
  --depth-invalid-rgb "$DEPTH_INVALID_RGB"
  --invalid-normal-rgb "$INVALID_NORMAL_RGB"
  --normal-encoding "$NORMAL_ENCODING"
  --raw-normal-dtype "$RAW_NORMAL_DTYPE"
  --overwrite
)
if [[ -n "$BUILDING_LIST" ]]; then
  args+=(--building-list "$BUILDING_LIST")
fi
if [[ -n "$TARGET_SAMPLES" ]]; then
  args+=(--target-samples "$TARGET_SAMPLES")
elif [[ -n "$SAMPLES_PER_BUILDING" ]]; then
  args+=(--samples-per-building "$SAMPLES_PER_BUILDING")
fi
if is_true "$ALL_FRAMES"; then
  args+=(--all-frames)
fi
if is_true "$WRITE_RAW_NORMALS"; then
  args+=(--write-raw-normals)
fi
if is_true "$INCLUDE_VAL_IN_METADATA"; then
  args+=(--include-val-in-metadata)
fi
if is_true "$STRICT"; then
  args+=(--strict)
fi

ovb_run_python "${args[@]}"

echo "Taskonomy preparation finished."
echo "  raw tars:    $RAW_TAR_ROOT"
echo "  depth root:  $DEPTH_OUT_ROOT"
echo "  depth tar:   $DEPTH_PREPARED_TAR"
echo "  normal root: $NORMALS_OUT_ROOT"
echo "  normal tar:  $NORMALS_PREPARED_TAR"
