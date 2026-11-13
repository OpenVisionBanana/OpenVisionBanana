#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Render Replica RGB-to-depth and RGB-to-surface-normal training data.

This script extracts Replica scene assets into scratch, renders samples with
Habitat-Sim, writes canonical prepared datasets, and persists prepared tars.

Usage:
  scripts/dataset_prep/prepare_replica_depth_normals.sh

Important environment variables:
  ROOT                         Repo root.
  USE_CONTAINER                1 to use SIF; 0 for the active environment. Default: 0.
  SIF                          Replica rendering container when USE_CONTAINER=1.
  PYTHON                       Python interpreter. Default: python.
  DATA_ROOT                    Persistent data root.
  REPLICA_ROOT                 Persistent Replica root.
  SCENE_ASSETS_TAR             Raw Replica scene-asset tar.
  SCRATCH_ROOT                 Usually under $TMPDIR.
  RAW_ROOT                     Extracted Replica root in scratch.
  PREPARED_ROOT                Scratch prepared-dataset parent.
  DEPTH_PREPARED_TAR           Final prepared depth tar.
  NORMALS_PREPARED_TAR         Final prepared normals tar.

Rendering:
  TASKS                        "depth normals", "depth", or "normals".
  NUM_SCENES                   "all" or integer. Default: all.
  SAMPLES_PER_SCENE            Default: 500.
  WIDTH, HEIGHT, HFOV          Default: 512, 512, 90.
  NORMAL_SOURCE                auto, rendered_from_mesh, or derived_from_depth.
  FORCE                        1 to rebuild outputs.
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
REPLICA_ROOT=${REPLICA_ROOT:-$DATA_ROOT/3d_geo/replica}
SCENE_ASSETS_TAR=${SCENE_ASSETS_TAR:-$REPLICA_ROOT/replica_v1_scene_assets.tar}
SCRATCH_ROOT=${SCRATCH_ROOT:-${TMPDIR:-/tmp/${USER:-$(id -un)}/replica_prep_${SLURM_JOB_ID:-$$}}/replica_prep}
RAW_ROOT=${RAW_ROOT:-$SCRATCH_ROOT/replica_v1}
PREPARED_ROOT=${PREPARED_ROOT:-$SCRATCH_ROOT/prepared}
DEPTH_OUT_ROOT=${DEPTH_OUT_ROOT:-$PREPARED_ROOT/replica_depth}
NORMALS_OUT_ROOT=${NORMALS_OUT_ROOT:-$PREPARED_ROOT/replica_surface_normals}
DEPTH_PREPARED_TAR=${DEPTH_PREPARED_TAR:-$REPLICA_ROOT/replica_depth.tar}
NORMALS_PREPARED_TAR=${NORMALS_PREPARED_TAR:-$REPLICA_ROOT/replica_surface_normals.tar}

TASKS=${TASKS:-depth normals}
FORCE=${FORCE:-0}
NUM_SCENES=${NUM_SCENES:-all}
SCENE_LIST=${SCENE_LIST:-}
SAMPLES_PER_SCENE=${SAMPLES_PER_SCENE:-500}
MAX_RETRY_FACTOR=${MAX_RETRY_FACTOR:-10}
SEED=${SEED:-42}
VAL_RATIO=${VAL_RATIO:-0}
WIDTH=${WIDTH:-512}
HEIGHT=${HEIGHT:-512}
HFOV=${HFOV:-90}
CAMERA_HEIGHT=${CAMERA_HEIGHT:-1.5}
PITCH_MIN=${PITCH_MIN:--15}
PITCH_MAX=${PITCH_MAX:-15}
MIN_DEPTH=${MIN_DEPTH:-0.1}
MAX_DEPTH=${MAX_DEPTH:-20.0}
MIN_VALID_FRACTION=${MIN_VALID_FRACTION:-0.95}
MIN_DEPTH_STD=${MIN_DEPTH_STD:-0.05}
MIN_RGB_MEAN=${MIN_RGB_MEAN:-2.0}
DEPTH_ENCODING=${DEPTH_ENCODING:-vision_banana_rgb}
DEPTH_INVALID_RGB=${DEPTH_INVALID_RGB:-0,0,0}
INVALID_NORMAL_RGB=${INVALID_NORMAL_RGB:-128,128,128}
NORMAL_ENCODING=${NORMAL_ENCODING:-vision_banana_x_left_red}
NORMAL_SOURCE=${NORMAL_SOURCE:-derived_from_depth}
DEPTH_CONVENTION=${DEPTH_CONVENTION:-projective_camera_z_meters}
SCENE_ID_MODE=${SCENE_ID_MODE:-auto}
GPU_DEVICE_ID=${GPU_DEVICE_ID:-0}
WRITE_RAW_NORMALS=${WRITE_RAW_NORMALS:-0}
RAW_NORMAL_DTYPE=${RAW_NORMAL_DTYPE:-float16}
ORIENT_Z_POSITIVE=${ORIENT_Z_POSITIVE:-1}
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

if [[ "$FORCE" != "1" ]]; then
  reuse_ok=1
  if task_enabled depth && [[ ! -s "$DEPTH_PREPARED_TAR" ]]; then
    reuse_ok=0
  fi
  if task_enabled normals && [[ ! -s "$NORMALS_PREPARED_TAR" ]]; then
    reuse_ok=0
  fi
  if [[ "$reuse_ok" == "1" ]]; then
    echo "Using existing Replica prepared tars for TASKS=$TASKS."
    task_enabled depth && echo "  depth tar:  $DEPTH_PREPARED_TAR"
    task_enabled normals && echo "  normal tar: $NORMALS_PREPARED_TAR"
    echo "Set FORCE=1 to rebuild."
    exit 0
  fi
fi

if [[ ! -d "$RAW_ROOT" ]]; then
  if [[ ! -s "$SCENE_ASSETS_TAR" ]]; then
    echo "ERROR: Replica scene-asset tar is missing: $SCENE_ASSETS_TAR" >&2
    echo "Run scripts/dataset_prep/download_replica.sh first." >&2
    exit 1
  fi
  mkdir -p "$SCRATCH_ROOT"
  echo "Extracting Replica scene assets into $SCRATCH_ROOT"
  tar -xf "$SCENE_ASSETS_TAR" -C "$SCRATCH_ROOT"
fi

if [[ ! -d "$RAW_ROOT" ]]; then
  echo "ERROR: expected extracted Replica root at $RAW_ROOT" >&2
  exit 1
fi

if [[ "$FORCE" == "1" ]]; then
  task_enabled depth && rm -rf "$DEPTH_OUT_ROOT" "$DEPTH_PREPARED_TAR" "$DEPTH_PREPARED_TAR.partial"
  task_enabled normals && rm -rf "$NORMALS_OUT_ROOT" "$NORMALS_PREPARED_TAR" "$NORMALS_PREPARED_TAR.partial"
fi

cleanup() {
  if [[ "$CLEAN_PREPARED_DIR_ON_EXIT" == "1" && "$PREPARED_ROOT" == "$SCRATCH_ROOT"/* ]]; then
    rm -rf "$PREPARED_ROOT"
  fi
}
trap cleanup EXIT

mkdir -p "$REPLICA_ROOT" "$SCRATCH_ROOT" "$PREPARED_ROOT"

# shellcheck disable=SC2206
task_args=($TASKS)
args=(
  dataset_prep/replica/prepare_replica_depth_normals.py
  --replica-root "$RAW_ROOT"
  --depth-out-root "$DEPTH_OUT_ROOT"
  --normals-out-root "$NORMALS_OUT_ROOT"
  --depth-prepared-tar "$DEPTH_PREPARED_TAR"
  --normals-prepared-tar "$NORMALS_PREPARED_TAR"
  --tasks "${task_args[@]}"
  --num-scenes "$NUM_SCENES"
  --samples-per-scene "$SAMPLES_PER_SCENE"
  --max-retry-factor "$MAX_RETRY_FACTOR"
  --seed "$SEED"
  --val-ratio "$VAL_RATIO"
  --width "$WIDTH"
  --height "$HEIGHT"
  --hfov "$HFOV"
  --camera-height "$CAMERA_HEIGHT"
  --pitch-min "$PITCH_MIN"
  --pitch-max "$PITCH_MAX"
  --min-depth "$MIN_DEPTH"
  --max-depth "$MAX_DEPTH"
  --min-depth-std "$MIN_DEPTH_STD"
  --min-valid-fraction "$MIN_VALID_FRACTION"
  --min-rgb-mean "$MIN_RGB_MEAN"
  --depth-encoding "$DEPTH_ENCODING"
  --depth-invalid-rgb "$DEPTH_INVALID_RGB"
  --invalid-normal-rgb "$INVALID_NORMAL_RGB"
  --normal-encoding "$NORMAL_ENCODING"
  --normal-source "$NORMAL_SOURCE"
  --depth-convention "$DEPTH_CONVENTION"
  --scene-id-mode "$SCENE_ID_MODE"
  --gpu-device-id "$GPU_DEVICE_ID"
  --raw-normal-dtype "$RAW_NORMAL_DTYPE"
  --overwrite
)
if [[ -n "$SCENE_LIST" ]]; then
  args+=(--scene-list "$SCENE_LIST")
fi
if is_true "$WRITE_RAW_NORMALS"; then
  args+=(--write-raw-normals)
fi
if ! is_true "$ORIENT_Z_POSITIVE"; then
  args+=(--no-orient-z-positive)
fi
if is_true "$INCLUDE_VAL_IN_METADATA"; then
  args+=(--include-val-in-metadata)
fi
if is_true "$STRICT"; then
  args+=(--strict)
fi

ovb_run_python "${args[@]}"

echo "Replica preparation finished."
echo "  raw root:   $RAW_ROOT"
echo "  depth tar:  $DEPTH_PREPARED_TAR"
echo "  normal tar: $NORMALS_PREPARED_TAR"
