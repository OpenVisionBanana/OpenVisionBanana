#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Prepare KITTI depth-selection validation data for depth and derived surface-normal evaluation.

Usage:
  scripts/dataset_prep/prepare_kitti.sh

Environment:
  ROOT                    Repo root.
  USE_CONTAINER           1 to use SIF; 0 for the active environment. Default: 0.
  SIF                     Container image when USE_CONTAINER=1.
  PYTHON                  Python interpreter. Default: python.
  DATA_ROOT               KITTI archive/prepared-data root.
  SELECTION_ZIP           KITTI data_depth_selection.zip path.
  TASK                    depth, surface_normals, or both. Default: both.
  DEPTH_OUT_ROOT          Prepared depth folder.
  NORMALS_OUT_ROOT        Prepared derived-normal folder.
  DEPTH_TAR_OUTPUT        Prepared depth tar.
  NORMALS_TAR_OUTPUT      Prepared derived-normal tar.
  DOWNLOAD_IF_MISSING     1 to download data_depth_selection.zip if missing. Default: 0.
  FORCE                   1 to rebuild output folders/tars. Default: 0.
  MIN_NORMAL_VALID_PIXELS Skip derived-normal samples below this valid-pixel count. Default: 4096.
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
SELECTION_ZIP=${SELECTION_ZIP:-$DATA_ROOT/data_depth_selection.zip}
TASK=${TASK:-both}
DEPTH_OUT_ROOT=${DEPTH_OUT_ROOT:-$DATA_ROOT/kitti_depth_selection}
NORMALS_OUT_ROOT=${NORMALS_OUT_ROOT:-$DATA_ROOT/kitti_depth_selection_surface_normals}
DEPTH_TAR_OUTPUT=${DEPTH_TAR_OUTPUT:-$DATA_ROOT/kitti_depth_selection.tar}
NORMALS_TAR_OUTPUT=${NORMALS_TAR_OUTPUT:-$DATA_ROOT/kitti_depth_selection_surface_normals.tar}
DOWNLOAD_IF_MISSING=${DOWNLOAD_IF_MISSING:-0}
FORCE=${FORCE:-0}
MIN_DEPTH=${MIN_DEPTH:-0.001}
MAX_DEPTH=${MAX_DEPTH:-80.0}
GRADIENT_RADIUS=${GRADIENT_RADIUS:-2}
MASK_EROSION_RADIUS=${MASK_EROSION_RADIUS:-0}
MAX_NEIGHBOR_DEPTH_RATIO=${MAX_NEIGHBOR_DEPTH_RATIO:-0.1}
MIN_NORMAL_VALID_PIXELS=${MIN_NORMAL_VALID_PIXELS:-4096}
INVALID_NORMAL_RGB=${INVALID_NORMAL_RGB:-128,128,128}
NORMAL_ENCODING=${NORMAL_ENCODING:-vision_banana_x_left_red}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ ! -f "$SELECTION_ZIP" ]] && is_true "$DOWNLOAD_IF_MISSING"; then
  DATA_ROOT="$DATA_ROOT" "$ROOT/scripts/dataset_prep/download_kitti_depth.sh"
fi

if [[ ! -f "$SELECTION_ZIP" ]]; then
  echo "ERROR: Missing KITTI data_depth_selection.zip: $SELECTION_ZIP" >&2
  exit 1
fi

if [[ "$TASK" == "depth" || "$TASK" == "both" ]]; then
  if [[ -s "$DEPTH_TAR_OUTPUT" ]] && ! is_true "$FORCE"; then
    echo "Using existing KITTI depth tar: $DEPTH_TAR_OUTPUT"
    echo "Set FORCE=1 to rebuild."
    DEPTH_DONE=1
  else
    DEPTH_DONE=0
  fi
else
  DEPTH_DONE=1
fi

if [[ "$TASK" == "surface_normals" || "$TASK" == "both" ]]; then
  if [[ -s "$NORMALS_TAR_OUTPUT" ]] && ! is_true "$FORCE"; then
    echo "Using existing KITTI surface-normal tar: $NORMALS_TAR_OUTPUT"
    echo "Set FORCE=1 to rebuild."
    NORMALS_DONE=1
  else
    NORMALS_DONE=0
  fi
else
  NORMALS_DONE=1
fi

if [[ "$DEPTH_DONE" == "1" && "$NORMALS_DONE" == "1" ]]; then
  exit 0
fi

if is_true "$FORCE"; then
  if [[ "$TASK" == "depth" || "$TASK" == "both" ]]; then
    rm -rf "$DEPTH_OUT_ROOT"
    rm -f "$DEPTH_TAR_OUTPUT" "$DEPTH_TAR_OUTPUT.partial"
  fi
  if [[ "$TASK" == "surface_normals" || "$TASK" == "both" ]]; then
    rm -rf "$NORMALS_OUT_ROOT"
    rm -f "$NORMALS_TAR_OUTPUT" "$NORMALS_TAR_OUTPUT.partial"
  fi
fi

args=(
  dataset_prep/kitti/prepare_kitti_eval.py
  --selection-zip "$SELECTION_ZIP"
  --task "$TASK"
  --depth-out-root "$DEPTH_OUT_ROOT"
  --normals-out-root "$NORMALS_OUT_ROOT"
  --depth-tar-output "$DEPTH_TAR_OUTPUT"
  --normals-tar-output "$NORMALS_TAR_OUTPUT"
  --min-depth "$MIN_DEPTH"
  --max-depth "$MAX_DEPTH"
  --gradient-radius "$GRADIENT_RADIUS"
  --mask-erosion-radius "$MASK_EROSION_RADIUS"
  --max-neighbor-depth-ratio "$MAX_NEIGHBOR_DEPTH_RATIO"
  --min-normal-valid-pixels "$MIN_NORMAL_VALID_PIXELS"
  --invalid-normal-rgb "$INVALID_NORMAL_RGB"
  --normal-encoding "$NORMAL_ENCODING"
  --overwrite
)

ovb_run_python "${args[@]}"

echo "KITTI preparation finished."
echo "  selection zip: $SELECTION_ZIP"
echo "  depth tar:     $DEPTH_TAR_OUTPUT"
echo "  normals tar:   $NORMALS_TAR_OUTPUT"
