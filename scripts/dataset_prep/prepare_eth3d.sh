#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Prepare ETH3D low-res two-view data for depth and derived surface-normal evaluation.

Usage:
  scripts/dataset_prep/prepare_eth3d.sh

Environment:
  ROOT                    Repo root.
  USE_CONTAINER           1 to use SIF; 0 for the active environment. Default: 0.
  SIF                     Container image when USE_CONTAINER=1.
  PYTHON                  Python interpreter. Default: python.
  DATA_ROOT               ETH3D raw/prepared root.
  TASK                    depth, surface_normals, or both. Default: both.
  DEPTH_OUT_ROOT          Prepared depth folder.
  NORMALS_OUT_ROOT        Prepared derived-normal folder.
  DEPTH_TAR_OUTPUT        Prepared depth tar.
  NORMALS_TAR_OUTPUT      Prepared derived-normal tar.
  DOWNLOAD_IF_MISSING     1 to download missing .7z archives. Default: 0.
  FORCE                   1 to rebuild output folders/tars. Default: 0.
  BASELINE_UNIT           auto, meters, or millimeters. Default: auto.
  INSTALL_PY7ZR_IF_MISSING
                          1 to install py7zr into PYDEPS_DIR if no extractor is
                          present in the container. Default: 0.
  PYDEPS_DIR              Local Python dependency target for py7zr.
  MIN_NORMAL_VALID_PIXELS Skip derived-normal samples below this count. Default: 4096.
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
TASK=${TASK:-both}
DEPTH_OUT_ROOT=${DEPTH_OUT_ROOT:-$DATA_ROOT/eth3d_low_res_two_view_depth}
NORMALS_OUT_ROOT=${NORMALS_OUT_ROOT:-$DATA_ROOT/eth3d_low_res_two_view_surface_normals}
DEPTH_TAR_OUTPUT=${DEPTH_TAR_OUTPUT:-$DATA_ROOT/eth3d_low_res_two_view_depth.tar}
NORMALS_TAR_OUTPUT=${NORMALS_TAR_OUTPUT:-$DATA_ROOT/eth3d_low_res_two_view_surface_normals.tar}
DOWNLOAD_IF_MISSING=${DOWNLOAD_IF_MISSING:-0}
FORCE=${FORCE:-0}
FORCE_EXTRACT=${FORCE_EXTRACT:-0}
BASELINE_UNIT=${BASELINE_UNIT:-auto}
INSTALL_PY7ZR_IF_MISSING=${INSTALL_PY7ZR_IF_MISSING:-0}
PYDEPS_DIR=${PYDEPS_DIR:-$DATA_ROOT/.python_deps/py7zr}
MIN_DEPTH=${MIN_DEPTH:-0.001}
MAX_DEPTH=${MAX_DEPTH:-1000.0}
GRADIENT_RADIUS=${GRADIENT_RADIUS:-2}
MASK_EROSION_RADIUS=${MASK_EROSION_RADIUS:-1}
MAX_NEIGHBOR_DEPTH_RATIO=${MAX_NEIGHBOR_DEPTH_RATIO:-0.1}
MIN_NORMAL_VALID_PIXELS=${MIN_NORMAL_VALID_PIXELS:-4096}
INVALID_NORMAL_RGB=${INVALID_NORMAL_RGB:-128,128,128}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ (! -f "$DATA_ROOT/two_view_training.7z" || ! -f "$DATA_ROOT/two_view_training_gt.7z") && ! -d "$DATA_ROOT/two_view_training" ]] && is_true "$DOWNLOAD_IF_MISSING"; then
  DATA_ROOT="$DATA_ROOT" "$ROOT/scripts/dataset_prep/download_eth3d.sh"
fi

if [[ ! -d "$DATA_ROOT/two_view_training" && (! -f "$DATA_ROOT/two_view_training.7z" || ! -f "$DATA_ROOT/two_view_training_gt.7z") ]]; then
  echo "ERROR: Missing ETH3D raw folders or .7z archives under: $DATA_ROOT" >&2
  exit 1
fi

if [[ "$TASK" == "depth" || "$TASK" == "both" ]]; then
  if [[ -s "$DEPTH_TAR_OUTPUT" ]] && ! is_true "$FORCE"; then
    echo "Using existing ETH3D depth tar: $DEPTH_TAR_OUTPUT"
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
    echo "Using existing ETH3D surface-normal tar: $NORMALS_TAR_OUTPUT"
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
  dataset_prep/eth3d/prepare_eth3d_eval.py
  --eth3d-root "$DATA_ROOT"
  --task "$TASK"
  --depth-out-root "$DEPTH_OUT_ROOT"
  --normals-out-root "$NORMALS_OUT_ROOT"
  --depth-tar-output "$DEPTH_TAR_OUTPUT"
  --normals-tar-output "$NORMALS_TAR_OUTPUT"
  --baseline-unit "$BASELINE_UNIT"
  --min-depth "$MIN_DEPTH"
  --max-depth "$MAX_DEPTH"
  --gradient-radius "$GRADIENT_RADIUS"
  --mask-erosion-radius "$MASK_EROSION_RADIUS"
  --max-neighbor-depth-ratio "$MAX_NEIGHBOR_DEPTH_RATIO"
  --min-normal-valid-pixels "$MIN_NORMAL_VALID_PIXELS"
  --invalid-normal-rgb "$INVALID_NORMAL_RGB"
  --overwrite
)

if is_true "$FORCE_EXTRACT"; then
  args+=(--force-extract)
fi

if is_true "$INSTALL_PY7ZR_IF_MISSING" && ! ovb_run_python -c 'import py7zr' >/dev/null 2>&1; then
  ovb_run_python -m pip install --quiet --target "$PYDEPS_DIR" py7zr
  export PYTHONPATH="$PYDEPS_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi
ovb_run_python "${args[@]}"

echo "ETH3D preparation finished."
echo "  raw root:    $DATA_ROOT"
echo "  depth tar:   $DEPTH_TAR_OUTPUT"
echo "  normals tar: $NORMALS_TAR_OUTPUT"
