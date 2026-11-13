#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Prepare DIODE validation data for depth and surface-normal evaluation.

Usage:
  scripts/dataset_prep/prepare_diode.sh

Environment:
  ROOT                    Repo root.
  USE_CONTAINER           1 to use SIF; 0 for the active environment. Default: 0.
  SIF                     Container image when USE_CONTAINER=1.
  PYTHON                  Python interpreter. Default: python.
  DATA_ROOT               Raw DIODE root.
  TASK                    depth, surface_normals, or both. Default: both.
  DOMAINS                 all, indoor, outdoor, or "indoor outdoor". Default: all.
  DEPTH_OUT_ROOT          Prepared depth folder.
  NORMALS_OUT_ROOT        Prepared surface-normal folder.
  DEPTH_TAR_OUTPUT        Prepared depth tar.
  NORMALS_TAR_OUTPUT      Prepared surface-normal tar.
  DOWNLOAD_IF_MISSING     1 to run download_diode.sh when raw files are missing. Default: 0.
  FORCE                   1 to rebuild output folders/tars.
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
DOMAINS=${DOMAINS:-all}
DEPTH_OUT_ROOT=${DEPTH_OUT_ROOT:-$DATA_ROOT/diode_depth}
NORMALS_OUT_ROOT=${NORMALS_OUT_ROOT:-$DATA_ROOT/diode_surface_normals}
DEPTH_TAR_OUTPUT=${DEPTH_TAR_OUTPUT:-$DATA_ROOT/diode_depth.tar}
NORMALS_TAR_OUTPUT=${NORMALS_TAR_OUTPUT:-$DATA_ROOT/diode_surface_normals.tar}
DOWNLOAD_IF_MISSING=${DOWNLOAD_IF_MISSING:-0}
FORCE=${FORCE:-0}
MIN_DEPTH=${MIN_DEPTH:-0.001}
MAX_DEPTH=${MAX_DEPTH:-300.0}
MIN_NORMAL_NORM=${MIN_NORMAL_NORM:-0.01}
INVALID_NORMAL_RGB=${INVALID_NORMAL_RGB:-128,128,128}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ ! -d "$DATA_ROOT/val" || ! -d "$DATA_ROOT/val_normal" || ! -d "$DATA_ROOT/data_list" ]]; then
  if is_true "$DOWNLOAD_IF_MISSING"; then
    DATA_ROOT="$DATA_ROOT" "$ROOT/scripts/dataset_prep/download_diode.sh"
  fi
fi

if [[ ! -d "$DATA_ROOT/val" ]]; then
  echo "ERROR: Missing DIODE val directory: $DATA_ROOT/val" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT/val_normal" ]]; then
  echo "ERROR: Missing DIODE val_normal directory: $DATA_ROOT/val_normal" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT/data_list" ]]; then
  echo "ERROR: Missing DIODE data_list directory: $DATA_ROOT/data_list" >&2
  exit 1
fi

if [[ "$TASK" == "depth" || "$TASK" == "both" ]]; then
  if [[ -s "$DEPTH_TAR_OUTPUT" ]] && ! is_true "$FORCE"; then
    DEPTH_DONE=1
    echo "Using existing DIODE depth tar: $DEPTH_TAR_OUTPUT"
    echo "Set FORCE=1 to rebuild."
  else
    DEPTH_DONE=0
  fi
else
  DEPTH_DONE=1
fi

if [[ "$TASK" == "surface_normals" || "$TASK" == "both" ]]; then
  if [[ -s "$NORMALS_TAR_OUTPUT" ]] && ! is_true "$FORCE"; then
    NORMALS_DONE=1
    echo "Using existing DIODE surface-normal tar: $NORMALS_TAR_OUTPUT"
    echo "Set FORCE=1 to rebuild."
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
  rm -rf "$DEPTH_OUT_ROOT" "$NORMALS_OUT_ROOT"
  rm -f "$DEPTH_TAR_OUTPUT" "$DEPTH_TAR_OUTPUT.partial" "$NORMALS_TAR_OUTPUT" "$NORMALS_TAR_OUTPUT.partial"
fi

args=(
  dataset_prep/diode/prepare_diode_eval.py
  --source-root "$DATA_ROOT"
  --task "$TASK"
  --domains "$DOMAINS"
  --depth-out-root "$DEPTH_OUT_ROOT"
  --normals-out-root "$NORMALS_OUT_ROOT"
  --depth-tar-output "$DEPTH_TAR_OUTPUT"
  --normals-tar-output "$NORMALS_TAR_OUTPUT"
  --min-depth "$MIN_DEPTH"
  --max-depth "$MAX_DEPTH"
  --min-normal-norm "$MIN_NORMAL_NORM"
  --invalid-normal-rgb "$INVALID_NORMAL_RGB"
  --overwrite
)

ovb_run_python "${args[@]}"

echo "DIODE preparation finished."
echo "  source root: $DATA_ROOT"
echo "  depth tar:   $DEPTH_TAR_OUTPUT"
echo "  normals tar: $NORMALS_TAR_OUTPUT"
