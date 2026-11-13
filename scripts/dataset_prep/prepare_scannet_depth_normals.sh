#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Prepare ScanNet v2 validation frames for depth and depth-derived normal evaluation.

The prepared output is one shared tar/folder used by both evaluators:
  scannet_depth_normals/
    images/, depths/, normals_depth/, masks/, intrinsics/
    scannet_val_debug.txt
    scannet_val_small.txt
    scannet_val_standard.txt
    scannet_val_large.txt
    scannet_val_giant.txt

Usage:
  scripts/dataset_prep/prepare_scannet_depth_normals.sh

Environment:
  ROOT                 Repo root.
  USE_CONTAINER        1 to use SIF; 0 for the active environment. Default: 0.
  SIF                  Container image when USE_CONTAINER=1.
  PYTHON               Python interpreter. Default: python.
  DATA_ROOT            Persistent ScanNet root. Default: /path/to/data/depth_est/scannet
  SOURCE_ROOT          Raw ScanNet root. Default: DATA_ROOT
  FRAMES_ROOT          Optional extracted scannet_frames_25k folder.
  VAL_SCENES_FILE      Optional scannetv2_val.txt path.
  OUT_ROOT             Prepared folder. Default: DATA_ROOT/scannet_depth_normals
  PREPARED_TAR         Prepared tar. Default: DATA_ROOT/scannet_depth_normals.tar
  DOWNLOAD_IF_MISSING  1 to run download_scannet.sh when assets are missing. Default: 0.
  FORCE                1 to rebuild output folder/tar. Default: 0.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

ROOT=${ROOT:-$DEFAULT_ROOT}
source "$ROOT/scripts/lib/runtime.sh"

DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT for this dataset}
SOURCE_ROOT=${SOURCE_ROOT:-$DATA_ROOT}
FRAMES_ROOT=${FRAMES_ROOT:-}
VAL_SCENES_FILE=${VAL_SCENES_FILE:-$SOURCE_ROOT/tasks/scannetv2_val.txt}
OUT_ROOT=${OUT_ROOT:-$DATA_ROOT/scannet_depth_normals}
PREPARED_TAR=${PREPARED_TAR:-$DATA_ROOT/scannet_depth_normals.tar}
DOWNLOAD_IF_MISSING=${DOWNLOAD_IF_MISSING:-0}
FORCE=${FORCE:-0}

MIN_DEPTH=${MIN_DEPTH:-0.001}
MAX_DEPTH=${MAX_DEPTH:-10.0}
MIN_VALID_FRACTION=${MIN_VALID_FRACTION:-0.05}
GRADIENT_RADIUS=${GRADIENT_RADIUS:-2}
MASK_EROSION_RADIUS=${MASK_EROSION_RADIUS:-1}
INVALID_NORMAL_RGB=${INVALID_NORMAL_RGB:-128,128,128}
NORMAL_ENCODING=${NORMAL_ENCODING:-vision_banana_x_left_red}
STRICT=${STRICT:-0}

if [[ ! -d "$SOURCE_ROOT/scannet_frames_25k" && ! -d "$SOURCE_ROOT/tasks/scannet_frames_25k" && ! -s "$SOURCE_ROOT/tasks/scannet_frames_25k.zip" ]]; then
  if is_true "$DOWNLOAD_IF_MISSING"; then
    DATA_ROOT="$SOURCE_ROOT" "$ROOT/scripts/dataset_prep/download_scannet.sh"
  fi
fi

if [[ ! -s "$VAL_SCENES_FILE" ]]; then
  if is_true "$DOWNLOAD_IF_MISSING"; then
    DATA_ROOT="$SOURCE_ROOT" "$ROOT/scripts/dataset_prep/download_scannet.sh"
  fi
fi

if [[ -s "$PREPARED_TAR" ]]; then
  if ! is_true "$FORCE"; then
    echo "Using existing ScanNet prepared tar: $PREPARED_TAR"
    echo "Set FORCE=1 to rebuild."
    exit 0
  fi
fi

args=(
  dataset_prep/scannet/prepare_scannet_depth_normals.py
  --source-root "$SOURCE_ROOT"
  --val-scenes-file "$VAL_SCENES_FILE"
  --out-root "$OUT_ROOT"
  --prepared-tar "$PREPARED_TAR"
  --min-depth "$MIN_DEPTH"
  --max-depth "$MAX_DEPTH"
  --min-valid-fraction "$MIN_VALID_FRACTION"
  --gradient-radius "$GRADIENT_RADIUS"
  --mask-erosion-radius "$MASK_EROSION_RADIUS"
  --invalid-normal-rgb "$INVALID_NORMAL_RGB"
  --normal-encoding "$NORMAL_ENCODING"
)
if [[ -n "$FRAMES_ROOT" ]]; then
  args+=(--frames-root "$FRAMES_ROOT")
fi
if is_true "$FORCE"; then
  args+=(--force)
fi
if is_true "$STRICT"; then
  args+=(--strict)
fi

ovb_run_python "${args[@]}"

echo "ScanNet preparation finished."
echo "  source root:  $SOURCE_ROOT"
echo "  prepared dir: $OUT_ROOT"
echo "  prepared tar: $PREPARED_TAR"
