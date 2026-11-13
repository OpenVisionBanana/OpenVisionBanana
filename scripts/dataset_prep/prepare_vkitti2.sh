#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Prepare Virtual KITTI 2 depth and derived surface-normal evaluation archives.

Usage:
  scripts/dataset_prep/prepare_vkitti2.sh

Environment:
  ROOT                    Repo root.
  USE_CONTAINER           1 to use SIF; 0 for the active environment. Default: 0.
  SIF                     Container image when USE_CONTAINER=1.
  PYTHON                  Python interpreter. Default: python.
  DATA_ROOT               VKITTI2 archive/prepared-data root.
  RGB_TAR                 vkitti_2.0.3_rgb.tar path.
  DEPTH_TAR               vkitti_2.0.3_depth.tar path.
  TEXTGT_TAR              vkitti_2.0.3_textgt.tar.gz path.
  TASK                    depth, surface_normals, or both. Default: both.
  SCENES                  Space/comma-separated scenes. Default: all five scenes.
  VARIANTS                Space/comma-separated variants. Default: clone.
  CAMERAS                 Space/comma-separated cameras. Default: Camera_0.
  LIMIT                   Optional sample limit for debug preparation.
  PREP_ROOT               Temporary prepared-folder root. Default: TMPDIR/vkitti2-prep-JOBID.
  DEPTH_OUT_ROOT          Prepared depth folder.
  NORMALS_OUT_ROOT        Prepared derived-normal folder.
  DEPTH_TAR_OUTPUT        Prepared depth tar.
  NORMALS_TAR_OUTPUT      Prepared derived-normal tar.
  Use .tar.gz output paths if compressed archives are preferred.
  KEEP_PREPARED_DIR       1 to keep auto-created PREP_ROOT folders after tar creation. Default: 0.
  DOWNLOAD_IF_MISSING     1 to download source archives if missing. Default: 0.
  FORCE                   1 to rebuild output folders/tars. Default: 0.
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
RGB_TAR=${RGB_TAR:-$DATA_ROOT/vkitti_2.0.3_rgb.tar}
DEPTH_TAR=${DEPTH_TAR:-$DATA_ROOT/vkitti_2.0.3_depth.tar}
TEXTGT_TAR=${TEXTGT_TAR:-$DATA_ROOT/vkitti_2.0.3_textgt.tar.gz}
TASK=${TASK:-both}
SCENES=${SCENES:-Scene01 Scene02 Scene06 Scene18 Scene20}
VARIANTS=${VARIANTS:-clone}
CAMERAS=${CAMERAS:-Camera_0}
LIMIT=${LIMIT:-}
DOWNLOAD_IF_MISSING=${DOWNLOAD_IF_MISSING:-0}
FORCE=${FORCE:-0}
MIN_DEPTH=${MIN_DEPTH:-0.001}
MAX_DEPTH=${MAX_DEPTH:-655.35}
DEPTH_PNG_SCALE=${DEPTH_PNG_SCALE:-100.0}
GRADIENT_RADIUS=${GRADIENT_RADIUS:-2}
MASK_EROSION_RADIUS=${MASK_EROSION_RADIUS:-0}
MAX_NEIGHBOR_DEPTH_RATIO=${MAX_NEIGHBOR_DEPTH_RATIO:-0.1}
MIN_NORMAL_VALID_PIXELS=${MIN_NORMAL_VALID_PIXELS:-4096}
INVALID_NORMAL_RGB=${INVALID_NORMAL_RGB:-128,128,128}
NORMAL_ENCODING=${NORMAL_ENCODING:-vision_banana_x_left_red}

tag_from_list() {
  printf '%s' "$1" | tr '[:upper:] ,' '[:lower:]__' | tr -s '_' | sed 's/^_//; s/_$//'
}

VARIANT_TAG=$(tag_from_list "$VARIANTS")
CAMERA_TAG=$(tag_from_list "$CAMERAS")
DATASET_TAG=vkitti2_${VARIANT_TAG}_${CAMERA_TAG}
TMPDIR=${TMPDIR:-/tmp/${USER:-$(id -un)}/open_visionbanana}
JOB_ID=${SLURM_JOB_ID:-manual-$$}
PREP_ROOT=${PREP_ROOT:-$TMPDIR/vkitti2-prep-$JOB_ID}
KEEP_PREPARED_DIR=${KEEP_PREPARED_DIR:-0}
AUTO_PREP_ROOT=0
if [[ -z "${DEPTH_OUT_ROOT+x}" ]]; then
  DEPTH_OUT_ROOT=$PREP_ROOT/${DATASET_TAG}_depth
  AUTO_PREP_ROOT=1
fi
if [[ -z "${NORMALS_OUT_ROOT+x}" ]]; then
  NORMALS_OUT_ROOT=$PREP_ROOT/${DATASET_TAG}_surface_normals
  AUTO_PREP_ROOT=1
fi
DEPTH_TAR_OUTPUT=${DEPTH_TAR_OUTPUT:-$DATA_ROOT/${DATASET_TAG}_depth.tar}
NORMALS_TAR_OUTPUT=${NORMALS_TAR_OUTPUT:-$DATA_ROOT/${DATASET_TAG}_surface_normals.tar}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if { [[ ! -f "$RGB_TAR" ]] || [[ ! -f "$DEPTH_TAR" ]] || [[ ! -f "$TEXTGT_TAR" ]]; } && is_true "$DOWNLOAD_IF_MISSING"; then
  DATA_ROOT="$DATA_ROOT" "$ROOT/scripts/dataset_prep/download_vkitti2.sh"
fi

for path in "$RGB_TAR" "$DEPTH_TAR" "$TEXTGT_TAR"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: Missing VKITTI2 source archive: $path" >&2
    exit 1
  fi
done

if [[ "$TASK" == "depth" || "$TASK" == "both" ]]; then
  if [[ -s "$DEPTH_TAR_OUTPUT" ]] && ! is_true "$FORCE"; then
    echo "Using existing VKITTI2 depth tar: $DEPTH_TAR_OUTPUT"
    DEPTH_DONE=1
  else
    DEPTH_DONE=0
  fi
else
  DEPTH_DONE=1
fi

if [[ "$TASK" == "surface_normals" || "$TASK" == "both" ]]; then
  if [[ -s "$NORMALS_TAR_OUTPUT" ]] && ! is_true "$FORCE"; then
    echo "Using existing VKITTI2 surface-normal tar: $NORMALS_TAR_OUTPUT"
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
  dataset_prep/vkitti2/prepare_vkitti2_eval.py
  --rgb-tar "$RGB_TAR"
  --depth-tar "$DEPTH_TAR"
  --textgt-tar "$TEXTGT_TAR"
  --task "$TASK"
  --scenes "$SCENES"
  --variants "$VARIANTS"
  --cameras "$CAMERAS"
  --depth-out-root "$DEPTH_OUT_ROOT"
  --normals-out-root "$NORMALS_OUT_ROOT"
  --depth-tar-output "$DEPTH_TAR_OUTPUT"
  --normals-tar-output "$NORMALS_TAR_OUTPUT"
  --depth-png-scale "$DEPTH_PNG_SCALE"
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
if [[ -n "$LIMIT" ]]; then args+=(--limit "$LIMIT"); fi

ovb_run_python "${args[@]}"

echo "Virtual KITTI 2 preparation finished."
echo "  depth tar:   $DEPTH_TAR_OUTPUT"
echo "  normals tar: $NORMALS_TAR_OUTPUT"

if [[ "$AUTO_PREP_ROOT" == "1" ]] && ! is_true "$KEEP_PREPARED_DIR"; then
  rm -rf "$PREP_ROOT"
fi
