#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Prepare iBims-1 standard depth and derived surface-normal evaluation datasets.

By default this prepares only the official/core 100-image benchmark. Auxiliary
augmented images are optional and must be enabled explicitly.

Usage:
  scripts/dataset_prep/prepare_ibims1.sh

Environment:
  ROOT                    Repo root.
  USE_CONTAINER           1 to use SIF; 0 for the active environment. Default: 0.
  SIF                     Container image when USE_CONTAINER=1.
  PYTHON                  Python interpreter. Default: python.
  SOURCE_ROOT             Raw iBims-1 download directory.
  TASK                    depth, surface_normals, or both. Default: both.
  DEPTH_OUT_ROOT          Prepared depth folder.
  NORMALS_OUT_ROOT        Prepared derived-normal folder.
  DEPTH_TAR_OUTPUT        Prepared depth tar.
  NORMALS_TAR_OUTPUT      Prepared derived-normal tar.
  DOWNLOAD_IF_MISSING     1 to download missing raw files. Default: 0.
  INCLUDE_AUX_AUGMENTED   1 to include optional aux_augmented rows/images. Default: 0.
  AUX_AUGMENTATIONS       Space-separated groups: appearance, geometric, all, or explicit names.
  DEPTH_VALIDITY_PROTOCOL nyu_10m or ibims_official. Default: nyu_10m.
                          ibims_official disables the 10 m prep-time upper-depth cutoff
                          and writes separate *_official outputs by default.
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
SOURCE_ROOT=${SOURCE_ROOT:-$DATA_ROOT/ibims1}
TASK=${TASK:-both}
DEPTH_VALIDITY_PROTOCOL=${DEPTH_VALIDITY_PROTOCOL:-nyu_10m}
if [[ "$DEPTH_VALIDITY_PROTOCOL" == "ibims_official" ]]; then
  DEPTH_OUT_ROOT=${DEPTH_OUT_ROOT:-$DATA_ROOT/ibims1_depth_official}
  NORMALS_OUT_ROOT=${NORMALS_OUT_ROOT:-$DATA_ROOT/ibims1_surface_normals_official}
  DEPTH_TAR_OUTPUT=${DEPTH_TAR_OUTPUT:-$DATA_ROOT/ibims1_depth_official.tar}
  NORMALS_TAR_OUTPUT=${NORMALS_TAR_OUTPUT:-$DATA_ROOT/ibims1_surface_normals_official.tar}
else
  DEPTH_OUT_ROOT=${DEPTH_OUT_ROOT:-$DATA_ROOT/ibims1_depth}
  NORMALS_OUT_ROOT=${NORMALS_OUT_ROOT:-$DATA_ROOT/ibims1_surface_normals}
  DEPTH_TAR_OUTPUT=${DEPTH_TAR_OUTPUT:-$DATA_ROOT/ibims1_depth.tar}
  NORMALS_TAR_OUTPUT=${NORMALS_TAR_OUTPUT:-$DATA_ROOT/ibims1_surface_normals.tar}
fi

DOWNLOAD_IF_MISSING=${DOWNLOAD_IF_MISSING:-0}
INCLUDE_AUX_AUGMENTED=${INCLUDE_AUX_AUGMENTED:-0}
AUX_AUGMENTATIONS=${AUX_AUGMENTATIONS:-appearance}
FORCE=${FORCE:-0}
MIN_DEPTH=${MIN_DEPTH:-0.001}
if [[ -z "${MAX_DEPTH+x}" ]]; then
  if [[ "$DEPTH_VALIDITY_PROTOCOL" == "ibims_official" ]]; then
    MAX_DEPTH=0
  else
    MAX_DEPTH=10.0
  fi
fi
GRADIENT_RADIUS=${GRADIENT_RADIUS:-2}
MASK_EROSION_RADIUS=${MASK_EROSION_RADIUS:-1}
INVALID_NORMAL_RGB=${INVALID_NORMAL_RGB:-128,128,128}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ ! -f "$SOURCE_ROOT/ibims1_core_mat.zip" || ! -f "$SOURCE_ROOT/imagelist.txt" ]]; then
  if is_true "$DOWNLOAD_IF_MISSING"; then
    DATA_ROOT="$SOURCE_ROOT" INCLUDE_AUX=1 "$ROOT/scripts/dataset_prep/download_ibims1.sh"
  fi
fi

if [[ ! -f "$SOURCE_ROOT/ibims1_core_mat.zip" ]]; then
  echo "ERROR: Missing iBims-1 core archive: $SOURCE_ROOT/ibims1_core_mat.zip" >&2
  exit 1
fi
if is_true "$INCLUDE_AUX_AUGMENTED" && [[ ! -f "$SOURCE_ROOT/ibims1_aux_augmented.zip" ]]; then
  echo "ERROR: INCLUDE_AUX_AUGMENTED=1 but missing $SOURCE_ROOT/ibims1_aux_augmented.zip" >&2
  exit 1
fi

if [[ "$TASK" == "depth" || "$TASK" == "both" ]]; then
  if [[ -s "$DEPTH_TAR_OUTPUT" ]] && ! is_true "$FORCE"; then
    echo "Using existing iBims-1 depth tar: $DEPTH_TAR_OUTPUT"
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
    echo "Using existing iBims-1 surface-normal tar: $NORMALS_TAR_OUTPUT"
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
  rm -rf "$DEPTH_OUT_ROOT" "$NORMALS_OUT_ROOT"
  rm -f "$DEPTH_TAR_OUTPUT" "$DEPTH_TAR_OUTPUT.partial" "$NORMALS_TAR_OUTPUT" "$NORMALS_TAR_OUTPUT.partial"
fi

args=(
  dataset_prep/ibims1/prepare_ibims1_eval.py
  --source-root "$SOURCE_ROOT"
  --task "$TASK"
  --depth-out-root "$DEPTH_OUT_ROOT"
  --normals-out-root "$NORMALS_OUT_ROOT"
  --depth-tar-output "$DEPTH_TAR_OUTPUT"
  --normals-tar-output "$NORMALS_TAR_OUTPUT"
  --min-depth "$MIN_DEPTH"
  --max-depth "$MAX_DEPTH"
  --depth-validity-protocol "$DEPTH_VALIDITY_PROTOCOL"
  --gradient-radius "$GRADIENT_RADIUS"
  --mask-erosion-radius "$MASK_EROSION_RADIUS"
  --invalid-normal-rgb "$INVALID_NORMAL_RGB"
  --overwrite
)
if is_true "$INCLUDE_AUX_AUGMENTED"; then
  args+=(--include-aux-augmented --aux-augmentations)
  # shellcheck disable=SC2206
  aux_parts=($AUX_AUGMENTATIONS)
  args+=("${aux_parts[@]}")
fi

ovb_run_python "${args[@]}"

echo "iBims-1 preparation finished."
echo "  source root: $SOURCE_ROOT"
echo "  validity protocol: $DEPTH_VALIDITY_PROTOCOL"
echo "  depth tar:   $DEPTH_TAR_OUTPUT"
echo "  normals tar: $NORMALS_TAR_OUTPUT"
