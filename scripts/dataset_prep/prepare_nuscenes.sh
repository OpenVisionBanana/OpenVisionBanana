#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Prepare nuScenes sparse LiDAR depth and LiDAR-derived surface-normal evaluation data.

Usage:
  scripts/dataset_prep/prepare_nuscenes.sh

Environment:
  VERSION                 v1.0-mini or v1.0-trainval. Default: v1.0-trainval.
  SPLIT                   test for mini or val for trainval.
  DATA_ROOT               Persistent nuScenes archive/output root.
  TMPDIR                  Temporary staging and preparation root for trainval.
  NUSCENES_ARCHIVE        Mini archive. Default: DATA_ROOT/v1.0-mini.tar.
  META_ARCHIVE            Trainval metadata. By default, auto-detects the
                          DATA_ROOT/v1.0-trainval_meta.tar or .tgz archive.
  BLOBS_GLOB              Trainval blob glob. Default: DATA_ROOT/v1.0-trainval??_blobs.tgz.
  SCENE_LIST              Official validation scene list.
  ARCHIVE_WORKERS         Concurrent blob readers. Default: 2.
  MIN_TMP_FREE_GB         Required free temporary space. Default: 90 for a full run.
  TASK                    depth, surface_normals, or both. Default: both.
  CAMERAS                 Space/comma-separated cameras. Default: CAM_FRONT.
  LIMIT                   Optional sample limit. Limited trainval outputs include a limit tag.
  DEPTH_TAR_OUTPUT        Persistent prepared depth tar.
  NORMALS_TAR_OUTPUT      Persistent prepared surface-normal tar.
  FORCE                   1 to rebuild outputs. Default: 0.
  KEEP_WORKDIR            1 to retain trainval temporary files for debugging. Default: 0.
  USE_CONTAINER           1 to use SIF; 0 for the active environment. Default: 0.
  SIF                     Container image when USE_CONTAINER=1.
  PYTHON                  Python interpreter. Default: python.
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
VERSION=${VERSION:-v1.0-trainval}
TASK=${TASK:-both}
CAMERAS=${CAMERAS:-CAM_FRONT}
LIMIT=${LIMIT:-}
FORCE=${FORCE:-0}
KEEP_WORKDIR=${KEEP_WORKDIR:-0}
MIN_DEPTH=${MIN_DEPTH:-1.0}
MAX_DEPTH=${MAX_DEPTH:-80.0}
NORMAL_K_NEIGHBORS=${NORMAL_K_NEIGHBORS:-20}
NORMAL_MIN_NEIGHBORS=${NORMAL_MIN_NEIGHBORS:-6}
INVALID_NORMAL_RGB=${INVALID_NORMAL_RGB:-128,128,128}
NORMAL_ENCODING=${NORMAL_ENCODING:-vision_banana_x_left_red}

if [[ -n "$LIMIT" && ( ! "$LIMIT" =~ ^[0-9]+$ || "$LIMIT" == 0 ) ]]; then
  echo "ERROR: LIMIT must be a positive integer; got $LIMIT" >&2
  exit 1
fi

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

case "$TASK" in
  depth|surface_normals|both) ;;
  *) echo "ERROR: TASK must be depth, surface_normals, or both; got $TASK" >&2; exit 1 ;;
esac

case "$VERSION" in
  v1.0-mini)
    SPLIT=${SPLIT:-test}
    if [[ "$SPLIT" != "test" ]]; then
      echo "ERROR: v1.0-mini preparation supports SPLIT=test only" >&2
      exit 1
    fi
    VERSION_TAG=mini
    NUSCENES_ARCHIVE=${NUSCENES_ARCHIVE:-$DATA_ROOT/v1.0-mini.tar}
    RAW_ROOT=${RAW_ROOT:-$DATA_ROOT/raw}
    ;;
  v1.0-trainval)
    SPLIT=${SPLIT:-val}
    if [[ "$SPLIT" != "val" ]]; then
      echo "ERROR: v1.0-trainval preparation currently supports SPLIT=val only" >&2
      exit 1
    fi
    VERSION_TAG=trainval_val
    if [[ -z "${META_ARCHIVE:-}" ]]; then
      if [[ -f "$DATA_ROOT/v1.0-trainval_meta.tar" ]]; then
        META_ARCHIVE=$DATA_ROOT/v1.0-trainval_meta.tar
      else
        META_ARCHIVE=$DATA_ROOT/v1.0-trainval_meta.tgz
      fi
    fi
    BLOBS_GLOB=${BLOBS_GLOB:-$DATA_ROOT/v1.0-trainval??_blobs.tgz}
    SCENE_LIST=${SCENE_LIST:-$ROOT/dataset_prep/nuscenes/splits/v1.0-trainval-val-scenes.txt}
    ARCHIVE_WORKERS=${ARCHIVE_WORKERS:-2}
    ;;
  *) echo "ERROR: Unsupported VERSION=$VERSION" >&2; exit 1 ;;
esac

CAMERA_TAG=$(printf '%s' "$CAMERAS" | tr '[:upper:] ,' '[:lower:]__' | tr -s '_' | sed 's/^_//; s/_$//')
LIMIT_TAG=
if [[ "$VERSION" == "v1.0-trainval" && -n "$LIMIT" ]]; then
  LIMIT_TAG=_limit${LIMIT}
fi
OUTPUT_STEM=nuscenes_${VERSION_TAG}_${CAMERA_TAG}${LIMIT_TAG}
DEPTH_TAR_OUTPUT=${DEPTH_TAR_OUTPUT:-$DATA_ROOT/${OUTPUT_STEM}_depth.tar}
NORMALS_TAR_OUTPUT=${NORMALS_TAR_OUTPUT:-$DATA_ROOT/${OUTPUT_STEM}_surface_normals.tar}

DEPTH_DONE=1
NORMALS_DONE=1
if [[ "$TASK" == "depth" || "$TASK" == "both" ]]; then
  DEPTH_DONE=0
  if [[ -s "$DEPTH_TAR_OUTPUT" ]] && ! is_true "$FORCE"; then
    echo "Using existing nuScenes depth tar: $DEPTH_TAR_OUTPUT"
    DEPTH_DONE=1
  fi
fi
if [[ "$TASK" == "surface_normals" || "$TASK" == "both" ]]; then
  NORMALS_DONE=0
  if [[ -s "$NORMALS_TAR_OUTPUT" ]] && ! is_true "$FORCE"; then
    echo "Using existing nuScenes surface-normal tar: $NORMALS_TAR_OUTPUT"
    NORMALS_DONE=1
  fi
fi
if [[ "$DEPTH_DONE" == 1 && "$NORMALS_DONE" == 1 ]]; then
  exit 0
fi
PREP_TASK=$TASK
if [[ "$TASK" == "both" && "$DEPTH_DONE" == 1 ]]; then PREP_TASK=surface_normals; fi
if [[ "$TASK" == "both" && "$NORMALS_DONE" == 1 ]]; then PREP_TASK=depth; fi

run_in_runtime() { ovb_run "$@"; }

publish_tar() {
  local source=$1 destination=$2 partial
  mkdir -p "$(dirname -- "$destination")"
  partial=${destination}.partial.${SLURM_JOB_ID:-$$}
  rm -f "$partial"
  if ! cp --reflink=auto "$source" "$partial"; then
    rm -f "$partial"
    return 1
  fi
  mv -f "$partial" "$destination"
}

if [[ "$VERSION" == "v1.0-mini" ]]; then
  if [[ ! -f "$NUSCENES_ARCHIVE" ]]; then
    DATA_ROOT="$DATA_ROOT" VERSION="$VERSION" NUSCENES_ARCHIVE="$NUSCENES_ARCHIVE" \
      "$ROOT/scripts/dataset_prep/download_nuscenes.sh"
  fi
  if [[ ! -d "$RAW_ROOT/$VERSION" || ! -d "$RAW_ROOT/samples" ]] || is_true "$FORCE"; then
    if is_true "$FORCE"; then rm -rf "$RAW_ROOT"; fi
    mkdir -p "$RAW_ROOT"
    extract_members=("$VERSION" "samples/LIDAR_TOP")
    read -r -a camera_parts <<<"${CAMERAS//,/ }"
    for camera in "${camera_parts[@]}"; do extract_members+=("samples/$camera"); done
    echo "Extracting mini nuScenes files into $RAW_ROOT"
    tar -xf "$NUSCENES_ARCHIVE" -C "$RAW_ROOT" "${extract_members[@]}"
  fi
  DEPTH_OUT_ROOT=${DEPTH_OUT_ROOT:-$DATA_ROOT/${OUTPUT_STEM}_depth}
  NORMALS_OUT_ROOT=${NORMALS_OUT_ROOT:-$DATA_ROOT/${OUTPUT_STEM}_surface_normals}
  if is_true "$FORCE"; then
    [[ "$PREP_TASK" == "surface_normals" ]] || rm -rf "$DEPTH_OUT_ROOT"
    [[ "$PREP_TASK" == "depth" ]] || rm -rf "$NORMALS_OUT_ROOT"
  fi
else
  [[ -f "$META_ARCHIVE" ]] || { echo "ERROR: Missing trainval metadata archive: $META_ARCHIVE" >&2; exit 1; }
  [[ -f "$SCENE_LIST" ]] || { echo "ERROR: Missing validation scene list: $SCENE_LIST" >&2; exit 1; }
  mapfile -t BLOB_ARCHIVES < <(compgen -G "$BLOBS_GLOB" | sort)
  if [[ ${#BLOB_ARCHIVES[@]} -ne 10 ]]; then
    echo "ERROR: Expected 10 blob archives matching $BLOBS_GLOB; found ${#BLOB_ARCHIVES[@]}" >&2
    exit 1
  fi

  TMPDIR=${TMPDIR:-/tmp/${USER:-$(id -un)}/open_visionbanana}
  mkdir -p "$TMPDIR"
  TMPDIR=$(cd -- "$TMPDIR" && pwd -P)
  MIN_TMP_FREE_GB=${MIN_TMP_FREE_GB:-$([[ -n "$LIMIT" ]] && echo 2 || echo 90)}
  available_kb=$(df -Pk "$TMPDIR" | awk 'NR==2 {print $4}')
  required_kb=$((MIN_TMP_FREE_GB * 1024 * 1024))
  if (( available_kb < required_kb )); then
    echo "ERROR: $TMPDIR has $((available_kb / 1024 / 1024)) GiB free; at least $MIN_TMP_FREE_GB GiB is required" >&2
    exit 1
  fi

  WORK_ROOT=${WORK_ROOT:-$TMPDIR/nuscenes-trainval-val-${SLURM_JOB_ID:-$$}}
  WORK_ROOT=$(realpath -m "$WORK_ROOT")
  case "$WORK_ROOT" in
    "$TMPDIR"/*) ;;
    *) echo "ERROR: trainval WORK_ROOT must be inside TMPDIR ($TMPDIR): $WORK_ROOT" >&2; exit 1 ;;
  esac
  RAW_ROOT=$WORK_ROOT/raw
  DEPTH_OUT_ROOT=${DEPTH_OUT_ROOT:-$WORK_ROOT/${OUTPUT_STEM}_depth}
  NORMALS_OUT_ROOT=${NORMALS_OUT_ROOT:-$WORK_ROOT/${OUTPUT_STEM}_surface_normals}
  LOCAL_DEPTH_TAR=$WORK_ROOT/${OUTPUT_STEM}_depth.tar
  LOCAL_NORMALS_TAR=$WORK_ROOT/${OUTPUT_STEM}_surface_normals.tar
  cleanup() {
    if ! is_true "$KEEP_WORKDIR"; then rm -rf "$WORK_ROOT"; fi
  }
  trap cleanup EXIT
  rm -rf "$WORK_ROOT"
  mkdir -p "$RAW_ROOT"

  stage_args=(
    "$PYTHON" dataset_prep/nuscenes/stage_nuscenes_split.py
    --metadata-archive "$META_ARCHIVE"
    --output-root "$RAW_ROOT"
    --version "$VERSION"
    --scene-list "$SCENE_LIST"
    --cameras "$CAMERAS"
    --workers "$ARCHIVE_WORKERS"
    --expected-archives 10
  )
  for archive in "${BLOB_ARCHIVES[@]}"; do stage_args+=(--blob "$archive"); done
  if [[ -n "$LIMIT" ]]; then
    stage_args+=(--limit "$LIMIT")
  elif [[ "$CAMERAS" == "CAM_FRONT" ]]; then
    stage_args+=(--expected-scenes 150 --expected-samples 6019)
  fi
  run_in_runtime "${stage_args[@]}"
fi

prep_args=(
  "$PYTHON" dataset_prep/nuscenes/prepare_nuscenes_eval.py
  --source-root "$RAW_ROOT"
  --version "$VERSION"
  --split "$SPLIT"
  --task "$PREP_TASK"
  --cameras "$CAMERAS"
  --depth-out-root "$DEPTH_OUT_ROOT"
  --normals-out-root "$NORMALS_OUT_ROOT"
  --depth-tar-output "${LOCAL_DEPTH_TAR:-$DEPTH_TAR_OUTPUT}"
  --normals-tar-output "${LOCAL_NORMALS_TAR:-$NORMALS_TAR_OUTPUT}"
  --min-depth "$MIN_DEPTH"
  --max-depth "$MAX_DEPTH"
  --normal-k-neighbors "$NORMAL_K_NEIGHBORS"
  --normal-min-neighbors "$NORMAL_MIN_NEIGHBORS"
  --invalid-normal-rgb "$INVALID_NORMAL_RGB"
  --normal-encoding "$NORMAL_ENCODING"
  --overwrite
)
if [[ "$VERSION" == "v1.0-trainval" ]]; then prep_args+=(--scene-list "$SCENE_LIST"); fi
if [[ -n "$LIMIT" ]]; then prep_args+=(--limit "$LIMIT"); fi
run_in_runtime "${prep_args[@]}"

if [[ "$VERSION" == "v1.0-trainval" ]]; then
  [[ "$PREP_TASK" == "surface_normals" ]] || publish_tar "$LOCAL_DEPTH_TAR" "$DEPTH_TAR_OUTPUT"
  [[ "$PREP_TASK" == "depth" ]] || publish_tar "$LOCAL_NORMALS_TAR" "$NORMALS_TAR_OUTPUT"
fi

echo "nuScenes preparation finished."
echo "  version/split: $VERSION/$SPLIT"
[[ "$PREP_TASK" == "surface_normals" ]] || echo "  depth tar:     $DEPTH_TAR_OUTPUT"
[[ "$PREP_TASK" == "depth" ]] || echo "  normals tar:   $NORMALS_TAR_OUTPUT"
