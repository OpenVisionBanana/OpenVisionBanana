#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Generate NYUv2 RGB-to-surface-normal evaluation data from nyu_depth_v2_labeled.mat.

Use cases:
  Case A: persistent-folder / simple setup
    Keep the generated nyu_surface_normals/ folder and tar on persistent storage.
  Case B: archive-aware evaluation
    Reuse nyu_surface_normals.tar and let the evaluator extract it into TMPDIR.

Usage:
  scripts/dataset_prep/prepare_nyu_surface_normals.sh

Important environment variables:
  ROOT                    Repo root. Defaults to the current checkout.
  USE_CONTAINER           1 to use SIF; 0 for the active environment. Default: 0.
  SIF                     Container image when USE_CONTAINER=1.
  PYTHON                  Python interpreter. Default: python.
  DATA_ROOT               Persistent depth data root.
  LABELED_MAT             NYUv2 labeled MAT file.
  SPLITS_MAT              Official NYUv2 split MAT file.
  OUT_ROOT                Generated dataset folder.
  TAR_OUTPUT              Generated tar archive.
  DOWNLOAD_IF_MISSING     1 to download LABELED_MAT/SPLITS_MAT if missing.
  FORCE                   1 to rebuild OUT_ROOT/TAR_OUTPUT.

Surface-normal controls:
  DEPTH_SOURCE            depths or rawDepths. Default: depths.
  MASK_SOURCE             rawDepths, depths, or both. Default: rawDepths.
  GRADIENT_RADIUS         Pixel radius for central-difference normals.
  MASK_EROSION_RADIUS     Erode valid-depth mask by this many pixels.
  MIN_DEPTH, MAX_DEPTH    Valid depth range in meters.
  WRITE_DEPTH_MM          1 to include debug depth_mm PNGs.
  LIMIT                   Optional per-split sample limit.
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
LABELED_MAT=${LABELED_MAT:-$DATA_ROOT/nyu_depth_v2_labeled.mat}
SPLITS_MAT=${SPLITS_MAT:-$DATA_ROOT/splits.mat}
OUT_ROOT=${OUT_ROOT:-$DATA_ROOT/nyu_surface_normals}
TAR_OUTPUT=${TAR_OUTPUT:-$DATA_ROOT/nyu_surface_normals.tar}

DOWNLOAD_IF_MISSING=${DOWNLOAD_IF_MISSING:-0}
FORCE=${FORCE:-0}
SPLIT=${SPLIT:-all}
LIMIT=${LIMIT:-}
DEPTH_SOURCE=${DEPTH_SOURCE:-depths}
MASK_SOURCE=${MASK_SOURCE:-rawDepths}
MIN_DEPTH=${MIN_DEPTH:-0.001}
MAX_DEPTH=${MAX_DEPTH:-10.0}
GRADIENT_RADIUS=${GRADIENT_RADIUS:-2}
MASK_EROSION_RADIUS=${MASK_EROSION_RADIUS:-1}
FX=${FX:-518.8579011745019}
FY=${FY:-519.4696111212749}
CX=${CX:-325.58244941119034}
CY=${CY:-253.73616633400465}
WRITE_DEPTH_MM=${WRITE_DEPTH_MM:-0}
INVALID_NORMAL_RGB=${INVALID_NORMAL_RGB:-128,128,128}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ ! -f "$LABELED_MAT" || ! -f "$SPLITS_MAT" ]]; then
  if is_true "$DOWNLOAD_IF_MISSING"; then
    DATA_ROOT="$DATA_ROOT" \
    LABELED_MAT="$LABELED_MAT" \
    SPLITS_MAT="$SPLITS_MAT" \
      "$ROOT/scripts/dataset_prep/download_nyu_depth_v2_labeled.sh"
  fi
fi

if [[ ! -f "$LABELED_MAT" ]]; then
  echo "ERROR: NYUv2 labeled MAT file not found: $LABELED_MAT" >&2
  echo "Download it with scripts/dataset_prep/download_nyu_depth_v2_labeled.sh" >&2
  exit 1
fi
if [[ ! -f "$SPLITS_MAT" ]]; then
  echo "ERROR: NYUv2 official splits file not found: $SPLITS_MAT" >&2
  echo "Download it with scripts/dataset_prep/download_nyu_depth_v2_labeled.sh" >&2
  exit 1
fi

if [[ -s "$TAR_OUTPUT" && "$FORCE" != "1" ]]; then
  echo "Using existing NYU surface-normal tar: $TAR_OUTPUT"
  echo "Set FORCE=1 to rebuild."
  exit 0
fi

if [[ "$FORCE" == "1" ]]; then
  rm -rf "$OUT_ROOT"
  rm -f "$TAR_OUTPUT" "$TAR_OUTPUT.partial"
fi

args=(
  dataset_prep/surface_normals/prepare_nyu_surface_normals.py
  --mat-path "$LABELED_MAT"
  --splits-mat "$SPLITS_MAT"
  --out-root "$OUT_ROOT"
  --tar-output "$TAR_OUTPUT"
  --split "$SPLIT"
  --depth-source "$DEPTH_SOURCE"
  --mask-source "$MASK_SOURCE"
  --min-depth "$MIN_DEPTH"
  --max-depth "$MAX_DEPTH"
  --gradient-radius "$GRADIENT_RADIUS"
  --mask-erosion-radius "$MASK_EROSION_RADIUS"
  --fx "$FX"
  --fy "$FY"
  --cx "$CX"
  --cy "$CY"
  --invalid-normal-rgb "$INVALID_NORMAL_RGB"
  --overwrite
)
if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi
if is_true "$WRITE_DEPTH_MM"; then
  args+=(--write-depth-mm)
fi

ovb_run_python "${args[@]}"

echo "NYUv2 surface-normal preparation finished."
echo "  labeled mat: $LABELED_MAT"
echo "  splits mat:  $SPLITS_MAT"
echo "  out root:    $OUT_ROOT"
echo "  tar output:  $TAR_OUTPUT"
