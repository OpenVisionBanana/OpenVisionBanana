#!/usr/bin/env bash
set -euo pipefail

# Rasterize the referring-segmentation eval masks (ReasonSeg + RefCOCOg UMD val)
# from their downloaded sources via dataset_prep/refseg/prepare_refseg_eval.py.
# CPU-only; override paths with environment variables instead of editing this file.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

show_help() {
  cat <<'EOF'
Prepare referring-segmentation eval masks (ReasonSeg val + RefCOCOg UMD val).

Usage:
  scripts/dataset_prep/prepare_refseg_eval.sh

Environment variables:
  ROOT             Repo root. Defaults to the current checkout.
  PYTHON           Python interpreter. Defaults to `python`.
  DATASETS         Which to prepare. Default: "reasonseg refcocog".
  REASONSEG_DIR    ReasonSeg val dir (<stem>.jpg/.json + masks/ output).
  REFCOCOG_DIR     RefCOCOg dir (refs(umd).p + instances.json).
  REFCOCOG_IMAGES  COCO train2014 image dir. Default: $REFCOCOG_DIR/train2014.
  SPLIT            RefCOCOg UMD split. Default: val.
  FORCE            1 to re-rasterize even if a masks/ dir already exists.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

is_true() { case "${1,,}" in 1|true|yes|y|on) return 0 ;; *) return 1 ;; esac; }

ROOT=${ROOT:-$DEFAULT_ROOT}
PYTHON=${PYTHON:-python}

DATASETS=${DATASETS:-"reasonseg refcocog"}
REFSEG_ROOT=${REFSEG_ROOT:-datasets_archives/refseg}
REASONSEG_DIR=${REASONSEG_DIR:-$REFSEG_ROOT/reasonseg_val}
REFCOCOG_DIR=${REFCOCOG_DIR:-$REFSEG_ROOT/refcocog}
REFCOCOG_IMAGES=${REFCOCOG_IMAGES:-$REFCOCOG_DIR/train2014}
SPLIT=${SPLIT:-val}
FORCE=${FORCE:-0}

run_in_env() {
  local command=$*
  ( cd "$ROOT" && bash -lc "$command" )
}

# masks dir non-empty -> already prepared (unless FORCE). Accept both the
# original repository-relative paths and portable absolute roots.
masks_present() {
  local path=$1
  [[ "$path" == /* ]] || path=$ROOT/$path
  compgen -G "$path/masks/*.png" > /dev/null 2>&1
}

for ds in $DATASETS; do
  case "$ds" in
    reasonseg)
      if ! is_true "$FORCE" && masks_present "$REASONSEG_DIR"; then
        echo "Using existing ReasonSeg masks: $REASONSEG_DIR/masks"
        continue
      fi
      printf -v cmd "%q " "$PYTHON" dataset_prep/refseg/prepare_refseg_eval.py reasonseg \
        --data-dir "$REASONSEG_DIR"
      run_in_env "$cmd"
      ;;
    refcocog)
      out_dir="$REFCOCOG_DIR/${SPLIT}_umd"
      if ! is_true "$FORCE" && masks_present "$out_dir"; then
        echo "Using existing RefCOCOg masks: $out_dir/masks"
        continue
      fi
      printf -v cmd "%q " "$PYTHON" dataset_prep/refseg/prepare_refseg_eval.py refcocog \
        --data-dir "$REFCOCOG_DIR" --images-dir "$REFCOCOG_IMAGES" --split "$SPLIT"
      run_in_env "$cmd"
      ;;
    *)
      echo "ERROR: unknown dataset '$ds' (expected reasonseg or refcocog)" >&2
      exit 1
      ;;
  esac
done

echo "Done. Prepared: $DATASETS"
