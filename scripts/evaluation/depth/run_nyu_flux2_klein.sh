#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

show_help() {
  cat <<'EOF'
Run NYU depth evaluation for either the base Flux2-Klein model or a fine-tuned
Open Vision Banana transformer.

Use cases:
  Case A: persistent-folder / simple setup
    Point DATA_ROOT at extracted NYU data and evaluate directly.
  Case B: scratch/TMPDIR + archive-aware setup
    Point SOURCE_ARCHIVE at nyu.tar and let the script extract into scratch.

Usage:
  scripts/evaluation/depth/run_nyu_flux2_klein.sh

Important environment variables:
  ROOT                     Repo root. Defaults to the current checkout.
  SIF                      Container image used for evaluation.
  MODEL                    Base model path or HF model ID.
  TRANSFORMER_PATH         Optional fine-tuned output/checkpoint/transformer dir.
  OUT_DIR                  Evaluation output directory.
  DATA_ROOT                Extracted NYU root or a parent containing nyu/.

Staging variables:
  SOURCE_ARCHIVE           Persistent NYU tar archive.
  DATA_ARCHIVE             Archive path the script will actually extract.
  COPY_ARCHIVE_TO_TMPDIR   1 to copy the archive into TMPDIR before extraction.
  STAGING_ROOT             Extraction directory.
  FORCE_EXTRACT            1 to re-extract the dataset.

Evaluation controls:
  SPLIT                    train, test, val, or all.
  STEPS                    Number of diffusion/inference steps.
  GUIDANCE_SCALE           Guidance scale passed to the pipeline.
  LIMIT                    Optional sample limit.
  DRY_RUN                  1 to resolve dataset pairs and exit before inference.
  NATIVE_RESOLUTION        1 to generate at input resolution.
  ALIGNMENT_MODE           auto, linear, log, inverse, or disparity-like mode.
  MAX_SHIFT_PX
  SHIFT_STEP
  MIN_OVERLAP_PIXELS
  MIN_DEPTH
  MAX_DEPTH
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

require_path() {
  local path=$1
  local label=$2
  if [[ ! -e "$path" ]]; then
    echo "ERROR: $label not found: $path" >&2
    exit 1
  fi
}

ROOT=${ROOT:-$DEFAULT_ROOT}
SIF=${SIF:?Set SIF to the evaluation container image, or use scripts/evaluate_depth.py}
PYTHON=${PYTHON:-/opt/conda/envs/open-vision-banana/bin/python}

LOCAL_MODEL=${LOCAL_MODEL:-}
BASE_MODEL=${BASE_MODEL:-black-forest-labs/FLUX.2-klein-base-9B}
MODEL=${MODEL:-$BASE_MODEL}
TRANSFORMER_PATH=${TRANSFORMER_PATH:-}

OUT_DIR=${OUT_DIR:-outputs/nyu_depth_base_flux2}
DATA_ROOT=${DATA_ROOT:-}
SPLIT=${SPLIT:-test}
STEPS=${STEPS:-4}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
LIMIT=${LIMIT:-}
DRY_RUN=${DRY_RUN:-0}
NATIVE_RESOLUTION=${NATIVE_RESOLUTION:-0}
ALIGNMENT_MODE=${ALIGNMENT_MODE:-auto}
MAX_SHIFT_PX=${MAX_SHIFT_PX:-8}
SHIFT_STEP=${SHIFT_STEP:-4}
MIN_OVERLAP_PIXELS=${MIN_OVERLAP_PIXELS:-4096}
MIN_DEPTH=${MIN_DEPTH:-0.001}
MAX_DEPTH=${MAX_DEPTH:-10.0}

TMPDIR=${TMPDIR:-/tmp/${USER:-$(id -un)}/open_visionbanana}
STAGING_ROOT=${STAGING_ROOT:-$TMPDIR/open_visionbanana_depth_eval}
SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-}
TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/nyu.tar}
DATA_ARCHIVE=${DATA_ARCHIVE:-$TMP_ARCHIVE}
COPY_ARCHIVE_TO_TMPDIR=${COPY_ARCHIVE_TO_TMPDIR:-1}
FORCE_EXTRACT=${FORCE_EXTRACT:-0}

if [[ -d "$LOCAL_MODEL" && "${MODEL}" == "${BASE_MODEL}" ]]; then
  BASE_MODEL=$LOCAL_MODEL
  MODEL=$BASE_MODEL
fi

if [[ -z "$TRANSFORMER_PATH" && -d "$MODEL/transformer" ]]; then
  TRANSFORMER_PATH=$MODEL
  MODEL=$BASE_MODEL
fi

stage_archive_if_needed() {
  if [[ -f "$DATA_ARCHIVE" ]]; then
    echo "Using NYU archive: $DATA_ARCHIVE"
    return
  fi
  if ! is_true "$COPY_ARCHIVE_TO_TMPDIR"; then
    require_path "$DATA_ARCHIVE" "NYU archive"
    return
  fi
  require_path "$SOURCE_ARCHIVE" "NYU source archive"
  mkdir -p "$(dirname "$DATA_ARCHIVE")"
  echo "Copying $SOURCE_ARCHIVE to $DATA_ARCHIVE"
  cp -f "$SOURCE_ARCHIVE" "$DATA_ARCHIVE"
}

extract_tar_if_needed() {
  local archive=$1
  local out_dir=$2
  local expected_dir=$3

  require_path "$archive" "NYU archive"
  mkdir -p "$out_dir"
  if is_true "$FORCE_EXTRACT"; then
    rm -rf "$out_dir/$expected_dir"
  fi
  if [[ -d "$out_dir/$expected_dir" ]]; then
    echo "Using existing extracted NYU data under: $out_dir/$expected_dir"
    return
  fi

  echo "Extracting $archive into $out_dir"
  tar -xf "$archive" -C "$out_dir"
}

if [[ -z "$DATA_ROOT" ]]; then
  DATA_ROOT=$STAGING_ROOT/nyu
fi
if [[ ! -f "$DATA_ROOT/nyu_test.txt" && ! -f "$DATA_ROOT/nyu_train.txt" ]]; then
  stage_archive_if_needed
  extract_tar_if_needed "$DATA_ARCHIVE" "$STAGING_ROOT" nyu
fi

eval_args=(
  evaluation/depth/inference_partial_flux2_klein_img2img.py
  --model "$MODEL"
  --data_root "$DATA_ROOT"
  --split "$SPLIT"
  --out_dir "$OUT_DIR"
  --steps "$STEPS"
  --guidance_scale "$GUIDANCE_SCALE"
  --alignment_mode "$ALIGNMENT_MODE"
  --max_shift_px "$MAX_SHIFT_PX"
  --shift_step "$SHIFT_STEP"
  --min_overlap_pixels "$MIN_OVERLAP_PIXELS"
  --min_depth "$MIN_DEPTH"
  --max_depth "$MAX_DEPTH"
)
if [[ -n "$TRANSFORMER_PATH" ]]; then
  eval_args+=(--transformer_path "$TRANSFORMER_PATH")
fi
if [[ -n "$LIMIT" ]]; then
  eval_args+=(--limit "$LIMIT")
fi
if is_true "$DRY_RUN"; then
  eval_args+=(--dry_run)
fi
if is_true "$NATIVE_RESOLUTION"; then
  eval_args+=(--native_resolution)
fi

echo "Running NYU depth evaluation"
echo "  repo root:         $ROOT"
echo "  model:             $MODEL"
echo "  transformer path:  ${TRANSFORMER_PATH:-<base model only>}"
echo "  data root:         $DATA_ROOT"
echo "  output dir:        $OUT_DIR"
echo "  split:             $SPLIT"
echo "  steps:             $STEPS"

printf -v eval_cmd "%q " "$PYTHON" "${eval_args[@]}"
printf -v root_q "%q" "$ROOT"

"$SIF" /bin/bash -lc "
set -euo pipefail
unset SSL_CERT_FILE
export SSL_CERT_FILE=/opt/conda/envs/open-vision-banana/lib/python3.10/site-packages/certifi/cacert.pem
cd $root_q
$eval_cmd
"
