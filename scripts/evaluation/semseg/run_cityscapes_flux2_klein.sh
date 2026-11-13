#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

show_help() {
  cat <<'EOF'
Run semantic-segmentation evaluation for Cityscapes or another supported layout
using either the base Flux2-Klein model or a fine-tuned Open Vision Banana
transformer.

Use case:
  Mostly Case A: persistent-folder / simple setup.

This launcher expects the evaluation dataset to already exist as an extracted
folder referenced by DATA_ROOT.

Usage:
  scripts/evaluation/semseg/run_cityscapes_flux2_klein.sh

Important environment variables:
  ROOT               Repo root. Defaults to the current checkout.
  SIF                Container image used for evaluation.
  MODEL              Base model path or HF model ID.
  TRANSFORMER_PATH   Optional fine-tuned output/checkpoint/transformer dir.
  OUT_DIR            Evaluation output directory.
  STAGE_QUALITATIVE_IN_TMPDIR
                     1 to run with OUT_DIR in TMPDIR, copy quantitative
                     root-level files to OUT_DIR, and tar qualitative
                     prediction folders into OUT_DIR.
  KEEP_TMPDIR_OUTPUTS
                     1 to keep staged TMPDIR outputs after archiving.

Dataset selection:
  DATASET            Dataset layout handled by the Python evaluator.
  DATA_ROOT          Dataset root with RGB and GT folders.
  CITYSCAPES_TAR     Cityscapes archive used when DATA_ROOT is not extracted.
  SPLIT              Dataset split, usually val.
  LIMIT              Optional sample limit.
  DRY_RUN            1 to resolve image/GT pairs and exit before inference.
  NATIVE_RESOLUTION  1 to generate at input resolution.

Inference controls:
  STEPS              Number of diffusion/inference steps.
  GUIDANCE_SCALE     Guidance scale passed to the pipeline.
  GENERATION_SEED    Base diffusion seed; combined with each image id so the
                     result is stable across sharding layouts.
  NUM_PROCESSES      Number of evaluator processes/GPUs. Defaults to 1.
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

has_any_dir() {
  local parent=$1
  shift
  local candidate
  for candidate in "$@"; do
    if [[ -d "$parent/$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

resolve_dir() {
  local parent=$1
  shift
  local candidate
  for candidate in "$@"; do
    if [[ -d "$parent/$candidate" ]]; then
      printf '%s\n' "$parent/$candidate"
      return
    fi
  done
  echo "ERROR: could not find extracted dataset directory under $parent (tried: $*)" >&2
  exit 1
}

extract_tar_if_needed() {
  local archive=$1
  local out_dir=$2
  shift 2
  if [[ ! -f "$archive" ]]; then
    echo "ERROR: archive not found: $archive" >&2
    exit 1
  fi
  mkdir -p "$out_dir"
  if has_any_dir "$out_dir" "$@"; then
    echo "Using existing extracted data under: $out_dir"
    return
  fi
  echo "Extracting $archive into $out_dir"
  tar -xf "$archive" -C "$out_dir"
}

ROOT=${ROOT:-$DEFAULT_ROOT}
SIF=${SIF:-}   # container image; the host environment is used when unset
PYTHON=${PYTHON:-/opt/conda/envs/open-vision-banana/bin/python}

LOCAL_MODEL=${LOCAL_MODEL:-}
BASE_MODEL=${BASE_MODEL:-black-forest-labs/FLUX.2-klein-base-9B}
MODEL=${MODEL:-$BASE_MODEL}
TRANSFORMER_PATH=${TRANSFORMER_PATH:-}
MODEL_FAMILY=${MODEL_FAMILY:-klein}
MODEL_OFFLOAD=${MODEL_OFFLOAD:-0}
OFFLOAD_MODE=${OFFLOAD_MODE:-}

OUT_DIR_EXPLICIT=${OUT_DIR:-}
OUT_DIR=${OUT_DIR:-outputs/cityscapes_base_flux2}
STAGE_QUALITATIVE_IN_TMPDIR=${STAGE_QUALITATIVE_IN_TMPDIR:-0}
KEEP_TMPDIR_OUTPUTS=${KEEP_TMPDIR_OUTPUTS:-0}
DATASET=${DATASET:-cityscapes}
DATA_ROOT=${DATA_ROOT:-}
CITYSCAPES_TAR=${CITYSCAPES_TAR:-$(dirname -- "$DATA_ROOT")/cityscapes.tar}
# COCO (coco-thing / coco-stuff) val archive; staged like cityscapes.tar when the
# extracted dataset/ tree isn't already present under DATA_ROOT.
COCO_TAR=${COCO_TAR:-$ROOT/datasets_archives/sem_segm/coco.tar}
SPLIT=${SPLIT:-val}
STEPS=${STEPS:-4}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-3.5}
# fixed | randomdraw | set | grouped.
#   fixed/randomdraw -> the full 19-class evaluator (inference_partial...). randomdraw
#     uses the fixed 19-class prompt but per-image random colors (LEGEND_SEED makes
#     them reproducible).
#   set -> the dynamic-subset evaluator (set_semseg.py): per-image class subset +
#     random colors + randomized prompt, scored over the selected classes. SUBSET_SEED
#     makes the eval set reproducible; EVAL_SET reuses a saved eval_set.json.
#   grouped -> the multi-pass full-taxonomy evaluator (grouped_semseg.py): per image,
#     ALL classes are partitioned into K = ceil(N/GROUP_SIZE) disjoint groups (seeded
#     by SUBSET_SEED), one pass per group with a training-style legend, merged by snap
#     distance into one full-taxonomy prediction -> mIoU comparable to fixed. ~K x cost.
PROMPT_STYLE=${PROMPT_STYLE:-fixed}
# randomdraw only: generate each image N times with independent color draws and
# majority-vote the decoded masks. 1 = the original single pass. N>1 costs N x the
# generation time and is NOT comparable with a repeats=1 score.
RANDOMDRAW_REPEATS=${RANDOMDRAW_REPEATS:-1}
LEGEND_SEED=${LEGEND_SEED:-0}
SUBSET_SEED=${SUBSET_SEED:-0}
# set only: force exactly this many classes per image instead of the random
# training M-rule. Empty = unchanged default behaviour. A fixed-K run is a
# different protocol and its scores do not compare with default-rule scores.
SET_CLASSES_PER_IMAGE=${SET_CLASSES_PER_IMAGE:-}
EVAL_SET=${EVAL_SET:-}
REGENERATE=${REGENERATE:-0}
# grouped-style knobs. FARTHEST_COLORS=1 (default): black-anchored max-min named
# colors, image-wide distinct; 0 = training-matched random dE>=25 draw (_rndcol).
GROUP_SIZE=${GROUP_SIZE:-5}
GROUP_BATCH_SIZE=${GROUP_BATCH_SIZE:-4}
GROUP_NOTATION=${GROUP_NOTATION:-rgb}
COLOR_MIN_DELTA_E=${COLOR_MIN_DELTA_E:-25.0}
FARTHEST_COLORS=${FARTHEST_COLORS:-1}
# Palette draw for the eval legends (diversify.COLOR_MAPPING_MODES); set it to the
# checkpoint's training color_mapping so eval colors come from the trained pool.
COLOR_MAPPING=${COLOR_MAPPING:-p200_randomnegdraw}
PANELS=${PANELS:-1}
# grouped only: 1 writes just what cannot be rebuilt (raws, resume_cache, legends,
# final maps, metrics); evaluation/semseg/rebuild_grouped_artifacts.py restores the rest.
MINIMAL_ARTIFACTS=${MINIMAL_ARTIFACTS:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}
# Cross-job array sharding (one 1-GPU shard per array task evaluating
# samples[SHARD_INDEX::NUM_SHARDS] into the shared OUT_DIR). Empty NUM_SHARDS keeps
# the single-job (optionally torchrun NUM_PROCESSES) behaviour. NO_AGGREGATE=1 makes
# each shard write only its shard_metrics; a separate aggregate job merges them.
NUM_SHARDS=${NUM_SHARDS:-}
SHARD_INDEX=${SHARD_INDEX:-0}
NO_AGGREGATE=${NO_AGGREGATE:-0}
# Sharded runs write predictions straight to the shared scratch OUT_DIR (disjoint
# files per shard), so the TMPDIR-staging/tar path must be off.
if [[ -n "$NUM_SHARDS" ]]; then
  STAGE_QUALITATIVE_IN_TMPDIR=0
fi
LIMIT=${LIMIT:-}
DRY_RUN=${DRY_RUN:-0}
NATIVE_RESOLUTION=${NATIVE_RESOLUTION:-0}
GENERATION_SEED=${GENERATION_SEED:-}
VISUAL_LEGEND_MODE=${VISUAL_LEGEND_MODE:-none}
# RESUME=1 reuses samples a previous run already finished in OUT_DIR instead of
# regenerating them (see --resume in inference_partial_flux2_klein_img2img.py).
# OFF by default: a plain rerun regenerates everything. Only the
# fixed/randomdraw evaluator supports it -- see the warning below.
RESUME=${RESUME:-0}
TMP_ROOT=${QUALITATIVE_TMPDIR:-${TMPDIR:-/tmp}}
JOB_ID=${SLURM_JOB_ID:-manual-$$}
JOB_STAGING_DIR="$TMP_ROOT/open-vision-banana-semseg-$JOB_ID"

if [[ -d "$LOCAL_MODEL" && "${MODEL}" == "${BASE_MODEL}" ]]; then
  BASE_MODEL=$LOCAL_MODEL
  MODEL=$BASE_MODEL
fi

if [[ -z "$TRANSFORMER_PATH" && -d "$MODEL/transformer" ]]; then
  TRANSFORMER_PATH=$MODEL
  MODEL=$BASE_MODEL
fi

if [[ "$DATASET" == "cityscapes" && ! -d "$DATA_ROOT/leftImg8bit/$SPLIT" ]]; then
  DATA_STAGE="$JOB_STAGING_DIR/data/sem_segm"
  extract_tar_if_needed "$CITYSCAPES_TAR" "$DATA_STAGE" cityscapes Cityscapes
  DATA_ROOT=$(resolve_dir "$DATA_STAGE" cityscapes Cityscapes)
fi

# COCO val (coco.tar has a top-level dataset/ tree). Extract once if the dataset/
# images aren't already under DATA_ROOT; the Python resolvers handle the dataset/
# fallback, so DATA_ROOT is just the dir holding dataset/.
if [[ "$DATASET" == coco-* \
      && ! -d "$DATA_ROOT/dataset/images/${SPLIT}2017" \
      && ! -d "$DATA_ROOT/images/${SPLIT}2017" ]]; then
  DATA_STAGE="$JOB_STAGING_DIR/data/coco"
  extract_tar_if_needed "$COCO_TAR" "$DATA_STAGE" dataset coco
  DATA_ROOT="$DATA_STAGE"
fi

PERSISTENT_OUT_DIR=$OUT_DIR
if [[ "$PERSISTENT_OUT_DIR" != /* ]]; then
  PERSISTENT_OUT_DIR="$ROOT/$PERSISTENT_OUT_DIR"
fi
RUN_OUT_DIR=$OUT_DIR
STAGING_DIR=
if is_true "$STAGE_QUALITATIVE_IN_TMPDIR" && ! is_true "$DRY_RUN"; then
  OUT_BASENAME=$(basename -- "$PERSISTENT_OUT_DIR")
  STAGING_DIR="$JOB_STAGING_DIR"
  RUN_OUT_DIR="$STAGING_DIR/$OUT_BASENAME"
  mkdir -p "$RUN_OUT_DIR" "$PERSISTENT_OUT_DIR"
fi

if [[ "$PROMPT_STYLE" == "set" ]]; then
  # Dynamic-subset evaluation (mirrors training diversification).
  eval_args=(
    evaluation/semseg/set_semseg.py
    --model "$MODEL"
    --dataset "$DATASET"
    --data_root "$DATA_ROOT"
    --split "$SPLIT"
    --out_dir "$RUN_OUT_DIR"
    --steps "$STEPS"
    --guidance_scale "$GUIDANCE_SCALE"
    --subset_seed "$SUBSET_SEED"
    --color_mapping "$COLOR_MAPPING"
  )
  # Empty (the default) leaves the training M-rule in place; only passed when set.
  if [[ -n "$SET_CLASSES_PER_IMAGE" ]]; then
    eval_args+=(--classes_per_image "$SET_CLASSES_PER_IMAGE")
  fi
  if [[ -n "$EVAL_SET" ]]; then
    eval_args+=(--eval-set "$EVAL_SET")
  fi
  if is_true "$REGENERATE"; then
    eval_args+=(--regenerate)
  fi
elif [[ "$PROMPT_STYLE" == "grouped" ]]; then
  # Grouped multi-pass full-taxonomy evaluation (K = ceil(classes/GROUP_SIZE) passes/image).
  eval_args=(
    evaluation/semseg/grouped_semseg.py
    --model "$MODEL"
    --dataset "$DATASET"
    --data_root "$DATA_ROOT"
    --split "$SPLIT"
    --out_dir "$RUN_OUT_DIR"
    --steps "$STEPS"
    --guidance_scale "$GUIDANCE_SCALE"
    --subset_seed "$SUBSET_SEED"
    --group_size "$GROUP_SIZE"
    --group_batch_size "$GROUP_BATCH_SIZE"
    --notation "$GROUP_NOTATION"
    --color_min_delta_e "$COLOR_MIN_DELTA_E"
    --color_mapping "$COLOR_MAPPING"
  )
  if is_true "$FARTHEST_COLORS"; then
    eval_args+=(--farthest_colors)
  else
    eval_args+=(--no-farthest_colors)
  fi
  if ! is_true "$PANELS"; then
    eval_args+=(--no_panels)
  fi
  if is_true "$MINIMAL_ARTIFACTS"; then
    eval_args+=(--minimal_artifacts)
  fi
  if is_true "$DRY_RUN"; then
    eval_args+=(--dry_run)
  fi
else
  eval_args=(
    evaluation/semseg/inference_partial_flux2_klein_img2img.py
    --model "$MODEL"
    --dataset "$DATASET"
    --data_root "$DATA_ROOT"
    --split "$SPLIT"
    --out_dir "$RUN_OUT_DIR"
    --steps "$STEPS"
    --guidance_scale "$GUIDANCE_SCALE"
    --prompt_style "$PROMPT_STYLE"
    --legend_seed "$LEGEND_SEED"
    --color_mapping "$COLOR_MAPPING"
    --randomdraw_repeats "$RANDOMDRAW_REPEATS"
    --visual_legend_mode "$VISUAL_LEGEND_MODE"
  )
  if is_true "$DRY_RUN"; then
    eval_args+=(--dry_run)   # set_semseg.py has no --dry_run
  fi
  if is_true "$RESUME"; then
    eval_args+=(--resume)
  fi
fi
if [[ -n "$GENERATION_SEED" ]]; then
  eval_args+=(--generation_seed "$GENERATION_SEED")
fi
# set_semseg.py / grouped_semseg.py have no --resume; warn instead of silently
# regenerating everything.
if is_true "$RESUME" && [[ "$PROMPT_STYLE" != "fixed" && "$PROMPT_STYLE" != "randomdraw" ]]; then
  echo "WARNING: RESUME=1 is ignored for prompt style '$PROMPT_STYLE'; only fixed/randomdraw" >&2
  echo "         support it. This run will regenerate every sample." >&2
fi
if [[ "$PROMPT_STYLE" == "fixed" || "$PROMPT_STYLE" == "randomdraw" ]]; then
  eval_args+=(--model_family "$MODEL_FAMILY")
  if is_true "$MODEL_OFFLOAD"; then eval_args+=(--model_offload); fi
  if [[ -n "$OFFLOAD_MODE" ]]; then eval_args+=(--offload_mode "$OFFLOAD_MODE"); fi
fi
if [[ -n "$NUM_SHARDS" ]]; then
  eval_args+=(--num_shards "$NUM_SHARDS" --shard_index "$SHARD_INDEX")
  if is_true "$NO_AGGREGATE"; then
    eval_args+=(--no_aggregate)
  fi
fi
if [[ -n "$TRANSFORMER_PATH" ]]; then
  eval_args+=(--transformer_path "$TRANSFORMER_PATH")
fi
if [[ -n "$LIMIT" ]]; then
  eval_args+=(--limit "$LIMIT")
fi
if is_true "$NATIVE_RESOLUTION"; then
  eval_args+=(--native_resolution)
fi

echo "Running semantic-segmentation evaluation"
echo "  repo root:         $ROOT"
echo "  dataset:           $DATASET"
echo "  data root:         $DATA_ROOT"
echo "  model:             $MODEL"
echo "  model family:      $MODEL_FAMILY (offload=${OFFLOAD_MODE:-$MODEL_OFFLOAD})"
echo "  transformer path:  ${TRANSFORMER_PATH:-<base model only>}"
echo "  output dir:        $PERSISTENT_OUT_DIR"
if [[ -n "$STAGING_DIR" ]]; then
  echo "  staging out dir:   $RUN_OUT_DIR"
fi
echo "  split:             $SPLIT"
echo "  steps:             $STEPS"
echo "  prompt style:      $PROMPT_STYLE"
echo "  visual legend:     $VISUAL_LEGEND_MODE"
echo "  randomdraw repeats:$RANDOMDRAW_REPEATS"
echo "  set classes/image: ${SET_CLASSES_PER_IMAGE:-<random M-rule>}"
echo "  color mapping:     $COLOR_MAPPING"
echo "  eval processes:    $NUM_PROCESSES"

# Cross-job sharding runs one process on this task's single GPU (no torchrun).
if [[ -z "$NUM_SHARDS" ]] && (( NUM_PROCESSES > 1 )); then
  launch_args=(
    "$PYTHON"
    -m torch.distributed.run
    --standalone
    --nproc_per_node "$NUM_PROCESSES"
    "${eval_args[@]}"
  )
else
  launch_args=("$PYTHON" "${eval_args[@]}")
fi

printf -v eval_cmd "%q " "${launch_args[@]}"
printf -v root_q "%q" "$ROOT"

# USE_CONTAINER=0 runs directly in the active environment (e.g. a venv);
# the default uses the Singularity image ($SIF).
if [[ "${USE_CONTAINER:-${SIF:+1}}" != "1" ]]; then
  ( cd "$ROOT" && eval "$eval_cmd" )
else
  "$SIF" /bin/bash -lc "
set -euo pipefail
unset SSL_CERT_FILE
export SSL_CERT_FILE=/opt/conda/envs/open-vision-banana/lib/python3.10/site-packages/certifi/cacert.pem
cd $root_q
$eval_cmd
"
fi

if [[ -n "$STAGING_DIR" ]]; then
  echo "Archiving qualitative outputs from TMPDIR"
  mkdir -p "$PERSISTENT_OUT_DIR"

  find "$RUN_OUT_DIR" -maxdepth 1 -type f -exec cp -f -t "$PERSISTENT_OUT_DIR" {} +

  qualitative_dirs=()
  for qualitative_dir in pred_rgb_raw pred_rgb_snapped pred_trainids panels legends visual_legends; do
    if [[ -d "$RUN_OUT_DIR/$qualitative_dir" ]]; then
      qualitative_dirs+=("$qualitative_dir")
    fi
  done

  if (( ${#qualitative_dirs[@]} > 0 )); then
    tar -C "$RUN_OUT_DIR" -czf "$PERSISTENT_OUT_DIR/qualitative_predictions.tar.gz" "${qualitative_dirs[@]}"
    printf "  archived:          %s\n" "$PERSISTENT_OUT_DIR/qualitative_predictions.tar.gz"
  else
    echo "  no qualitative prediction directories found to archive"
  fi

  echo "  quantitative dir:  $PERSISTENT_OUT_DIR"

  if ! is_true "$KEEP_TMPDIR_OUTPUTS"; then
    rm -rf "$STAGING_DIR"
  fi
fi
