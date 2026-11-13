#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
source "$ROOT/scripts/lib/runtime.sh"

show_help() {
  cat <<'EOF'
Run GenAI-Bench image-generation evaluation for FLUX.2-Klein.

Usage:
  scripts/evaluation/image_generation/run_genai_bench_flux2_klein.sh

Important environment variables:
  ROOT                 Repo root.
  SIF                  Generation container image.
  SCORE_SIF            Scoring container image.
  SCORE_MODEL_PATH     Local Qwen2.5-VL evaluator snapshot for offline scoring.
  MODEL                Base model path or Hugging Face model ID.
  MODEL_TAG            Output/checkpoint grouping tag.
  CKPT                 Optional checkpoint folder name.
  TRANSFORMER_PATH     Optional fine-tuned output/checkpoint/transformer dir.
  PROMPT_DIR           Directory for prepared prompt JSONL.
  PROMPTS_JSONL        Prepared prompt JSONL path.
  DATASET_NAME         Hugging Face prompt dataset name.
  FALLBACK_DATASET_NAME
                       Fallback Hugging Face prompt dataset name.
  GENAI_SOURCE_DIR     Optional local prompt source directory.
  OUT_DIR              Persistent output directory.
  RUN_PREP             1 to prepare prompts.
  RUN_GENERATE         1 to generate images.
  RUN_SCORE            1 to score generated images.
  EVAL_STEPS           Diffusion/inference steps.
  GUIDANCE_SCALE       Guidance scale.
  LIMIT                Optional prompt limit.
  NUM_SHARDS           Manual shard count.
  SHARD_INDEX          Manual shard index.
  STAGE_GENERATIONS_IN_TMPDIR
                       1 to generate/score in TMPDIR, then archive/copy results.
  KEEP_TMPDIR_OUTPUTS  1 to keep staged TMPDIR outputs.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

SIF=${SIF:-}
SCORE_SIF=${SCORE_SIF:-}
PYTHON=${PYTHON:-python}
SCORE_PYTHON=${SCORE_PYTHON:-python}

BASE_MODEL=${BASE_MODEL:-black-forest-labs/FLUX.2-klein-base-9B}
LOCAL_MODEL=${LOCAL_MODEL:-}
MODEL=${MODEL:-$BASE_MODEL}
if [[ -d "$LOCAL_MODEL" && "$MODEL" == "$BASE_MODEL" ]]; then
  MODEL=$LOCAL_MODEL
fi

SCRATCH_ROOT=${SCRATCH_ROOT:-${TMPDIR:-/tmp}}
CKPT_ROOT=${CKPT_ROOT:-$SCRATCH_ROOT/checkpoints/open_visionbanana}
MODEL_TAG=${MODEL_TAG:-f2k9b_pretrained}
CKPT=${CKPT:-}
OUTPUT_DIR=${OUTPUT_DIR:-$CKPT_ROOT/$MODEL_TAG}
TRANSFORMER_PATH=${TRANSFORMER_PATH:-}
MODEL_FAMILY=${MODEL_FAMILY:-klein}
LORA_PATH=${LORA_PATH:-}
MODEL_OFFLOAD=${MODEL_OFFLOAD:-0}
if [[ -z "$TRANSFORMER_PATH" && -n "$CKPT" ]]; then
  TRANSFORMER_PATH="$OUTPUT_DIR/$CKPT/transformer"
fi
if [[ -z "$TRANSFORMER_PATH" && -d "$OUTPUT_DIR/transformer" ]]; then
  TRANSFORMER_PATH="$OUTPUT_DIR"
fi

PROMPT_DIR=${PROMPT_DIR:-$SCRATCH_ROOT/genai_bench_prompts}
PROMPTS_JSONL=${PROMPTS_JSONL:-$PROMPT_DIR/prompts.jsonl}
DATASET_NAME=${DATASET_NAME:-BaiqiL/GenAI-Bench}
FALLBACK_DATASET_NAME=${FALLBACK_DATASET_NAME:-BaiqiL/GenAI-Bench-1600}
DATASET_SPLIT=${DATASET_SPLIT:-}
GENAI_SOURCE_DIR=${GENAI_SOURCE_DIR:-}
PREP_LIMIT=${PREP_LIMIT:-}
PREP_NO_STREAMING=${PREP_NO_STREAMING:-0}

EVAL_STEPS=${EVAL_STEPS:-28}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-3.5}
HEIGHT=${HEIGHT:-}
WIDTH=${WIDTH:-}
SEED=${SEED:-0}
PROMPT_PREFIX=${PROMPT_PREFIX:-}
LIMIT=${LIMIT:-}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_INDEX=${SHARD_INDEX:-0}
OVERWRITE=${OVERWRITE:-0}
DRY_RUN=${DRY_RUN:-0}

EVALUATOR=${EVALUATOR:-qwen2.5-vl-7b}
SCORE_MODEL_PATH=${SCORE_MODEL_PATH:-}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-8}
SCORE_DEVICE=${SCORE_DEVICE:-cuda}
SCORE_CACHE_DIR=${SCORE_CACHE_DIR:-}
SCORE_OUT_DIR=${SCORE_OUT_DIR:-}

RUN_PREP=${RUN_PREP:-1}
RUN_GENERATE=${RUN_GENERATE:-1}
RUN_SCORE=${RUN_SCORE:-1}
STAGE_GENERATIONS_IN_TMPDIR=${STAGE_GENERATIONS_IN_TMPDIR:-0}
KEEP_TMPDIR_OUTPUTS=${KEEP_TMPDIR_OUTPUTS:-0}
GENAI_TMPDIR=${GENAI_TMPDIR:-${TMPDIR:-/tmp}}

OUT_DIR=${OUT_DIR:?Set OUT_DIR to the persistent GenAI-Bench result directory}

run_python() {
  local sif=$1
  local python_bin=$2
  shift 2
  if [[ -n "$sif" ]]; then
    ovb_container_exec --nv "$sif" "$python_bin" "$@"
  else
    "$python_bin" "$@"
  fi
}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

copy_staged_metadata() {
  local run_dir=$1
  local persistent_dir=$2
  mkdir -p "$persistent_dir"
  find "$run_dir" -maxdepth 1 -type f ! -name "generated_images.tar.gz" -exec cp -f -t "$persistent_dir" {} +
  if [[ -d "$run_dir/shard_manifests" ]]; then
    rm -rf "$persistent_dir/shard_manifests"
    cp -a "$run_dir/shard_manifests" "$persistent_dir/"
  fi
  if [[ -d "$run_dir/scores" ]]; then
    rm -rf "$persistent_dir/scores"
    cp -a "$run_dir/scores" "$persistent_dir/"
    python - "$persistent_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for metrics_path in root.glob("scores/*/metrics.json"):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    data["generation_dir"] = str(root)
    score_tensor = metrics_path.parent / "scores.pt"
    if score_tensor.exists():
        data["score_tensor"] = str(score_tensor)
    metrics_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
for per_prompt_path in root.glob("scores/*/per_prompt.jsonl"):
    rows = []
    for line in per_prompt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image = Path(str(row.get("image", "")))
        if image.is_absolute() and image.name:
            row["image"] = f"images/{image.name}"
        rows.append(row)
    per_prompt_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
PY
  fi
}

archive_staged_images() {
  local run_dir=$1
  local persistent_dir=$2
  copy_staged_metadata "$run_dir" "$persistent_dir"
  if [[ -d "$run_dir/images" ]]; then
    echo "Archiving GenAI-Bench generated images from TMPDIR"
    tar -C "$run_dir" -czf "$run_dir/generated_images.tar.gz" images
    mv -f "$run_dir/generated_images.tar.gz" "$persistent_dir/generated_images.tar.gz"
    echo "  archived images: $persistent_dir/generated_images.tar.gz"
  else
    echo "  no images directory found to archive"
  fi
  echo "  metadata dir:    $persistent_dir"
}

ensure_images_for_scoring() {
  local run_dir=$1
  if [[ -d "$run_dir/images" ]]; then
    return
  fi
  if [[ -f "$run_dir/generated_images.tar.gz" ]]; then
    echo "Extracting archived GenAI-Bench images for scoring: $run_dir/generated_images.tar.gz"
    tar -C "$run_dir" -xzf "$run_dir/generated_images.tar.gz" images
  fi
}

cd "$ROOT"

PERSISTENT_OUT_DIR=$OUT_DIR
RUN_OUT_DIR=$OUT_DIR
STAGING_DIR=
if is_true "$STAGE_GENERATIONS_IN_TMPDIR" && is_true "$RUN_GENERATE" && ! is_true "$DRY_RUN"; then
  if [[ "$NUM_SHARDS" != "1" || "$SHARD_INDEX" != "0" ]]; then
    echo "ERROR: STAGE_GENERATIONS_IN_TMPDIR=1 currently supports one unsharded generation run only." >&2
    echo "       Use NUM_SHARDS=1 SHARD_INDEX=0, or disable staging for sharded array runs." >&2
    exit 1
  fi
  JOB_ID=${SLURM_JOB_ID:-manual-$$}
  STAGING_DIR="$GENAI_TMPDIR/open-vision-banana-genai-$JOB_ID"
  RUN_OUT_DIR="$STAGING_DIR/output"
  mkdir -p "$RUN_OUT_DIR" "$PERSISTENT_OUT_DIR"
fi

if [[ "$RUN_PREP" == "1" ]]; then
  prep_args=(
    evaluation/image_generation/genai_bench/prepare_prompts.py
    --dataset_name "$DATASET_NAME"
    --fallback_dataset_name "$FALLBACK_DATASET_NAME"
    --out_dir "$PROMPT_DIR"
  )
  if [[ -n "$DATASET_SPLIT" ]]; then
    prep_args+=(--split "$DATASET_SPLIT")
  fi
  if [[ -n "$GENAI_SOURCE_DIR" ]]; then
    prep_args+=(--source_dir "$GENAI_SOURCE_DIR")
  fi
  if [[ -n "$PREP_LIMIT" ]]; then
    prep_args+=(--limit "$PREP_LIMIT")
  fi
  if [[ "$PREP_NO_STREAMING" == "1" ]]; then
    prep_args+=(--no_streaming)
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    prep_args+=(--overwrite)
  fi
  echo "Preparing GenAI-Bench prompts"
  run_python "$SIF" "$PYTHON" "${prep_args[@]}"
fi

if [[ "$RUN_GENERATE" == "1" ]]; then
  gen_args=(
    evaluation/image_generation/genai_bench/generate_flux2_klein.py
    --prompts_jsonl "$PROMPTS_JSONL"
    --out_dir "$RUN_OUT_DIR"
    --model "$MODEL"
    --model_tag "$MODEL_TAG"
    --checkpoint "$CKPT"
    --steps "$EVAL_STEPS"
    --guidance_scale "$GUIDANCE_SCALE"
    --seed "$SEED"
    --prompt_prefix "$PROMPT_PREFIX"
    --num_shards "$NUM_SHARDS"
    --shard_index "$SHARD_INDEX"
  )
  if [[ -n "$TRANSFORMER_PATH" ]]; then
    gen_args+=(--transformer_path "$TRANSFORMER_PATH")
  fi
  gen_args+=(--model_family "$MODEL_FAMILY")
  [[ -n "$LORA_PATH" ]] && gen_args+=(--lora_path "$LORA_PATH")
  [[ "$MODEL_OFFLOAD" == "1" ]] && gen_args+=(--model_offload)
  if [[ -n "$HEIGHT" ]]; then
    gen_args+=(--height "$HEIGHT")
  fi
  if [[ -n "$WIDTH" ]]; then
    gen_args+=(--width "$WIDTH")
  fi
  if [[ -n "$LIMIT" ]]; then
    gen_args+=(--limit "$LIMIT")
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    gen_args+=(--overwrite)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    gen_args+=(--dry_run)
  fi

  echo "Generating GenAI-Bench images"
  echo "  model:            $MODEL"
  echo "  transformer path: ${TRANSFORMER_PATH:-<base model only>}"
  echo "  output dir:       $PERSISTENT_OUT_DIR"
  if [[ "$RUN_OUT_DIR" != "$PERSISTENT_OUT_DIR" ]]; then
    echo "  staging out dir:  $RUN_OUT_DIR"
  fi
  run_python "$SIF" "$PYTHON" "${gen_args[@]}"
fi

if [[ "$RUN_SCORE" == "1" ]]; then
  ensure_images_for_scoring "$RUN_OUT_DIR"
  score_args=(
    evaluation/image_generation/genai_bench/score_t2v_metrics.py
    --generation_dir "$RUN_OUT_DIR"
    --evaluator "$EVALUATOR"
    --device "$SCORE_DEVICE"
    --batch_size "$SCORE_BATCH_SIZE"
  )
  if [[ -n "$SCORE_CACHE_DIR" ]]; then
    score_args+=(--cache_dir "$SCORE_CACHE_DIR")
  fi
  if [[ -n "$SCORE_MODEL_PATH" ]]; then
    score_args+=(--checkpoint "$SCORE_MODEL_PATH")
  fi
  if [[ -n "$SCORE_OUT_DIR" ]]; then
    score_args+=(--out_dir "$SCORE_OUT_DIR")
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    score_args+=(--overwrite)
  fi

  echo "Scoring GenAI-Bench images"
  echo "  evaluator: $EVALUATOR"
  run_python "$SCORE_SIF" "$SCORE_PYTHON" "${score_args[@]}"
fi

if [[ -n "$STAGING_DIR" ]]; then
  archive_staged_images "$RUN_OUT_DIR" "$PERSISTENT_OUT_DIR"
  if ! is_true "$KEEP_TMPDIR_OUTPUTS"; then
    rm -rf "$STAGING_DIR"
  fi
fi
