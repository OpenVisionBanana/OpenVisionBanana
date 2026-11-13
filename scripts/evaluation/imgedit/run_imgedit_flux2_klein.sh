#!/usr/bin/env bash
set -euo pipefail

# Naming convention:
#   eval:             imgedit_basic / imgedit_uge
#   eval params:      steps28_gs4.0
#   eval out:         outputs/${MODEL_TAG}/${CKPT}/imgedit_basic_steps28_gs4.0
# For pretrained base evaluation:
#   MODEL_TAG=f2k9b_pretrained TRANSFORMER_PATH= bash run_imgedit_flux2_klein.sh

MODEL_ID=${MODEL_ID:-black-forest-labs/FLUX.2-klein-base-9B}
BENCHMARK_ROOT=${BENCHMARK_ROOT:?Set BENCHMARK_ROOT to the ImgEdit-Bench Benchmark directory}
JUDGE_MODEL=${JUDGE_MODEL:-ImgEdit_Judge}

: "${MODEL_TAG:?Set MODEL_TAG, e.g. f2k9b_pretrained or f2k9b_bddacdc-id2color_r512_d0-s8-out_lr1e-6_b1-ga2-gpu3_20k_wu100_offload}"
OUTPUT_DIR=${OUTPUT_DIR:-finetuned_models/${MODEL_TAG}}
CKPT=${CKPT:-checkpoint-5000}
TRANSFORMER_PATH=${TRANSFORMER_PATH-$OUTPUT_DIR/$CKPT/transformer}
OUTPUT_MODEL_TAG=${OUTPUT_MODEL_TAG:-${MODEL_TAG}}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
NUM_PROCESSES=${NUM_PROCESSES:-3}
# Manual cross-job sharding: when NUM_SHARDS is set, inference runs as a single
# process on one GPU and only evaluates samples[SHARD_INDEX::NUM_SHARDS], writing
# disjoint <key>.png into the shared out_dir (skip-if-exists makes it idempotent).
# Leave NUM_SHARDS empty for the default torchrun (NUM_PROCESSES GPUs) behaviour.
NUM_SHARDS=${NUM_SHARDS:-}
SHARD_INDEX=${SHARD_INDEX:-0}
# Pipeline-stage toggles so generation (GPU-parallel, sharded) and the judge
# (one pass over the full folder) can run as separate SLURM jobs.
RUN_INFERENCE=${RUN_INFERENCE:-1}
RUN_JUDGE=${RUN_JUDGE:-1}
# Judge-side cross-job sharding, mirroring NUM_SHARDS for generation. The default
# (empty) keeps the torchrun path below, which is single-node; set these to spread
# the judge over a multi-node allocation. No rank can merge in that mode -- each
# shard writes shard_results/rank_*_of_*.json and a final JUDGE_MERGE_SHARDS=1 pass
# (CPU-only, no model load) merges them and writes the aggregates.
JUDGE_NUM_SHARDS=${JUDGE_NUM_SHARDS:-}
JUDGE_SHARD_INDEX=${JUDGE_SHARD_INDEX:-0}
JUDGE_MERGE_SHARDS=${JUDGE_MERGE_SHARDS:-0}
# First N samples, applied to generation AND the judge. Both derive their key list
# as sort_json_keys(edit_info) sliced to LIMIT, so they stay in agreement. Empty =
# the full suite. A capped run must not be reported against a full-suite one.
LIMIT=${LIMIT:-}
EVAL_STEPS=${EVAL_STEPS:-28}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-3.5}
# Task tag prepended to every edit instruction. `-` not `:-`: an explicitly empty
# PROMPT_PREFIX sends the bare instruction (the control for the base model, which
# was never trained with the tag); unset gets the default.
PROMPT_PREFIX=${PROMPT_PREFIX-[TASK: IMAGE_EDITING]}
JUDGE_EXTRA_ARGS=${JUDGE_EXTRA_ARGS:-}
# Generation seed. Empty (default) leaves the generator unseeded (not reproducible).
# Repeat runs must use a different seed per repeat.
SEED=${SEED:-}
EVAL_PARAMS_TAG=${EVAL_PARAMS_TAG:-steps${EVAL_STEPS}_gs${GUIDANCE_SCALE}}
if [[ -n "${TRANSFORMER_PATH:-}" ]]; then
  MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT:-outputs/${OUTPUT_MODEL_TAG}/${CKPT}}
else
  MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT:-outputs/${OUTPUT_MODEL_TAG}}
fi

IMGEDIT_BASIC_OUT_DIR=${IMGEDIT_BASIC_OUT_DIR:-${MODEL_OUTPUT_ROOT}/imgedit_basic_${EVAL_PARAMS_TAG}}
IMGEDIT_UGE_OUT_DIR=${IMGEDIT_UGE_OUT_DIR:-${MODEL_OUTPUT_ROOT}/imgedit_uge_${EVAL_PARAMS_TAG}}

run_imgedit_suite() {
  local suite=$1
  local out_dir=$2

  echo "Running ImgEdit ${suite}"
  echo "  model tag:        $OUTPUT_MODEL_TAG"
  echo "  checkpoint:       ${CKPT:-<none>}"
  echo "  eval params:      $EVAL_PARAMS_TAG"
  echo "  transformer path: ${TRANSFORMER_PATH:-<base model only>}"
  echo "  output dir:       $out_dir"
  echo "  seed:             ${SEED:-<ambient RNG>}"
  echo "  prompt prefix:    ${PROMPT_PREFIX:-<none: bare instruction>}"

  inference_args=(
    evaluation/imgedit/inference_flux2_klein_imgedit.py
    --suite "$suite"
    --benchmark_root "$BENCHMARK_ROOT"
    --model "$MODEL_ID"
    --out_dir "$out_dir"
    --steps "$EVAL_STEPS"
    --guidance_scale "$GUIDANCE_SCALE"
    --prompt_prefix "$PROMPT_PREFIX"
  )

  if [[ -n "$SEED" ]]; then
    inference_args+=(--seed "$SEED")
  fi

  if [[ -n "$LIMIT" ]]; then
    inference_args+=(--limit "$LIMIT")
  fi

  if [[ -n "${TRANSFORMER_PATH:-}" ]]; then
    inference_args+=(--transformer_path "$TRANSFORMER_PATH")
  fi

  if [[ -n "$NUM_SHARDS" ]]; then
    echo "  sharding:         shard $SHARD_INDEX of $NUM_SHARDS (single GPU)"
    inference_args+=(--num_shards "$NUM_SHARDS" --shard_index "$SHARD_INDEX")
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python "${inference_args[@]}"
  else
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    python -m torch.distributed.run --standalone --nproc_per_node "$NUM_PROCESSES" \
      "${inference_args[@]}"
  fi
}

if [[ "$RUN_INFERENCE" == "1" ]]; then
  run_imgedit_suite basic "$IMGEDIT_BASIC_OUT_DIR"
  run_imgedit_suite uge "$IMGEDIT_UGE_OUT_DIR"
fi

# Merge the per-shard generation manifests into the final manifest.jsonl here,
# at judge time, when every generation shard is guaranteed finished.
merge_shard_manifests() {
  local out_dir=$1
  local sm="$out_dir/shard_manifests"
  if compgen -G "$sm/rank_*_of_*.jsonl" >/dev/null 2>&1; then
    cat "$sm"/rank_*_of_*.jsonl > "$out_dir/manifest.jsonl"
    echo "Merged $(ls "$sm"/rank_*_of_*.jsonl | wc -l) shard manifests -> $out_dir/manifest.jsonl"
  fi
}

if [[ "$RUN_JUDGE" == "1" ]]; then
  # Data-parallel judging: one full judge per GPU scoring keys[rank::N], rank 0
  # merging (device_map="auto" alone would pipeline-split one judge across GPUs).
  JUDGE_NUM_PROCESSES=${JUDGE_NUM_PROCESSES:-$NUM_PROCESSES}
  judge_launcher=(python)
  judge_shard_args=()
  if [[ "$JUDGE_MERGE_SHARDS" == "1" ]]; then
    judge_shard_args=(--merge_shards)
    echo "Merging judge shard results (CPU only, no model load)"
  elif [[ -n "$JUDGE_NUM_SHARDS" ]]; then
    judge_shard_args=(--num_shards "$JUDGE_NUM_SHARDS" --shard_index "$JUDGE_SHARD_INDEX")
    echo "Judging shard $JUDGE_SHARD_INDEX of $JUDGE_NUM_SHARDS on GPUs $CUDA_VISIBLE_DEVICES"
  else
    if [[ "$JUDGE_NUM_PROCESSES" -gt 1 ]]; then
      judge_launcher=(python -m torch.distributed.run --standalone --nproc_per_node "$JUDGE_NUM_PROCESSES")
    fi
    echo "Judging with $JUDGE_NUM_PROCESSES process(es) on GPUs $CUDA_VISIBLE_DEVICES"
  fi

  run_judge_suite() {
    local suite=$1 out_dir=$2
    # Concatenating the generation manifests is a single-writer step; judge shards
    # skip it, the merge pass (and the unsharded path) does it.
    if [[ -z "$JUDGE_NUM_SHARDS" || "$JUDGE_MERGE_SHARDS" == "1" ]]; then
      merge_shard_manifests "$out_dir"
    fi
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    "${judge_launcher[@]}" evaluation/imgedit/judge_imgedit_qwen.py \
      --suite "$suite" \
      --benchmark_root "$BENCHMARK_ROOT" \
      --judge_model "$JUDGE_MODEL" \
      --result_img_folder "$out_dir" \
      ${LIMIT:+--limit "$LIMIT"} \
      ${judge_shard_args+"${judge_shard_args[@]}"} \
      $JUDGE_EXTRA_ARGS
  }

  run_judge_suite basic "$IMGEDIT_BASIC_OUT_DIR"
  run_judge_suite uge   "$IMGEDIT_UGE_OUT_DIR"
fi