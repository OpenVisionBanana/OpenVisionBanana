#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

show_help() {
  cat <<'EOF'
Launch mixed Open Vision Banana partial-transformer fine-tuning from an
already prepared dataset-specs JSON.

Use case:
  Persistent-folder / simple setup.

Use this launcher when your prepared datasets already live at stable,
persistent paths and you do not want to stage raw data through TMPDIR first.

Usage:
  scripts/training/run_open_visionbanana_pool_flux2_klein.sh

Important environment variables:
  ROOT                         Repo root. Defaults to the current checkout.
  SIF                          Container image used for training.
  USE_CONTAINER                1 to run through SIF, 0 to run in the active env.
  DATASET_SPECS                Prepared dataset pool JSON to train on.
  OUTPUT_DIR                   Directory that will receive checkpoints/model.
  CUDA_VISIBLE_DEVICES         GPU list, for example 0 or 0,1,2,3.
  NUM_PROCESSES                Accelerate process count, TOTAL across nodes.
                               Defaults to GPU count times NUM_MACHINES.
  NUM_MACHINES                 Node count (default 1). For multi-node, run this
                               launcher once per node (srun --ntasks-per-node=1)
                               with MAIN_PROCESS_IP/MAIN_PROCESS_PORT set.
  MACHINE_RANK                 This node's rank. Defaults to SLURM_NODEID.
  MAIN_PROCESS_IP              Rendezvous host (first node); required when
                               NUM_MACHINES > 1.
  MAIN_PROCESS_PORT            Rendezvous port (default 29500).
  MODEL                        Base model path or HF model ID.
  LOCAL_MODEL                  Preferred local model directory fallback.
  TRAIN_SCRIPT                 Training entrypoint. Defaults to the standard
                               partial-finetune trainer.

Common training hyperparameters:
  RESOLUTION                   Training resolution.
  ASPECT_RATIO_BUCKETS         Optional semicolon-separated H,W buckets.
  TRAIN_BATCH_SIZE             Per-device batch size.
  PROMPT_EMBEDDING_BATCH_SIZE  Batch size used while caching text embeddings.
  DATALOADER_NUM_WORKERS       Per-rank loader workers (default 4). 0 loads
                               batches synchronously in the training process.
  BUCKET_INDEX_CACHE_DIR       Optional durable cache for deterministic
                               aspect-ratio bucket assignments.
  GRADIENT_ACCUMULATION_STEPS  Gradient accumulation factor.
  LEARNING_RATE                Optimizer learning rate.
  MAX_TRAIN_STEPS              Total optimization steps.
  CHECKPOINTING_STEPS          Save interval for resumable checkpoints.
  CHECKPOINTS_TOTAL_LIMIT      Maximum number of checkpoints to keep.
  TRAIN_LAST_TRANSFORMER_BLOCKS
  TRAIN_LAST_SINGLE_TRANSFORMER_BLOCKS
  TRAIN_OUTPUT_LAYERS          1 to train norm_out/proj_out, 0 to keep frozen.
  TRAIN_INPUT_LAYERS           1 to also train input/conditioning embedders.
  TRAIN_ALL_TRANSFORMER_BLOCKS 1 to train all double/single blocks plus output
                               layers, matching temp/open_visionbanana's
                               full-block setup without training every parameter.
  TRAIN_FULL_TRANSFORMER       1 to train every transformer parameter.
  LOSS_MASK_POLICY             disabled (default), auto, or enabled.
  TRAINABLE_SCOPE_CONTRACT     release_colorexp (default) or separate_embeddings.
  ALLOW_MISSING_GEOMETRY_MASKS Legacy reproduction only; default 0.
  ALLOW_LEGACY_RNG_RESUME      Explicit legacy checkpoint migration; default 0.
  LOSS_MASK_MODE               hard (default) or soft.
  LOSS_MASK_MIN_VALID_FRACTION Latent validity threshold (default 0.5).
  LOSS_MASK_EROSION_RADIUS     Pre-downsampling mask erosion (default 0).
  LOSS_MASK_REDUCTION          per_sample (default) or global_valid.
  DEPTH_ENCODING               vision_banana_rgb, barron_scalar_grayscale,
                               relative_inverse_grayscale, or
                               relative_linear_grayscale.
  DEPTH_TARGET_RESIZE_POLICY   metric_field_then_encode (default) or encoded_rgb.
  NORMAL_TARGET_RESIZE_POLICY  vector_field_renormalize_then_encode (default)
                               or encoded_rgb.
  NORMAL_TARGET_ENCODING       vision_banana_x_left_red (default) or
                               legacy_direct_xyz.
  DEPTH_PROMPT_CONTRACT        Canonical, row, or campaign depth prompt contract.
  NORMAL_PROMPT_CONTRACT       Canonical, row, or campaign normal prompt contract.
  FSDP_TRANSFORMER_DTYPE_POLICY
                               fp32_all (legacy default), auto, bf16, or
                               fp32_trainable.

Optional behavior flags:
  CACHE_LATENTS                1 to precompute VAE latents.
  TASK_HOMOGENEOUS_BATCHES     1 to group batches by bucket and task/source.
  PROMPT_ENCODING_MODE         auto, on_the_fly, precompute, or packed_precompute.
  SAMPLING_STRATEGY            with_replacement or without_replacement.
  MAX_REFSEG_EXPRESSIONS_PER_IMAGE
                               Distinct RefSeg expressions retained per source image.
  PACKED_PROMPT_SHARD_BYTES    Approximate packed prompt shard size.
  VALIDATION_PROMPTS_FILE      Fixed text-only prompt text/JSONL file.
  VALIDATION_NUM_INFERENCE_STEPS
                               Steps for fixed text-only validation.
  VALIDATION_HEIGHT/WIDTH      Optional fixed text-only validation size.
  RESAMPLE_DATASET_EACH_EPOCH  1 to resample dataset-pool rows each epoch.
  VALIDATION_PROMPT            Prompt used for periodic validation.
  VALIDATION_IMAGE             Conditioning image used for validation.
  NUM_VALIDATION_IMAGES        Images per validation run.
  VALIDATION_EPOCHS            Validation frequency in epochs.
  SKIP_FINAL_INFERENCE         1 to skip final validation inference.
  RESUME_FROM_CHECKPOINT       Checkpoint path or the literal value "latest".
  DISTRIBUTED_STRATEGY         ddp or fsdp. Optional alias for FSDP_TRANSFORMER.
  FSDP_TRANSFORMER             1 passes --fsdp_transformer, 0 passes
                               --no-fsdp_transformer. Defaults to 0.
  FSDP_TRANSFORMER_CLS_TO_WRAP Optional comma-separated FSDP block class names.

Typical example:
  CUDA_VISIBLE_DEVICES=0 \
  DATASET_SPECS=/path/to/open_visionbanana_mix_v1.json \
  OUTPUT_DIR=/path/to/finetuned_model \
  scripts/training/run_open_visionbanana_pool_flux2_klein.sh
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

require_file() {
  local path=$1
  local label=$2
  if [[ ! -f "$path" ]]; then
    echo "ERROR: $label not found: $path" >&2
    exit 1
  fi
}

ROOT=${ROOT:-$DEFAULT_ROOT}
SIF=${SIF:-}
USE_CONTAINER=${USE_CONTAINER:-0}
source "$ROOT/scripts/lib/runtime.sh"

DATA_ROOT=${DATA_ROOT:-$ROOT/datasets_archives}
DATASET_SPECS=${DATASET_SPECS:-${PREPARED_ROOT:-$DATA_ROOT/open_visionbanana_prepared}/open_visionbanana_mix_v1.json}

LOCAL_MODEL=${LOCAL_MODEL:-}
MODEL=${MODEL:-black-forest-labs/FLUX.2-klein-base-9B}
if [[ -d "$LOCAL_MODEL" && "${MODEL}" == "black-forest-labs/FLUX.2-klein-base-9B" ]]; then
  MODEL=$LOCAL_MODEL
fi

OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/finetuned_models/open-visionbanana-mix-v1}
LOGGING_DIR=${LOGGING_DIR:-logs}
PRECOMPUTE_CACHE_DIR=${PRECOMPUTE_CACHE_DIR:-}
BUCKET_INDEX_CACHE_DIR=${BUCKET_INDEX_CACHE_DIR:-}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_dreambooth_partial_flux2_klein_img2img.py}
TRAIN_METHOD=${TRAIN_METHOD:-partial_full_rank}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-$LORA_RANK}
LORA_DROPOUT=${LORA_DROPOUT:-0}
LORA_LAYERS=${LORA_LAYERS:-}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
REPORT_TO=${REPORT_TO:-tensorboard}
INSTANCE_PROMPT=${INSTANCE_PROMPT:-[TASK: IMAGE_EDITING] Edit the image according to the instruction.}
OPTIMIZER=${OPTIMIZER:-AdamW}
USE_8BIT_ADAM=${USE_8BIT_ADAM:-0}

RESOLUTION=${RESOLUTION:-512}
ASPECT_RATIO_BUCKETS=${ASPECT_RATIO_BUCKETS:-}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-512}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
PROMPT_EMBEDDING_BATCH_SIZE=${PROMPT_EMBEDDING_BATCH_SIZE:-16}
DATASET_INDEX_NUM_WORKERS=${DATASET_INDEX_NUM_WORKERS:-16}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
DATALOADER_TIMEOUT_SECONDS=${DATALOADER_TIMEOUT_SECONDS:-0}
ROW_LOAD_TIMEOUT_SECONDS=${ROW_LOAD_TIMEOUT_SECONDS:-0}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-2}
DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS:-0}
ALLOW_RESUME_CODE_CHANGE=${ALLOW_RESUME_CODE_CHANGE:-0}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
LR_SCHEDULER=${LR_SCHEDULER:-constant}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-3.5}
WEIGHTING_SCHEME=${WEIGHTING_SCHEME:-none}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-500}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-5000}
CHECKPOINTS_TOTAL_LIMIT=${CHECKPOINTS_TOTAL_LIMIT:-5}
PERMANENT_CHECKPOINT_STEPS=${PERMANENT_CHECKPOINT_STEPS:-}
TRAIN_LAST_TRANSFORMER_BLOCKS=${TRAIN_LAST_TRANSFORMER_BLOCKS:-0}
TRAIN_LAST_SINGLE_TRANSFORMER_BLOCKS=${TRAIN_LAST_SINGLE_TRANSFORMER_BLOCKS:-1}
TRAIN_OUTPUT_LAYERS=${TRAIN_OUTPUT_LAYERS:-1}
TRAIN_INPUT_LAYERS=${TRAIN_INPUT_LAYERS:-0}
TRAIN_ALL_TRANSFORMER_BLOCKS=${TRAIN_ALL_TRANSFORMER_BLOCKS:-0}
TRAIN_FULL_TRANSFORMER=${TRAIN_FULL_TRANSFORMER:-0}
TRAIN_NON_BLOCK_LAYERS=${TRAIN_NON_BLOCK_LAYERS:-0}
TRAINABLE_SCOPE_CONTRACT=${TRAINABLE_SCOPE_CONTRACT:-release_colorexp}
TRAIN_TRANSFORMER_BLOCK_INDICES=${TRAIN_TRANSFORMER_BLOCK_INDICES:-}
TRAIN_SINGLE_TRANSFORMER_BLOCK_INDICES=${TRAIN_SINGLE_TRANSFORMER_BLOCK_INDICES:-}
LOSS_MASK_POLICY=${LOSS_MASK_POLICY:-disabled}
LOSS_MASK_MODE=${LOSS_MASK_MODE:-hard}
LOSS_MASK_MIN_VALID_FRACTION=${LOSS_MASK_MIN_VALID_FRACTION:-0.5}
LOSS_MASK_EROSION_RADIUS=${LOSS_MASK_EROSION_RADIUS:-0}
LOSS_MASK_REDUCTION=${LOSS_MASK_REDUCTION:-per_sample}
LOSS_MASK_MIN_SAMPLE_VALID_FRACTION=${LOSS_MASK_MIN_SAMPLE_VALID_FRACTION:-0.0}
DEPTH_ENCODING=${DEPTH_ENCODING:-vision_banana_rgb}
DEPTH_TARGET_RESIZE_POLICY=${DEPTH_TARGET_RESIZE_POLICY:-metric_field_then_encode}
NORMAL_TARGET_RESIZE_POLICY=${NORMAL_TARGET_RESIZE_POLICY:-vector_field_renormalize_then_encode}
NORMAL_TARGET_ENCODING=${NORMAL_TARGET_ENCODING:-vision_banana_x_left_red}
NORMAL_PROMPT_CONTRACT=${NORMAL_PROMPT_CONTRACT:-vision_banana_v3}
DEPTH_PROMPT_CONTRACT=${DEPTH_PROMPT_CONTRACT:-canonical_v2}
ALLOW_UNVERIFIED_NORMAL_SOURCES=${ALLOW_UNVERIFIED_NORMAL_SOURCES:-0}
ALLOW_LEGACY_HYPERSIM_RANGE_DEPTH=${ALLOW_LEGACY_HYPERSIM_RANGE_DEPTH:-0}
PALETTE_MANIFOLD_LOSS_WEIGHT=${PALETTE_MANIFOLD_LOSS_WEIGHT:-0}
UNIT_NORMAL_LOSS_WEIGHT=${UNIT_NORMAL_LOSS_WEIGHT:-0}
ANGULAR_NORMAL_AUX_WEIGHT=${ANGULAR_NORMAL_AUX_WEIGHT:-0}
METRIC_DEPTH_AUX_WEIGHT=${METRIC_DEPTH_AUX_WEIGHT:-0}
METRIC_DEPTH_AUX_MAX_DEPTH=${METRIC_DEPTH_AUX_MAX_DEPTH:-80}
DATASET_POOL_SEED=${DATASET_POOL_SEED:-42}
SAMPLES_PER_EPOCH=${SAMPLES_PER_EPOCH:-}
RESAMPLE_DATASET_EACH_EPOCH=${RESAMPLE_DATASET_EACH_EPOCH:-1}
TASK_HOMOGENEOUS_BATCHES=${TASK_HOMOGENEOUS_BATCHES:-0}
CACHE_LATENTS=${CACHE_LATENTS:-0}
PROMPT_ENCODING_MODE=${PROMPT_ENCODING_MODE:-packed_precompute}
PROMPT_PRECOMPUTE_MAX_UNIQUE=${PROMPT_PRECOMPUTE_MAX_UNIQUE:-2048}
ALLOW_LARGE_PROMPT_PRECOMPUTE_CACHE=${ALLOW_LARGE_PROMPT_PRECOMPUTE_CACHE:-0}
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-with_replacement}
MAX_REFSEG_EXPRESSIONS_PER_IMAGE=${MAX_REFSEG_EXPRESSIONS_PER_IMAGE:-5}
PACKED_PROMPT_SHARD_BYTES=${PACKED_PROMPT_SHARD_BYTES:-1073741824}
OFFLOAD=${OFFLOAD:-0}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
ALLOW_TF32=${ALLOW_TF32:-0}
SEED=${SEED:-}

VALIDATION_PROMPT=${VALIDATION_PROMPT:-}
VALIDATION_PROMPTS_FILE=${VALIDATION_PROMPTS_FILE:-}
VALIDATION_NUM_INFERENCE_STEPS=${VALIDATION_NUM_INFERENCE_STEPS:-28}
VALIDATION_HEIGHT=${VALIDATION_HEIGHT:-}
VALIDATION_WIDTH=${VALIDATION_WIDTH:-}
VALIDATION_IMAGE=${VALIDATION_IMAGE:-}
FINAL_VALIDATION_PROMPT=${FINAL_VALIDATION_PROMPT:-}
NUM_VALIDATION_IMAGES=${NUM_VALIDATION_IMAGES:-}
VALIDATION_EPOCHS=${VALIDATION_EPOCHS:-}
SKIP_FINAL_INFERENCE=${SKIP_FINAL_INFERENCE:-0}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
FSDP_TRANSFORMER=${FSDP_TRANSFORMER:-0}
FSDP_TRANSFORMER_CLS_TO_WRAP=${FSDP_TRANSFORMER_CLS_TO_WRAP:-}
FSDP_TRANSFORMER_DTYPE_POLICY=${FSDP_TRANSFORMER_DTYPE_POLICY:-fp32_all}
DISTRIBUTED_STRATEGY=${DISTRIBUTED_STRATEGY:-ddp}
case "${DISTRIBUTED_STRATEGY,,}" in
  "") ;;
  ddp) FSDP_TRANSFORMER=0 ;;
  fsdp) FSDP_TRANSFORMER=1 ;;
  *)
    echo "ERROR: unknown DISTRIBUTED_STRATEGY '$DISTRIBUTED_STRATEGY' (use ddp or fsdp)." >&2
    exit 1 ;;
esac

require_file "$DATASET_SPECS" "dataset specs"
mkdir -p "$OUTPUT_DIR"

if [[ "$LOGGING_DIR" = /* ]]; then
  RESOLVED_LOGGING_DIR=$LOGGING_DIR
else
  RESOLVED_LOGGING_DIR=$OUTPUT_DIR/$LOGGING_DIR
fi
mkdir -p "$RESOLVED_LOGGING_DIR"

IFS=',' read -r -a gpu_array <<< "$CUDA_VISIBLE_DEVICES"
# Multi-node: this script must run once per node (srun --ntasks-per-node=1);
# each instance spawns the node's local ranks and all ranks rendezvous at
# MAIN_PROCESS_IP:MAIN_PROCESS_PORT. NUM_PROCESSES is the TOTAL across nodes.
NUM_MACHINES=${NUM_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-${SLURM_NODEID:-0}}
NUM_PROCESSES=${NUM_PROCESSES:-$(( ${#gpu_array[@]} * NUM_MACHINES ))}

accelerate_args=(
  launch
  --num_processes "$NUM_PROCESSES"
  --num_machines "$NUM_MACHINES"
  --mixed_precision "$MIXED_PRECISION"
  --dynamo_backend=no
)
if (( NUM_MACHINES > 1 )); then
  if [[ -z "${MAIN_PROCESS_IP:-}" ]]; then
    echo "ERROR: MAIN_PROCESS_IP must be set when NUM_MACHINES > 1" >&2
    exit 1
  fi
  accelerate_args+=(
    --machine_rank "$MACHINE_RANK"
    --main_process_ip "$MAIN_PROCESS_IP"
    --main_process_port "${MAIN_PROCESS_PORT:-29500}"
  )
fi
if (( NUM_PROCESSES > 1 )); then
  accelerate_args+=(--multi_gpu)
fi

train_args=(
  "$TRAIN_SCRIPT"
  --train_method "$TRAIN_METHOD"
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --pretrained_model_name_or_path "$MODEL"
  --dataset_specs "$DATASET_SPECS"
  --depth_target_encoding "$DEPTH_ENCODING"
  --depth_target_resize_policy "$DEPTH_TARGET_RESIZE_POLICY"
  --normal_target_resize_policy "$NORMAL_TARGET_RESIZE_POLICY"
  --normal_target_encoding "$NORMAL_TARGET_ENCODING"
  --depth_prompt_contract "$DEPTH_PROMPT_CONTRACT"
  --normal_prompt_contract "$NORMAL_PROMPT_CONTRACT"
  --palette_manifold_loss_weight "$PALETTE_MANIFOLD_LOSS_WEIGHT"
  --unit_normal_loss_weight "$UNIT_NORMAL_LOSS_WEIGHT"
  --angular_normal_aux_weight "$ANGULAR_NORMAL_AUX_WEIGHT"
  --metric_depth_aux_weight "$METRIC_DEPTH_AUX_WEIGHT"
  --metric_depth_aux_max_depth "$METRIC_DEPTH_AUX_MAX_DEPTH"
  --image_column image
  --cond_image_column conditioning_image
  --caption_column text
  --instance_prompt "$INSTANCE_PROMPT"
  --output_dir "$OUTPUT_DIR"
  --logging_dir "$LOGGING_DIR"
  --resolution "$RESOLUTION"
  --max_sequence_length "$MAX_SEQUENCE_LENGTH"
  --mixed_precision "$MIXED_PRECISION"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --prompt_embedding_batch_size "$PROMPT_EMBEDDING_BATCH_SIZE"
  --dataset_index_num_workers "$DATASET_INDEX_NUM_WORKERS"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --dataloader_timeout_seconds "$DATALOADER_TIMEOUT_SECONDS"
  --row_load_timeout_seconds "$ROW_LOAD_TIMEOUT_SECONDS"
  --dataloader_prefetch_factor "$DATALOADER_PREFETCH_FACTOR"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --lr_scheduler "$LR_SCHEDULER"
  --adam_weight_decay "$WEIGHT_DECAY"
  --guidance_scale "$GUIDANCE_SCALE"
  --weighting_scheme "$WEIGHTING_SCHEME"
  --lr_warmup_steps "$LR_WARMUP_STEPS"
  --optimizer "$OPTIMIZER"
  --max_train_steps "$MAX_TRAIN_STEPS"
  --train_last_transformer_blocks "$TRAIN_LAST_TRANSFORMER_BLOCKS"
  --train_last_single_transformer_blocks "$TRAIN_LAST_SINGLE_TRANSFORMER_BLOCKS"
  --trainable_scope_contract "$TRAINABLE_SCOPE_CONTRACT"
  --checkpointing_steps "$CHECKPOINTING_STEPS"
  --checkpoints_total_limit "$CHECKPOINTS_TOTAL_LIMIT"
  --dataset_pool_seed "$DATASET_POOL_SEED"
  --report_to "$REPORT_TO"
  --prompt_encoding_mode "$PROMPT_ENCODING_MODE"
  --prompt_precompute_max_unique "$PROMPT_PRECOMPUTE_MAX_UNIQUE"
  --sampling_strategy "$SAMPLING_STRATEGY"
  --max_refseg_expressions_per_image "$MAX_REFSEG_EXPRESSIONS_PER_IMAGE"
  --packed_prompt_shard_bytes "$PACKED_PROMPT_SHARD_BYTES"
  --validation_num_inference_steps "$VALIDATION_NUM_INFERENCE_STEPS"
)
if is_true "$ALLOW_RESUME_CODE_CHANGE"; then
  train_args+=(--allow_resume_code_change)
fi
if is_true "${ALLOW_LEGACY_RNG_RESUME:-0}"; then
  train_args+=(--allow_legacy_rng_resume)
fi
if is_true "$ALLOW_UNVERIFIED_NORMAL_SOURCES"; then
  train_args+=(--allow_unverified_normal_sources)
fi
if is_true "$ALLOW_LEGACY_HYPERSIM_RANGE_DEPTH"; then
  train_args+=(--allow_legacy_hypersim_range_depth)
fi
if is_true "$ALLOW_LARGE_PROMPT_PRECOMPUTE_CACHE"; then
  train_args+=(--allow_large_prompt_precompute_cache)
fi
if [[ -n "$LORA_LAYERS" ]]; then
  train_args+=(--lora_layers "$LORA_LAYERS")
fi
if [[ -n "$PERMANENT_CHECKPOINT_STEPS" ]]; then
  train_args+=(--permanent_checkpoint_steps "$PERMANENT_CHECKPOINT_STEPS")
fi
if [[ -n "$ASPECT_RATIO_BUCKETS" ]]; then
  train_args+=(--aspect_ratio_buckets "$ASPECT_RATIO_BUCKETS")
fi

if is_true "$GRADIENT_CHECKPOINTING"; then
  train_args+=(--gradient_checkpointing)
fi
if is_true "$TRAIN_OUTPUT_LAYERS"; then
  train_args+=(--train_output_layers)
else
  train_args+=(--no-train_output_layers)
fi
if is_true "$TRAIN_INPUT_LAYERS"; then
  train_args+=(--train_input_layers)
else
  train_args+=(--no-train_input_layers)
fi
if is_true "$TRAIN_ALL_TRANSFORMER_BLOCKS"; then
  train_args+=(--train_all_transformer_blocks)
fi
if is_true "$TRAIN_FULL_TRANSFORMER"; then
  train_args+=(--train_full_transformer)
fi
if is_true "$TRAIN_NON_BLOCK_LAYERS"; then
  train_args+=(--train_non_block_layers)
fi
if [[ -n "$TRAIN_TRANSFORMER_BLOCK_INDICES" ]]; then
  train_args+=(--train_transformer_block_indices "$TRAIN_TRANSFORMER_BLOCK_INDICES")
fi
if [[ -n "$TRAIN_SINGLE_TRANSFORMER_BLOCK_INDICES" ]]; then
  train_args+=(--train_single_transformer_block_indices "$TRAIN_SINGLE_TRANSFORMER_BLOCK_INDICES")
fi
if is_true "$RESAMPLE_DATASET_EACH_EPOCH"; then
  train_args+=(--resample_dataset_each_epoch)
else
  train_args+=(--no-resample_dataset_each_epoch)
fi
if is_true "$CACHE_LATENTS"; then
  train_args+=(--cache_latents)
fi
if is_true "$OFFLOAD"; then
  train_args+=(--offload)
fi
if is_true "$USE_8BIT_ADAM"; then
  train_args+=(--use_8bit_adam)
fi
if is_true "$DATALOADER_PERSISTENT_WORKERS"; then
  train_args+=(--dataloader_persistent_workers)
fi
if is_true "$TASK_HOMOGENEOUS_BATCHES"; then
  train_args+=(--task_homogeneous_batches)
fi
if is_true "$ALLOW_TF32"; then
  train_args+=(--allow_tf32)
fi
if is_true "$SKIP_FINAL_INFERENCE"; then
  train_args+=(--skip_final_inference)
fi
if [[ -n "$SAMPLES_PER_EPOCH" ]]; then
  train_args+=(--samples_per_epoch "$SAMPLES_PER_EPOCH")
fi
if [[ -n "$PRECOMPUTE_CACHE_DIR" ]]; then
  train_args+=(--precompute_cache_dir "$PRECOMPUTE_CACHE_DIR")
fi
if [[ -n "$BUCKET_INDEX_CACHE_DIR" ]]; then
  train_args+=(--bucket_index_cache_dir "$BUCKET_INDEX_CACHE_DIR")
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  train_args+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
if [[ -n "$SEED" ]]; then
  train_args+=(--seed "$SEED")
fi
if [[ -n "$VALIDATION_PROMPT" ]]; then
  train_args+=(--validation_prompt "$VALIDATION_PROMPT")
fi
if [[ -n "$VALIDATION_PROMPTS_FILE" ]]; then
  train_args+=(--validation_prompts_file "$VALIDATION_PROMPTS_FILE")
fi
if [[ -n "$VALIDATION_HEIGHT" || -n "$VALIDATION_WIDTH" ]]; then
  if [[ -z "$VALIDATION_HEIGHT" || -z "$VALIDATION_WIDTH" ]]; then
    echo "ERROR: set both VALIDATION_HEIGHT and VALIDATION_WIDTH, or neither." >&2
    exit 2
  fi
  train_args+=(--validation_height "$VALIDATION_HEIGHT" --validation_width "$VALIDATION_WIDTH")
fi
if [[ -n "$VALIDATION_IMAGE" ]]; then
  train_args+=(--validation_image "$VALIDATION_IMAGE")
fi
if [[ -n "$FINAL_VALIDATION_PROMPT" ]]; then
  train_args+=(--final_validation_prompt "$FINAL_VALIDATION_PROMPT")
fi
if [[ -n "$NUM_VALIDATION_IMAGES" ]]; then
  train_args+=(--num_validation_images "$NUM_VALIDATION_IMAGES")
fi
if [[ -n "$VALIDATION_EPOCHS" ]]; then
  train_args+=(--validation_epochs "$VALIDATION_EPOCHS")
fi
if is_true "$FSDP_TRANSFORMER"; then
  train_args+=(--fsdp_transformer)
else
  train_args+=(--no-fsdp_transformer)
fi
if [[ -n "$FSDP_TRANSFORMER_CLS_TO_WRAP" ]]; then
  train_args+=(--fsdp_transformer_cls_to_wrap "$FSDP_TRANSFORMER_CLS_TO_WRAP")
fi
train_args+=(--loss_mask_policy "$LOSS_MASK_POLICY")
if is_true "${ALLOW_MISSING_GEOMETRY_MASKS:-0}"; then
  train_args+=(--allow_missing_geometry_masks)
fi
train_args+=(--loss_mask_mode "$LOSS_MASK_MODE")
train_args+=(--loss_mask_min_valid_fraction "$LOSS_MASK_MIN_VALID_FRACTION")
train_args+=(--loss_mask_erosion_radius "$LOSS_MASK_EROSION_RADIUS")
train_args+=(--loss_mask_reduction "$LOSS_MASK_REDUCTION")
train_args+=(--loss_mask_min_sample_valid_fraction "$LOSS_MASK_MIN_SAMPLE_VALID_FRACTION")
train_args+=(--fsdp_transformer_dtype_policy "$FSDP_TRANSFORMER_DTYPE_POLICY")

echo "Launching persistent-data training run"
echo "  repo root:              $ROOT"
echo "  use container:          $USE_CONTAINER"
echo "  container:              $SIF"
echo "  dataset specs:          $DATASET_SPECS"
echo "  model:                  $MODEL"
echo "  train script:           $TRAIN_SCRIPT"
echo "  output dir:             $OUTPUT_DIR"
echo "  logging dir:            $RESOLVED_LOGGING_DIR"
echo "  visible GPUs:           $CUDA_VISIBLE_DEVICES"
echo "  accelerate processes:   $NUM_PROCESSES"
echo "  machines / node rank:   $NUM_MACHINES / $MACHINE_RANK"
echo "  batch size/device:      $TRAIN_BATCH_SIZE"
echo "  aspect buckets:         ${ASPECT_RATIO_BUCKETS:-${RESOLUTION}x${RESOLUTION}}"
echo "  grad accumulation:      $GRADIENT_ACCUMULATION_STEPS"
echo "  dataloader workers:     $DATALOADER_NUM_WORKERS (prefetch $DATALOADER_PREFETCH_FACTOR, persistent $DATALOADER_PERSISTENT_WORKERS)"
echo "  row/load timeouts:      ${ROW_LOAD_TIMEOUT_SECONDS}s / ${DATALOADER_TIMEOUT_SECONDS}s"
echo "  allow code-only resume: $ALLOW_RESUME_CODE_CHANGE"
echo "  max train steps:        $MAX_TRAIN_STEPS"
echo "  guidance scale:         $GUIDANCE_SCALE"
echo "  optimizer:              $OPTIMIZER"
echo "  use 8-bit Adam:         $USE_8BIT_ADAM"
echo "  fsdp transformer:       $FSDP_TRANSFORMER"
echo "  fsdp dtype policy:      $FSDP_TRANSFORMER_DTYPE_POLICY"
echo "  loss mask policy:       $LOSS_MASK_POLICY"
echo "  train all blocks:       $TRAIN_ALL_TRANSFORMER_BLOCKS"
echo "  train full transformer: $TRAIN_FULL_TRANSFORMER"
echo "  train input layers:     $TRAIN_INPUT_LAYERS"
echo "  cache latents:          $CACHE_LATENTS"
echo "  bucket-index cache:     ${BUCKET_INDEX_CACHE_DIR:-disabled}"
echo "  prompt encoding:        $PROMPT_ENCODING_MODE"
echo "  sampling strategy:      $SAMPLING_STRATEGY"
echo "  max RefSeg expressions: $MAX_REFSEG_EXPRESSIONS_PER_IMAGE"
echo "  packed prompt shard B:  $PACKED_PROMPT_SHARD_BYTES"
echo "  validation prompts:     ${VALIDATION_PROMPTS_FILE:-<none>}"
echo "  offload:                $OFFLOAD"
echo "  task-homogeneous batch: $TASK_HOMOGENEOUS_BATCHES"

printf -v accelerate_cmd "%q " accelerate "${accelerate_args[@]}" "${train_args[@]}"
printf -v root_q "%q" "$ROOT"
printf -v visible_gpus_q "%q" "$CUDA_VISIBLE_DEVICES"

if [[ "$USE_CONTAINER" == "0" ]]; then
  bash -lc "
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$visible_gpus_q
cd $root_q
$accelerate_cmd
"
else
  [[ -n "$SIF" ]] || { echo "ERROR: set SIF when USE_CONTAINER=1" >&2; exit 2; }
  ovb_container_exec --nv "$SIF" /bin/bash -lc "
set -euo pipefail
unset SSL_CERT_FILE
export SSL_CERT_FILE=/opt/conda/envs/open-vision-banana/lib/python3.10/site-packages/certifi/cacert.pem
export CUDA_VISIBLE_DEVICES=$visible_gpus_q
cd $root_q
$accelerate_cmd
"
fi
