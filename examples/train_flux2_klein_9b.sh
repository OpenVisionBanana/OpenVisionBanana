#!/usr/bin/env bash
# Portable four-GPU FLUX.2-Klein 9B fine-tuning example. No scheduler is assumed.
# Settings follow the released zero-shot 9B model (TRAIN.md), with a smaller
# batch layout and gradient checkpointing for four GPUs. Set
# LOSS_MASK_POLICY=auto for the non-zero-shot pool.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${MODEL:-black-forest-labs/FLUX.2-klein-base-9B}
DATASET_SPECS=${DATASET_SPECS:?Set DATASET_SPECS to a prepared-pool JSON file}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/outputs/flux2-klein-9b}
NUM_GPUS=${NUM_GPUS:-4}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-100000}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACCUM=${GRAD_ACCUM:-16}
LOSS_MASK_POLICY=${LOSS_MASK_POLICY:-disabled}

accelerate launch --num_processes "$NUM_GPUS" \
  "$ROOT/train_dreambooth_partial_flux2_klein_img2img.py" \
  --pretrained_model_name_or_path "$MODEL" \
  --dataset_specs "$DATASET_SPECS" \
  --output_dir "$OUTPUT_DIR" \
  --train_method partial_full_rank \
  --train_all_transformer_blocks \
  --train_non_block_layers \
  --train_output_layers \
  --fsdp_transformer \
  --resolution 512 \
  --train_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --max_train_steps "$MAX_TRAIN_STEPS" \
  --learning_rate 2e-6 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 1000 \
  --guidance_scale 3.5 \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --checkpointing_steps 1000 \
  --permanent_checkpoint_steps 12500,25000,37500,50000,62500,75000,87500,100000 \
  --checkpoints_total_limit 5 \
  --seed 42 \
  --depth_target_encoding relative_inverse_grayscale \
  --depth_target_resize_policy metric_field_then_encode \
  --depth_prompt_contract campaign_detailed_inverse \
  --normal_target_encoding vision_banana_x_left_red \
  --normal_target_resize_policy vector_field_renormalize_then_encode \
  --normal_prompt_contract campaign_short \
  --loss_mask_policy "$LOSS_MASK_POLICY" \
  --loss_mask_erosion_radius 0 \
  "$@"
