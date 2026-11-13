#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Run surface-normal evaluation through the geometry-v6 protocol for NYUv2,
iBims-1, DIODE, KITTI, ETH3D, nuScenes, Virtual KITTI 2, ScanNet, or Sintel.

Usage:
  DATASET=nyu SOURCE_ARCHIVE=/path/to/nyu_surface_normals.tar \
    scripts/evaluation/surface_normals/run_surface_normals.sh

Dataset controls:
  DATASET                  nyu, ibims1, diode, kitti, eth3d, nuscenes, vkitti2, scannet, or sintel. Default: nyu.
  ARCHIVE_ROOT             Root containing depth/ and surface_normals/ prepared tars.
  DATA_ROOT                Extracted prepared dataset root.
  SOURCE_ARCHIVE           Persistent tar archive copied/extracted to TMPDIR.
  EXPECTED_DIR             Top-level folder expected after tar extraction.
  CASES                    Space-separated cases. KITTI: depth_selection. ETH3D: low_res_two_view. nuScenes: cam_front. VKITTI2: clone_camera_0.
  AUX_AUGMENTATIONS        Space-separated iBims aux groups. Default: appearance.
  IBIMS1_NORMALS_PROTOCOL  nyu_10m or ibims_official. Default: ibims_official.
                           ibims_official uses surface_normals/ibims1_official.tar by default.
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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

ROOT=${ROOT:-$DEFAULT_ROOT}
SIF=${SIF:-}
PYTHON=${PYTHON:-python}
USE_CONTAINER=${USE_CONTAINER:-0}
DATASET=${DATASET:-nyu}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-${PREPARED_ROOT:-}}

LOCAL_MODEL=${LOCAL_MODEL:-}
BASE_MODEL=${BASE_MODEL:-black-forest-labs/FLUX.2-klein-base-9B}
MODEL=${MODEL:-$BASE_MODEL}
TRANSFORMER_PATH=${TRANSFORMER_PATH:-}
MODEL_FAMILY=${MODEL_FAMILY:-klein}
LORA_PATH=${LORA_PATH:-}
MODEL_OFFLOAD=${MODEL_OFFLOAD:-0}
if [[ -d "$LOCAL_MODEL" && "${MODEL}" == "${BASE_MODEL}" ]]; then BASE_MODEL=$LOCAL_MODEL; MODEL=$BASE_MODEL; fi
if [[ -z "$TRANSFORMER_PATH" && -d "$MODEL/transformer" && ! -f "$MODEL/model_index.json" ]]; then TRANSFORMER_PATH=$MODEL; MODEL=$BASE_MODEL; fi

OUT_DIR=${OUT_DIR:-outputs/${DATASET}_surface_normals_base_flux2}
STAGE_QUALITATIVE_IN_TMPDIR=${STAGE_QUALITATIVE_IN_TMPDIR:-1}
KEEP_TMPDIR_OUTPUTS=${KEEP_TMPDIR_OUTPUTS:-0}
DURABLE_EVAL_RESUME=${DURABLE_EVAL_RESUME:-1}
RESUME_CHUNK_SIZE=${RESUME_CHUNK_SIZE:-50}
EVAL_NUM_GPUS=${EVAL_NUM_GPUS:-${NUM_PROCESSES:-1}}
EVAL_SHARD_SYNC_TIMEOUT_SECONDS=${EVAL_SHARD_SYNC_TIMEOUT_SECONDS:-3600}
EVAL_GPU_LAUNCH_STAGGER_SECONDS=${EVAL_GPU_LAUNCH_STAGGER_SECONDS:-0}
DATA_ROOT=${DATA_ROOT:-}
if [[ -z "${SPLIT+x}" ]]; then
  if [[ "$DATASET" == "nuscenes" ]]; then SPLIT=val; else SPLIT=test; fi
fi
CASES=${CASES:-standard}
AUX_AUGMENTATIONS=${AUX_AUGMENTATIONS:-appearance}
IBIMS1_NORMALS_PROTOCOL=${IBIMS1_NORMALS_PROTOCOL:-ibims_official}
STEPS=${STEPS:-28}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
SEED=${SEED:-0}
GEOMETRY_SEED_MODE=${GEOMETRY_SEED_MODE:-fixed_per_sample_reset}
PROMPT=${PROMPT:-}
LIMIT=${LIMIT:-}
SAMPLE_MANIFEST=${SAMPLE_MANIFEST:-}
SAMPLE_MANIFEST_ROOT=${SAMPLE_MANIFEST_ROOT:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/manifests}}
PRIMARY_CASE=${PRIMARY_CASE:-}
EVAL_PROTOCOL_VERSION=${EVAL_PROTOCOL_VERSION:-geometry-eval-v6}
DATASET_ROLE=${DATASET_ROLE:-}
DRY_RUN=${DRY_RUN:-0}
NATIVE_RESOLUTION=${NATIVE_RESOLUTION:-0}
MIN_NORMAL_NORM=${MIN_NORMAL_NORM:-0.01}
MIN_VALID_PIXELS=${MIN_VALID_PIXELS:-4096}
EVAL_MASK_EROSION_RADIUS=${EVAL_MASK_EROSION_RADIUS:-0}
DIAGNOSTIC_MASK_EROSION_RADIUS=${DIAGNOSTIC_MASK_EROSION_RADIUS:-0}
BIDIRECTIONAL_NORMALS=${BIDIRECTIONAL_NORMALS:-0}
REPORT_BIDIRECTIONAL_DIAGNOSTIC=${REPORT_BIDIRECTIONAL_DIAGNOSTIC:-1}
ANGLE_VIS_MAX=${ANGLE_VIS_MAX:-90.0}
PREDICTION_NORMAL_ENCODING=${PREDICTION_NORMAL_ENCODING:-vision_banana_x_left_red}
GT_NORMAL_ENCODING=${GT_NORMAL_ENCODING:-auto}
GT_NORMAL_PERMUTATION=${GT_NORMAL_PERMUTATION:-}
GT_NORMAL_SIGNS=${GT_NORMAL_SIGNS:-}

TMPDIR=${TMPDIR:-/tmp/${USER:-$(id -un)}/open_visionbanana}
mkdir -p "$TMPDIR"
TMP_ROOT=${QUALITATIVE_TMPDIR:-${TMPDIR:-/tmp}}
JOB_ID=${SLURM_JOB_ID:-manual-$$}
JOB_STAGING_DIR="$TMP_ROOT/open-vision-banana-surface-normals-$DATASET-$JOB_ID"
EVAL_SHARD_SYNC_DIR=${EVAL_SHARD_SYNC_DIR:-$TMPDIR/open-vb-normal-shards-$DATASET-$JOB_ID}
STAGING_ROOT=${STAGING_ROOT:-$TMPDIR/open_visionbanana_surface_normals_eval_$DATASET}
COPY_ARCHIVE_TO_TMPDIR=${COPY_ARCHIVE_TO_TMPDIR:-1}
FORCE_EXTRACT=${FORCE_EXTRACT:-0}

case "$DATASET" in
  nyu)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/surface_normals/nyu.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-nyu_surface_normals}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/nyu_surface_normals.tar}
    ;;
  ibims1)
    if [[ "$IBIMS1_NORMALS_PROTOCOL" == "ibims_official" ]]; then
      SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/surface_normals/ibims1_official.tar}}
      EXPECTED_DIR=${EXPECTED_DIR:-ibims}
      TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/ibims1_surface_normals_official.tar}
    else
      SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/surface_normals/ibims1.tar}}
      EXPECTED_DIR=${EXPECTED_DIR:-ibims1_surface_normals}
      TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/ibims1_surface_normals.tar}
    fi
    ;;
  diode)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/surface_normals/diode.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-diode_surface_normals}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/diode_surface_normals.tar}
    ;;
  kitti)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/surface_normals/kitti.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-kitti_depth_selection_surface_normals}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/kitti_depth_selection_surface_normals.tar}
    ;;
  eth3d)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/surface_normals/eth3d.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-eth3d_low_res_two_view_surface_normals}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/eth3d_low_res_two_view_surface_normals.tar}
    ;;
  nuscenes)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/surface_normals/nuscenes.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-nuscenes_trainval_val_cam_front_surface_normals}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/nuscenes_trainval_val_cam_front_surface_normals.tar}
    ;;
  vkitti2)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/surface_normals/vkitti2.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-vkitti2_dsine_surface_normals}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/vkitti2_clone_camera_0_surface_normals.tar}
    ;;
  scannet)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/surface_normals/scannet.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-scannet_depth_normals}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/scannet.tar}
    ;;
  sintel)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/surface_normals/sintel.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-sintel_surface_normals}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/sintel_surface_normals.tar}
    ;;
  *)
    echo "ERROR: Unsupported DATASET=$DATASET" >&2
    exit 1
    ;;
esac


if [[ -z "$SAMPLE_MANIFEST" && -n "$SAMPLE_MANIFEST_ROOT" && -f "$SAMPLE_MANIFEST_ROOT/$DATASET.jsonl" ]]; then
  SAMPLE_MANIFEST=$SAMPLE_MANIFEST_ROOT/$DATASET.jsonl
fi
DATA_ARCHIVE=${DATA_ARCHIVE:-$TMP_ARCHIVE}

if [[ -z "$DATASET_ROLE" ]]; then
  case "$DATASET" in
    nyu|diode|scannet|vkitti2|sintel) DATASET_ROLE=vision_banana_primary ;;
    ibims1|eth3d|kitti|nuscenes) DATASET_ROLE=extended_diagnostic ;;
  esac
fi
if [[ -z "$PRIMARY_CASE" && "$DATASET" == diode ]]; then PRIMARY_CASE=indoor; fi

stage_archive_if_needed() {
  if is_true "$FORCE_EXTRACT" && is_true "$COPY_ARCHIVE_TO_TMPDIR" && [[ -f "$SOURCE_ARCHIVE" ]]; then
    mkdir -p "$(dirname "$DATA_ARCHIVE")"
    echo "Refreshing $DATASET archive cache: $DATA_ARCHIVE"
    cp -f "$SOURCE_ARCHIVE" "$DATA_ARCHIVE"
    return
  fi
  if [[ -f "$DATA_ARCHIVE" ]]; then echo "Using $DATASET archive: $DATA_ARCHIVE"; return; fi
  if ! is_true "$COPY_ARCHIVE_TO_TMPDIR"; then require_path "$DATA_ARCHIVE" "$DATASET archive"; return; fi
  require_path "$SOURCE_ARCHIVE" "$DATASET source archive"
  mkdir -p "$(dirname "$DATA_ARCHIVE")"
  cp -f "$SOURCE_ARCHIVE" "$DATA_ARCHIVE"
}

extract_tar_if_needed() {
  require_path "$DATA_ARCHIVE" "$DATASET archive"
  mkdir -p "$STAGING_ROOT"
  if is_true "$FORCE_EXTRACT"; then rm -rf "$STAGING_ROOT/$EXPECTED_DIR"; fi
  if [[ -d "$STAGING_ROOT/$EXPECTED_DIR" ]]; then echo "Using existing extracted $DATASET data under: $STAGING_ROOT/$EXPECTED_DIR"; return; fi
  tar -xf "$DATA_ARCHIVE" -C "$STAGING_ROOT"
}

if [[ -z "$DATA_ROOT" ]]; then DATA_ROOT=$STAGING_ROOT/$EXPECTED_DIR; fi
if [[ ! -d "$DATA_ROOT" ]] || is_true "$FORCE_EXTRACT"; then stage_archive_if_needed; extract_tar_if_needed; fi

PERSISTENT_OUT_DIR=$OUT_DIR
if [[ "$PERSISTENT_OUT_DIR" != /* ]]; then PERSISTENT_OUT_DIR="$ROOT/$PERSISTENT_OUT_DIR"; fi
RESUME_STATE_DIR=${RESUME_STATE_DIR:-$PERSISTENT_OUT_DIR/.resume_v1}
RUN_OUT_DIR=$OUT_DIR
STAGING_DIR=
if is_true "$STAGE_QUALITATIVE_IN_TMPDIR" && ! is_true "$DRY_RUN"; then
  OUT_BASENAME=$(basename -- "$PERSISTENT_OUT_DIR")
  STAGING_DIR="$JOB_STAGING_DIR"
  RUN_OUT_DIR="$STAGING_DIR/$OUT_BASENAME"
  mkdir -p "$RUN_OUT_DIR" "$PERSISTENT_OUT_DIR"
fi

eval_args=(
  evaluation/surface_normals/inference_partial_flux2_klein_img2img.py
  --dataset "$DATASET"
  --model "$MODEL"
  --data_root "$DATA_ROOT"
  --split "$SPLIT"
  --out_dir "$RUN_OUT_DIR"
  --steps "$STEPS"
  --guidance_scale "$GUIDANCE_SCALE"
  --min_normal_norm "$MIN_NORMAL_NORM"
  --min_valid_pixels "$MIN_VALID_PIXELS"
  --eval_mask_erosion_radius "$EVAL_MASK_EROSION_RADIUS"
  --diagnostic_mask_erosion_radius "$DIAGNOSTIC_MASK_EROSION_RADIUS"
  --angle_vis_max "$ANGLE_VIS_MAX"
  --prediction_normal_encoding "$PREDICTION_NORMAL_ENCODING"
  --gt_normal_encoding "$GT_NORMAL_ENCODING"
  --protocol_version "$EVAL_PROTOCOL_VERSION"
  --dataset_role "$DATASET_ROLE"
  --cases
)
if [[ -n "$GT_NORMAL_PERMUTATION" ]]; then eval_args+=("--gt_normal_permutation=$GT_NORMAL_PERMUTATION"); fi
if [[ -n "$GT_NORMAL_SIGNS" ]]; then eval_args+=("--gt_normal_signs=$GT_NORMAL_SIGNS"); fi
# shellcheck disable=SC2206
case_parts=($CASES)
eval_args+=("${case_parts[@]}" --aux-augmentations)
# shellcheck disable=SC2206
aux_parts=($AUX_AUGMENTATIONS)
eval_args+=("${aux_parts[@]}")

eval_args+=(--model_family "$MODEL_FAMILY")
if [[ -n "$TRANSFORMER_PATH" ]]; then eval_args+=(--transformer_path "$TRANSFORMER_PATH"); fi
if [[ -n "$LORA_PATH" ]]; then eval_args+=(--lora_path "$LORA_PATH"); fi
if [[ "$MODEL_OFFLOAD" == "1" ]]; then eval_args+=(--model_offload); fi
if [[ -n "$PROMPT" ]]; then eval_args+=(--prompt "$PROMPT"); fi
if [[ -n "$LIMIT" ]]; then eval_args+=(--limit "$LIMIT"); fi
if [[ -n "$SAMPLE_MANIFEST" ]]; then eval_args+=(--sample_manifest "$SAMPLE_MANIFEST"); fi
if [[ -n "$PRIMARY_CASE" ]]; then eval_args+=(--primary_case "$PRIMARY_CASE"); fi
if [[ "$GEOMETRY_SEED_MODE" == legacy_unseeded ]]; then
  eval_args+=(--legacy_unseeded)
else
  eval_args+=(--seed "$SEED")
fi
if is_true "$DRY_RUN"; then eval_args+=(--dry_run); fi
if is_true "$NATIVE_RESOLUTION"; then eval_args+=(--native_resolution); fi
if is_true "$BIDIRECTIONAL_NORMALS"; then eval_args+=(--bidirectional_normals); fi
if is_true "$REPORT_BIDIRECTIONAL_DIAGNOSTIC"; then eval_args+=(--report_bidirectional_diagnostic); fi
if is_true "$DURABLE_EVAL_RESUME" && ! is_true "$DRY_RUN"; then
  if [[ -z "${RESUME_DATA_IDENTITY:-}" ]]; then
    if [[ -f "$SOURCE_ARCHIVE" ]]; then
      RESUME_DATA_IDENTITY=$(stat -Lc '%n|%s|%Y' "$SOURCE_ARCHIVE")
    else
      RESUME_DATA_IDENTITY=$(realpath "$DATA_ROOT")
    fi
  fi
  eval_args+=(
    --resume_state_dir "$RESUME_STATE_DIR"
    --resume_chunk_size "$RESUME_CHUNK_SIZE"
    --resume_data_identity "$RESUME_DATA_IDENTITY"
  )
fi
if ! [[ "$EVAL_NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EVAL_NUM_GPUS must be a positive integer, got: $EVAL_NUM_GPUS" >&2
  exit 2
fi
if (( EVAL_NUM_GPUS > 1 )); then
  [[ "$GEOMETRY_SEED_MODE" != legacy_unseeded ]] || {
    echo "ERROR: multi-GPU evaluation requires deterministic per-sample seeds." >&2
    exit 2
  }
  mkdir -p "$EVAL_SHARD_SYNC_DIR"
  eval_args+=(
    --num_shards "$EVAL_NUM_GPUS"
    --shard_sync_dir "$EVAL_SHARD_SYNC_DIR"
    --shard_sync_timeout_seconds "$EVAL_SHARD_SYNC_TIMEOUT_SECONDS"
  )
fi

echo "Running $DATASET surface-normal evaluation"
echo "  data root:        $DATA_ROOT"
echo "  output dir:       $PERSISTENT_OUT_DIR"
echo "  cases:            $CASES"
echo "  aux augmentations:$AUX_AUGMENTATIONS"
echo "  protocol:         $EVAL_PROTOCOL_VERSION ($DATASET_ROLE)"
echo "  generation seed:  $SEED ($GEOMETRY_SEED_MODE)"
echo "  evaluator GPUs:   $EVAL_NUM_GPUS"
echo "  prediction encoding: $PREDICTION_NORMAL_ENCODING"
echo "  GT to canonical:  encoding=$GT_NORMAL_ENCODING perm=${GT_NORMAL_PERMUTATION:-auto} signs=${GT_NORMAL_SIGNS:-auto}"
if [[ -n "$SAMPLE_MANIFEST" ]]; then echo "  sample manifest:  $SAMPLE_MANIFEST"; fi
if [[ -n "$PRIMARY_CASE" ]]; then echo "  primary case:     $PRIMARY_CASE"; fi
if [[ "$DATASET" == "ibims1" ]]; then echo "  iBims protocol:   $IBIMS1_NORMALS_PROTOCOL"; fi

OVB_CONTAINER_GPU=${OVB_CONTAINER_GPU:-1}
source "$ROOT/scripts/lib/runtime.sh"
run_evaluation() {
  if (( EVAL_NUM_GPUS == 1 )); then
    ovb_run_python "${eval_args[@]}"
    return
  fi
  local -a pids=()
  local gpu_index status remaining
  for ((gpu_index=0; gpu_index<EVAL_NUM_GPUS; gpu_index++)); do
    (CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONUNBUFFERED=1 \
      ovb_run_python "${eval_args[@]}" --shard_index "$gpu_index") &
    pids+=("$!")
    if [[ "$EVAL_GPU_LAUNCH_STAGGER_SECONDS" != 0 && "$gpu_index" -lt $((EVAL_NUM_GPUS - 1)) ]]; then
      sleep "$EVAL_GPU_LAUNCH_STAGGER_SECONDS"
    fi
  done
  remaining=${#pids[@]}
  while (( remaining > 0 )); do
    if wait -n; then
      :
    else
      status=$?
      kill "${pids[@]}" 2>/dev/null || true
      wait 2>/dev/null || true
      return "$status"
    fi
    remaining=$((remaining - 1))
  done
}
run_evaluation
if (( EVAL_NUM_GPUS > 1 )); then rm -rf "$EVAL_SHARD_SYNC_DIR"; fi

if [[ -n "$STAGING_DIR" ]]; then
  mkdir -p "$PERSISTENT_OUT_DIR"
  find "$RUN_OUT_DIR" -maxdepth 1 -type f -exec cp -f -t "$PERSISTENT_OUT_DIR" {} +
  prediction_dirs=()
  for prediction_dir in pred_rgb_raw pred_normals_encoded angular_error_gray; do
    if [[ -d "$RUN_OUT_DIR/$prediction_dir" ]]; then prediction_dirs+=("$prediction_dir"); fi
  done
  if (( ${#prediction_dirs[@]} > 0 )); then
    final_archive_tmp="$PERSISTENT_OUT_DIR/.qualitative_predictions.tar.gz.tmp-$JOB_ID"
    tar -C "$RUN_OUT_DIR" -czf "$final_archive_tmp" "${prediction_dirs[@]}"
    mv -f "$final_archive_tmp" "$PERSISTENT_OUT_DIR/qualitative_predictions.tar.gz"
  fi
  if ! is_true "$KEEP_TMPDIR_OUTPUTS"; then rm -rf "$STAGING_DIR"; fi
fi
if ! is_true "$DRY_RUN"; then
  completion_tmp="$PERSISTENT_OUT_DIR/.EVALUATION_COMPLETE.tmp-$JOB_ID"
  printf 'completed_at=%s\njob_id=%s\n' "$(date --iso-8601=seconds)" "$JOB_ID" > "$completion_tmp"
  mv -f "$completion_tmp" "$PERSISTENT_OUT_DIR/EVALUATION_COMPLETE"
  if is_true "$DURABLE_EVAL_RESUME"; then rm -rf "$RESUME_STATE_DIR"; fi
fi
