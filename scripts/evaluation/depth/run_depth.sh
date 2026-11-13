#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Run depth evaluation through the geometry-v5 protocol. The supported datasets
are NYUv2, iBims-1, DIODE, KITTI depth-selection, KITTI Eigen, ETH3D,
nuScenes, Virtual KITTI 2, and ScanNet.

Usage:
  DATASET=nyu SOURCE_ARCHIVE=/path/to/nyu.tar scripts/evaluation/depth/run_depth.sh

Dataset controls:
  DATASET                  nyu, ibims1, diode, kitti, kitti_eigen, eth3d,
                           nuscenes, vkitti2, or scannet. Default: nyu.
  ARCHIVE_ROOT             Root containing depth/ and surface_normals/ prepared tars.
  DATA_ROOT                Extracted prepared dataset root.
  SOURCE_ARCHIVE           Persistent tar archive copied/extracted to TMPDIR.
  EXPECTED_DIR             Top-level folder expected after tar extraction.
  CASES                    Space-separated cases. KITTI: depth_selection. ETH3D: low_res_two_view. nuScenes: cam_front. VKITTI2: clone_camera_0.
  AUX_AUGMENTATIONS        Space-separated iBims aux groups. Default: appearance.
  IBIMS1_DEPTH_PROTOCOL    nyu_10m or ibims_official. Default: ibims_official.
                           ibims_official uses depth/ibims1_official.tar by default.

Other controls match run_nyu_flux2_klein.sh.
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

if [[ -d "$LOCAL_MODEL" && "${MODEL}" == "${BASE_MODEL}" ]]; then
  BASE_MODEL=$LOCAL_MODEL
  MODEL=$BASE_MODEL
fi
if [[ -z "$TRANSFORMER_PATH" && -d "$MODEL/transformer" && ! -f "$MODEL/model_index.json" ]]; then
  TRANSFORMER_PATH=$MODEL
  MODEL=$BASE_MODEL
fi

OUT_DIR=${OUT_DIR:-outputs/${DATASET}_depth_base_flux2}
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
IBIMS1_DEPTH_PROTOCOL=${IBIMS1_DEPTH_PROTOCOL:-ibims_official}
STEPS=${STEPS:-28}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
SEED=${SEED:-0}
GEOMETRY_SEED_MODE=${GEOMETRY_SEED_MODE:-fixed_per_sample_reset}
DEPTH_ENCODING=${DEPTH_ENCODING:-vision_banana_rgb}
EVAL_PROTOCOL_VERSION=${EVAL_PROTOCOL_VERSION:-geometry-eval-v5}
DATASET_ROLE=${DATASET_ROLE:-}
PRIMARY_CASE=${PRIMARY_CASE:-}
PROMPT=${PROMPT:-}
HELDOUT_DEPTH_CALIBRATION_RECORD=${HELDOUT_DEPTH_CALIBRATION_RECORD:-}
DEPTH_CALIBRATION_CANDIDATES_ONLY=${DEPTH_CALIBRATION_CANDIDATES_ONLY:-0}
DEPTH_CALIBRATION_RECORD_OUTPUT=${DEPTH_CALIBRATION_RECORD_OUTPUT:-}
LIMIT=${LIMIT:-}
SAMPLE_MANIFEST=${SAMPLE_MANIFEST:-}
SAMPLE_MANIFEST_ROOT=${SAMPLE_MANIFEST_ROOT:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/manifests}}
DRY_RUN=${DRY_RUN:-0}
NATIVE_RESOLUTION=${NATIVE_RESOLUTION:-0}
ALIGNMENT_MODE=${ALIGNMENT_MODE:-auto}
MAX_SHIFT_PX=${MAX_SHIFT_PX:-8}
SHIFT_STEP=${SHIFT_STEP:-4}
MIN_OVERLAP_PIXELS=${MIN_OVERLAP_PIXELS:-4096}
MIN_DEPTH=${MIN_DEPTH:-0.001}
if [[ -z "${MAX_DEPTH+x}" ]]; then
  if [[ "$DATASET" == "diode" ]]; then
    MAX_DEPTH=300.0
  elif [[ "$DATASET" == "kitti" || "$DATASET" == "kitti_eigen" ]]; then
    MAX_DEPTH=80.0
  elif [[ "$DATASET" == "eth3d" ]]; then
    MAX_DEPTH=80.0
  elif [[ "$DATASET" == "nuscenes" ]]; then
    MAX_DEPTH=80.0
  elif [[ "$DATASET" == "vkitti2" ]]; then
    MAX_DEPTH=80.0
  elif [[ "$DATASET" == "ibims1" && "$IBIMS1_DEPTH_PROTOCOL" == "ibims_official" ]]; then
    MAX_DEPTH=10.0
  else
    MAX_DEPTH=10.0
  fi
fi

TMPDIR=${TMPDIR:-/tmp/${USER:-$(id -un)}/open_visionbanana}
mkdir -p "$TMPDIR"
TMP_ROOT=${QUALITATIVE_TMPDIR:-${TMPDIR:-/tmp}}
JOB_ID=${SLURM_JOB_ID:-manual-$$}
JOB_STAGING_DIR="$TMP_ROOT/open-vision-banana-depth-$DATASET-$JOB_ID"
EVAL_SHARD_SYNC_DIR=${EVAL_SHARD_SYNC_DIR:-$TMPDIR/open-vb-depth-shards-$DATASET-$JOB_ID}
STAGING_ROOT=${STAGING_ROOT:-$TMPDIR/open_visionbanana_depth_eval_$DATASET}
COPY_ARCHIVE_TO_TMPDIR=${COPY_ARCHIVE_TO_TMPDIR:-1}
FORCE_EXTRACT=${FORCE_EXTRACT:-0}

case "$DATASET" in
  nyu)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/depth/nyu.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-nyu}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/nyu.tar}
    ;;
  ibims1)
    if [[ "$IBIMS1_DEPTH_PROTOCOL" == "ibims_official" ]]; then
      SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/depth/ibims1_official.tar}}
      EXPECTED_DIR=${EXPECTED_DIR:-ibims1_depth_official}
      TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/ibims1_depth_official.tar}
    else
      SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/depth/ibims1.tar}}
      EXPECTED_DIR=${EXPECTED_DIR:-ibims1_depth}
      TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/ibims1_depth.tar}
    fi
    ;;
  diode)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/depth/diode.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-diode_depth}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/diode_depth.tar}
    ;;
  kitti)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/depth/kitti.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-kitti_depth_selection}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/kitti_depth_selection.tar}
    ;;
  kitti_eigen)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/depth/kitti_eigen.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-2011_09_26}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/kitti_eigen.tar}
    ;;
  eth3d)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/depth/eth3d.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-eth3d_low_res_two_view_depth}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/eth3d_low_res_two_view_depth.tar}
    ;;
  nuscenes)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/depth/nuscenes.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-nuscenes_trainval_val_cam_front_depth}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/nuscenes_trainval_val_cam_front_depth.tar}
    ;;
  vkitti2)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/depth/vkitti2.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-vkitti2_clone_camera_0_depth}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/vkitti2_clone_camera_0_depth.tar}
    ;;
  scannet)
    SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-${ARCHIVE_ROOT:+$ARCHIVE_ROOT/depth/scannet.tar}}
    EXPECTED_DIR=${EXPECTED_DIR:-scannet_depth_normals}
    TMP_ARCHIVE=${TMP_ARCHIVE:-$TMPDIR/scannet.tar}
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
    nyu|ibims1|eth3d|diode|kitti|nuscenes) DATASET_ROLE=vision_banana_primary ;;
    vkitti2|scannet) DATASET_ROLE=extended_diagnostic ;;
    kitti_eigen) DATASET_ROLE=affine_diagnostic_only ;;
  esac
fi
if [[ -z "$PRIMARY_CASE" && "$DATASET" == diode ]]; then PRIMARY_CASE=indoor; fi
if [[ "$DATASET" == nuscenes && "$MIN_OVERLAP_PIXELS" == 4096 ]]; then
  MIN_OVERLAP_PIXELS=10
fi

stage_archive_if_needed() {
  if is_true "$FORCE_EXTRACT" && is_true "$COPY_ARCHIVE_TO_TMPDIR" && [[ -f "$SOURCE_ARCHIVE" ]]; then
    mkdir -p "$(dirname "$DATA_ARCHIVE")"
    echo "Refreshing $DATASET archive cache: $DATA_ARCHIVE"
    cp -f "$SOURCE_ARCHIVE" "$DATA_ARCHIVE"
    return
  fi
  if [[ -f "$DATA_ARCHIVE" ]]; then
    echo "Using $DATASET archive: $DATA_ARCHIVE"
    return
  fi
  if ! is_true "$COPY_ARCHIVE_TO_TMPDIR"; then
    require_path "$DATA_ARCHIVE" "$DATASET archive"
    return
  fi
  require_path "$SOURCE_ARCHIVE" "$DATASET source archive"
  mkdir -p "$(dirname "$DATA_ARCHIVE")"
  echo "Copying $SOURCE_ARCHIVE to $DATA_ARCHIVE"
  cp -f "$SOURCE_ARCHIVE" "$DATA_ARCHIVE"
}

extract_tar_if_needed() {
  require_path "$DATA_ARCHIVE" "$DATASET archive"
  mkdir -p "$STAGING_ROOT"
  if is_true "$FORCE_EXTRACT"; then
    rm -rf "$STAGING_ROOT/$EXPECTED_DIR"
  fi
  if [[ -d "$STAGING_ROOT/$EXPECTED_DIR" ]]; then
    echo "Using existing extracted $DATASET data under: $STAGING_ROOT/$EXPECTED_DIR"
    return
  fi
  echo "Extracting $DATA_ARCHIVE into $STAGING_ROOT"
  tar -xf "$DATA_ARCHIVE" -C "$STAGING_ROOT"
}

if [[ -z "$DATA_ROOT" ]]; then
  if [[ "$DATASET" == kitti_eigen ]]; then
    # Marigold stores projected GT drives beside date-grouped RGB directories.
    # The date folder is an extraction sentinel, not the evaluator root.
    DATA_ROOT=$STAGING_ROOT
  else
    DATA_ROOT=$STAGING_ROOT/$EXPECTED_DIR
  fi
fi
DATA_SENTINEL=$DATA_ROOT
if [[ "$DATASET" == kitti_eigen ]]; then
  DATA_SENTINEL=$DATA_ROOT/$EXPECTED_DIR
fi
if [[ ! -d "$DATA_SENTINEL" ]] || is_true "$FORCE_EXTRACT"; then
  stage_archive_if_needed
  extract_tar_if_needed
fi

PERSISTENT_OUT_DIR=$OUT_DIR
if [[ "$PERSISTENT_OUT_DIR" != /* ]]; then
  PERSISTENT_OUT_DIR="$ROOT/$PERSISTENT_OUT_DIR"
fi
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
  evaluation/depth/inference_partial_flux2_klein_img2img.py
  --dataset "$DATASET"
  --model "$MODEL"
  --data_root "$DATA_ROOT"
  --split "$SPLIT"
  --out_dir "$RUN_OUT_DIR"
  --steps "$STEPS"
  --guidance_scale "$GUIDANCE_SCALE"
  --depth_encoding "$DEPTH_ENCODING"
  --alignment_mode "$ALIGNMENT_MODE"
  --max_shift_px "$MAX_SHIFT_PX"
  --shift_step "$SHIFT_STEP"
  --min_overlap_pixels "$MIN_OVERLAP_PIXELS"
  --min_depth "$MIN_DEPTH"
  --max_depth "$MAX_DEPTH"
  --protocol_version "$EVAL_PROTOCOL_VERSION"
  --dataset_role "$DATASET_ROLE"
  --cases
)
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
if [[ -n "$HELDOUT_DEPTH_CALIBRATION_RECORD" ]]; then
  eval_args+=(--heldout_depth_calibration_record "$HELDOUT_DEPTH_CALIBRATION_RECORD")
fi
if is_true "$DEPTH_CALIBRATION_CANDIDATES_ONLY"; then
  [[ -z "$HELDOUT_DEPTH_CALIBRATION_RECORD" ]] || {
    echo "ERROR: calibration-candidate mode cannot also load a calibration record." >&2
    exit 2
  }
  eval_args+=(--depth_calibration_candidates_only)
fi
if [[ -n "$DEPTH_CALIBRATION_RECORD_OUTPUT" ]] && ! is_true "$DEPTH_CALIBRATION_CANDIDATES_ONLY"; then
  echo "ERROR: DEPTH_CALIBRATION_RECORD_OUTPUT requires DEPTH_CALIBRATION_CANDIDATES_ONLY=1." >&2
  exit 2
fi
if [[ "$GEOMETRY_SEED_MODE" == legacy_unseeded ]]; then
  eval_args+=(--legacy_unseeded)
else
  eval_args+=(--seed "$SEED")
fi
if is_true "$DRY_RUN"; then eval_args+=(--dry_run); fi
if is_true "$NATIVE_RESOLUTION"; then eval_args+=(--native_resolution); fi
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

echo "Running $DATASET depth evaluation"
echo "  data root:        $DATA_ROOT"
echo "  output dir:       $PERSISTENT_OUT_DIR"
echo "  cases:            $CASES"
echo "  aux augmentations:$AUX_AUGMENTATIONS"
echo "  protocol:         $EVAL_PROTOCOL_VERSION ($DATASET_ROLE)"
echo "  generation seed:  $SEED ($GEOMETRY_SEED_MODE)"
echo "  evaluator GPUs:   $EVAL_NUM_GPUS"
if [[ -n "$SAMPLE_MANIFEST" ]]; then echo "  sample manifest:  $SAMPLE_MANIFEST"; fi
if [[ -n "$PRIMARY_CASE" ]]; then echo "  primary case:     $PRIMARY_CASE"; fi
if is_true "$DEPTH_CALIBRATION_CANDIDATES_ONLY"; then
  echo "  mode:             held-out affine calibration candidates"
fi
if [[ -n "$HELDOUT_DEPTH_CALIBRATION_RECORD" ]]; then
  echo "  calibration record:$HELDOUT_DEPTH_CALIBRATION_RECORD"
fi
if is_true "$DURABLE_EVAL_RESUME" && ! is_true "$DRY_RUN"; then
  echo "  durable resume:   $RESUME_STATE_DIR"
fi
if [[ "$DATASET" == "ibims1" ]]; then echo "  iBims protocol:   $IBIMS1_DEPTH_PROTOCOL"; fi

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
  for prediction_dir in pred_rgb_raw pred_depth_gray pred_depth_direct_mm pred_depth_direct_vis pred_vb_rgb_path_distance pred_depth_aligned_mm pred_depth_aligned_vis pred_depth_aligned_noshift_mm pred_depth_aligned_noshift_vis; do
    if [[ -d "$RUN_OUT_DIR/$prediction_dir" ]]; then prediction_dirs+=("$prediction_dir"); fi
  done
  if (( ${#prediction_dirs[@]} > 0 )); then
    final_archive_tmp="$PERSISTENT_OUT_DIR/.qualitative_predictions.tar.gz.tmp-$JOB_ID"
    tar -C "$RUN_OUT_DIR" -czf "$final_archive_tmp" "${prediction_dirs[@]}"
    mv -f "$final_archive_tmp" "$PERSISTENT_OUT_DIR/qualitative_predictions.tar.gz"
  fi
  if ! is_true "$KEEP_TMPDIR_OUTPUTS"; then rm -rf "$STAGING_DIR"; fi
fi
if [[ -n "$DEPTH_CALIBRATION_RECORD_OUTPUT" ]]; then
  split_name=$($PYTHON - "$DATASET" "$SPLIT" <<'PY'
import sys
from evaluation.depth import datasets
print(datasets.canonicalize_split_name(sys.argv[1], sys.argv[2]))
PY
)
  if [[ "$DEPTH_CALIBRATION_RECORD_OUTPUT" != /* ]]; then
    DEPTH_CALIBRATION_RECORD_OUTPUT="$ROOT/$DEPTH_CALIBRATION_RECORD_OUTPUT"
  fi
  "$PYTHON" "$ROOT/scripts/evaluation/depth/write_heldout_depth_calibration_record.py" \
    --results "$PERSISTENT_OUT_DIR/${DATASET}_${split_name}_results.json" \
    --per-sample "$PERSISTENT_OUT_DIR/${DATASET}_${split_name}_per_sample.jsonl" \
    --output "$DEPTH_CALIBRATION_RECORD_OUTPUT"
fi
if ! is_true "$DRY_RUN"; then
  completion_tmp="$PERSISTENT_OUT_DIR/.EVALUATION_COMPLETE.tmp-$JOB_ID"
  printf 'completed_at=%s\njob_id=%s\n' "$(date --iso-8601=seconds)" "$JOB_ID" > "$completion_tmp"
  mv -f "$completion_tmp" "$PERSISTENT_OUT_DIR/EVALUATION_COMPLETE"
  if is_true "$DURABLE_EVAL_RESUME"; then rm -rf "$RESUME_STATE_DIR"; fi
fi
