#!/usr/bin/env bash
set -euo pipefail

# Run SA-Co/Gold instance-segmentation evaluation with the base Flux2-Klein model
# or a fine-tuned Open Vision Banana transformer. Mirrors
# scripts/evaluation/semseg/run_cityscapes_flux2_klein.sh, including its tar-staging
# flow: the canonical data source is SA-Co-Gold.tar, extracted once into
# SACO_STAGE_DIR (see stage_saco_gold.sh) instead of reading a pre-extracted folder.
#
# Important environment variables:
#   ROOT               Repo root. Defaults to the current checkout.
#   SIF                Singularity image (used only when USE_CONTAINER!=0).
#   MODEL              Base model path or HF id.
#   TRANSFORMER_PATH   Optional fine-tuned output/checkpoint/transformer dir.
#   OUT_DIR            Evaluation output directory.
#   SACO_ROOT          SA-Co-Gold root (sa1b-images/, metaclip-images/, gt-annotations/).
#                      If it lacks gt-annotations/, SACO_TAR is staged into
#                      SACO_STAGE_DIR and SACO_ROOT is re-pointed at the extract.
#   SACO_TAR           SA-Co-Gold archive (default datasets_archives/instanceseg/SA-Co-Gold.tar).
#   SACO_STAGE_DIR     Where SACO_TAR is extracted -- node-local disk (default under
#                      $TMPDIR) so the eval reads images from fast local storage.
#   SUBSETS            Space-separated subset list (default: all 7).
#   LIMIT              Cap queries per subset (seeded sample; empty = full set).
#   SAMPLE_SEED        Seed for the per-subset LIMIT sample (default 0).
#   MAX_INSTANCES      Only process pairs with <= N GT instances (empty = no limit).
#   GATE               file | oracle | none  (default file).
#   GATE_FILE          gate_decisions.json (for GATE=file). If GATE=file and this
#                      file is missing, it is auto-generated with Qwen3-VL first.
#   REQUIRE_COMPLETE_GATE  1 to fail if any selected query lacks a gate decision.
#   QWEN_MODEL         Qwen3-VL id/path used when auto-generating a missing gate.
#   GATE_BACKEND       Auto-gen backend when GATE=file & file missing (qwen|oracle).
#   STEPS              Diffusion steps (default 28).
#   ISEG_BATCH_SIZE    Per-GPU one-image-per-prompt FLUX batch size (default 1).
#   GUIDANCE_SCALE     Guidance scale (default 3.5).
#   PROMPT_SET         legend | ordered | legacy | a fixed-K draw mode
#                      (fixeddraw_32, randomdraw_32, fixeddraw_32_lr,
#                      randomdraw_32_unlisted,
#                      fixeddraw_32_ordered_unlisted,
#                      fixeddraw_32_mixedrepresent, fixeddraw_64, randomdraw_64)
#                      -- must match the wording the checkpoint was trained on
#                      (empty -> evaluator default "fixeddraw_32_lr").
#   PROMPT_TEMPLATE_INDEX  Force canonical base template 0, 1, 2, or 3 for a
#                      fixed-K draw mode, while retaining its color legend and
#                      ordering. Intended for controlled prompt ablations.
#   LEGEND_LEN         PROMPT_SET=legend: constant convention entries per prompt
#                      (empty -> evaluator default 10).
#   NOTATION           PROMPT_SET=legend: mix (default) | name | rgb.
#   NATIVE_RESOLUTION  1 to generate at input resolution.
#   INSTANCE_SCORE     Constant per-instance confidence (default 1.0).
#   TAU/GAMMA/THETA_SIZE/THETA_EROSION Decoder defaults: 58 / 1 / 8e-4 / 0.1.
#   SNAPPING/GAPFILL/GRABCUT  Independent 1/0 switches (all default 1).
#   GAPFILL_RADIUS/GAPFILL_THRESHOLD  Defaults 3 / 58, using unsnapped RGB.
#   REFINEMENT_ORDER   gapfill_grabcut (default) | grabcut_gapfill.
#   GRABCUT_ITERATIONS/BAND_MODE/RADIUS  Defaults 2 / fixed / 3.
#   GRABCUT_BAND_SCALE/MIN_RADIUS       Adaptive-band defaults 0.05 / 3.
#   GRABCUT_RING_SCALE/CROP_MARGIN/MIN_AREA_RATIO Defaults 1 / 5 / 0.9.
#   POSTPROCESS_ONLY/RAW_RGB_ROOT  CPU-only saved-RGB refinement into a new OUT_DIR.
#   RANDOMDRAW_REPEATS N palette-assignment generations for explicit randomdraw
#                      modes, or fixed draw modes with RANDOMIZE_PALETTE=1.
#   GENERATION_SEED_REPEATS  N diffusion seeds per palette (requires GENERATION_SEED).
#                      Members use inclusive-majority instance consensus by default.
#   ISEG_CONSENSUS_MODE inclusive_majority (default: 50/50 ties keep foreground)
#                      or strict_majority (requires more than half the votes).
#   REPEAT_ANALYSIS    1 (default): keep per-repeat masks; CPU merge scores every
#                      standalone repeat and prefix, and renders sparse diagnostics.
#   RUN_EVAL           1 to score with CGF1 after dumping predictions (default 1).
#   IOU_TYPE           segm (default) | bbox.
#   SAVE_RGB/SAVE_PANELS  1 to save generated images / qualitative panels.
#   RESUME_FROM_RGB    1 to replay verified saved RGBs and generate missing batches.
#   ARTIFACT_STAGE_DIR  Optional node-local PNG spool.
#   ARTIFACT_SYNC_DIR/SLOTS  Shared copy semaphore; default two concurrent writers.
#   LOG_BATCH_TIMING   1 to separate GPU generation time from decode and I/O time.
#   PANEL_EVERY       With SAVE_PANELS, retain the first and every Nth
#                     gate-positive query within each subset across all shards.
#   NUM_PROCESSES      GPUs for single-job torchrun sharding (default 1).
#   NUM_SHARDS/SHARD_INDEX  Cross-job array sharding (overrides NUM_PROCESSES).
#   DRY_RUN            1 to resolve samples + gate and exit before model load.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

is_true() { case "${1,,}" in 1|true|yes|y|on) return 0 ;; *) return 1 ;; esac }

# shellcheck source=scripts/evaluation/instanceseg/stage_saco_gold.sh
. "$SCRIPT_DIR/stage_saco_gold.sh"

ROOT=${ROOT:-$DEFAULT_ROOT}
SIF=${SIF:-}   # container image; the host environment is used when unset
PYTHON=${PYTHON:-/opt/conda/envs/open-vision-banana/bin/python}

LOCAL_MODEL=${LOCAL_MODEL:-}
BASE_MODEL=${BASE_MODEL:-black-forest-labs/FLUX.2-klein-base-9B}
MODEL=${MODEL:-$BASE_MODEL}
TRANSFORMER_PATH=${TRANSFORMER_PATH:-}

OUT_DIR=${OUT_DIR:-outputs/sacogold_base_flux2}
SACO_ROOT=${SACO_ROOT:-$ROOT/datasets_archives/instanceseg/SA-Co-Gold}
SACO_TAR=${SACO_TAR:-$ROOT/datasets_archives/instanceseg/SA-Co-Gold.tar}
SACO_STAGE_DIR=${SACO_STAGE_DIR:-${TMPDIR:-/tmp}/open-vision-banana-instanceseg-${SLURM_JOB_ID:-manual-$$}/data}
SUBSETS=${SUBSETS:-}
LIMIT=${LIMIT:-}
SAMPLE_SEED=${SAMPLE_SEED:-0}
MAX_INSTANCES=${MAX_INSTANCES:-}
GATE=${GATE:-file}
GATE_FILE=${GATE_FILE:-}
REQUIRE_COMPLETE_GATE=${REQUIRE_COMPLETE_GATE:-0}
QWEN_MODEL=${QWEN_MODEL:-}
GATE_BACKEND=${GATE_BACKEND:-qwen}
STEPS=${STEPS:-28}
ISEG_BATCH_SIZE=${ISEG_BATCH_SIZE:-1}
case "$ISEG_BATCH_SIZE" in
  ""|*[!0-9]*|0)
    echo "ERROR: ISEG_BATCH_SIZE must be a positive integer, got $ISEG_BATCH_SIZE" >&2
    exit 1
    ;;
esac
GUIDANCE_SCALE=${GUIDANCE_SCALE:-3.5}
GENERATION_SEED=${GENERATION_SEED:-}
GENERATION_SEED_REPEATS=${GENERATION_SEED_REPEATS:-1}
RANDOMDRAW_REPEATS=${RANDOMDRAW_REPEATS:-1}
ISEG_CONSENSUS_MODE=${ISEG_CONSENSUS_MODE:-inclusive_majority}
case "$ISEG_CONSENSUS_MODE" in
  inclusive_majority|strict_majority) ;;
  *) echo "ERROR: ISEG_CONSENSUS_MODE must be inclusive_majority or strict_majority" >&2; exit 1 ;;
esac
REPEAT_INDEX_OFFSET=${REPEAT_INDEX_OFFSET:-0}   # continue a finished sweep at this palette index
RANDOMIZE_PALETTE=${RANDOMIZE_PALETTE:-0}
REPEAT_ANALYSIS=${REPEAT_ANALYSIS:-1}
PROMPT_SET=${PROMPT_SET:-}         # empty -> evaluator default (fixeddraw_32_lr); must match the checkpoint's training prompts
LEGEND_LEN=${LEGEND_LEN:-}         # PROMPT_SET=legend only
NOTATION=${NOTATION:-}             # PROMPT_SET=legend only
PROMPT_TEMPLATE_INDEX=${PROMPT_TEMPLATE_INDEX:-}
if [[ -n "$PROMPT_TEMPLATE_INDEX" && ! "$PROMPT_TEMPLATE_INDEX" =~ ^[0-3]$ ]]; then
  echo "ERROR: PROMPT_TEMPLATE_INDEX must be 0, 1, 2, or 3; got $PROMPT_TEMPLATE_INDEX" >&2
  exit 1
fi
NATIVE_RESOLUTION=${NATIVE_RESOLUTION:-0}
INSTANCE_SCORE=${INSTANCE_SCORE:-1.0}
TAU=${TAU:-}
GAMMA=${GAMMA:-1.0}
THETA_SIZE=${THETA_SIZE:-0.0008}
THETA_EROSION=${THETA_EROSION:-0.1}
SNAPPING=${SNAPPING:-1}
GAPFILL=${GAPFILL:-1}
GRABCUT=${GRABCUT:-1}
GAPFILL_RADIUS=${GAPFILL_RADIUS:-3}
GAPFILL_THRESHOLD=${GAPFILL_THRESHOLD:-58}
REFINEMENT_ORDER=${REFINEMENT_ORDER:-gapfill_grabcut}
GRABCUT_ITERATIONS=${GRABCUT_ITERATIONS:-2}
GRABCUT_BAND_MODE=${GRABCUT_BAND_MODE:-fixed}
GRABCUT_RADIUS=${GRABCUT_RADIUS:-3}
GRABCUT_BAND_SCALE=${GRABCUT_BAND_SCALE:-0.05}
GRABCUT_MIN_RADIUS=${GRABCUT_MIN_RADIUS:-3}
GRABCUT_RING_SCALE=${GRABCUT_RING_SCALE:-1}
GRABCUT_CROP_MARGIN=${GRABCUT_CROP_MARGIN:-5}
GRABCUT_MIN_AREA_RATIO=${GRABCUT_MIN_AREA_RATIO:-0.9}
POSTPROCESS_ONLY=${POSTPROCESS_ONLY:-0}
RAW_RGB_ROOT=${RAW_RGB_ROOT:-}
RUN_EVAL=${RUN_EVAL:-1}
IOU_TYPE=${IOU_TYPE:-segm}
SAVE_RGB=${SAVE_RGB:-0}
RESUME_FROM_RGB=${RESUME_FROM_RGB:-0}
SAVE_PANELS=${SAVE_PANELS:-1}
PANEL_EVERY=${PANEL_EVERY:-}      # with SAVE_PANELS, save 1st + every Nth positive query per subset globally (default 500; 1 = all)
PANEL_EVERY_REPEAT=${PANEL_EVERY_REPEAT:-0}  # 1 to panel every palette draw, not just the run's first
NUM_PROCESSES=${NUM_PROCESSES:-1}
NUM_SHARDS=${NUM_SHARDS:-}
SHARD_INDEX=${SHARD_INDEX:-0}
NO_AGGREGATE=${NO_AGGREGATE:-0}   # shards write rank_*_of_N.json and return (no in-process merge)
MERGE_ONLY=${MERGE_ONLY:-0}       # skip model+generation; merge shard preds + score (CPU-only)
DRY_RUN=${DRY_RUN:-0}

# Resolve a fine-tuned transformer dir handed in as MODEL, and the local snapshot.
if [[ -d "$LOCAL_MODEL" && "${MODEL}" == "${BASE_MODEL}" ]]; then
  BASE_MODEL=$LOCAL_MODEL
  MODEL=$BASE_MODEL
fi
if [[ -z "$TRANSFORMER_PATH" && -d "$MODEL/transformer" ]]; then
  TRANSFORMER_PATH=$MODEL
  MODEL=$BASE_MODEL
fi

# Stage the SA-Co/Gold data from its tar (idempotent; no-op if SACO_ROOT already
# holds the extracted tree). Done before the gate block + eval.
SACO_ROOT=$(stage_saco_gold "$SACO_ROOT" "$SACO_TAR" "$SACO_STAGE_DIR")

run_in_env() {
  # Run a pre-quoted command string in the eval environment (host venv or SIF).
  local cmd="$1" rq
  printf -v rq "%q" "$ROOT"
  if [[ "${USE_CONTAINER:-${SIF:+1}}" != "1" ]]; then
    ( cd "$ROOT" && eval "$cmd" )
  else
    "$SIF" /bin/bash -lc "
set -euo pipefail
unset SSL_CERT_FILE
export SSL_CERT_FILE=/opt/conda/envs/open-vision-banana/lib/python3.10/site-packages/certifi/cacert.pem
cd $rq
$cmd
"
  fi
}

# Build a missing gate file with Qwen3-VL. It covers the requested SUBSETS in
# full (LIMIT / MAX_INSTANCES not applied) so the cache is reusable across
# checkpoints. GATE=oracle needs no file.
if [[ "$GATE" == "file" && -n "$GATE_FILE" && ! -f "$GATE_FILE" ]] && ! is_true "$DRY_RUN" && ! is_true "$MERGE_ONLY" && ! is_true "$POSTPROCESS_ONLY"; then
  echo "Gate file not found: $GATE_FILE"
  echo "Generating presence gate with Qwen3-VL (backend=$GATE_BACKEND) over subsets='${SUBSETS:-<all 7>}'."
  echo "  one-time + reused across checkpoints; for the full 168k set prefer the sharded gate job, or GATE=oracle for smoke runs."
  gate_args=(
    evaluation/instanceseg/gate_saco_gold_qwen.py
    --out "$GATE_FILE"
    --gate "$GATE_BACKEND"
    --saco_root "$SACO_ROOT"
    --sample_seed "$SAMPLE_SEED"
  )
  if [[ -n "$QWEN_MODEL" ]]; then gate_args+=(--qwen_model "$QWEN_MODEL"); fi
  if [[ -n "$SUBSETS" ]]; then gate_args+=(--subsets $SUBSETS); fi
  gate_launch=("$PYTHON" "${gate_args[@]}")
  printf -v gate_cmd "%q " "${gate_launch[@]}"
  run_in_env "$gate_cmd"
fi

eval_args=(
  evaluation/instanceseg/inference_flux2_klein_instanceseg.py
  --model "$MODEL"
  --out_dir "$OUT_DIR"
  --saco_root "$SACO_ROOT"
  --gate "$GATE"
  --steps "$STEPS"
  --guidance_scale "$GUIDANCE_SCALE"
  --batch_size "$ISEG_BATCH_SIZE"
  --instance_score "$INSTANCE_SCORE"
  --sample_seed "$SAMPLE_SEED"
  --randomdraw_repeats "$RANDOMDRAW_REPEATS"
  --generation_seed_repeats "$GENERATION_SEED_REPEATS"
  --consensus_mode "$ISEG_CONSENSUS_MODE"
  --gamma "$GAMMA" --theta_size "$THETA_SIZE" --theta_erosion "$THETA_EROSION"
  --gapfill_radius "$GAPFILL_RADIUS" --gapfill_threshold "$GAPFILL_THRESHOLD"
  --refinement_order "$REFINEMENT_ORDER"
  --grabcut_iterations "$GRABCUT_ITERATIONS" --grabcut_band_mode "$GRABCUT_BAND_MODE"
  --grabcut_radius "$GRABCUT_RADIUS" --grabcut_band_scale "$GRABCUT_BAND_SCALE"
  --grabcut_min_radius "$GRABCUT_MIN_RADIUS" --grabcut_ring_scale "$GRABCUT_RING_SCALE"
  --grabcut_crop_margin "$GRABCUT_CROP_MARGIN" --grabcut_min_area_ratio "$GRABCUT_MIN_AREA_RATIO"
  --iou_type "$IOU_TYPE"
)
for setting in SNAPPING GAPFILL GRABCUT; do
  flag=${setting,,}
  if is_true "${!setting}"; then eval_args+=("--$flag"); else eval_args+=("--no-$flag"); fi
done
if is_true "$POSTPROCESS_ONLY"; then eval_args+=(--postprocess_only); fi
if [[ -n "$RAW_RGB_ROOT" ]]; then eval_args+=(--raw_rgb_root "$RAW_RGB_ROOT"); fi
# Only passed when set.
if [[ "$REPEAT_INDEX_OFFSET" != 0 ]]; then
  eval_args+=(--repeat_index_offset "$REPEAT_INDEX_OFFSET")
fi
if [[ -n "$TAU" ]]; then eval_args+=(--tau "$TAU"); fi
if [[ -n "$GENERATION_SEED" ]]; then eval_args+=(--generation_seed "$GENERATION_SEED"); fi
if is_true "$RANDOMIZE_PALETTE"; then eval_args+=(--randomize_palette); fi
if is_true "$REPEAT_ANALYSIS"; then eval_args+=(--repeat_analysis); else eval_args+=(--no-repeat_analysis); fi
if [[ -n "$TRANSFORMER_PATH" ]]; then eval_args+=(--transformer_path "$TRANSFORMER_PATH"); fi
if [[ -n "$PROMPT_SET" ]]; then eval_args+=(--prompt_set "$PROMPT_SET"); fi
if [[ -n "$LEGEND_LEN" ]]; then eval_args+=(--legend_len "$LEGEND_LEN"); fi
if [[ -n "$NOTATION" ]]; then eval_args+=(--notation "$NOTATION"); fi
if [[ -n "$PROMPT_TEMPLATE_INDEX" ]]; then
  eval_args+=(--prompt_template_index "$PROMPT_TEMPLATE_INDEX")
fi
if [[ -n "$SUBSETS" ]]; then eval_args+=(--subsets $SUBSETS); fi
if [[ -n "$LIMIT" ]]; then eval_args+=(--limit "$LIMIT"); fi
if [[ -n "$MAX_INSTANCES" ]]; then eval_args+=(--max_instances "$MAX_INSTANCES"); fi
if [[ "$GATE" == "file" && -n "$GATE_FILE" ]]; then eval_args+=(--gate_file "$GATE_FILE"); fi
if is_true "$REQUIRE_COMPLETE_GATE"; then eval_args+=(--require_complete_gate); fi
if is_true "$NATIVE_RESOLUTION"; then eval_args+=(--native_resolution); fi
if is_true "$SAVE_RGB"; then eval_args+=(--save_rgb); fi
if is_true "$RESUME_FROM_RGB"; then eval_args+=(--resume_from_rgb); fi
if [[ -n ${ARTIFACT_STAGE_DIR:-} ]] && ! is_true "$MERGE_ONLY"; then
  eval_args+=(--artifact_stage_dir "$ARTIFACT_STAGE_DIR"
              --artifact_sync_slots "${ARTIFACT_SYNC_SLOTS:-2}")
  if [[ -n ${ARTIFACT_SYNC_DIR:-} ]]; then eval_args+=(--artifact_sync_dir "$ARTIFACT_SYNC_DIR"); fi
fi
if is_true "${LOG_BATCH_TIMING:-0}"; then eval_args+=(--log_batch_timing); fi
if is_true "$SAVE_PANELS"; then eval_args+=(--save_panels); fi
if [[ -n "$PANEL_EVERY" ]]; then eval_args+=(--panel_every "$PANEL_EVERY"); fi
if is_true "$PANEL_EVERY_REPEAT"; then eval_args+=(--panels_every_repeat); fi
if is_true "$RUN_EVAL"; then eval_args+=(--run_eval); fi
if is_true "$DRY_RUN"; then eval_args+=(--dry_run); fi
if [[ -n "$NUM_SHARDS" ]]; then eval_args+=(--num_shards "$NUM_SHARDS" --shard_index "$SHARD_INDEX"); fi
if is_true "$NO_AGGREGATE"; then eval_args+=(--no_aggregate); fi
if is_true "$MERGE_ONLY"; then eval_args+=(--merge_only); fi

echo "Running SA-Co/Gold instance-segmentation evaluation"
echo "  repo root:         $ROOT"
echo "  saco root:         $SACO_ROOT    (tar: $SACO_TAR)"
echo "  model:             $MODEL"
echo "  transformer path:  ${TRANSFORMER_PATH:-<base model only>}"
echo "  output dir:        $OUT_DIR"
echo "  subsets:           ${SUBSETS:-<all 7>}    limit: ${LIMIT:-<full>}    max_instances: ${MAX_INSTANCES:-<none>}"
echo "  gate:              $GATE    gate file: ${GATE_FILE:-<none>}"
echo "  steps:             $STEPS    batch: $ISEG_BATCH_SIZE    eval processes: $NUM_PROCESSES    shards: ${NUM_SHARDS:-1}"
echo "  decoder tau/gamma: ${TAU:-58} / $GAMMA    theta_size/erosion: $THETA_SIZE / $THETA_EROSION"
echo "  refinement:        snap=$SNAPPING gapfill=$GAPFILL grabcut=$GRABCUT order=$REFINEMENT_ORDER"
echo "  prompt template:   ${PROMPT_TEMPLATE_INDEX:-per-pair draw-mode selection}"
echo "  randomdraw repeats:$RANDOMDRAW_REPEATS"
echo "  consensus mode:   $ISEG_CONSENSUS_MODE"
echo "  replay saved RGB:  $RESUME_FROM_RGB"
echo "  mode:              $(is_true "$MERGE_ONLY" && echo merge_only || { is_true "$NO_AGGREGATE" && echo shard-only || echo generate+merge; })"

# merge_only is CPU-only (no model) -> plain python, never torchrun. Cross-job array
# sharding also runs one plain process on this task's GPU (no torchrun); torchrun is
# only for single-job intra-node multi-GPU (NUM_PROCESSES>1, no NUM_SHARDS).
if is_true "$MERGE_ONLY" || is_true "$POSTPROCESS_ONLY"; then
  launch_args=("$PYTHON" "${eval_args[@]}")
elif [[ -z "$NUM_SHARDS" ]] && (( NUM_PROCESSES > 1 )); then
  launch_args=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NUM_PROCESSES" "${eval_args[@]}")
else
  launch_args=("$PYTHON" "${eval_args[@]}")
fi

printf -v eval_cmd "%q " "${launch_args[@]}"
run_in_env "$eval_cmd"
