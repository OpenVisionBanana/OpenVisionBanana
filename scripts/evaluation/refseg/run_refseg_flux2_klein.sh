#!/usr/bin/env bash
set -euo pipefail

# Run referring-segmentation evaluation (RefCOCOg UMD val -> cIoU, or
# ReasonSeg val -> gIoU) with the base Flux2-Klein model or a fine-tuned
# Open Vision Banana transformer. Override paths via environment variables.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

show_help() {
  cat <<'EOF'
Run referring-segmentation evaluation for RefCOCOg (cIoU) or ReasonSeg (gIoU).

Usage:
  DATASET=refcocog  scripts/evaluation/refseg/run_refseg_flux2_klein.sh
  DATASET=reasonseg scripts/evaluation/refseg/run_refseg_flux2_klein.sh

Environment variables:
  ROOT             Repo root. Defaults to the current checkout.
  PYTHON           Python interpreter. Defaults to `python`.
  MODEL            Base model path or HF id.
  LOCAL_MODEL      Local FLUX.2-klein snapshot (auto-used as MODEL if present).
                   Defaults to the scratch HF cache snapshot.
  SCRATCH_ROOT     Scratch base for checkpoints + caches. Default: dss scratch.
  CKPT_ROOT        Checkpoint root. Default: $SCRATCH_ROOT/checkpoints/open_visionbanana.
  MODEL_TAG        Training-run tag; resolves $CKPT_ROOT/$MODEL_TAG/$CKPT/transformer.
  CKPT             Checkpoint name. Default: checkpoint-20000.
  TRANSFORMER_PATH Explicit transformer dir (overrides MODEL_TAG; else base model).
  OUT_DIR          Output dir. Default follows the shared eval convention:
                   $SCRATCH_ROOT/eval_outputs/open_visionbanana/$MODEL_TAG/$CKPT/refseg_<dataset>-val_steps<N>.
                   For reasonseg with rewrite ON the folder is
                   refseg_reasonseg-<rewrite_tag>-val_steps<N>, where <rewrite_tag> is
                   taken from the cache filename (reasonseg_rewrites_<tag>.json).

  DATASET          refcocog | reasonseg. Default: refcocog.
  INDEX            refcocog: refcocog_val_umd.json from prepare_refseg_eval.py.
  IMAGES_ROOT      refcocog: override COCO train2014 dir (else index paths).
  PER_OBJECT       refcocog: 1 = one sample per object instead of per sentence.
  DATA_DIR         reasonseg: ReasonSeg val dir (<stem>.jpg/.json + masks/).
  REASONSEG_ALL_EXPR  reasonseg: 1 = one sample per paraphrase (else per image).
  EXPR_INDEX       reasonseg: paraphrase index when one-per-image. Default 0.
  REASONSEG_REWRITE   reasonseg: 1 (default) rewrites implicit queries to explicit
                   referring expressions via the cached Qwen3-VL pass; 0 = raw-query baseline.
  REASONSEG_REWRITE_FILE  reasonseg: cached rewrites JSON (from rewrite_reasonseg_qwen.py).

  PROMPT_TEMPLATE  Prompt; must contain "{expr}". Default: white-on-black mask.
  STEPS            Inference steps. Default 28.
  GUIDANCE_SCALE   Guidance scale. Default 4.0.
  NATIVE_RESOLUTION  1 to generate at input resolution.
  SAVE_MASKS       1 to also save decoded binary masks.
  SAVE_PANELS      1 (default) saves input|ground-truth|output panels with the
                   prompt as a header; 0 to skip (e.g. full 4,896-sample runs).
  LIMIT            Optional sample limit.
  DRY_RUN          1 to resolve samples and exit before model load.
  NUM_PROCESSES    Evaluator processes/GPUs (torchrun sharding). Default 1.
  NUM_SHARDS       Cross-job sharding: total shards. Each runs one process on one
                   GPU over samples[SHARD_INDEX::NUM_SHARDS]. Unlike torchrun this
                   is not bound to a node, so shards can span an allocation.
                   Rank 0 waits for every shard_metrics/ file and merges.
  SHARD_INDEX      This shard's index. Default 0.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

is_true() { case "${1,,}" in 1|true|yes|y|on) return 0 ;; *) return 1 ;; esac; }

ROOT=${ROOT:-$DEFAULT_ROOT}
PYTHON=${PYTHON:-python}

# Checkpoints + HF cache live under SCRATCH_ROOT.
SCRATCH_ROOT=${SCRATCH_ROOT:-${SCRATCH_DLR:-/dss/dsstbyfs02/scratch/02/${USER:-$(id -un)}}}
LOCAL_MODEL=${LOCAL_MODEL:-$SCRATCH_ROOT/cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-9B/snapshots/32773329fbe7e81a90ef971740e8ba4b0364ecf3}
BASE_MODEL=${BASE_MODEL:-black-forest-labs/FLUX.2-klein-base-9B}
MODEL=${MODEL:-$BASE_MODEL}
# Resolve a fine-tuned transformer from the training-run layout
# $CKPT_ROOT/$MODEL_TAG/$CKPT/transformer when MODEL_TAG is set; otherwise use an
# explicit TRANSFORMER_PATH (and fall back to the base model if neither is given).
CKPT_ROOT=${CKPT_ROOT:-$SCRATCH_ROOT/checkpoints/open_visionbanana}
MODEL_TAG=${MODEL_TAG:-ovb_f2k9b_bddacdc-pico-id-cloneverify_r512_d0-s24-out_lr1e-5_b1-ga2-gpu4_20k_wu100_offload}
CKPT=${CKPT:-checkpoint-20000}
if [[ -z "${TRANSFORMER_PATH:-}" && -n "$MODEL_TAG" ]]; then
  TRANSFORMER_PATH="$CKPT_ROOT/$MODEL_TAG/$CKPT/transformer"
fi
TRANSFORMER_PATH=${TRANSFORMER_PATH:-}

DATASET=${DATASET:-refcocog}

REFSEG_ROOT=${REFSEG_ROOT:-datasets_archives/refseg}
INDEX=${INDEX:-$REFSEG_ROOT/refcocog/val_umd/refcocog_val_umd.json}
IMAGES_ROOT=${IMAGES_ROOT:-}
PER_OBJECT=${PER_OBJECT:-0}
DATA_DIR=${DATA_DIR:-$REFSEG_ROOT/reasonseg_val}
REASONSEG_ALL_EXPR=${REASONSEG_ALL_EXPR:-0}
EXPR_INDEX=${EXPR_INDEX:-0}
# reasonseg: rewrite implicit queries to explicit referring expressions via the
# cached Qwen3-VL pass (ON by default). Set REASONSEG_REWRITE=0 for the raw-query
# baseline. The cache is produced once by rewrite_reasonseg_qwen.py and reused.
REASONSEG_REWRITE=${REASONSEG_REWRITE:-1}
REASONSEG_REWRITE_FILE=${REASONSEG_REWRITE_FILE:-$REFSEG_ROOT/reasonseg_rewrites_qwen3vl32bthink.json}

PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-}
STEPS=${STEPS:-28}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-3.5}
COLOR_SEED=${COLOR_SEED:-0}
GENERATION_SEED=${GENERATION_SEED:-}
NATIVE_RESOLUTION=${NATIVE_RESOLUTION:-0}
SAVE_MASKS=${SAVE_MASKS:-0}
SAVE_PANELS=${SAVE_PANELS:-1}
LIMIT=${LIMIT:-}
DRY_RUN=${DRY_RUN:-0}

# Output dir follows the shared eval convention used by the other suites:
#   $SCRATCH_ROOT/eval_outputs/open_visionbanana/$MODEL_TAG/$CKPT/<suite>_<benchmark>-<split>_steps<N>
# MODEL_TAG falls back to "base_flux2" for a base-model run with no tag.
# reasonseg with rewrite ON appends the rewrite model (from the cache filename
# reasonseg_rewrites_<tag>.json) so different rewriters get distinct dirs;
# refcocog / --no-rewrite keep the plain <benchmark> name.
REFSEG_SUITE_DS="$DATASET"
if [[ "$DATASET" == "reasonseg" ]] && is_true "$REASONSEG_REWRITE"; then
  _rw_base=$(basename "$REASONSEG_REWRITE_FILE" .json)
  REFSEG_SUITE_DS="reasonseg-${_rw_base#reasonseg_rewrites_}"
fi
OUT_DIR=${OUT_DIR:-$SCRATCH_ROOT/eval_outputs/open_visionbanana/${MODEL_TAG}/$CKPT/refseg_${REFSEG_SUITE_DS}-val_steps${STEPS}}
NUM_PROCESSES=${NUM_PROCESSES:-1}
# Cross-job sharding, as in the semseg/instanceseg wrappers. The evaluator already
# accepts --num_shards/--shard_index; this only plumbs them through so shards can
# be spread over a multi-node allocation, which torchrun --standalone cannot do.
NUM_SHARDS=${NUM_SHARDS:-}
SHARD_INDEX=${SHARD_INDEX:-0}
NO_AGGREGATE=${NO_AGGREGATE:-0}

if [[ "$DATASET" != "refcocog" && "$DATASET" != "reasonseg" ]]; then
  echo "ERROR: DATASET must be refcocog or reasonseg, got '$DATASET'" >&2
  exit 1
fi

# Prefer a local model snapshot, and treat a transformer dir passed as MODEL.
if [[ -d "$LOCAL_MODEL" && "$MODEL" == "$BASE_MODEL" ]]; then
  BASE_MODEL=$LOCAL_MODEL
  MODEL=$BASE_MODEL
fi
if [[ -z "$TRANSFORMER_PATH" && -d "$MODEL/transformer" ]]; then
  TRANSFORMER_PATH=$MODEL
  MODEL=$BASE_MODEL
fi

eval_args=(
  evaluation/refseg/inference_flux2_klein_refseg.py
  --dataset "$DATASET"
  --model "$MODEL"
  --out_dir "$OUT_DIR"
  --steps "$STEPS"
  --guidance_scale "$GUIDANCE_SCALE"
  --color_seed "$COLOR_SEED"
)
[[ -n "$GENERATION_SEED" ]] && eval_args+=(--generation_seed "$GENERATION_SEED")
if [[ "$DATASET" == "refcocog" ]]; then
  eval_args+=(--index "$INDEX")
  [[ -n "$IMAGES_ROOT" ]] && eval_args+=(--images_root "$IMAGES_ROOT")
  is_true "$PER_OBJECT" && eval_args+=(--per_object)
else
  eval_args+=(--data_dir "$DATA_DIR" --reasonseg_expr_index "$EXPR_INDEX")
  is_true "$REASONSEG_ALL_EXPR" && eval_args+=(--reasonseg_all_expr)
  if is_true "$REASONSEG_REWRITE"; then
    eval_args+=(--reasonseg_rewrite_file "$REASONSEG_REWRITE_FILE")
  else
    eval_args+=(--no-reasonseg_rewrite)
  fi
fi
[[ -n "$NUM_SHARDS" ]] && eval_args+=(--num_shards "$NUM_SHARDS" --shard_index "$SHARD_INDEX")
is_true "$NO_AGGREGATE" && eval_args+=(--no_aggregate)
[[ -n "$PROMPT_TEMPLATE" ]] && eval_args+=(--prompt_template "$PROMPT_TEMPLATE")
[[ -n "$TRANSFORMER_PATH" ]] && eval_args+=(--transformer_path "$TRANSFORMER_PATH")
[[ -n "$LIMIT" ]] && eval_args+=(--limit "$LIMIT")
is_true "$NATIVE_RESOLUTION" && eval_args+=(--native_resolution)
is_true "$SAVE_MASKS" && eval_args+=(--save_masks)
is_true "$SAVE_PANELS" || eval_args+=(--no-save-panels)
is_true "$DRY_RUN" && eval_args+=(--dry_run)

echo "Running referring-segmentation evaluation"
echo "  repo root:        $ROOT"
echo "  dataset:          $DATASET  (headline: $([[ $DATASET == refcocog ]] && echo cIoU || echo gIoU))"
echo "  model:            $MODEL"
echo "  transformer path: ${TRANSFORMER_PATH:-<base model only>}"
echo "  output dir:       $OUT_DIR"
echo "  eval processes:   $NUM_PROCESSES${NUM_SHARDS:+  (shard $SHARD_INDEX of $NUM_SHARDS, single GPU)}"

# A shard is one plain process on this task's GPU; torchrun is only for the
# single-job intra-node case (NUM_PROCESSES>1 with no NUM_SHARDS).
if [[ -z "$NUM_SHARDS" ]] && (( NUM_PROCESSES > 1 )); then
  launch_args=("$PYTHON" -m torch.distributed.run --standalone
               --nproc_per_node "$NUM_PROCESSES" "${eval_args[@]}")
else
  launch_args=("$PYTHON" "${eval_args[@]}")
fi
printf -v eval_cmd "%q " "${launch_args[@]}"

( cd "$ROOT" && eval "$eval_cmd" )
