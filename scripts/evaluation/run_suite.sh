#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
source "$ROOT/scripts/lib/runtime.sh"

show_help() {
  cat <<'EOF'
Portable front door for depth, surface-normal, or GenAI-Bench evaluation.

Usage:
  scripts/evaluation/run_suite.sh --suite depth|normals|genai [options]

Options:
  --model-family klein
  --root PATH
  --model PATH
  --checkpoint PATH       Full-rank transformer/checkpoint artifact.
  --lora PATH             Standard Diffusers LoRA artifact.
  --out-dir PATH
  --geometry-profile full|ablation
  --strict                Require every dataset in a geometry suite.
  --available             Evaluate every prepared archive present (default).
  --preflight-only

Existing evaluator environment variables are forwarded unchanged.
EOF
}

SUITE=
MODEL_FAMILY=${MODEL_FAMILY:-klein}
MODEL_ARG=
TRANSFORMER_PATH=${TRANSFORMER_PATH:-}
LORA_PATH=${LORA_PATH:-}
OUT_DIR=${OUT_DIR:-}
STRICT_DATA=0
PREFLIGHT_ONLY=0
STORAGE_ROOT_ARG=
GEOMETRY_PROFILE=${GEOMETRY_PROFILE:-full}
IBIMS1_DEPTH_PROTOCOL=${IBIMS1_DEPTH_PROTOCOL:-ibims_official}
IBIMS1_NORMALS_PROTOCOL=${IBIMS1_NORMALS_PROTOCOL:-ibims_official}

while (($#)); do
  case "$1" in
    --suite) SUITE=${2:?Missing value for --suite}; shift 2 ;;
    --root) STORAGE_ROOT_ARG=${2:?Missing value for --root}; shift 2 ;;
    --model-family) MODEL_FAMILY=${2:?Missing value for --model-family}; shift 2 ;;
    --model) MODEL_ARG=${2:?Missing value for --model}; shift 2 ;;
    --checkpoint) TRANSFORMER_PATH=${2:?Missing value for --checkpoint}; shift 2 ;;
    --lora) LORA_PATH=${2:?Missing value for --lora}; shift 2 ;;
    --out-dir) OUT_DIR=${2:?Missing value for --out-dir}; shift 2 ;;
    --geometry-profile) GEOMETRY_PROFILE=${2:?Missing value for --geometry-profile}; shift 2 ;;
    --strict) STRICT_DATA=1; shift ;;
    --available) STRICT_DATA=0; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; show_help >&2; exit 2 ;;
  esac
done

[[ -n "$SUITE" ]] || { echo "ERROR: --suite is required." >&2; exit 2; }
case "$GEOMETRY_PROFILE" in full|ablation) ;; *) echo "ERROR: --geometry-profile must be full or ablation" >&2; exit 2 ;; esac
[[ -z "$TRANSFORMER_PATH" || -z "$LORA_PATH" ]] || {
  echo "ERROR: --checkpoint and --lora are mutually exclusive." >&2
  exit 2
}
case "$MODEL_FAMILY" in
  klein) default_model_name=FLUX.2-klein-base-9B; configured_model=${KLEIN_MODEL:-${MODEL:-}} ;;
  *) echo "ERROR: --model-family must be klein" >&2; exit 2 ;;
esac

STORAGE_ROOT=${STORAGE_ROOT_ARG:-${OVB_STORAGE_ROOT:-}}
if [[ -n "$STORAGE_ROOT_ARG" ]]; then
  export OVB_STORAGE_ROOT="$STORAGE_ROOT_ARG"
  MODEL_ROOT=$STORAGE_ROOT_ARG/models
  PREPARED_ROOT=$STORAGE_ROOT_ARG/prepared
  RAW_ROOT=$STORAGE_ROOT_ARG/raw
  OUTPUT_ROOT=$STORAGE_ROOT_ARG/outputs
  GENAI_SOURCE_DIR=$STORAGE_ROOT_ARG/raw/genai-bench
  PROMPTS_JSONL=$STORAGE_ROOT_ARG/prepared/genai-bench/prompts.jsonl
  configured_model=
fi
MODEL_ROOT=${MODEL_ROOT:-${STORAGE_ROOT:+$STORAGE_ROOT/models}}
SCORE_MODEL_PATH=${SCORE_MODEL_PATH:-${MODEL_ROOT:+$MODEL_ROOT/Qwen2.5-VL-7B-Instruct}}
export SCORE_MODEL_PATH
PREPARED_ROOT=${PREPARED_ROOT:-${STORAGE_ROOT:+$STORAGE_ROOT/prepared}}
if [[ -n "$STORAGE_ROOT_ARG" ]]; then
  GEOMETRY_ARCHIVE_BASE=$PREPARED_ROOT
else
  GEOMETRY_ARCHIVE_BASE=${ARCHIVE_ROOT:-$PREPARED_ROOT}
fi
if [[ -z "${GEOMETRY_ARCHIVE_ROOT:-}" && "$GEOMETRY_PROFILE" == ablation ]]; then
  GEOMETRY_ARCHIVE_ROOT=${GEOMETRY_ARCHIVE_BASE:+$GEOMETRY_ARCHIVE_BASE/profiles/ablation}
else
  GEOMETRY_ARCHIVE_ROOT=${GEOMETRY_ARCHIVE_ROOT:-$GEOMETRY_ARCHIVE_BASE}
fi
RAW_ROOT=${RAW_ROOT:-${STORAGE_ROOT:+$STORAGE_ROOT/raw}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${STORAGE_ROOT:+$STORAGE_ROOT/outputs}}
GENAI_SOURCE_DIR=${GENAI_SOURCE_DIR:-${RAW_ROOT:+$RAW_ROOT/genai-bench}}
PROMPTS_JSONL=${PROMPTS_JSONL:-${PREPARED_ROOT:+$PREPARED_ROOT/genai-bench/prompts.jsonl}}
MODEL=${MODEL_ARG:-${configured_model:-${MODEL_ROOT:+$MODEL_ROOT/$default_model_name}}}
if [[ -z "$MODEL" || ( -z "$OUT_DIR" && -z "$OUTPUT_ROOT" ) ]]; then
  echo "ERROR: set OVB_STORAGE_ROOT or provide --model and --out-dir with the required data roots." >&2
  exit 2
fi

case "$SUITE" in
  depth)
    preflight_profile=depth-eval
    archive_folder=depth
    if [[ "$GEOMETRY_PROFILE" == ablation ]]; then
      all_datasets=(nuscenes scannet vkitti2)
    else
      all_datasets=(nyu ibims1 eth3d diode kitti nuscenes vkitti2 scannet kitti_eigen)
    fi
    launcher=$SCRIPT_DIR/depth/run_all.sh
    ;;
  normals|surface_normals)
    SUITE=normals
    preflight_profile=normals-eval
    archive_folder=surface_normals
    if [[ "$GEOMETRY_PROFILE" == ablation ]]; then
      all_datasets=(nuscenes scannet vkitti2)
    else
      all_datasets=(nyu ibims1 diode kitti eth3d nuscenes vkitti2 scannet sintel)
    fi
    launcher=$SCRIPT_DIR/surface_normals/run_all.sh
    ;;
  genai)
    preflight_profile=genai-eval
    archive_folder=
    all_datasets=()
    launcher=$SCRIPT_DIR/image_generation/run_genai_bench.sh
    ;;
  *) echo "ERROR: --suite must be depth, normals, or genai" >&2; exit 2 ;;
esac

if [[ -z "$OUT_DIR" ]]; then
  if [[ -n "$archive_folder" ]]; then
    OUT_DIR=$OUTPUT_ROOT/evaluation/$SUITE/$GEOMETRY_PROFILE/$MODEL_FAMILY
  else
    OUT_DIR=$OUTPUT_ROOT/evaluation/$SUITE/$MODEL_FAMILY
  fi
fi

geometry_archive_filename() {
  local dataset=$1
  if [[ "$dataset" != ibims1 ]]; then
    printf '%s.tar\n' "$dataset"
  elif [[ "$archive_folder" == depth && "$IBIMS1_DEPTH_PROTOCOL" == ibims_official ]]; then
    printf 'ibims1_official.tar\n'
  elif [[ "$archive_folder" == surface_normals && "$IBIMS1_NORMALS_PROTOCOL" == ibims_official ]]; then
    printf 'ibims1_official.tar\n'
  else
    printf 'ibims1.tar\n'
  fi
}

preflight=(scripts/preflight.py --profile "$preflight_profile" --model-family "$MODEL_FAMILY" --model "$MODEL")
[[ -n "$STORAGE_ROOT" ]] || preflight+=(--explicit-paths)
if [[ -n "$archive_folder" ]]; then
  preflight+=(
    --archive-root "$GEOMETRY_ARCHIVE_ROOT"
    --geometry-profile "$GEOMETRY_PROFILE"
    --ibims1-depth-protocol "$IBIMS1_DEPTH_PROTOCOL"
    --ibims1-normals-protocol "$IBIMS1_NORMALS_PROTOCOL"
  )
fi
[[ -n "$TRANSFORMER_PATH" ]] && preflight+=(--transformer-path "$TRANSFORMER_PATH")
[[ -n "$LORA_PATH" ]] && preflight+=(--lora-path "$LORA_PATH")
[[ "$STRICT_DATA" == 1 ]] && preflight+=(--strict-data)
if [[ "$SUITE" == genai && -n "$PROMPTS_JSONL" ]]; then preflight+=(--prompts-jsonl "$PROMPTS_JSONL"); fi
if ! ovb_is_true "${SKIP_PREFLIGHT:-0}"; then
  ovb_run_python "${preflight[@]}"
fi
[[ "$PREFLIGHT_ONLY" == 1 ]] && exit 0

DATASETS=${DATASETS:-}
if [[ -n "$archive_folder" && -z "$DATASETS" ]]; then
  if [[ "$STRICT_DATA" == 1 ]]; then
    DATASETS="${all_datasets[*]}"
  else
    available=()
    missing=()
    for dataset in "${all_datasets[@]}"; do
      archive_name=$(geometry_archive_filename "$dataset")
      if [[ -f "$GEOMETRY_ARCHIVE_ROOT/$archive_folder/$archive_name" ]]; then
        available+=("$dataset")
      else
        missing+=("$dataset")
      fi
    done
    ((${#available[@]})) || { echo "ERROR: no prepared $SUITE archives found." >&2; exit 1; }
    DATASETS="${available[*]}"
    echo "Evaluating available $SUITE datasets: $DATASETS"
    ((${#missing[@]} == 0)) || echo "Not prepared: ${missing[*]}"
  fi
fi

mkdir -p "$OUT_DIR"
exec env \
  MODEL_FAMILY="$MODEL_FAMILY" MODEL="$MODEL" LOCAL_MODEL="$MODEL" \
  TRANSFORMER_PATH="$TRANSFORMER_PATH" LORA_PATH="$LORA_PATH" \
  PREPARED_ROOT="$PREPARED_ROOT" ARCHIVE_ROOT="$GEOMETRY_ARCHIVE_ROOT" \
  SAMPLE_MANIFEST_ROOT="$GEOMETRY_ARCHIVE_ROOT/manifests" RAW_ROOT="$RAW_ROOT" \
  GENAI_SOURCE_DIR="$GENAI_SOURCE_DIR" PROMPTS_JSONL="$PROMPTS_JSONL" \
  SCORE_MODEL_PATH="$SCORE_MODEL_PATH" \
  OUTPUT_ROOT="$OUTPUT_ROOT" OUT_DIR="$OUT_DIR" DATASETS="$DATASETS" \
  GEOMETRY_PROFILE="$GEOMETRY_PROFILE" \
  IBIMS1_DEPTH_PROTOCOL="$IBIMS1_DEPTH_PROTOCOL" \
  IBIMS1_NORMALS_PROTOCOL="$IBIMS1_NORMALS_PROTOCOL" \
  USE_CONTAINER="${USE_CONTAINER:-0}" SIF="${SIF:-}" PYTHON="${PYTHON:-python}" \
  bash "$launcher"
