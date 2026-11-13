#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
source "$ROOT/scripts/lib/runtime.sh"

show_help() {
  cat <<'EOF'
Portable front door for FLUX.2-Klein training.

Usage:
  scripts/training/run_training.sh [options]

Options:
  --profile geometry|image-generation|legacy-pool
  --root PATH
  --model-family klein
  --model PATH
  --dataset-specs PATH
  --output-dir PATH
  --preflight-only

All existing training environment variables remain supported and are forwarded
to the original family-specific launcher.
EOF
}

TRAIN_PROFILE=${TRAIN_PROFILE:-geometry}
MODEL_FAMILY=${MODEL_FAMILY:-klein}
MODEL_ARG=
DATASET_SPECS=${DATASET_SPECS:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
PREFLIGHT_ONLY=0
STORAGE_ROOT_ARG=

while (($#)); do
  case "$1" in
    --profile) TRAIN_PROFILE=${2:?Missing value for --profile}; shift 2 ;;
    --root) STORAGE_ROOT_ARG=${2:?Missing value for --root}; shift 2 ;;
    --model-family) MODEL_FAMILY=${2:?Missing value for --model-family}; shift 2 ;;
    --model) MODEL_ARG=${2:?Missing value for --model}; shift 2 ;;
    --dataset-specs) DATASET_SPECS=${2:?Missing value for --dataset-specs}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?Missing value for --output-dir}; shift 2 ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; show_help >&2; exit 2 ;;
  esac
done

case "$MODEL_FAMILY" in
  klein) default_model_name=FLUX.2-klein-base-9B; configured_model=${KLEIN_MODEL:-${MODEL:-}}; launcher=$SCRIPT_DIR/run_open_visionbanana_pool_flux2_klein.sh ;;
  *) echo "ERROR: --model-family must be klein" >&2; exit 2 ;;
esac
case "$TRAIN_PROFILE" in
  geometry) default_specs_rel=training/geometry_mix.json ;;
  image-generation) default_specs_rel=pd12m/500k/dataset_specs.json ;;
  legacy-pool) default_specs_rel=open_visionbanana_mix_v1.json ;;
  *) echo "ERROR: --profile must be geometry, image-generation, or legacy-pool" >&2; exit 2 ;;
esac

STORAGE_ROOT=${STORAGE_ROOT_ARG:-${OVB_STORAGE_ROOT:-}}
if [[ -n "$STORAGE_ROOT_ARG" ]]; then
  export OVB_STORAGE_ROOT="$STORAGE_ROOT_ARG"
  MODEL_ROOT=$STORAGE_ROOT_ARG/models
  PREPARED_ROOT=$STORAGE_ROOT_ARG/prepared
  OUTPUT_ROOT=$STORAGE_ROOT_ARG/outputs
  configured_model=
fi
MODEL_ROOT=${MODEL_ROOT:-${STORAGE_ROOT:+$STORAGE_ROOT/models}}
PREPARED_ROOT=${PREPARED_ROOT:-${STORAGE_ROOT:+$STORAGE_ROOT/prepared}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${STORAGE_ROOT:+$STORAGE_ROOT/outputs}}
MODEL=${MODEL_ARG:-${configured_model:-${MODEL_ROOT:+$MODEL_ROOT/$default_model_name}}}
DATASET_SPECS=${DATASET_SPECS:-${PREPARED_ROOT:+$PREPARED_ROOT/$default_specs_rel}}
if [[ -z "$MODEL" || -z "$DATASET_SPECS" ]]; then
  echo "ERROR: set OVB_STORAGE_ROOT or provide --model and --dataset-specs." >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  [[ -n "$OUTPUT_ROOT" ]] || { echo "ERROR: set OVB_STORAGE_ROOT, OUTPUT_ROOT, or --output-dir." >&2; exit 2; }
  spec_tag=$(basename "${DATASET_SPECS%.json}")
  OUTPUT_DIR=$OUTPUT_ROOT/training/${MODEL_FAMILY}_${spec_tag}
fi

preflight=(scripts/preflight.py --profile train --model-family "$MODEL_FAMILY" --model "$MODEL" --dataset-specs "$DATASET_SPECS")
if ! ovb_is_true "${SKIP_PREFLIGHT:-0}"; then
  ovb_run_python "${preflight[@]}"
fi
[[ "$PREFLIGHT_ONLY" == 1 ]] && exit 0

mkdir -p "$OUTPUT_DIR"
exec env \
  MODEL_FAMILY="$MODEL_FAMILY" MODEL="$MODEL" LOCAL_MODEL="$MODEL" \
  DATASET_SPECS="$DATASET_SPECS" OUTPUT_DIR="$OUTPUT_DIR" \
  USE_CONTAINER="${USE_CONTAINER:-0}" SIF="${SIF:-}" PYTHON="${PYTHON:-python}" \
  bash "$launcher"
