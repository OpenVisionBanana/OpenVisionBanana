#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Acquire a PD12M subset as normalized WebDataset shards.

Required environment: RAW_ROOT. Optional runtime variables: USE_CONTAINER,
SIF, and PYTHON. Run prepare_pd12m.sh afterward for offline validation.
EOF
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then show_help; exit 0; fi

# Acquisition stage: select PD12M metadata and download normalized WebDataset
# shards into persistent raw storage. Preparation/validation is a separate step.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
source "$ROOT/scripts/lib/runtime.sh"

RAW_ROOT=${RAW_ROOT:?Set RAW_ROOT to persistent raw-archive storage}
PRESET=${PRESET:-500k}
PD12M_RAW_DIR=${PD12M_RAW_DIR:-$RAW_ROOT/pd12m/$PRESET}
WORKERS=${WORKERS:-32}
SEED=${SEED:-42}
METADATA_JSONL=${METADATA_JSONL:-}
FORCE_SELECTION=${FORCE_SELECTION:-0}
FORCE_SHARDS=${FORCE_SHARDS:-0}

args=(dataset_prep/pd12m/prepare_pd12m.py --out-dir "$PD12M_RAW_DIR" --preset "$PRESET" --stage all --workers "$WORKERS" --seed "$SEED")
[[ -n "$METADATA_JSONL" ]] && args+=(--metadata-jsonl "$METADATA_JSONL")
ovb_is_true "$FORCE_SELECTION" && args+=(--force-selection)
ovb_is_true "$FORCE_SHARDS" && args+=(--force-shards)

ovb_run_python "${args[@]}"
echo "PD12M acquisition complete: $PD12M_RAW_DIR"
echo "Run scripts/dataset_prep/prepare_pd12m.sh to validate and publish its dataset spec."
