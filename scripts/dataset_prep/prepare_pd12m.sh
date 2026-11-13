#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Validate acquired PD12M shards and publish a portable dataset specification.

Required environment: RAW_ROOT and PREPARED_ROOT. Optional runtime variables:
USE_CONTAINER, SIF, and PYTHON. This stage performs no downloads.
EOF
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then show_help; exit 0; fi

# Offline preparation stage: validate acquired PD12M shards and publish a small
# dataset-spec artifact. The large WebDataset tars remain in RAW_ROOT and are
# not duplicated.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
source "$ROOT/scripts/lib/runtime.sh"

RAW_ROOT=${RAW_ROOT:?Set RAW_ROOT to persistent raw-archive storage}
PREPARED_ROOT=${PREPARED_ROOT:?Set PREPARED_ROOT to persistent prepared storage}
PRESET=${PRESET:-500k}
PD12M_RAW_DIR=${PD12M_RAW_DIR:-$RAW_ROOT/pd12m/$PRESET}
PD12M_PREPARED_DIR=${PD12M_PREPARED_DIR:-$PREPARED_ROOT/pd12m/$PRESET}
PD12M_WEIGHT=${PD12M_WEIGHT:-1.0}
SAMPLES_PER_EPOCH=${SAMPLES_PER_EPOCH:-50000}

ovb_run_python dataset_prep/pd12m/validate_pd12m.py "$PD12M_RAW_DIR"
mkdir -p "$PD12M_PREPARED_DIR"
spec_partial=$PD12M_PREPARED_DIR/dataset_specs.json.partial
ovb_run_python scripts/training/create_pd12m_mix_config.py \
  --out "$spec_partial" \
  --pd12m-shards "$PD12M_RAW_DIR/shards/*.tar" \
  --pd12m-weight "$PD12M_WEIGHT" \
  --samples-per-epoch "$SAMPLES_PER_EPOCH"
mv -f "$spec_partial" "$PD12M_PREPARED_DIR/dataset_specs.json"
printf 'raw_dir=%s\nvalidated_at=%s\n' "$PD12M_RAW_DIR" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$PD12M_PREPARED_DIR/COMPLETE"
echo "PD12M prepared spec: $PD12M_PREPARED_DIR/dataset_specs.json"
