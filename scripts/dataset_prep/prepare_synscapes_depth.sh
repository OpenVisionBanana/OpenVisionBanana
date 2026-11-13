#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Prepare an acquired Synscapes source tar for depth evaluation.

Required environment: RAW_ROOT and PREPARED_ROOT. Optional runtime variables:
USE_CONTAINER, SIF, and PYTHON. Acquisition is a separate step.
EOF
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then show_help; exit 0; fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
source "$ROOT/scripts/lib/runtime.sh"

RAW_ROOT=${RAW_ROOT:?Set RAW_ROOT to persistent raw-archive storage}
PREPARED_ROOT=${PREPARED_ROOT:?Set PREPARED_ROOT to persistent prepared storage}
SOURCE_TAR=${SOURCE_TAR:-$RAW_ROOT/synscapes/synscapes.tar}
OUT_ROOT=${OUT_ROOT:-${TMPDIR:-/tmp}/open_visionbanana/synscapes_depth}
TAR_OUTPUT=${TAR_OUTPUT:-$PREPARED_ROOT/depth/synscapes.tar}
LIMIT=${LIMIT:-}
SEED=${SEED:-42}
FORCE=${FORCE:-0}

[[ -f "$SOURCE_TAR" ]] || { echo "ERROR: missing Synscapes source tar: $SOURCE_TAR" >&2; exit 2; }
mkdir -p "$(dirname "$TAR_OUTPUT")"
if ovb_is_true "$FORCE"; then
  rm -rf "$OUT_ROOT"
  rm -f "$TAR_OUTPUT"
elif [[ -s "$TAR_OUTPUT" ]]; then
  echo "Using existing prepared Synscapes depth archive: $TAR_OUTPUT"
  exit 0
fi
args=(dataset_prep/depth/prepare_synscapes_depth.py --synscapes-tar "$SOURCE_TAR" --out-root "$OUT_ROOT" --prepared-tar "$TAR_OUTPUT" --seed "$SEED")
[[ -n "$LIMIT" ]] && args+=(--limit "$LIMIT")
ovb_run_python "${args[@]}"
