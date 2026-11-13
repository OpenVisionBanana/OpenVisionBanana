#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
source "$ROOT/scripts/lib/runtime.sh"
SIF=${SIF:-}
PYTHON=${PYTHON:-python3}
STAGE=${STAGE:-preflight}
DEST=${DEST:?Set DEST to the final local FLUX.2 snapshot directory}
REVISION=${REVISION:-main}
REPO_ID=${REPO_ID:-black-forest-labs/FLUX.2-klein-base-9B}
export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}

args=("$ROOT/scripts/models/download_flux2.py" --stage "$STAGE" --dest "$DEST" --repo-id "$REPO_ID" --revision "$REVISION")
if [[ -n "$SIF" ]]; then
  ovb_container_exec "$SIF" "$PYTHON" "${args[@]}"
  exit $?
fi
exec "$PYTHON" "${args[@]}"
