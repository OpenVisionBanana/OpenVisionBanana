#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
source "$ROOT/scripts/lib/runtime.sh"
PYTHON=${PYTHON:-python3}
STAGE=${STAGE:-preflight}
REPO_ID=${REPO_ID:?Set REPO_ID}
DEST=${DEST:?Set DEST}
REVISION=${REVISION:-main}
export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}

args=("$ROOT/scripts/models/download_hf_snapshot.py" --stage "$STAGE" --repo-id "$REPO_ID" --dest "$DEST" --revision "$REVISION")
if [[ -n "${SIF:-}" ]]; then
  ovb_container_exec "$SIF" "$PYTHON" "${args[@]}"
  exit $?
fi
exec "$PYTHON" "${args[@]}"
