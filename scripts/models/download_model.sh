#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

show_help() {
  cat <<'EOF'
Securely preflight, download, or verify a supported local model snapshot.

Usage:
  scripts/models/download_model.sh --root PATH --family FAMILY --stage preflight|download|verify

Families:
  klein, klein4b, qwen3vl8b, qwen3vl32b, qwen2vl7b, qwen-image-edit

The destination defaults below MODEL_ROOT derived from OVB_STORAGE_ROOT.
DEST, REPO_ID, REVISION, PYTHON, SIF, and HF_TOKEN_PATH remain overridable.
EOF
}

MODEL_FAMILY=${MODEL_FAMILY:-klein}
STAGE=${STAGE:-preflight}
STORAGE_ROOT_ARG=
while (($#)); do
  case "$1" in
    --root) STORAGE_ROOT_ARG=${2:?Missing value for --root}; shift 2 ;;
    --family) MODEL_FAMILY=${2:?Missing value for --family}; shift 2 ;;
    --stage) STAGE=${2:?Missing value for --stage}; shift 2 ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$STORAGE_ROOT_ARG" ]] || export OVB_STORAGE_ROOT="$STORAGE_ROOT_ARG"
[[ -z "$STORAGE_ROOT_ARG" ]] || MODEL_ROOT=$STORAGE_ROOT_ARG/models
[[ -z "$STORAGE_ROOT_ARG" ]] || unset DEST
MODEL_ROOT=${MODEL_ROOT:-${OVB_STORAGE_ROOT:+$OVB_STORAGE_ROOT/models}}
[[ -n "$MODEL_ROOT" || -n "${DEST:-}" ]] || {
  echo "ERROR: set OVB_STORAGE_ROOT, MODEL_ROOT, or DEST." >&2
  exit 2
}
case "$MODEL_FAMILY" in
  klein)
    REPO_ID=${REPO_ID:-black-forest-labs/FLUX.2-klein-base-9B}
    DEST=${DEST:-$MODEL_ROOT/FLUX.2-klein-base-9B}
    ;;
  klein4b)
    REPO_ID=${REPO_ID:-black-forest-labs/FLUX.2-klein-base-4B}
    DEST=${DEST:-$MODEL_ROOT/FLUX.2-klein-base-4B}
    ;;
  qwen3vl8b)
    REPO_ID=${REPO_ID:-Qwen/Qwen3-VL-8B-Instruct}
    DEST=${DEST:-$MODEL_ROOT/Qwen3-VL-8B-Instruct}
    generic=1
    ;;
  qwen3vl32b)
    REPO_ID=${REPO_ID:-Qwen/Qwen3-VL-32B-Thinking}
    DEST=${DEST:-$MODEL_ROOT/Qwen3-VL-32B-Thinking}
    generic=1
    ;;
  qwen2vl7b)
    REPO_ID=${REPO_ID:-Qwen/Qwen2.5-VL-7B-Instruct}
    DEST=${DEST:-$MODEL_ROOT/Qwen2.5-VL-7B-Instruct}
    generic=1
    ;;
  qwen-image-edit)
    REPO_ID=${REPO_ID:-Qwen/Qwen-Image-Edit-2511}
    DEST=${DEST:-$MODEL_ROOT/Qwen-Image-Edit-2511}
    generic=1
    ;;
  *) echo "ERROR: unsupported --family '$MODEL_FAMILY'" >&2; exit 2 ;;
esac

if [[ "${generic:-0}" == 1 ]]; then
  exec env REPO_ID="$REPO_ID" DEST="$DEST" STAGE="$STAGE" \
    "$SCRIPT_DIR/download_hf_snapshot.sh"
fi
exec env MODEL_FAMILY="$MODEL_FAMILY" REPO_ID="$REPO_ID" DEST="$DEST" STAGE="$STAGE" \
  "$SCRIPT_DIR/download_flux2.sh"
