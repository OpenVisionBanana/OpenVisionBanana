#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Download a Qwen VQA judge model into a Hugging Face cache for offline scoring.

Run this on a login/transfer node with internet. It only needs one of:
  - hf
  - huggingface-cli
  - python with huggingface_hub installed

Usage:
  scripts/evaluation/image_generation/download_qwen_judge_model.sh

Environment variables:
  MODEL_ID   Hugging Face repo id. Default: Qwen/Qwen3.5-9B
  HF_HOME    Required cache directory shared with compute nodes.
  HF_TOKEN   Optional token for gated/private repos.

Then score with:
  HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  EVALUATOR=qwen3.5-9b SCORE_BATCH_SIZE=1 RUN_PREP=0 RUN_GENERATE=0 RUN_SCORE=1 \
  scripts/evaluation/image_generation/run_genai_bench_flux2_klein.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-9B}
HF_HOME=${HF_HOME:?Set HF_HOME to a persistent Hugging Face cache directory}
export HF_HOME

mkdir -p "$HF_HOME"

echo "Downloading judge model"
echo "  MODEL_ID: $MODEL_ID"
echo "  HF_HOME:  $HF_HOME"

if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL_ID"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$MODEL_ID"
else
  python - "$MODEL_ID" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

model_id = sys.argv[1]
token = os.environ.get("HF_TOKEN")
snapshot_download(repo_id=model_id, token=token)
PY
fi

echo "Done. Keep HF_HOME pointed at this cache when scoring:"
echo "  export HF_HOME=$HF_HOME"
