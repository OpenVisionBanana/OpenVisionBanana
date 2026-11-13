#!/usr/bin/env bash
set -euo pipefail

# Populate a persistent Hugging Face datasets cache for offline SteerViT
# preparation. Source RGB images are acquired separately.

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Cache SteerViT metadata/masks for offline preparation.

Required environment:
  HF_HOME          Persistent Hugging Face cache directory.

Optional environment:
  PYTHON           Python with `datasets` installed (default: python3).
  HF_DATASETS      Comma-separated dataset IDs (default: JonaRuthardt/SteerViT).
  STEERVIT_SPLITS  Comma-separated splits (default: train,validation).

The Hugging Face token, when required, is read by the standard Hub client. It
is never accepted as a command-line argument or printed by this script.
EOF
  exit 0
fi

HF_HOME=${HF_HOME:?Set HF_HOME to a persistent cache directory}
PYTHON=${PYTHON:-python3}
HF_DATASETS=${HF_DATASETS:-JonaRuthardt/SteerViT}
STEERVIT_SPLITS=${STEERVIT_SPLITS:-train,validation}
mkdir -p "$HF_HOME"

# This is explicitly the networked acquisition stage.
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME HF_DATASETS STEERVIT_SPLITS

"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

from datasets import load_dataset

dataset_ids = [value.strip() for value in os.environ["HF_DATASETS"].split(",") if value.strip()]
splits = [value.strip() for value in os.environ["STEERVIT_SPLITS"].split(",") if value.strip()]
if not dataset_ids or not splits:
    raise SystemExit("HF_DATASETS and STEERVIT_SPLITS must not be empty")

cached = []
for dataset_id in dataset_ids:
    for split in splits:
        dataset = load_dataset(dataset_id, split=split)
        cached.append({"dataset": dataset_id, "split": split, "rows": len(dataset)})
        print(f"Cached {dataset_id} [{split}]: {len(dataset)} rows")

marker = Path(os.environ["HF_HOME"]) / "open_visionbanana_steervit_cache.json"
temporary = marker.with_suffix(marker.suffix + ".partial")
temporary.write_text(json.dumps({"datasets": cached}, indent=2) + "\n", encoding="utf-8")
temporary.replace(marker)
print(f"SteerViT cache ready: {marker}")
PY
