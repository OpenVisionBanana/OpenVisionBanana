#!/usr/bin/env bash
# Download FoodSeg103 from the Hugging Face hub, materialize the parquet rows as
# raw images + grayscale class-id masks, and pack a tar that the normal staging
# path can extract.
#
# Output:
#   datasets_archives/sem_segm/foodseg103.tar -> foodseg103/{images,labels}/{train,val}/*.png
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_foodseg103.sh
#   WORK=$TMPDIR/foodseg103_download ...      # use node-local scratch
#   FOODSEG103_LIMIT=32 ...                   # quick smoke download, per split
#   KEEP_EXTRACTED=1 ...                      # keep materialized folder beside tar
#   FOODSEG103_HF_CACHE=$TMPDIR/hf_cache ...  # override temporary HF cache root
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/sem_segm}
TMP_PARENT=${TMPDIR:-$ARCHIVE_DIR}
WORK=${WORK:-$TMP_PARENT/open_visionbanana_foodseg103_download}
FOODSEG103_TAR=${FOODSEG103_TAR:-$ARCHIVE_DIR/foodseg103.tar}
FOODSEG103_BUILD_ROOT=${FOODSEG103_BUILD_ROOT:-$WORK/foodseg103_build}
FOODSEG103_HF_CACHE=${FOODSEG103_HF_CACHE:-$WORK/huggingface_cache}
FOODSEG103_REPO=${FOODSEG103_REPO:-EduardoPacheco/FoodSeg103}
FOODSEG103_SPLITS=${FOODSEG103_SPLITS:-train,validation}
FOODSEG103_LIMIT=${FOODSEG103_LIMIT:-}
PYTHON=${PYTHON:-python}
FORCE=${FORCE:-0}
KEEP_EXTRACTED=${KEEP_EXTRACTED:-0}

mkdir -p "$ARCHIVE_DIR" "$WORK" "$FOODSEG103_BUILD_ROOT"

if [[ "$FORCE" == "1" ]]; then
  rm -rf "$FOODSEG103_BUILD_ROOT/foodseg103" "$FOODSEG103_TAR"
fi

if [[ -f "$FOODSEG103_TAR" && "$FORCE" != "1" ]]; then
  echo "FoodSeg103 tar already exists: $FOODSEG103_TAR"
  echo "Set FORCE=1 to rebuild it."
  exit 0
fi

env \
  ROOT="$ROOT" \
  FOODSEG103_REPO="$FOODSEG103_REPO" \
  FOODSEG103_SPLITS="$FOODSEG103_SPLITS" \
  FOODSEG103_LIMIT="$FOODSEG103_LIMIT" \
  FOODSEG103_OUT="$FOODSEG103_BUILD_ROOT/foodseg103" \
  FOODSEG103_HF_CACHE="$FOODSEG103_HF_CACHE" \
  "$PYTHON" - <<'PY'
import json
import os
import sys
from pathlib import Path

cache_root = Path(os.environ["FOODSEG103_HF_CACHE"])
os.environ["HF_HOME"] = str(cache_root)
os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")

import numpy as np
from PIL import Image

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "Missing Python package 'datasets'. Install it in the active environment "
        "or run inside the project data-prep container."
    ) from exc

root = Path(os.environ["ROOT"])
sys.path.insert(0, str(root))
from common.semseg.taxonomy import FOODSEG103_CLASSES  # noqa: E402

repo = os.environ["FOODSEG103_REPO"]
out = Path(os.environ["FOODSEG103_OUT"])
splits = [s.strip() for s in os.environ["FOODSEG103_SPLITS"].split(",") if s.strip()]
limit_text = os.environ.get("FOODSEG103_LIMIT", "").strip()
limit = int(limit_text) if limit_text else None
split_alias = {"validation": "val", "val": "val", "train": "train"}

out.mkdir(parents=True, exist_ok=True)
(out / "id2label.json").write_text(
    json.dumps({str(i): name for i, name in enumerate(FOODSEG103_CLASSES)}, indent=2) + "\n",
    encoding="utf-8",
)

summary = {}
for hf_split in splits:
    local_split = split_alias.get(hf_split, hf_split)
    image_dir = out / "images" / local_split
    label_dir = out / "labels" / local_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(repo, split=hf_split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    count = 0
    for row in dataset:
        sample_id = int(row["id"])
        stem = f"{local_split}_{sample_id:06d}"

        image = row["image"].convert("RGB")
        image.save(image_dir / f"{stem}.png")

        label = row["label"]
        arr = np.asarray(label)
        if arr.ndim == 3:
            if not (np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2])):
                raise ValueError(f"Label for {hf_split}/{sample_id} is RGB, not class ids.")
            arr = arr[..., 0]
        arr = arr.astype(np.uint8, copy=False)
        if int(arr.max(initial=0)) >= len(FOODSEG103_CLASSES):
            raise ValueError(f"Label for {hf_split}/{sample_id} contains class id {int(arr.max())}.")
        Image.fromarray(arr).save(label_dir / f"{stem}.png")
        count += 1

    summary[local_split] = count
    print(f"{hf_split} -> {local_split}: wrote {count} samples")

(out / "source.json").write_text(
    json.dumps(
        {
            "source": repo,
            "splits": splits,
            "local_splits": summary,
            "format": "images/{split}/*.png + labels/{split}/*.png, label pixels are raw FoodSeg103 ids 0..103",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo "Packing $FOODSEG103_TAR"
tar -C "$FOODSEG103_BUILD_ROOT" -cf "$FOODSEG103_TAR.partial" foodseg103
mv -f "$FOODSEG103_TAR.partial" "$FOODSEG103_TAR"

if [[ "$KEEP_EXTRACTED" != "1" ]]; then
  rm -rf "$FOODSEG103_BUILD_ROOT"
fi

cat <<EOF2

Done:
  $FOODSEG103_TAR

Use it through the normal staging launcher:
  FOODSEG103_TAR=$FOODSEG103_TAR bash scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh

Or run semseg directly:
  KEEP_EXTRACTED=1 bash scripts/dataset_prep/data_download_scripts/download_datasets_foodseg103.sh
  python dataset_prep/semseg/dataset_prep.py --foodseg103-root $FOODSEG103_BUILD_ROOT/foodseg103 --out-root /tmp/foodseg103_semseg
EOF2
