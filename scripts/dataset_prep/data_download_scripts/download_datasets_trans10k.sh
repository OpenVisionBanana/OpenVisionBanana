#!/usr/bin/env bash
# Download Trans10K-v2 (transparent-object segmentation) from the Hugging Face hub,
# materialize the parquet rows as raw images + class-id masks, and pack a tar that
# the normal staging path can extract.
#
# The mirror stores masks as RGB PNGs painted with the official Trans10K palette.
# This folds them to uint8 class ids so the reader in common/semseg/taxonomy.py
# stays a plain lookup, matching FoodSeg103 and PASCAL-Context. Class names and
# index order follow the official Trans10K-v2 taxonomy, not the mirror's README
# (which renumbers the classes and calls index 8 "wall" instead of "floor glass").
#
# The parquet rows carry the original Trans10K file names, so stems stay traceable
# back to the upstream release (e.g. images/val/10000.jpg <- 10000.jpg / 10000_mask.png).
#
# Output:
#   datasets_archives/sem_segm/trans10k.tar -> trans10k/images/{train,val,test}/*.jpg
#                                              trans10k/labels/{train,val,test}/*.png
#                                              trans10k/{id2label,palette,source}.json
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_trans10k.sh
#   WORK=$TMPDIR/trans10k_download ...      # use node-local scratch (needs ~4 GB free)
#   TRANS10K_LIMIT=32 ...                   # quick smoke run, per split (never packs a tar)
#   TRANS10K_SPLITS=train,validation ...    # skip the 4428-image test split
#   TRANS10K_WORKERS=8 ...                  # mask conversion / encode threads
#   KEEP_EXTRACTED=1 ...                    # keep the materialized folder beside the tar
#   TRANS10K_HF_CACHE=$TMPDIR/hf_cache ...  # override temporary HF cache root
#   FORCE=1 ...                             # rebuild even if the tar already exists
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/sem_segm}
TMP_PARENT=${TMPDIR:-$ARCHIVE_DIR}
WORK=${WORK:-$TMP_PARENT/open_visionbanana_trans10k_download}
TRANS10K_TAR=${TRANS10K_TAR:-$ARCHIVE_DIR/trans10k.tar}
TRANS10K_BUILD_ROOT=${TRANS10K_BUILD_ROOT:-$WORK/trans10k_build}
TRANS10K_HF_CACHE=${TRANS10K_HF_CACHE:-$WORK/huggingface_cache}
TRANS10K_REPO=${TRANS10K_REPO:-rdyzakya/Trans10K-v2}
TRANS10K_SPLITS=${TRANS10K_SPLITS:-train,validation,test}
TRANS10K_LIMIT=${TRANS10K_LIMIT:-}
TRANS10K_WORKERS=${TRANS10K_WORKERS:-8}
PYTHON=${PYTHON:-python}
FORCE=${FORCE:-0}
KEEP_EXTRACTED=${KEEP_EXTRACTED:-0}

mkdir -p "$ARCHIVE_DIR" "$WORK" "$TRANS10K_BUILD_ROOT"

if [[ "$FORCE" == "1" ]]; then
  rm -rf "$TRANS10K_BUILD_ROOT/trans10k" "$TRANS10K_TAR"
fi

if [[ -f "$TRANS10K_TAR" && "$FORCE" != "1" ]]; then
  echo "Trans10K-v2 tar already exists: $TRANS10K_TAR"
  echo "Set FORCE=1 to rebuild it."
  exit 0
fi

env \
  ROOT="$ROOT" \
  TRANS10K_REPO="$TRANS10K_REPO" \
  TRANS10K_SPLITS="$TRANS10K_SPLITS" \
  TRANS10K_LIMIT="$TRANS10K_LIMIT" \
  TRANS10K_WORKERS="$TRANS10K_WORKERS" \
  TRANS10K_OUT="$TRANS10K_BUILD_ROOT/trans10k" \
  TRANS10K_HF_CACHE="$TRANS10K_HF_CACHE" \
  "$PYTHON" - <<'PY'
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

cache_root = Path(os.environ["TRANS10K_HF_CACHE"])
os.environ["HF_HOME"] = str(cache_root)
os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError as exc:
    raise SystemExit(
        "Missing Python package 'huggingface_hub'. Install it in the active "
        "environment or run inside the project data-prep container."
    ) from exc

root = Path(os.environ["ROOT"])
sys.path.insert(0, str(root))
from common.semseg.taxonomy import TRANS10K_CLASSES, TRANS10K_PALETTE  # noqa: E402

repo = os.environ["TRANS10K_REPO"]
out = Path(os.environ["TRANS10K_OUT"])
splits = [s.strip() for s in os.environ["TRANS10K_SPLITS"].split(",") if s.strip()]
limit_text = os.environ.get("TRANS10K_LIMIT", "").strip()
limit = int(limit_text) if limit_text else None
workers = max(1, int(os.environ.get("TRANS10K_WORKERS", "8")))
split_alias = {"validation": "val", "val": "val", "train": "train", "test": "test"}

# --- RGB mask -> class id ---------------------------------------------------- #
# Colors are looked up exactly; the few off-palette pixels fall back to the
# nearest palette entry.
palette = TRANS10K_PALETTE.astype(np.int64)
pal_keys = (palette[:, 0] << 16) | (palette[:, 1] << 8) | palette[:, 2]
pal_order = np.argsort(pal_keys)
pal_sorted = pal_keys[pal_order]


def rgb_to_ids(rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """HxWx3 uint8 -> (HxW uint8 class ids, #pixels that needed the nearest-color fallback)."""
    key = ((rgb[..., 0].astype(np.int64) << 16)
           | (rgb[..., 1].astype(np.int64) << 8)
           | rgb[..., 2].astype(np.int64))
    uniq, inverse = np.unique(key.ravel(), return_inverse=True)
    pos = np.clip(np.searchsorted(pal_sorted, uniq), 0, len(pal_sorted) - 1)
    exact = pal_sorted[pos] == uniq
    ids = np.empty(uniq.shape, dtype=np.uint8)
    ids[exact] = pal_order[pos[exact]]
    off = 0
    if not exact.all():
        stray = uniq[~exact]
        cols = np.stack([(stray >> 16) & 255, (stray >> 8) & 255, stray & 255], axis=1)
        ids[~exact] = ((cols[:, None, :] - palette[None, :, :]) ** 2).sum(2).argmin(1)
        off = int(np.isin(key.ravel(), stray).sum())
    return ids[inverse].reshape(key.shape), off


# --- locate the parquet shards ----------------------------------------------- #
repo_files = [f for f in list_repo_files(repo, repo_type="dataset")
              if f.startswith("data/") and f.endswith(".parquet")]
if not repo_files:
    raise SystemExit(f"No data/*.parquet shards found in the dataset repo {repo}.")
shards: dict[str, list[str]] = {}
for name in sorted(repo_files):
    shards.setdefault(Path(name).name.split("-", 1)[0], []).append(name)

unknown = [s for s in splits if s not in shards]
if unknown:
    raise SystemExit(f"Requested splits {unknown} are not in {repo} (available: {sorted(shards)}).")

out.mkdir(parents=True, exist_ok=True)
(out / "id2label.json").write_text(
    json.dumps({str(i): name for i, name in enumerate(TRANS10K_CLASSES)}, indent=2) + "\n",
    encoding="utf-8",
)
(out / "palette.json").write_text(
    json.dumps([[int(v) for v in row] for row in TRANS10K_PALETTE], indent=2) + "\n",
    encoding="utf-8",
)

summary: dict[str, int] = {}
off_palette_total = 0
for hf_split in splits:
    local_split = split_alias.get(hf_split, hf_split)
    image_dir = out / "images" / local_split
    label_dir = out / "labels" / local_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    def convert(row: tuple[dict, dict]) -> int:
        image_cell, mask_cell = row
        # The parquet keeps the upstream file names: "10000.jpg" / "10000_mask.png".
        stem = Path(image_cell["path"]).stem
        suffix = Path(image_cell["path"]).suffix.lower() or ".jpg"
        # Image bytes are written through untouched -- no decode/re-encode, so the
        # JPEGs stay bit-identical to the release instead of being recompressed.
        (image_dir / f"{stem}{suffix}").write_bytes(image_cell["bytes"])

        with Image.open(io.BytesIO(mask_cell["bytes"])) as mask:
            arr = np.asarray(mask.convert("RGB"))
        ids, off = rgb_to_ids(arr)
        Image.fromarray(ids).save(label_dir / f"{stem}.png")
        return off

    count = 0
    for shard in shards[hf_split]:
        if limit is not None and count >= limit:
            break
        local_shard = hf_hub_download(repo, shard, repo_type="dataset")
        parquet = pq.ParquetFile(local_shard)
        for group in range(parquet.num_row_groups):
            if limit is not None and count >= limit:
                break
            table = parquet.read_row_group(group, columns=["image", "mask"])
            rows = list(zip(table.column("image").to_pylist(), table.column("mask").to_pylist()))
            if limit is not None:
                rows = rows[:limit - count]
            with ThreadPoolExecutor(max_workers=workers) as ex:
                off_palette_total += sum(ex.map(convert, rows))
            count += len(rows)

    summary[local_split] = count
    print(f"{hf_split} -> {local_split}: wrote {count} samples")

if off_palette_total:
    print(f"NOTE: {off_palette_total} pixels were off-palette and took the nearest palette color.")

(out / "source.json").write_text(
    json.dumps(
        {
            "source": repo,
            "upstream": "https://github.com/xieenze/Trans2Seg (arXiv 2101.08461)",
            "splits": splits,
            "local_splits": summary,
            "num_classes": len(TRANS10K_CLASSES),
            "class_source": (
                "segmentron/data/dataloader/transparent11.py + segmentron/utils/visualize.py "
                "(trans10kv2pallete) in xieenze/Trans2Seg; the HF mirror's README uses a "
                "different numbering and names index 8 'wall' rather than 'floor glass'."
            ),
            "off_palette_pixels": off_palette_total,
            "format": (
                "images/{split}/*.jpg are the upstream JPEGs byte-for-byte; "
                "labels/{split}/*.png are uint8 class ids, 0 = background, 1..11 = the "
                "Trans10K-v2 transparent classes, folded from the release's RGB masks "
                "via the official palette."
            ),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

# A TRANS10K_LIMIT run materializes only a few images per split and never packs,
# so a partial tar cannot be mistaken for a complete one.
if [[ -n "$TRANS10K_LIMIT" ]]; then
  cat <<EOF2

Smoke run (TRANS10K_LIMIT=$TRANS10K_LIMIT): no tar was packed.
Inspect the partial output, then re-run without TRANS10K_LIMIT to build for real:
  ls $TRANS10K_BUILD_ROOT/trans10k/labels/val | head
  python dataset_prep/semseg/dataset_prep.py --trans10k-root $TRANS10K_BUILD_ROOT/trans10k \\
    --out-root /tmp/trans10k_semseg --splits train,val --dry-run
EOF2
  exit 0
fi

echo "Packing $TRANS10K_TAR"
tar -C "$TRANS10K_BUILD_ROOT" -cf "$TRANS10K_TAR.new" trans10k
mv -f "$TRANS10K_TAR.new" "$TRANS10K_TAR"

if [[ "$KEEP_EXTRACTED" != "1" ]]; then
  echo "== Removing the materialized tree ($WORK) (set KEEP_EXTRACTED=1 to keep it) =="
  rm -rf "$WORK"
fi

cat <<EOF2

Done:
  $TRANS10K_TAR

Use it through the normal staging launcher:
  TRANS10K_TAR=$TRANS10K_TAR bash scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh

Or run semseg directly:
  KEEP_EXTRACTED=1 bash scripts/dataset_prep/data_download_scripts/download_datasets_trans10k.sh
  python dataset_prep/semseg/dataset_prep.py --trans10k-root $TRANS10K_BUILD_ROOT/trans10k \\
    --out-root /tmp/trans10k_semseg
EOF2