#!/usr/bin/env bash
# Add PASCAL-Context 59-class semantic labels to the existing voc.tar.
#
# PASCAL-Context re-annotates the PASCAL VOC 2010 trainval images with whole-scene
# labels. The images themselves are already inside voc.tar (VOC2012 JPEGImages is a
# superset of VOC2010 trainval), so this script reuses them instead of re-downloading
# 2 GB: it extracts voc.tar, drops the context labels in beside the VOC 21-class
# masks, and repacks the same tar.
#
# Ground truth comes from the Stanford trainval.tar.gz (.mat LabelMaps over the full
# 459-class vocabulary). NOTE: the sibling 59_context_labels.tar.gz on that page is
# NOT ground truth -- the page describes it as "the segmentation results ... the
# result of a 59-way classifier" on the val subset. It is downloaded here only to
# recover the official train/val id split (its 5105 file names ARE the val ids);
# its pixels are never used. Set PASCAL_CONTEXT_SPLIT=all to skip that download and
# put every image in train.
#
# The 59 classes and their palette follow mmsegmentation (index 0 = background,
# 1..59 alphabetical); pascal_context.py is vendored next to the labels so the
# class->color mapping travels with the data.
#
# Output (inside the repacked voc.tar):
#   voc/images/*.jpg                                   (unchanged, shared with VOC)
#   voc/SegmentationClass/, voc/sets/                   (unchanged VOC 21-class data)
#   voc/pascal_context_59classlabels/labels/*.png       uint8, 0=background, 1..59
#   voc/pascal_context_59classlabels/sets/{train,val}.txt
#   voc/pascal_context_59classlabels/pascal_context.py  mmseg classes + palette
#   voc/pascal_context_59classlabels/59_labels.txt      official Stanford name list
#   voc/pascal_context_59classlabels/{id2label,source}.json
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_pascal_context.sh
#   WORK=$TMPDIR/pc_download ...          # use node-local scratch (needs ~6 GB free)
#   PASCAL_CONTEXT_LIMIT=32 ...           # quick smoke run (converts N images only)
#   PASCAL_CONTEXT_SPLIT=all ...          # skip the val-id download; everything -> train
#   PASCAL_CONTEXT_WORKERS=8 ...          # .mat -> png conversion threads
#   KEEP_EXTRACTED=1 ...                  # keep the extracted voc/ tree beside the tar
#   FORCE=1 ...                           # rebuild even if the labels are already in the tar
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/sem_segm}
TMP_PARENT=${TMPDIR:-$ARCHIVE_DIR}
WORK=${WORK:-$TMP_PARENT/open_visionbanana_pascal_context_download}
VOC_TAR=${VOC_TAR:-$ARCHIVE_DIR/voc.tar}
VOC_BUILD_ROOT=${VOC_BUILD_ROOT:-$WORK/voc_build}
PC_SRC=${PC_SRC:-$WORK/pascal_context_src}
PYTHON=${PYTHON:-python}
FORCE=${FORCE:-0}
KEEP_EXTRACTED=${KEEP_EXTRACTED:-0}
PASCAL_CONTEXT_LIMIT=${PASCAL_CONTEXT_LIMIT:-}
PASCAL_CONTEXT_SPLIT=${PASCAL_CONTEXT_SPLIT:-official}
PASCAL_CONTEXT_WORKERS=${PASCAL_CONTEXT_WORKERS:-8}

STANFORD_BASE=${STANFORD_BASE:-https://cs.stanford.edu/~roozbeh/pascal-context}
PC_TRAINVAL_URL=${PC_TRAINVAL_URL:-$STANFORD_BASE/trainval.tar.gz}
PC_59LABELS_URL=${PC_59LABELS_URL:-$STANFORD_BASE/59_labels.txt}
PC_VALIDS_URL=${PC_VALIDS_URL:-$STANFORD_BASE/59_context_labels.tar.gz}
# Pinned mmsegmentation revision.
MMSEG_PASCAL_CONTEXT_URL=${MMSEG_PASCAL_CONTEXT_URL:-https://raw.githubusercontent.com/open-mmlab/mmsegmentation/b040e147adfa027bbc071b624bedf0ae84dfc922/mmseg/datasets/pascal_context.py}
VOC_URL=${VOC_URL:-https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar}

PC_DIR_NAME=pascal_context_59classlabels
VOC_DIR="$VOC_BUILD_ROOT/voc"
PC_DIR="$VOC_DIR/$PC_DIR_NAME"

mkdir -p "$ARCHIVE_DIR" "$WORK" "$VOC_BUILD_ROOT" "$PC_SRC"

if [[ "$FORCE" != "1" && -f "$VOC_TAR" ]] && tar -tf "$VOC_TAR" "voc/$PC_DIR_NAME/labels" >/dev/null 2>&1; then
  echo "voc.tar already carries $PC_DIR_NAME: $VOC_TAR"
  echo "Set FORCE=1 to rebuild it."
  exit 0
fi

# --------------------------------------------------------------------------- #
# 1. VOC images: reuse the existing tar, or build one from the VOC2012 release. #
# --------------------------------------------------------------------------- #
if [[ -f "$VOC_TAR" ]]; then
  echo "== Extracting existing $VOC_TAR =="
  rm -rf "$VOC_DIR"
  tar -C "$VOC_BUILD_ROOT" -xf "$VOC_TAR"
  [[ -d "$VOC_DIR/images" ]] || { echo "ERROR: $VOC_TAR has no voc/images/ directory." >&2; exit 1; }
else
  echo "== No $VOC_TAR; downloading Pascal-VOC 2012 trainval (~2 GB) =="
  wget -nv -c -O "$WORK/VOCtrainval_11-May-2012.tar" "$VOC_URL"
  echo "== Extracting =="
  tar -C "$WORK" -xf "$WORK/VOCtrainval_11-May-2012.tar"
  V="$WORK/VOCdevkit/VOC2012"
  mkdir -p "$VOC_DIR"
  # Same layout build_voc_pairs() expects (see download_datasets_semseg_ade_voc.sh).
  cp -r "$V/JPEGImages"             "$VOC_DIR/images"
  cp -r "$V/SegmentationClass"      "$VOC_DIR/SegmentationClass"
  cp -r "$V/ImageSets/Segmentation" "$VOC_DIR/sets"
fi

# --------------------------------------------------------------------------- #
# 2. PASCAL-Context sources.                                                   #
# --------------------------------------------------------------------------- #
mkdir -p "$PC_DIR"
echo "== Downloading PASCAL-Context ground truth (~31 MB) =="
wget -nv -c -O "$PC_SRC/trainval.tar.gz" "$PC_TRAINVAL_URL"
if [[ ! -d "$PC_SRC/trainval" ]]; then
  tar -C "$PC_SRC" -xzf "$PC_SRC/trainval.tar.gz"
fi

echo "== Downloading the 59-class name list + mmseg class/color mapping =="
wget -nv -c -O "$PC_DIR/59_labels.txt" "$PC_59LABELS_URL"
wget -nv -c -O "$PC_DIR/pascal_context.py" "$MMSEG_PASCAL_CONTEXT_URL"

VAL_IDS_TAR=""
if [[ "$PASCAL_CONTEXT_SPLIT" == "official" ]]; then
  # Downloaded for its FILE NAMES only -- these are the official val ids.
  echo "== Downloading the official val id list (~14 MB, names only) =="
  wget -nv -c -O "$PC_SRC/59_context_labels.tar.gz" "$PC_VALIDS_URL"
  VAL_IDS_TAR="$PC_SRC/59_context_labels.tar.gz"
fi

# --------------------------------------------------------------------------- #
# 3. Convert the 459-class .mat LabelMaps to 59-class PNGs.                    #
# --------------------------------------------------------------------------- #
env \
  PC_SRC="$PC_SRC" \
  PC_DIR="$PC_DIR" \
  VOC_DIR="$VOC_DIR" \
  VAL_IDS_TAR="$VAL_IDS_TAR" \
  PASCAL_CONTEXT_LIMIT="$PASCAL_CONTEXT_LIMIT" \
  PASCAL_CONTEXT_WORKERS="$PASCAL_CONTEXT_WORKERS" \
  PC_TRAINVAL_URL="$PC_TRAINVAL_URL" \
  MMSEG_PASCAL_CONTEXT_URL="$MMSEG_PASCAL_CONTEXT_URL" \
  "$PYTHON" - <<'PY'
import json
import os
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from scipy.io import loadmat
except ImportError as exc:
    raise SystemExit(
        "Missing Python package 'scipy' (needed to read the .mat LabelMaps). "
        "Install it in the active environment or run inside the data-prep container."
    ) from exc

pc_src = Path(os.environ["PC_SRC"])
pc_dir = Path(os.environ["PC_DIR"])
voc_dir = Path(os.environ["VOC_DIR"])
limit_text = os.environ.get("PASCAL_CONTEXT_LIMIT", "").strip()
limit = int(limit_text) if limit_text else None
workers = max(1, int(os.environ.get("PASCAL_CONTEXT_WORKERS", "8")))

# --- the 60 class names + palette, parsed from the vendored mmseg module ------ #
src = (pc_dir / "pascal_context.py").read_text(encoding="utf-8")
classes_block = re.search(r"classes=\((.*?)\),\s*\n\s*palette=", src, re.S)
palette_block = re.search(r"palette=\[(.*?)\]\)", src, re.S)
if not classes_block or not palette_block:
    raise SystemExit(
        f"Could not parse classes/palette out of pascal_context.py "
        f"(downloaded from {os.environ['MMSEG_PASCAL_CONTEXT_URL']})."
    )
class_names = re.findall(r"'([^']+)'", classes_block.group(1))
palette = [[int(v) for v in row.split(",")]
           for row in re.findall(r"\[([0-9,\s]+)\]", palette_block.group(1))]
if len(class_names) != 60:
    raise SystemExit(f"Expected 60 mmseg classes (background + 59), got {len(class_names)}.")

# --- 459-class GT name -> id, from the labels.txt inside trainval.tar.gz ------ #
gt_name_to_id = {}
for line in (pc_src / "labels.txt").read_text(encoding="utf-8").splitlines():
    if ":" not in line:
        continue
    raw_id, name = line.split(":", 1)
    gt_name_to_id[name.strip()] = int(raw_id)

missing = [n for n in class_names[1:] if n not in gt_name_to_id]
if missing:
    raise SystemExit(f"These mmseg classes have no match in the 459-class GT vocabulary: {missing}")

# Everything outside the 59 selected classes collapses to background (0), which is
# how the 60-way PASCAL-Context benchmark is defined.
lut = np.zeros(max(gt_name_to_id.values()) + 1, dtype=np.uint8)
for new_idx, name in enumerate(class_names[1:], start=1):
    lut[gt_name_to_id[name]] = new_idx

# --- split ids --------------------------------------------------------------- #
mat_paths = sorted((pc_src / "trainval").glob("*.mat"))
if not mat_paths:
    raise SystemExit(f"No .mat annotations under {pc_src / 'trainval'}")
all_stems = [p.stem for p in mat_paths]

val_stems: set[str] = set()
val_ids_tar = os.environ.get("VAL_IDS_TAR", "").strip()
if val_ids_tar:
    with tarfile.open(val_ids_tar) as tf:
        val_stems = {Path(n).stem for n in tf.getnames() if n.endswith(".png")}
    unknown = val_stems - set(all_stems)
    if unknown:
        raise SystemExit(f"{len(unknown)} val ids are absent from the GT annotations, e.g. {sorted(unknown)[:3]}")
    print(f"Official split: {len(all_stems) - len(val_stems)} train / {len(val_stems)} val")
else:
    print(f"PASCAL_CONTEXT_SPLIT=all: every one of the {len(all_stems)} images goes to train")

if limit is not None:
    mat_paths = mat_paths[:limit]

# --- convert ----------------------------------------------------------------- #
label_dir = pc_dir / "labels"
label_dir.mkdir(parents=True, exist_ok=True)


def convert(mat_path: Path) -> tuple[str, bool]:
    """.mat LabelMap (459-class uint16) -> uint8 PNG in the 60-class index space."""
    raw = loadmat(mat_path)["LabelMap"].astype(np.int64)
    if raw.max(initial=0) >= len(lut):
        raise ValueError(f"{mat_path.name} holds label id {int(raw.max())} beyond the 459-class vocabulary")
    Image.fromarray(lut[raw]).save(label_dir / f"{mat_path.stem}.png")
    # The image must exist in voc/images/ for the pair to be usable downstream.
    return mat_path.stem, (voc_dir / "images" / f"{mat_path.stem}.jpg").exists()


with ThreadPoolExecutor(max_workers=workers) as ex:
    results = list(ex.map(convert, mat_paths))

written = [stem for stem, _ in results]
orphans = [stem for stem, has_image in results if not has_image]
print(f"Converted {len(written)} label maps -> {label_dir}")
if orphans:
    print(f"WARNING: {len(orphans)} labels have no matching voc/images/*.jpg, e.g. {orphans[:3]}")
    if len(orphans) > len(written) * 0.02:
        raise SystemExit("Too many labels without images -- is voc.tar missing the VOC2010 JPEGs?")

# --- sets, id2label, provenance ---------------------------------------------- #
sets_dir = pc_dir / "sets"
sets_dir.mkdir(parents=True, exist_ok=True)
written_set = set(written)
train_ids = sorted(s for s in written_set if s not in val_stems)
val_ids = sorted(s for s in written_set if s in val_stems)
(sets_dir / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
(sets_dir / "val.txt").write_text("\n".join(val_ids) + "\n" if val_ids else "", encoding="utf-8")
print(f"Wrote sets/train.txt ({len(train_ids)}) and sets/val.txt ({len(val_ids)})")

(pc_dir / "id2label.json").write_text(
    json.dumps({str(i): name for i, name in enumerate(class_names)}, indent=2) + "\n",
    encoding="utf-8",
)
(pc_dir / "palette.json").write_text(json.dumps(palette, indent=2) + "\n", encoding="utf-8")
(pc_dir / "source.json").write_text(
    json.dumps(
        {
            "annotations": os.environ["PC_TRAINVAL_URL"],
            "class_mapping": os.environ["MMSEG_PASCAL_CONTEXT_URL"],
            "images": "voc/images (VOC2012 JPEGImages; a superset of VOC2010 trainval)",
            "num_classes": len(class_names),
            "counts": {"train": len(train_ids), "val": len(val_ids)},
            "format": (
                "labels/*.png are uint8, pixel value = class index: 0 = background, "
                "1..59 = the mmseg PascalContext classes in METAINFO order. Classes "
                "outside the 59 are folded into background, per the standard 60-way "
                "PASCAL-Context protocol."
            ),
            "split_source": (
                "official val ids recovered from 59_context_labels.tar.gz file names"
                if val_ids_tar else "PASCAL_CONTEXT_SPLIT=all (no val split)"
            ),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

# --------------------------------------------------------------------------- #
# 4. Repack voc.tar (write beside it first, then swap, so a failure mid-write  #
#    can't destroy the existing archive).                                      #
#                                                                              #
# A PASCAL_CONTEXT_LIMIT run converts only a handful of images, so it NEVER    #
# repacks -- otherwise a smoke test would replace a complete voc.tar with a    #
# partial one. It leaves the extracted tree in place to inspect instead.       #
# --------------------------------------------------------------------------- #
if [[ -n "$PASCAL_CONTEXT_LIMIT" ]]; then
  cat <<EOF2

Smoke run (PASCAL_CONTEXT_LIMIT=$PASCAL_CONTEXT_LIMIT): $VOC_TAR was NOT modified.
Inspect the partial output, then re-run without PASCAL_CONTEXT_LIMIT to build for real:
  ls $PC_DIR/labels | head
  python dataset_prep/semseg/dataset_prep.py --pascal-context-root $VOC_DIR \\
    --out-root /tmp/pascal_context_semseg --splits train,val --dry-run
EOF2
  exit 0
fi

echo "== Repacking $VOC_TAR =="
tar -C "$VOC_BUILD_ROOT" -cf "$VOC_TAR.new" voc
mv -f "$VOC_TAR.new" "$VOC_TAR"

if [[ "$KEEP_EXTRACTED" != "1" ]]; then
  echo "== Removing the extracted tree ($WORK) (set KEEP_EXTRACTED=1 to keep it) =="
  rm -rf "$WORK"
fi

cat <<EOF2

Done:
  $VOC_TAR   (voc/ + voc/$PC_DIR_NAME/)

The VOC 21-class data is untouched, so the existing --voc-root flow still works:
  VOC_TAR=$VOC_TAR bash scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh

Feed the context labels to semseg prep:
  python dataset_prep/semseg/dataset_prep.py --pascal-context-root <extracted voc dir> \\
    --out-root /tmp/pascal_context_semseg --splits train,val
EOF2
