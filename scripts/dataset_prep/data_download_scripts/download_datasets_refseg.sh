#!/usr/bin/env bash
# Download the referring-segmentation EVAL datasets (RefCOCOg UMD val + ReasonSeg
# val) and arrange them into the exact layouts dataset_prep/refseg/prepare_refseg_eval.py
# expects. Unlike the semseg sources these are kept as extracted directories under
# datasets_archives/refseg/ (the refseg eval job reads them in place; no .tar staging):
#
#   datasets_archives/refseg/refcocog/refs(umd).p          (--data-dir)
#   datasets_archives/refseg/refcocog/instances.json
#   datasets_archives/refseg/refcocog/train2014/*.jpg      (--images-dir)
#   datasets_archives/refseg/reasonseg_val/<stem>.jpg+.json (--data-dir)
#
# After this, rasterize the masks with:
#   scripts/dataset_prep/prepare_refseg_eval.sh
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_refseg.sh
#   REASONSEG_ZIP=/path/to/val.zip ...      # use an already-downloaded ReasonSeg val.zip
#   REASONSEG_GDRIVE_ID=<file_id> ...       # fetch ReasonSeg val.zip via gdown by file id
#   KEEP_DOWNLOADS=1 ...                     # keep the _download_tmp/ working dir
#   SKIP_REFCOCOG=1 ...                      # only handle ReasonSeg
#   SKIP_REASONSEG=1 ...                     # only handle RefCOCOg
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/refseg}
WORK=${WORK:-$ARCHIVE_DIR/_download_tmp}
mkdir -p "$ARCHIVE_DIR" "$WORK"

# ---------------------------------------------------------------------------- #
# RefCOCOg (UMD split) -- referring expressions over COCO train2014 images.
# refcocog.zip (licheng "refer" release) extracts a refcocog/ dir with the
# pickled refs + instances.json; the images come from the COCO train2014 zip.
# build via prepare_refseg_eval.py refcocog --data-dir <refcocog> --images-dir <train2014>.
# ---------------------------------------------------------------------------- #
REFCOCOG_DIR="$ARCHIVE_DIR/refcocog"
REFCOCOG_ZIP_URL=${REFCOCOG_ZIP_URL:-https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip}
if [[ "${SKIP_REFCOCOG:-0}" != "1" ]]; then
if [[ ! -f "$REFCOCOG_DIR/refs(umd).p" || ! -f "$REFCOCOG_DIR/instances.json" ]]; then
  echo "== Downloading RefCOCOg annotations (refcocog.zip) =="
  wget -c -O "$WORK/refcocog.zip" "$REFCOCOG_ZIP_URL"
  echo "== Extracting refcocog.zip =="
  unzip -q -o "$WORK/refcocog.zip" -d "$ARCHIVE_DIR"   # -> $ARCHIVE_DIR/refcocog/
else
  echo "== RefCOCOg annotations already present, skipping =="
fi

TRAIN2014_URL=${TRAIN2014_URL:-http://images.cocodataset.org/zips/train2014.zip}
# Reuse the shared COCO zip from download_datasets_coco.sh when present.
TRAIN2014_ZIP=${TRAIN2014_ZIP:-$ROOT/datasets_archives/shared_seg/coco2014/train2014.zip}
if ! compgen -G "$REFCOCOG_DIR/train2014/COCO_train2014_*.jpg" > /dev/null 2>&1; then
  if [[ -s "$TRAIN2014_ZIP" ]]; then
    train2014_zip=$TRAIN2014_ZIP
    echo "== Using COCO train2014 images from $train2014_zip =="
  else
    echo "== Downloading COCO train2014 images (~13 GB) =="
    train2014_zip=$WORK/train2014.zip
    wget -c -O "$train2014_zip" "$TRAIN2014_URL"
  fi
  echo "== Extracting train2014.zip =="
  unzip -q -o "$train2014_zip" -d "$WORK"          # -> $WORK/train2014/
  mkdir -p "$REFCOCOG_DIR"
  rm -rf "$REFCOCOG_DIR/train2014"
  mv "$WORK/train2014" "$REFCOCOG_DIR/train2014"
else
  echo "== COCO train2014 images already present, skipping =="
fi
echo "   -> RefCOCOg ready: $REFCOCOG_DIR  (refs(umd).p, instances.json, train2014/)"
else
  echo "== RefCOCOg skipped (SKIP_REFCOCOG=1) =="
fi

# ---------------------------------------------------------------------------- #
# ReasonSeg val (LISA release) -- <stem>.jpg next to a <stem>.json LabelMe file.
# Distributed as val.zip in a Google Drive folder; not directly wget-able, so we
# accept a local zip (REASONSEG_ZIP) or fetch it with gdown by file id
# (REASONSEG_GDRIVE_ID). The zip extracts a val/ dir that we flatten to reasonseg_val/.
#   Google Drive folder: https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy
# ---------------------------------------------------------------------------- #
REASONSEG_DIR="$ARCHIVE_DIR/reasonseg_val"
if [[ "${SKIP_REASONSEG:-0}" != "1" ]]; then
if ! compgen -G "$REASONSEG_DIR/*.json" > /dev/null 2>&1; then
  zip_path=""
  if [[ -n "${REASONSEG_ZIP:-}" ]]; then
    zip_path="$REASONSEG_ZIP"
  elif [[ -n "${REASONSEG_GDRIVE_ID:-}" ]]; then
    echo "== Fetching ReasonSeg val.zip via gdown (id=$REASONSEG_GDRIVE_ID) =="
    if ! command -v gdown > /dev/null 2>&1; then
      echo "ERROR: gdown not installed (pip install gdown), or set REASONSEG_ZIP to a local val.zip." >&2
      exit 1
    fi
    gdown -O "$WORK/val.zip" "$REASONSEG_GDRIVE_ID"
    zip_path="$WORK/val.zip"
  else
    cat >&2 <<EOF
ReasonSeg val.zip lives in a Google Drive folder and cannot be wget'd directly:
  https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy
Re-run with one of:
  REASONSEG_ZIP=/path/to/val.zip        (already downloaded)
  REASONSEG_GDRIVE_ID=<file_id>         (fetch via gdown; pip install gdown)
EOF
    exit 1
  fi
  echo "== Extracting ReasonSeg val.zip =="
  unzip -q -o "$zip_path" -d "$WORK"
  # The zip nests the pairs under val/ (LISA layout); flatten to reasonseg_val/.
  src_dir="$WORK/val"
  [[ -d "$src_dir" ]] || src_dir=$(dirname "$(find "$WORK" -name '*.json' -path '*val*' | head -1)")
  mkdir -p "$REASONSEG_DIR"
  cp -r "$src_dir"/. "$REASONSEG_DIR"/
else
  echo "== ReasonSeg val already present, skipping =="
fi
echo "   -> ReasonSeg ready: $REASONSEG_DIR  (<stem>.jpg/.json)"
else
  echo "== ReasonSeg skipped (SKIP_REASONSEG=1) =="
fi

if [[ "${KEEP_DOWNLOADS:-0}" != "1" ]]; then
  echo "== Cleaning up $WORK (set KEEP_DOWNLOADS=1 to keep) =="
  rm -rf "$WORK"
fi

cat <<EOF

Done. Selected eval sources are arranged under $ARCHIVE_DIR.
Next: rasterize the GT masks with
  scripts/dataset_prep/prepare_refseg_eval.sh
EOF
