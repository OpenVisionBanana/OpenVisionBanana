#!/usr/bin/env bash
# Build datasets_archives/sem_segm/coco.tar, the COCO-Object-80 (and COCO-Stuff)
# semantic-segmentation evaluation set on COCO 2017 val (5,000 images):
#
#   dataset/images/val2017/*.jpg                                 cocodataset.org val2017
#   dataset/annotations/val2017/*.png                            COCO-Stuff stuffthingmaps (val)
#   dataset/annotations/panoptic_annotations/panoptic_val2017.json   category ids
#   dataset/annotations/stuff_annotations/stuff_val2017.json
#   dataset/annotations/stuff_annotations/stuff_val2017_pixelmaps/*.png
#
# All sources are public. The official zips are kept with the other COCO 2017
# downloads in shared_seg/coco2017/ (val2017.zip is reused if
# download_datasets_coco.sh already fetched it):
#   val2017.zip, stuffthingmaps_trainval2017.zip,
#   panoptic_annotations_trainval2017.zip, stuff_annotations_trainval2017.zip
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_coco_object80.sh
#   KEEP_DOWNLOADS=1 ...   # keep the extraction work dir
#   FORCE=1 ...            # rebuild an existing coco.tar
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/sem_segm}
COCO2017_ARCHIVE_DIR=${COCO2017_ARCHIVE_DIR:-$(dirname -- "$ARCHIVE_DIR")/shared_seg/coco2017}
KEEP_DOWNLOADS=${KEEP_DOWNLOADS:-0}
FORCE=${FORCE:-0}

VAL_URL=${VAL_URL:-http://images.cocodataset.org/zips/val2017.zip}
STUFFTHING_URL=${STUFFTHING_URL:-http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip}
PANOPTIC_URL=${PANOPTIC_URL:-http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip}
STUFF_URL=${STUFF_URL:-http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip}

OUT_TAR=$ARCHIVE_DIR/coco.tar
DOWNLOADS=$COCO2017_ARCHIVE_DIR
WORK=$ARCHIVE_DIR/_coco_object80_tmp

if [[ "$FORCE" != 1 && -s "$OUT_TAR" ]]; then
  echo "== Using existing $OUT_TAR (FORCE=1 rebuilds) =="
  exit 0
fi
command -v unzip > /dev/null 2>&1 || { echo "ERROR: unzip not found." >&2; exit 1; }
mkdir -p "$DOWNLOADS"

fetch() {
  local url=$1 out=$DOWNLOADS/$(basename -- "$1")
  if [[ ! -s "$out" ]]; then
    echo "== Downloading $(basename -- "$out") ==" >&2
    wget -c -O "$out.partial" "$url" >&2
    mv -f "$out.partial" "$out"
  fi
  printf '%s\n' "$out"
}

val_zip=$(fetch "$VAL_URL")
stuffthing_zip=$(fetch "$STUFFTHING_URL")
panoptic_zip=$(fetch "$PANOPTIC_URL")
stuff_zip=$(fetch "$STUFF_URL")

rm -rf "$WORK"
dataset=$WORK/dataset
mkdir -p "$dataset/images" "$dataset/annotations/panoptic_annotations" \
  "$dataset/annotations/stuff_annotations" "$WORK/unpacked"

echo "== Extracting =="
unzip -q -o "$val_zip" 'val2017/*' -d "$dataset/images"
unzip -q -o "$stuffthing_zip" 'val2017/*' -d "$dataset/annotations"
unzip -q -o -j "$panoptic_zip" 'annotations/panoptic_val2017.json' \
  -d "$dataset/annotations/panoptic_annotations"
unzip -q -o -j "$stuff_zip" 'annotations/stuff_val2017.json' 'annotations/stuff_val2017_pixelmaps.zip' \
  -d "$WORK/unpacked"
mv "$WORK/unpacked/stuff_val2017.json" "$dataset/annotations/stuff_annotations/"
unzip -q -o "$WORK/unpacked/stuff_val2017_pixelmaps.zip" -d "$dataset/annotations/stuff_annotations"

for dir in images/val2017 annotations/val2017 annotations/stuff_annotations/stuff_val2017_pixelmaps; do
  found=$(find "$dataset/$dir" -type f | wc -l)
  echo "   $dir: $found"
  [[ "$found" == 5000 ]] || { echo "ERROR: expected 5000 files in $dir." >&2; exit 1; }
done

echo "== Packing $OUT_TAR =="
tar -cf "$OUT_TAR.partial" -C "$WORK" dataset
mv -f "$OUT_TAR.partial" "$OUT_TAR"
[[ "$KEEP_DOWNLOADS" == 1 ]] || rm -rf "$WORK"
echo "Done: $OUT_TAR"
