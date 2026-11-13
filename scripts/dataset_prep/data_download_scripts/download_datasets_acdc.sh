#!/usr/bin/env bash
# Build datasets_archives/sem_segm/acdc.tar from the official ACDC release
# (https://acdc.vision.ee.ethz.ch/download, registration required):
#
#   acdc/ACDC/rgb_anon/<condition>/<split>/<sequence>/*_rgb_anon.png
#   acdc/ACDC/gt/<condition>/{train,val}/<sequence>/*_gt_labelTrainIds.png (+ labelIds, ...)
#   acdc/ACDC/gt_panoptic/{train,val}_gt_panoptic.json + <condition>/... panoptic PNGs
#
# Semantic segmentation reads rgb_anon + gt (*_gt_labelTrainIds.png); instance
# segmentation reads gt_panoptic. Both pools use the train and val splits.
#
# Manual download (default):
#   1. Log in at https://acdc.vision.ee.ethz.ch/download and download
#        rgb_anon_trainvaltest.zip   anonymized RGB images
#        gt_trainval.zip             semantic segmentation labels
#        gt_panoptic_trainval.zip    panoptic segmentation labels
#   2. Put them in $ACDC_DIR (default datasets_archives/sem_segm/acdc/).
#   3. Run this script.
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_acdc.sh
#   ACDC_ALLOW_NO_PANOPTIC=1 ...  # semantic only; ACDC then drops out of instanceseg
#   KEEP_DOWNLOADS=1 ...          # keep the extraction work dir
#   FORCE=1 ...                   # rebuild an existing acdc.tar
#
# The downloaded zips are never deleted.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/sem_segm}
ACDC_DIR=${ACDC_DIR:-$ARCHIVE_DIR/acdc}
ACDC_ALLOW_NO_PANOPTIC=${ACDC_ALLOW_NO_PANOPTIC:-0}
KEEP_DOWNLOADS=${KEEP_DOWNLOADS:-0}
FORCE=${FORCE:-0}

OUT_TAR=$ARCHIVE_DIR/acdc.tar
WORK=${WORK:-$ARCHIVE_DIR/_acdc_tmp}
# Labelled train+val frames in the release (1,600 + 406).
EXPECTED_LABELLED=2006

if [[ "$FORCE" != 1 && -s "$OUT_TAR" ]]; then
  echo "== Using existing $OUT_TAR (FORCE=1 rebuilds) =="
  exit 0
fi
command -v unzip > /dev/null 2>&1 || { echo "ERROR: unzip not found." >&2; exit 1; }
mkdir -p "$ACDC_DIR"

find_zip() {
  find "$ACDC_DIR" -maxdepth 2 -type f -iname "$1" -print -quit
}
rgb_zip=$(find_zip 'rgb_anon*.zip')
gt_zip=$(find_zip 'gt_trainval*.zip')
panoptic_zip=$(find_zip 'gt_panoptic*.zip')

missing=()
[[ -n "$rgb_zip" ]] || missing+=(rgb_anon_trainvaltest.zip)
[[ -n "$gt_zip" ]] || missing+=(gt_trainval.zip)
[[ -n "$panoptic_zip" || "$ACDC_ALLOW_NO_PANOPTIC" == 1 ]] || missing+=(gt_panoptic_trainval.zip)
if (( ${#missing[@]} )); then
  cat >&2 <<EOF
ACDC zip(s) not found in $ACDC_DIR: ${missing[*]}
Log in at https://acdc.vision.ee.ethz.ch/download, download them into that
folder (or set ACDC_DIR), then re-run.
EOF
  exit 1
fi

rm -rf "$WORK"
mkdir -p "$WORK/unpacked" "$WORK/acdc/ACDC"
for zip_path in "$rgb_zip" "$gt_zip" ${panoptic_zip:+"$panoptic_zip"}; do
  echo "== Extracting $(basename -- "$zip_path") =="
  unzip -q -o "$zip_path" -d "$WORK/unpacked"
done

# The zips may or may not wrap their folders in a top-level directory.
for name in rgb_anon gt gt_panoptic; do
  [[ "$name" == gt_panoptic && -z "$panoptic_zip" ]] && continue
  src=$(find "$WORK/unpacked" -maxdepth 3 -type d -name "$name" -print -quit)
  [[ -n "$src" ]] || { echo "ERROR: no $name/ folder in the extracted ACDC zips." >&2; exit 1; }
  mv "$src" "$WORK/acdc/ACDC/$name"
done

labelled=$(find "$WORK/acdc/ACDC/gt" -type f -name '*_gt_labelTrainIds.png' | wc -l)
images=$(find "$WORK/acdc/ACDC/rgb_anon" -type f -name '*_rgb_anon.png' -path '*/train/*' -o \
         -type f -name '*_rgb_anon.png' -path '*/val/*' | wc -l)
echo "   train+val: $images RGB images, $labelled labelTrainIds masks"
[[ "$labelled" == "$EXPECTED_LABELLED" ]] || {
  echo "ERROR: expected $EXPECTED_LABELLED *_gt_labelTrainIds.png masks." >&2
  exit 1
}
if [[ -n "$panoptic_zip" ]]; then
  for split in train val; do
    [[ -f "$WORK/acdc/ACDC/gt_panoptic/${split}_gt_panoptic.json" ]] || {
      echo "ERROR: gt_panoptic/${split}_gt_panoptic.json missing." >&2
      exit 1
    }
  done
fi

echo "== Packing $OUT_TAR =="
tar -cf "$OUT_TAR.partial" -C "$WORK" acdc
mv -f "$OUT_TAR.partial" "$OUT_TAR"
[[ "$KEEP_DOWNLOADS" == 1 ]] || rm -rf "$WORK"
echo "Done: $OUT_TAR"
