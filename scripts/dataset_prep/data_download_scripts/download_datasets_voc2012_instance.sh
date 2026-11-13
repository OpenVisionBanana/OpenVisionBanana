#!/usr/bin/env bash
# Download Pascal VOC 2012 and pack the full VOCdevkit for instance-segmentation
# prep. Unlike semseg's voc.tar it keeps SegmentationObject (per-pixel instance ids):
#
#   datasets_archives/instanceseg/VOC2012.tar
#       -> VOCdevkit/VOC2012/{JPEGImages,SegmentationObject,SegmentationClass,ImageSets,...}
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_voc2012_instance.sh
#   KEEP_DOWNLOADS=1 ...   # keep the downloaded tar and extraction work dir
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/instanceseg}
KEEP_DOWNLOADS=${KEEP_DOWNLOADS:-0}
VOC_URL=${VOC_URL:-http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar}

VOC2012_TAR=$ARCHIVE_DIR/VOC2012.tar
WORK=$ARCHIVE_DIR/_voc2012_instance_tmp

if [[ -s "$VOC2012_TAR" ]]; then
  echo "== Using existing $VOC2012_TAR =="
  exit 0
fi

mkdir -p "$WORK"
echo "== Downloading Pascal VOC 2012 trainval (~2 GB) =="
wget -c -O "$WORK/VOCtrainval_11-May-2012.tar" "$VOC_URL"
rm -rf "$WORK/VOCdevkit"
tar -C "$WORK" -xf "$WORK/VOCtrainval_11-May-2012.tar"
[[ -d "$WORK/VOCdevkit/VOC2012/SegmentationObject" ]] || {
  echo "ERROR: VOCdevkit/VOC2012/SegmentationObject missing after extraction." >&2
  exit 1
}
echo "== Packing $VOC2012_TAR =="
tar -C "$WORK" -cf "$VOC2012_TAR.partial" VOCdevkit
mv -f "$VOC2012_TAR.partial" "$VOC2012_TAR"
[[ "$KEEP_DOWNLOADS" == 1 ]] || rm -rf "$WORK"
echo "Done: $VOC2012_TAR"
