#!/usr/bin/env bash
# Download the ADE20K instance-segmentation release (SceneParsing ChallengeData2017,
# 100 thing classes) and pack it for instance-segmentation prep:
#
#   datasets_archives/instanceseg/ade20k_instanceseg/ade20k_instanceseg.tar
#       -> ade20k_instanceseg/{annotations_instance,images}/
#   datasets_archives/instanceseg/ade20k_instanceseg/{instanceInfo100_train.txt,
#       instanceInfo100_val.txt,categoryMapping.txt}
#
# The class-name tables sit beside the tar; staging copies them next to the extract.
# They are not in the release tars; they come from the challenge toolkit
# (github.com/CSAILVision/placeschallenge, instancesegmentation/).
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_ade20k_instance.sh
#   KEEP_DOWNLOADS=1 ...   # keep the downloaded tars and extraction work dir
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/instanceseg}
KEEP_DOWNLOADS=${KEEP_DOWNLOADS:-0}
ADE_ANN_URL=${ADE_ANN_URL:-http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar}
ADE_IMG_URL=${ADE_IMG_URL:-http://sceneparsing.csail.mit.edu/data/ChallengeData2017/images.tar}
ADE_TABLES_URL=${ADE_TABLES_URL:-https://raw.githubusercontent.com/CSAILVision/placeschallenge/de354d418657765406a82b3416da7ee5682a7f1f/instancesegmentation}

ADE_DIR=${ADE_DIR:-$ARCHIVE_DIR/ade20k_instanceseg}
ADE_TAR=$ADE_DIR/ade20k_instanceseg.tar
WORK=$ARCHIVE_DIR/_ade20k_instance_tmp
TABLES=(instanceInfo100_train.txt instanceInfo100_val.txt categoryMapping.txt)

fetch_tables() {
  local name
  for name in "${TABLES[@]}"; do
    [[ -s "$ADE_DIR/$name" ]] && continue
    echo "== Downloading $name =="
    wget -q -O "$ADE_DIR/$name.partial" "$ADE_TABLES_URL/$name"
    mv -f "$ADE_DIR/$name.partial" "$ADE_DIR/$name"
  done
}

mkdir -p "$ADE_DIR"
if [[ -s "$ADE_TAR" ]]; then
  echo "== Using existing $ADE_TAR =="
  fetch_tables
  exit 0
fi

mkdir -p "$WORK/ann" "$WORK/img"
echo "== Downloading ADE20K-instance annotations + images =="
wget -c -O "$WORK/annotations_instance.tar" "$ADE_ANN_URL"
wget -c -O "$WORK/images.tar" "$ADE_IMG_URL"
tar -C "$WORK/ann" -xf "$WORK/annotations_instance.tar"
tar -C "$WORK/img" -xf "$WORK/images.tar"

ann_src=$(find "$WORK/ann" -type d -name annotations_instance -print -quit)
img_src=$(find "$WORK/img" -type d -name images -print -quit)
[[ -n "$ann_src" && -n "$img_src" ]] || {
  echo "ERROR: annotations_instance/ or images/ not found in the ADE20K downloads." >&2
  exit 1
}
payload=$WORK/ade20k_instanceseg
rm -rf "$payload"
mkdir -p "$payload"
mv "$ann_src" "$payload/annotations_instance"
mv "$img_src" "$payload/images"
echo "== Packing $ADE_TAR =="
tar -C "$WORK" -cf "$ADE_TAR.partial" ade20k_instanceseg
mv -f "$ADE_TAR.partial" "$ADE_TAR"

fetch_tables
[[ "$KEEP_DOWNLOADS" == 1 ]] || rm -rf "$WORK"
echo "Done: $ADE_TAR"
