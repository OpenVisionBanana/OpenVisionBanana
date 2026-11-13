#!/usr/bin/env bash
# Download the official COCO 2014 and 2017 zips (https://cocodataset.org/#download)
# into the shared-segmentation archive folder. The zips are kept as-is; each
# consumer extracts what it needs:
#
#   datasets_archives/shared_seg/coco2014/
#     train2014.zip                  RefCOCO / RefCOCO+ / RefCOCOg / gRefCOCO photos
#     val2014.zip                    LVIS refseg photos
#     annotations_trainval2014.zip
#   datasets_archives/shared_seg/coco2017/
#     train2017.zip                  COCONut-XL (download_datasets_coconut_xlarge.sh)
#     unlabeled2017.zip              COCONut-B images
#     val2017.zip
#     annotations_trainval2017.zip
#   (download_datasets_coco_object80.sh adds the COCO-Stuff and panoptic
#    annotation zips for the COCO-Object-80 benchmark to coco2017/.)
#
# download_datasets_refseg.sh reuses coco2014/train2014.zip, and
# download_datasets_coconut_xlarge.sh links coco2017/{train,unlabeled}2017.zip
# into its download folder instead of fetching them again.
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_coco.sh
#   COCO_FILES=train2014,val2014 ...           # subset of the names above
#   SHARED_SEG_ROOT=/data/ovb/shared_seg ...   # another destination
#   VERIFY=1 ...                               # test each zip after download
#
# About 80 GB in total (unlabeled2017 ~19 GB, train2017 ~19 GB, train2014 ~13 GB).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
SHARED_SEG_ROOT=${SHARED_SEG_ROOT:-$ROOT/datasets_archives/shared_seg}
COCO_FILES=${COCO_FILES:-train2014,val2014,annotations_trainval2014,train2017,unlabeled2017,val2017,annotations_trainval2017}
VERIFY=${VERIFY:-0}

declare -A URLS=(
  [train2014]=http://images.cocodataset.org/zips/train2014.zip
  [val2014]=http://images.cocodataset.org/zips/val2014.zip
  [annotations_trainval2014]=http://images.cocodataset.org/annotations/annotations_trainval2014.zip
  [train2017]=http://images.cocodataset.org/zips/train2017.zip
  [unlabeled2017]=http://images.cocodataset.org/zips/unlabeled2017.zip
  [val2017]=http://images.cocodataset.org/zips/val2017.zip
  [annotations_trainval2017]=http://images.cocodataset.org/annotations/annotations_trainval2017.zip
)

IFS=, read -r -a files <<< "$COCO_FILES"
for name in "${files[@]}"; do
  [[ -n "${URLS[$name]+x}" ]] || {
    echo "ERROR: unknown COCO file '$name'; choose from: ${!URLS[*]}" >&2
    exit 2
  }
done
command -v wget > /dev/null 2>&1 || { echo "ERROR: wget not found." >&2; exit 1; }

for name in "${files[@]}"; do
  year=${name: -4}
  dest_dir=$SHARED_SEG_ROOT/coco$year
  dest=$dest_dir/$name.zip
  mkdir -p "$dest_dir"
  if [[ -s "$dest" ]]; then
    echo "== Using existing $dest =="
    continue
  fi
  # Resumable; the final name appears only after a complete download.
  echo "== Downloading $name.zip -> $dest_dir =="
  wget -c -O "$dest.partial" "${URLS[$name]}"
  if [[ "$VERIFY" == 1 ]]; then
    echo "== Testing $name.zip =="
    unzip -tq "$dest.partial"
  fi
  mv -f "$dest.partial" "$dest"
done

echo
echo "Done. COCO zips are under $SHARED_SEG_ROOT/coco{2014,2017}/."
