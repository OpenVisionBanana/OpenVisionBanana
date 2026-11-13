#!/usr/bin/env bash
# Build datasets_archives/sem_segm/cityscapes.tar from the official Cityscapes
# release (https://www.cityscapes-dataset.com/, registration required):
#
#   cityscapes/leftImg8bit/{train,val}/<city>/*_leftImg8bit.png
#   cityscapes/gtFine/{train,val,test}/<city>/*_gtFine_{polygons.json,labelIds,
#       instanceIds,color,labelTrainIds}.png
#
# The zips are extracted in place, so sem_segm/cityscapes/ is also the extracted
# tree the semseg evaluator reads (it prefers that folder over cityscapes.tar).
#
# The *_labelTrainIds.png masks the semseg converter and evaluator read are not
# shipped; they are rendered from the polygons with the official
# cityscapesscripts createTrainIdLabelImgs. Test images are unlabelled and
# omitted. Cityscapes val is the semseg benchmark; only train enters the
# non-zero-shot pool (CITYSCAPES_SPLITS=train).
#
# Manual download (default):
#   1. Log in at https://www.cityscapes-dataset.com/downloads/ and download
#        gtFine_trainvaltest.zip      (~241 MB)
#        leftImg8bit_trainvaltest.zip (~11 GB)
#   2. Put both in $CITYSCAPES_DIR (default datasets_archives/sem_segm/cityscapes/).
#   3. Run this script.
#
# Requirements: unzip, and a Python with cityscapesscripts
# (pip install cityscapesscripts), selected with PYTHON.
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_cityscapes.sh
#   CITYSCAPES_CSDOWNLOAD=1 ...   # fetch the zips with csDownload (asks for your login)
#   FORCE=1 ...                   # rebuild an existing cityscapes.tar
#
# The downloaded zips are never deleted.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/sem_segm}
CITYSCAPES_DIR=${CITYSCAPES_DIR:-$ARCHIVE_DIR/cityscapes}
CITYSCAPES_CSDOWNLOAD=${CITYSCAPES_CSDOWNLOAD:-0}
PYTHON=${PYTHON:-python3}
FORCE=${FORCE:-0}

OUT_TAR=$ARCHIVE_DIR/cityscapes.tar
ZIPS=(gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip)

if [[ "$FORCE" != 1 && -s "$OUT_TAR" ]]; then
  # The converters and the evaluator read the extracted trees next to the tar;
  # recover them from the tar when only cityscapes.tar was staged.
  if [[ ! -d "$CITYSCAPES_DIR/gtFine" || ! -d "$CITYSCAPES_DIR/leftImg8bit" ]]; then
    echo "== Extracting gtFine/ and leftImg8bit/ from existing $OUT_TAR =="
    tar -xf "$OUT_TAR" -C "$CITYSCAPES_DIR" --strip-components=1 \
      "$(basename -- "$CITYSCAPES_DIR")/gtFine" "$(basename -- "$CITYSCAPES_DIR")/leftImg8bit"
  fi
  echo "== Using existing $OUT_TAR (FORCE=1 rebuilds) =="
  exit 0
fi
command -v unzip > /dev/null 2>&1 || { echo "ERROR: unzip not found." >&2; exit 1; }
"$PYTHON" -c 'import cityscapesscripts.preparation.createTrainIdLabelImgs' 2> /dev/null || {
  echo "ERROR: $PYTHON cannot import cityscapesscripts (pip install cityscapesscripts)." >&2
  exit 1
}
mkdir -p "$CITYSCAPES_DIR"

missing=()
for name in "${ZIPS[@]}"; do
  [[ -s "$CITYSCAPES_DIR/$name" ]] || missing+=("$name")
done
if (( ${#missing[@]} )) && [[ "$CITYSCAPES_CSDOWNLOAD" == 1 ]]; then
  echo "== Fetching ${missing[*]} with csDownload =="
  "$PYTHON" -m cityscapesscripts.download.downloader -d "$CITYSCAPES_DIR" -r "${missing[@]}"
  missing=()
  for name in "${ZIPS[@]}"; do
    [[ -s "$CITYSCAPES_DIR/$name" ]] || missing+=("$name")
  done
fi
if (( ${#missing[@]} )); then
  cat >&2 <<EOF
Cityscapes zip(s) not found in $CITYSCAPES_DIR: ${missing[*]}
Log in at https://www.cityscapes-dataset.com/downloads/, download them into that
folder (or set CITYSCAPES_DIR), then re-run. CITYSCAPES_CSDOWNLOAD=1
fetches them with the official csDownload tool instead.
EOF
  exit 1
fi

rm -rf "$CITYSCAPES_DIR/gtFine" "$CITYSCAPES_DIR/leftImg8bit"
echo "== Extracting gtFine_trainvaltest.zip =="
unzip -q -o "$CITYSCAPES_DIR/gtFine_trainvaltest.zip" 'gtFine/*' -d "$CITYSCAPES_DIR"
echo "== Extracting leftImg8bit train/val from leftImg8bit_trainvaltest.zip =="
unzip -q -o "$CITYSCAPES_DIR/leftImg8bit_trainvaltest.zip" \
  'leftImg8bit/train/*' 'leftImg8bit/val/*' -d "$CITYSCAPES_DIR"

echo "== Rendering *_gtFine_labelTrainIds.png (createTrainIdLabelImgs) =="
CITYSCAPES_DATASET="$CITYSCAPES_DIR" "$PYTHON" -m cityscapesscripts.preparation.createTrainIdLabelImgs
echo

count() { find "$CITYSCAPES_DIR/$1" -type f -name "$2" | wc -l; }
declare -A expected=(
  ["leftImg8bit/train *_leftImg8bit.png"]=2975
  ["leftImg8bit/val *_leftImg8bit.png"]=500
  ["gtFine *_gtFine_labelTrainIds.png"]=5000
)
for key in "${!expected[@]}"; do
  read -r dir pattern <<< "$key"
  found=$(count "$dir" "$pattern")
  echo "   $dir/$pattern: $found"
  [[ "$found" == "${expected[$key]}" ]] || {
    echo "ERROR: expected ${expected[$key]} files for $dir/$pattern." >&2
    exit 1
  }
done

# Pack the extracted tree only, not the zips beside it.
echo "== Packing $OUT_TAR =="
tar -cf "$OUT_TAR.partial" -C "$(dirname -- "$CITYSCAPES_DIR")" \
  --no-recursion "$(basename -- "$CITYSCAPES_DIR")" \
  --recursion "$(basename -- "$CITYSCAPES_DIR")/gtFine" "$(basename -- "$CITYSCAPES_DIR")/leftImg8bit"
mv -f "$OUT_TAR.partial" "$OUT_TAR"
echo "Done: $OUT_TAR"
