#!/usr/bin/env bash
# Build datasets_archives/sem_segm/bdd_10k.tar from the official BDD100K
# 10K semantic-segmentation release (http://bdd-data.berkeley.edu/, registration
# required):
#
#   BDD/images/{train,val,test}/<stem>.jpg
#   BDD/labels/{train,val}/<stem>_train_id.png          Cityscapes 19-class train ids
#   BDD/color_labels/{train,val}/<stem>_train_color.png
#
# The semseg converter pairs BDD/labels/<split>/*_train_id.png with the image of
# the same stem. Two official downloads produce this layout:
#
#   legacy:  bdd100k_seg.zip                         (bdd100k/seg/{images,labels,color_labels})
#   current: bdd100k_images_10k.zip                  (bdd100k/images/10k/{train,val,test})
#            + bdd100k_sem_seg_labels_trainval.zip   (bdd100k/labels/sem_seg/{masks,colormaps})
#
# The paper pool used the legacy layout; the current labels are a later revision,
# so a pool built from them can differ slightly.
#
# Manual download (default):
#   1. Log in at http://bdd-data.berkeley.edu/ and download either the legacy zip or
#      both current zips listed above.
#   2. Put them in $BDD_DIR (default datasets_archives/sem_segm/bdd100k/).
#   3. Run this script.
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_bdd.sh
#   KEEP_DOWNLOADS=1 ...   # keep the extraction work dir
#   FORCE=1 ...            # rebuild an existing bdd_10k.tar
#
# The downloaded zips are never deleted.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/sem_segm}
BDD_DIR=${BDD_DIR:-$ARCHIVE_DIR/bdd100k}
KEEP_DOWNLOADS=${KEEP_DOWNLOADS:-0}
FORCE=${FORCE:-0}

OUT_TAR=$ARCHIVE_DIR/bdd_10k.tar
WORK=${WORK:-$ARCHIVE_DIR/_bdd_tmp}
declare -A EXPECTED=([train]=7000 [val]=1000)

if [[ "$FORCE" != 1 && -s "$OUT_TAR" ]]; then
  echo "== Using existing $OUT_TAR (FORCE=1 rebuilds) =="
  exit 0
fi
command -v unzip > /dev/null 2>&1 || { echo "ERROR: unzip not found." >&2; exit 1; }
mkdir -p "$BDD_DIR"

find_zip() {
  find "$BDD_DIR" -maxdepth 2 -type f -name "$1" -print -quit
}
legacy_zip=$(find_zip 'bdd100k_seg.zip')
images_zip=$(find_zip 'bdd100k_images_10k.zip')
labels_zip=$(find_zip 'bdd100k_sem_seg_labels_trainval.zip')

if [[ -z "$legacy_zip" && ( -z "$images_zip" || -z "$labels_zip" ) ]]; then
  cat >&2 <<EOF
BDD100K zips not found in $BDD_DIR. Log in at http://bdd-data.berkeley.edu/
and download either
  bdd100k_seg.zip                                  (legacy, as used for the paper)
or both
  bdd100k_images_10k.zip + bdd100k_sem_seg_labels_trainval.zip
into that folder (or set BDD_DIR), then re-run.
EOF
  exit 1
fi

rm -rf "$WORK"
mkdir -p "$WORK/unpacked" "$WORK/BDD"

find_dir() {
  # Print the directory matching path pattern $1 under the extracted zips.
  find "$WORK/unpacked" -type d -path "$1" -print -quit
}

if [[ -n "$legacy_zip" ]]; then
  echo "== Extracting $(basename -- "$legacy_zip") (legacy layout) =="
  unzip -q -o "$legacy_zip" -d "$WORK/unpacked"
  seg=$(find_dir '*/seg')
  [[ -n "$seg" && -d "$seg/images" && -d "$seg/labels" ]] || {
    echo "ERROR: $legacy_zip has no seg/{images,labels} tree." >&2
    exit 1
  }
  for name in images labels color_labels; do
    [[ -d "$seg/$name" ]] && mv "$seg/$name" "$WORK/BDD/$name"
  done
else
  echo "== Extracting $(basename -- "$images_zip") and $(basename -- "$labels_zip") (current layout) =="
  unzip -q -o "$images_zip" -d "$WORK/unpacked"
  unzip -q -o "$labels_zip" -d "$WORK/unpacked"
  images=$(find_dir '*/images/10k')
  masks=$(find_dir '*/sem_seg/masks')
  colormaps=$(find_dir '*/sem_seg/colormaps')
  [[ -n "$images" && -n "$masks" ]] || {
    echo "ERROR: expected images/10k/ and labels/sem_seg/masks/ in the BDD100K zips." >&2
    exit 1
  }
  mv "$images" "$WORK/BDD/images"
  # Rename to the legacy suffixes the converter pairs on.
  for split in train val; do
    mkdir -p "$WORK/BDD/labels/$split" "$WORK/BDD/color_labels/$split"
    for mask in "$masks/$split"/*.png; do
      stem=$(basename -- "$mask" .png)
      mv "$mask" "$WORK/BDD/labels/$split/${stem}_train_id.png"
    done
    if [[ -n "$colormaps" && -d "$colormaps/$split" ]]; then
      for color in "$colormaps/$split"/*.png; do
        stem=$(basename -- "$color" .png)
        mv "$color" "$WORK/BDD/color_labels/$split/${stem}_train_color.png"
      done
    fi
  done
fi

for split in train val; do
  labels=$(find "$WORK/BDD/labels/$split" -type f -name '*_train_id.png' | wc -l)
  unmatched=0
  while IFS= read -r mask; do
    stem=$(basename -- "$mask" _train_id.png)
    [[ -f "$WORK/BDD/images/$split/$stem.jpg" ]] || ((unmatched += 1))
  done < <(find "$WORK/BDD/labels/$split" -type f -name '*_train_id.png')
  echo "   $split: $labels masks, $unmatched without an image"
  [[ "$labels" == "${EXPECTED[$split]}" && "$unmatched" == 0 ]] || {
    echo "ERROR: expected ${EXPECTED[$split]} masks with matching images for $split." >&2
    exit 1
  }
done

echo "== Packing $OUT_TAR =="
tar -cf "$OUT_TAR.partial" -C "$WORK" BDD
mv -f "$OUT_TAR.partial" "$OUT_TAR"
[[ "$KEEP_DOWNLOADS" == 1 ]] || rm -rf "$WORK"
echo "Done: $OUT_TAR"
