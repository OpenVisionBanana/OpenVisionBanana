#!/usr/bin/env bash
# Download ADE20k (150-class SceneParsing) + Pascal-VOC 2012 for semseg dataset
# prep, arrange them into the exact layouts dataset_prep/semseg/dataset_prep.py
# expects, and pack each as a .tar in datasets_archives/sem_segm/ (same convention
# as cityscapes.tar / acdc.tar / bdd_10k.tar).
#
# After running, the prepared roots are:
#   datasets_archives/sem_segm/ADEChallengeData2016/   (--ade20k-root)
#   datasets_archives/sem_segm/voc/                    (--voc-root)
# and the tars:
#   datasets_archives/sem_segm/ade20k.tar
#   datasets_archives/sem_segm/voc.tar
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_semseg_ade_voc.sh
#   WANT_VOC_AUG=1 ...             # also fetch SBD augmented masks (trainaug split)
#   KEEP_EXTRACTED=1 ...          # keep the extracted dirs after tarring
#   SKIP_ADE=1 / SKIP_VOC=1 ...    # build only one of the two archives
#   FORCE=1 ...                   # repack an existing ade20k.tar / voc.tar
#
# An existing voc.tar is kept by default: download_datasets_pascal_context.sh adds
# the PASCAL-Context labels to it, and repacking from voc/ would drop them.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/sem_segm}
WORK=${WORK:-$ARCHIVE_DIR/_download_tmp}
SKIP_ADE=${SKIP_ADE:-0}
SKIP_VOC=${SKIP_VOC:-0}
FORCE=${FORCE:-0}
mkdir -p "$ARCHIVE_DIR" "$WORK"

# ---------------------------------------------------------------------------- #
# ADE20k -- ADEChallengeData2016 (the 150-class scene-parsing benchmark).
# Layout it ships in == exactly what build_ade20k_pairs() reads:
#   ADEChallengeData2016/images/{training,validation}/*.jpg
#   ADEChallengeData2016/annotations/{training,validation}/*.png   (pixel = class id; 0 = ignore, 1..150)
# ---------------------------------------------------------------------------- #
ADE_URL="http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
ADE_DIR="$ARCHIVE_DIR/ADEChallengeData2016"
if [[ "$SKIP_ADE" == "1" ]]; then
  echo "== ADE20k skipped (SKIP_ADE=1) =="
elif [[ "$FORCE" != "1" && -s "$ARCHIVE_DIR/ade20k.tar" ]]; then
  echo "== Using existing $ARCHIVE_DIR/ade20k.tar (FORCE=1 repacks) =="
else
  if [[ ! -d "$ADE_DIR/images/training" ]]; then
    echo "== Downloading ADEChallengeData2016 (~1 GB) =="
    wget -c -O "$WORK/ADEChallengeData2016.zip" "$ADE_URL"
    echo "== Extracting =="
    unzip -q -o "$WORK/ADEChallengeData2016.zip" -d "$ARCHIVE_DIR"
  else
    echo "== ADEChallengeData2016 already present, skipping download =="
  fi
  echo "== Packing ade20k.tar =="
  tar -C "$ARCHIVE_DIR" -cf "$ARCHIVE_DIR/ade20k.tar.partial" ADEChallengeData2016
  mv -f "$ARCHIVE_DIR/ade20k.tar.partial" "$ARCHIVE_DIR/ade20k.tar"
  echo "   -> $ARCHIVE_DIR/ade20k.tar   (--ade20k-root $ADE_DIR)"
fi

# ---------------------------------------------------------------------------- #
# Pascal-VOC 2012. The raw VOCdevkit/VOC2012 has JPEGImages / SegmentationClass /
# ImageSets/Segmentation; build_voc_pairs() wants those renamed into:
#   voc/images/*.jpg              (<- JPEGImages)
#   voc/SegmentationClass/*.png   (<- SegmentationClass; palette PNG, index = class id, 255 = void)
#   voc/sets/{train,val}.txt      (<- ImageSets/Segmentation)
# Optionally voc/SegmentationClassAug/ + voc/sets/trainaug.txt (SBD) for trainaug.
# ---------------------------------------------------------------------------- #
VOC_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
VOC_DIR="$ARCHIVE_DIR/voc"
if [[ "$SKIP_VOC" == "1" ]]; then
  echo "== Pascal-VOC skipped (SKIP_VOC=1) =="
elif [[ "$FORCE" != "1" && -s "$ARCHIVE_DIR/voc.tar" ]]; then
  echo "== Using existing $ARCHIVE_DIR/voc.tar (FORCE=1 repacks) =="
else
if [[ ! -d "$VOC_DIR/SegmentationClass" ]]; then
  echo "== Downloading Pascal-VOC 2012 trainval (~2 GB) =="
  wget -c -O "$WORK/VOCtrainval_11-May-2012.tar" "$VOC_URL"
  echo "== Extracting =="
  tar -C "$WORK" -xf "$WORK/VOCtrainval_11-May-2012.tar"
  V="$WORK/VOCdevkit/VOC2012"
  mkdir -p "$VOC_DIR"
  cp -r "$V/JPEGImages"               "$VOC_DIR/images"
  cp -r "$V/SegmentationClass"        "$VOC_DIR/SegmentationClass"
  cp -r "$V/ImageSets/Segmentation"   "$VOC_DIR/sets"
else
  echo "== voc/ already present, skipping download =="
fi

if [[ "${WANT_VOC_AUG:-0}" == "1" && ! -d "$VOC_DIR/SegmentationClassAug" ]]; then
  # SBD augmented masks (Hariharan et al.) give a ~10.5k-image trainaug split.
  echo "== Fetching SBD augmented masks (trainaug) =="
  echo "   NOTE: download SegmentationClassAug.zip from the SBD/DeepLab mirror,"
  echo "   place it at $WORK/SegmentationClassAug.zip, then re-run with WANT_VOC_AUG=1."
  if [[ -f "$WORK/SegmentationClassAug.zip" ]]; then
    unzip -q -o "$WORK/SegmentationClassAug.zip" -d "$VOC_DIR"
    # trainaug list ships with most mirrors; otherwise union train+val SBD ids here.
    [[ -f "$WORK/trainaug.txt" ]] && cp "$WORK/trainaug.txt" "$VOC_DIR/sets/trainaug.txt"
  fi
fi
echo "== Packing voc.tar =="
tar -C "$ARCHIVE_DIR" -cf "$ARCHIVE_DIR/voc.tar.partial" voc
mv -f "$ARCHIVE_DIR/voc.tar.partial" "$ARCHIVE_DIR/voc.tar"
echo "   -> $ARCHIVE_DIR/voc.tar   (--voc-root $VOC_DIR)"
fi

if [[ "${KEEP_EXTRACTED:-0}" != "1" ]]; then
  echo "== Cleaning intermediate downloads ($WORK) (set KEEP_EXTRACTED=1 to keep) =="
  rm -rf "$WORK"
fi

cat <<EOF

Done. Use with dataset_prep, e.g.:
  python dataset_prep/semseg/dataset_prep.py \\
    --ade20k-root $ADE_DIR \\
    --voc-root $VOC_DIR \\
    --splits train,val --out-root /path/to/prepared --variants-per-image 1
EOF
