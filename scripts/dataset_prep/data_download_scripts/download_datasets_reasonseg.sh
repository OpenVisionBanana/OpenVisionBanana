#!/usr/bin/env bash
# Arrange the LISA ReasonSeg release (train / val / test) into the layouts the
# repository consumes:
#
#   datasets_archives/refseg/reasonseg_train_test/train.tar   (train/<stem>.jpg + .json)
#   datasets_archives/refseg/reasonseg_train_test/test.tar    (test/<stem>.jpg + .json)
#   datasets_archives/refseg/reasonseg_val/<stem>.jpg + .json
#
# train.tar / test.tar feed dataset_prep/refseg/prepare_reasonseg.py (the
# refseg_reasonseg training member; REASONSEG_REFSEG_ROOT). reasonseg_val/ is the
# evaluation set; rasterize its masks afterwards with
# scripts/dataset_prep/prepare_refseg_eval.sh. Training excludes val images.
#
# ReasonSeg is distributed as one zip per split in a Google Drive folder, which
# cannot be fetched with wget:
#   https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy
#
# Manual download (default):
#   1. Open the folder above and download train.zip, val.zip and test.zip.
#   2. Put train.zip and test.zip in datasets_archives/refseg/reasonseg_train_test/
#      and val.zip in datasets_archives/refseg/reasonseg_val/.
#   3. Run this script.
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_reasonseg.sh
#   REASONSEG_SPLITS=train ...                # only some of train,test,val
#   KEEP_DOWNLOADS=1 ...                      # keep the extraction work dir
#   FORCE=1 ...                               # rebuild existing outputs
#
# The downloaded zips are never deleted.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/refseg}
REASONSEG_SPLITS=${REASONSEG_SPLITS:-train,test,val}
REASONSEG_FOLDER_URL=${REASONSEG_FOLDER_URL:-https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy}
KEEP_DOWNLOADS=${KEEP_DOWNLOADS:-0}
FORCE=${FORCE:-0}

TRAIN_TEST_DIR=$ARCHIVE_DIR/reasonseg_train_test
VAL_DIR=$ARCHIVE_DIR/reasonseg_val
WORK=$ARCHIVE_DIR/_reasonseg_tmp

# Image/annotation pairs in the LISA release; a mismatch is reported, not fatal.
declare -A EXPECTED=([train]=239 [test]=779 [val]=200)

IFS=, read -r -a splits <<< "$REASONSEG_SPLITS"
for split in "${splits[@]}"; do
  [[ -n "${EXPECTED[$split]+x}" ]] || {
    echo "ERROR: REASONSEG_SPLITS must be a subset of train,test,val (got '$split')." >&2
    exit 2
  }
done
command -v unzip > /dev/null 2>&1 || { echo "ERROR: unzip not found." >&2; exit 1; }
mkdir -p "$TRAIN_TEST_DIR" "$VAL_DIR"

is_done() {
  local split=$1
  if [[ "$split" == val ]]; then
    compgen -G "$VAL_DIR/*.json" > /dev/null
  else
    [[ -s "$TRAIN_TEST_DIR/$split.tar" ]]
  fi
}

# Each split's zip sits in its own dataset folder (a subfolder is tolerated).
find_zip() {
  find "$TRAIN_TEST_DIR" "$VAL_DIR" -maxdepth 3 -type f -iname "$1.zip" -print -quit
}

pending=()
for split in "${splits[@]}"; do
  if [[ "$FORCE" != 1 ]] && is_done "$split"; then
    echo "== ReasonSeg $split already present, skipping =="
  else
    pending+=("$split")
  fi
done

missing=()
for split in "${pending[@]}"; do
  [[ -n "$(find_zip "$split")" ]] || missing+=("$split.zip")
done
if (( ${#missing[@]} )); then
  cat >&2 <<EOF
ReasonSeg zip(s) not found: ${missing[*]}
Download them by hand from the LISA Google Drive folder:
  $REASONSEG_FOLDER_URL
put train.zip and test.zip in $TRAIN_TEST_DIR and val.zip in $VAL_DIR, then re-run.
EOF
  exit 1
fi

# Extract <split>.zip and print the directory that holds its <stem>.json files.
extract_split() {
  local split=$1 zip_path=$2 dest=$WORK/$split
  rm -rf "$dest"
  mkdir -p "$dest"
  unzip -q -o "$zip_path" -d "$dest"
  local first_json
  first_json=$(find "$dest" -type f -name '*.json' -not -path '*/__MACOSX/*' -print -quit)
  [[ -n "$first_json" ]] || { echo "ERROR: no annotations in $zip_path" >&2; exit 1; }
  dirname -- "$first_json"
}

check_pairs() {
  local split=$1 dir=$2 json image
  json=$(find "$dir" -maxdepth 1 -type f -name '*.json' | wc -l)
  image=$(find "$dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
  echo "   $split: $json annotations, $image images"
  if [[ "$json" != "${EXPECTED[$split]}" || "$image" != "${EXPECTED[$split]}" ]]; then
    echo "   WARNING: expected ${EXPECTED[$split]} pairs for $split; check the download." >&2
  fi
}

for split in "${pending[@]}"; do
  zip_path=$(find_zip "$split")
  echo "== Extracting ReasonSeg $split from $zip_path =="
  src=$(extract_split "$split" "$zip_path")
  check_pairs "$split" "$src"

  if [[ "$split" == val ]]; then
    # Same flattened layout as download_datasets_refseg.sh. Existing masks/ is kept.
    mkdir -p "$VAL_DIR"
    find "$src" -maxdepth 1 -type f \( -name '*.json' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
      -exec cp -f {} "$VAL_DIR"/ \;
    echo "   -> $VAL_DIR"
  else
    # prepare_reasonseg.py reads <split>.tar with members under <split>/.
    pack=$WORK/pack_$split
    rm -rf "$pack"
    mkdir -p "$pack/$split" "$TRAIN_TEST_DIR"
    find "$src" -maxdepth 1 -type f \( -name '*.json' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
      -exec mv -f {} "$pack/$split"/ \;
    tar -cf "$TRAIN_TEST_DIR/$split.tar.partial" -C "$pack" "$split"
    mv -f "$TRAIN_TEST_DIR/$split.tar.partial" "$TRAIN_TEST_DIR/$split.tar"
    echo "   -> $TRAIN_TEST_DIR/$split.tar"
  fi
done

if [[ "$KEEP_DOWNLOADS" != 1 ]]; then
  rm -rf "$WORK"
fi

cat <<EOF

Done. ReasonSeg is arranged under $ARCHIVE_DIR.
  training member: REASONSEG_REFSEG_ENABLE=1 scripts/dataset_prep/prepare_open_visionbanana_pool.sh
                   (or dataset_prep/refseg/prepare_reasonseg.py --reasonseg-root $TRAIN_TEST_DIR)
  evaluation:      scripts/dataset_prep/prepare_refseg_eval.sh
EOF
