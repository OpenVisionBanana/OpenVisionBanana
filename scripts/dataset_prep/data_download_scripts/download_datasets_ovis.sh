#!/usr/bin/env bash
# Download/repack OVIS for Open Vision Banana instance-segmentation prep.
#
# Output:
#   datasets_archives/sem_segm/ovis.tar -> OVIS/{train,valid|val,annotations_*.json}
#
# The official files are downloaded with gdown using the Google Drive ids from
# the OVIS download page. If you already have an extracted copy, set
# OVIS_EXTRACTED_ROOT=/path/to/OVIS to skip downloading and only validate + pack.
set -euo pipefail

show_help() {
  cat <<'EOF'
Download/repack OVIS for Open Vision Banana instance-segmentation prep.

Outputs:
  $ARCHIVE_DIR/ovis.tar

Configuration:
  ROOT=/path/to/open_visionbanana       repo/data root used for defaults
  ARCHIVE_DIR=$ROOT/datasets_archives/sem_segm
  OVIS_TAR=$ARCHIVE_DIR/ovis.tar
  TMPDIR=/fast/scratch                  scratch parent for temporary work
  WORK=$TMPDIR/open_visionbanana_ovis_download
  PACKAGE_DIR=$WORK/packages            downloaded zip/json location
  EXTRACT_ROOT=$WORK/extracted          temporary unzip location
  OVIS_EXTRACTED_ROOT=/tmp/data/OVIS    existing extracted OVIS directory
  FORCE=0                               redownload/repack existing outputs
  KEEP_EXTRACTED=0                      keep $WORK after successful packing

Download commands used when OVIS_EXTRACTED_ROOT is not set:
  gdown 11DPalTyrI2OdrDai7pG45gn3W10P1XO5 # train.zip
  gdown 171Z_pqcVkbJQ5lDMVM_mSfscjOPLBW7Z # valid.zip
  gdown 1gIPS-LGJLNhWc4_QGBTGAcSDMtVOaixk # annotations_train.json
  gdown 165KJVcFDjiW75PlEqpMcJhG2ChEz6E5w # annotations_valid.json

Examples:
  bash scripts/dataset_prep/data_download_scripts/download_datasets_ovis.sh
  OVIS_EXTRACTED_ROOT=/tmp/data/OVIS bash scripts/dataset_prep/data_download_scripts/download_datasets_ovis.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

ROOT=${ROOT:-$HOME/open_visionbanana}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/sem_segm}
OVIS_TAR=${OVIS_TAR:-$ARCHIVE_DIR/ovis.tar}
TMP_PARENT=${TMPDIR:-/tmp/${USER:-user}}
WORK=${WORK:-$TMP_PARENT/open_visionbanana_ovis_download}
PACKAGE_DIR=${PACKAGE_DIR:-$WORK/packages}
EXTRACT_ROOT=${EXTRACT_ROOT:-$WORK/extracted}
OVIS_EXTRACTED_ROOT=${OVIS_EXTRACTED_ROOT:-}
FORCE=${FORCE:-0}
KEEP_EXTRACTED=${KEEP_EXTRACTED:-0}

TRAIN_ZIP_ID=${TRAIN_ZIP_ID:-11DPalTyrI2OdrDai7pG45gn3W10P1XO5}
VALID_ZIP_ID=${VALID_ZIP_ID:-171Z_pqcVkbJQ5lDMVM_mSfscjOPLBW7Z}
TRAIN_ANN_ID=${TRAIN_ANN_ID:-1gIPS-LGJLNhWc4_QGBTGAcSDMtVOaixk}
VALID_ANN_ID=${VALID_ANN_ID:-165KJVcFDjiW75PlEqpMcJhG2ChEz6E5w}

mkdir -p "$ARCHIVE_DIR"

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required tool not found in PATH: $1" >&2
    exit 1
  fi
}

download_drive_file() {
  local file_id=$1
  local out=$2
  if [[ "$FORCE" == "1" ]]; then
    rm -f "$out"
  fi
  if [[ -s "$out" ]]; then
    echo "== Already present: $out =="
    return
  fi
  require_tool gdown
  mkdir -p "$(dirname "$out")"
  echo "== Downloading OVIS file =="
  echo "   gdown $file_id -O $out"
  gdown "$file_id" -O "$out"
  [[ -s "$out" ]] || { echo "ERROR: gdown did not produce $out" >&2; exit 1; }
}

extract_zip_if_needed() {
  local zip_path=$1
  local expected_dir=$2
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$expected_dir"
  fi
  if [[ -d "$expected_dir" ]]; then
    echo "== Already extracted: $expected_dir =="
    return
  fi
  require_tool unzip
  mkdir -p "$EXTRACT_ROOT/OVIS"
  echo "== Extracting $zip_path =="
  unzip -q -o "$zip_path" -d "$EXTRACT_ROOT/OVIS"
}

resolve_ovis_dir() {
  local root=$1
  for candidate in "$root" "$root/OVIS" "$root/ovis"; do
    if [[ -f "$candidate/annotations_train.json" && -d "$candidate/train" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "ERROR: could not find OVIS under $root (expected annotations_train.json and train/)." >&2
  exit 1
}

validate_ovis_dir() {
  local dir=$1
  local missing=0
  for path in "$dir/annotations_train.json" "$dir/train"; do
    if [[ ! -e "$path" ]]; then
      echo "ERROR: missing required OVIS path: $path" >&2
      missing=1
    fi
  done
  if { [[ -d "$dir/valid" ]] || [[ -d "$dir/val" ]]; } && [[ ! -f "$dir/annotations_valid.json" ]]; then
    echo "ERROR: found a validation frame directory but missing annotations_valid.json" >&2
    missing=1
  fi
  if [[ -f "$dir/annotations_valid.json" && ! -d "$dir/valid" && ! -d "$dir/val" ]]; then
    echo "WARNING: annotations_valid.json exists but no valid/ or val/ directory is present; validation split will be skipped." >&2
  fi
  if [[ "$missing" != "0" ]]; then
    exit 1
  fi
}

if [[ "$FORCE" == "1" ]]; then
  rm -f "$OVIS_TAR"
fi
if [[ -s "$OVIS_TAR" && "$FORCE" != "1" ]]; then
  echo "== OVIS tar already present: $OVIS_TAR =="
  exit 0
fi

if [[ -n "$OVIS_EXTRACTED_ROOT" ]]; then
  OVIS_DIR=$(resolve_ovis_dir "$OVIS_EXTRACTED_ROOT")
else
  require_tool unzip
  mkdir -p "$PACKAGE_DIR" "$EXTRACT_ROOT/OVIS"
  download_drive_file "$TRAIN_ZIP_ID" "$PACKAGE_DIR/train.zip"
  download_drive_file "$VALID_ZIP_ID" "$PACKAGE_DIR/valid.zip"
  download_drive_file "$TRAIN_ANN_ID" "$EXTRACT_ROOT/OVIS/annotations_train.json"
  download_drive_file "$VALID_ANN_ID" "$EXTRACT_ROOT/OVIS/annotations_valid.json"
  extract_zip_if_needed "$PACKAGE_DIR/train.zip" "$EXTRACT_ROOT/OVIS/train"
  extract_zip_if_needed "$PACKAGE_DIR/valid.zip" "$EXTRACT_ROOT/OVIS/valid"
  OVIS_DIR=$(resolve_ovis_dir "$EXTRACT_ROOT")
fi

validate_ovis_dir "$OVIS_DIR"

echo "== Packing OVIS tar =="
mkdir -p "$(dirname "$OVIS_TAR")"
tar -C "$(dirname "$OVIS_DIR")" -cf "$OVIS_TAR.partial" "$(basename "$OVIS_DIR")"
mv -f "$OVIS_TAR.partial" "$OVIS_TAR"
echo "   -> $OVIS_TAR"

if [[ -z "$OVIS_EXTRACTED_ROOT" && "$KEEP_EXTRACTED" != "1" ]]; then
  rm -rf "$WORK"
fi
