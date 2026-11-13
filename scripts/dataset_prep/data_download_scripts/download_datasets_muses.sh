#!/usr/bin/env bash
# Download MUSES frame-camera + semantic + panoptic train/val data, verify the
# layout expected by the Open Vision Banana converters, and pack it as one tar:
#
#   datasets_archives/sem_segm/muses.tar  ->  muses/{frame_camera,gt_semantic,gt_panoptic,...}
#
# The tar is consumed by scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh.
# Semantic segmentation uses frame_camera + gt_semantic labelTrainIds; instance
# segmentation uses frame_camera + gt_panoptic COCO-style panoptic annotations.
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_muses.sh
#
# Repack an already extracted download:
#   MUSES_EXTRACTED_ROOT=/tmp/data/MUSES \
#     bash scripts/dataset_prep/data_download_scripts/download_datasets_muses.sh
#
# Keep panoptic out when you only want semantic segmentation:
#   WANT_PANOPTIC=0 bash scripts/dataset_prep/data_download_scripts/download_datasets_muses.sh
set -euo pipefail

show_help() {
  cat <<'EOF'
Download/repack MUSES for Open Vision Banana dataset prep.

Outputs:
  $ARCHIVE_DIR/muses.tar

Configuration:
  ROOT=/path/to/open_visionbanana       repo/data root used for defaults
  ARCHIVE_DIR=$ROOT/datasets_archives/sem_segm
  MUSES_TAR=$ARCHIVE_DIR/muses.tar
  TMPDIR=/fast/scratch                  scratch parent for temporary work
  WORK=$TMPDIR/open_visionbanana_muses_download
  PACKAGE_DIR=$WORK/packages            downloaded zip location
  EXTRACT_ROOT=$WORK/extracted          temporary unzip location
  MUSES_EXTRACTED_ROOT=/path/to/MUSES   existing folder containing muses/
  WANT_PANOPTIC=1                       include gt_panoptic for instance seg
  FORCE=0                               redownload/repack existing outputs
  KEEP_EXTRACTED=0                      keep $WORK after successful packing

Examples:
  bash scripts/dataset_prep/data_download_scripts/download_datasets_muses.sh
  MUSES_EXTRACTED_ROOT=/tmp/data/MUSES bash scripts/dataset_prep/data_download_scripts/download_datasets_muses.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

ROOT=${ROOT:-$HOME/open_visionbanana}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/sem_segm}
MUSES_TAR=${MUSES_TAR:-$ARCHIVE_DIR/muses.tar}
TMP_PARENT=${TMPDIR:-/tmp/${USER:-user}}
WORK=${WORK:-$TMP_PARENT/open_visionbanana_muses_download}
PACKAGE_DIR=${PACKAGE_DIR:-$WORK/packages}
EXTRACT_ROOT=${EXTRACT_ROOT:-$WORK/extracted}
MUSES_EXTRACTED_ROOT=${MUSES_EXTRACTED_ROOT:-}
WANT_PANOPTIC=${WANT_PANOPTIC:-1}
FORCE=${FORCE:-0}
KEEP_EXTRACTED=${KEEP_EXTRACTED:-0}

FRAME_URL=${FRAME_URL:-https://muses.ethz.ch/MUSES_packages/frame_camera_trainvaltest.zip}
SEMANTIC_URL=${SEMANTIC_URL:-https://muses.ethz.ch/MUSES_packages/gt_semantic_trainval.zip}
PANOPTIC_URL=${PANOPTIC_URL:-https://muses.ethz.ch/MUSES_packages/gt_panoptic_trainval.zip}

mkdir -p "$ARCHIVE_DIR"

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required tool not found in PATH: $1" >&2
    exit 1
  fi
}

require_tool tar

resolve_muses_dir() {
  local root=$1
  if [[ -d "$root/muses" ]]; then
    printf '%s\n' "$root/muses"
    return
  fi
  if [[ -d "$root/frame_camera" ]]; then
    printf '%s\n' "$root"
    return
  fi
  echo "ERROR: could not find a MUSES root at $root (expected muses/ or frame_camera/)." >&2
  exit 1
}

download_file() {
  local url=$1
  local out=$2
  local tmp="${out}.partial"
  mkdir -p "$(dirname "$out")"
  if [[ "$FORCE" == "1" ]]; then
    rm -f "$out" "$tmp"
  fi
  if [[ -s "$out" ]]; then
    echo "== Already present: $out =="
    return
  fi
  require_tool wget
  echo "== Downloading $url =="
  echo "   -> $out"
  if ! wget -c -O "$tmp" "$url"; then
    cat >&2 <<EOF
ERROR: failed to download $url
MUSES downloads may require website registration. Download the zip manually,
place it at $out, and rerun this script; or set MUSES_EXTRACTED_ROOT to an
already extracted folder containing muses/.
EOF
    exit 1
  fi
  mv -f "$tmp" "$out"
}

extract_zip() {
  local zip_path=$1
  require_tool unzip
  echo "== Extracting $zip_path to $EXTRACT_ROOT =="
  unzip -q -o "$zip_path" -d "$EXTRACT_ROOT"
}

validate_muses_dir() {
  local dir=$1
  local missing=0
  for path in \
    "$dir/frame_camera/train" \
    "$dir/frame_camera/val" \
    "$dir/frame_camera/test" \
    "$dir/gt_semantic/train" \
    "$dir/gt_semantic/val"; do
    if [[ ! -d "$path" ]]; then
      echo "ERROR: missing required MUSES path: $path" >&2
      missing=1
    fi
  done
  if [[ "$WANT_PANOPTIC" == "1" ]]; then
    for path in \
      "$dir/gt_panoptic/train" \
      "$dir/gt_panoptic/val" \
      "$dir/gt_panoptic/train.json" \
      "$dir/gt_panoptic/val.json"; do
      if [[ ! -e "$path" ]]; then
        echo "ERROR: missing required MUSES panoptic path: $path" >&2
        missing=1
      fi
    done
  fi
  if [[ "$missing" != "0" ]]; then
    exit 1
  fi
}

if [[ "$FORCE" == "1" ]]; then
  rm -f "$MUSES_TAR"
fi

if [[ -s "$MUSES_TAR" && "$FORCE" != "1" ]]; then
  echo "== MUSES tar already present: $MUSES_TAR =="
  exit 0
fi

if [[ -n "$MUSES_EXTRACTED_ROOT" ]]; then
  MUSES_DIR=$(resolve_muses_dir "$MUSES_EXTRACTED_ROOT")
else
  mkdir -p "$PACKAGE_DIR" "$EXTRACT_ROOT"
  download_file "$FRAME_URL" "$PACKAGE_DIR/frame_camera_trainvaltest.zip"
  download_file "$SEMANTIC_URL" "$PACKAGE_DIR/gt_semantic_trainval.zip"
  if [[ "$WANT_PANOPTIC" == "1" ]]; then
    download_file "$PANOPTIC_URL" "$PACKAGE_DIR/gt_panoptic_trainval.zip"
  fi

  extract_zip "$PACKAGE_DIR/frame_camera_trainvaltest.zip"
  extract_zip "$PACKAGE_DIR/gt_semantic_trainval.zip"
  if [[ "$WANT_PANOPTIC" == "1" ]]; then
    extract_zip "$PACKAGE_DIR/gt_panoptic_trainval.zip"
  fi
  MUSES_DIR=$(resolve_muses_dir "$EXTRACT_ROOT")
fi

validate_muses_dir "$MUSES_DIR"

echo "== Packing MUSES tar =="
mkdir -p "$(dirname "$MUSES_TAR")"
tar -C "$(dirname "$MUSES_DIR")" -cf "$MUSES_TAR.partial" "$(basename "$MUSES_DIR")"
mv -f "$MUSES_TAR.partial" "$MUSES_TAR"
echo "   -> $MUSES_TAR"

if [[ -z "$MUSES_EXTRACTED_ROOT" && "$KEEP_EXTRACTED" != "1" ]]; then
  echo "== Cleaning temporary files: $WORK (set KEEP_EXTRACTED=1 to keep) =="
  rm -rf "$WORK"
fi

cat <<EOF

Done. Use with the full pool builder:
  MUSES_TAR=$MUSES_TAR \\
    bash scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh

Or direct converter smoke tests:
  python dataset_prep/semseg/dataset_prep.py --muses-root $MUSES_DIR --out-root /tmp/muses_semseg
  python dataset_prep/instanceseg/dataset_prep.py --muses-root $MUSES_DIR --out-root /tmp/muses_instanceseg
EOF
