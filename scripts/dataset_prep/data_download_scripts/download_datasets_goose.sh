#!/usr/bin/env bash
# Download the GOOSE (not GOOSE-Ex) 2D train/val image dataset, normalize it to
# the documented layout, and pack it as one tar for the Open Vision Banana pool:
#
#   datasets_archives/sem_segm/goose.tar -> goose/{images,labels,goose_label_mapping.csv,...}
#
# Only the 2D image/annotation zips are downloaded. 3D point clouds and GOOSE-Ex
# are intentionally excluded.
set -euo pipefail

show_help() {
  cat <<'EOF'
Download/repack GOOSE 2D train/val for Open Vision Banana dataset prep.

Outputs:
  $ARCHIVE_DIR/goose.tar

Configuration:
  ROOT=/path/to/open_visionbanana       repo/data root used for defaults
  ARCHIVE_DIR=$ROOT/datasets_archives/sem_segm
  GOOSE_TAR=$ARCHIVE_DIR/goose.tar
  TMPDIR=/fast/scratch                  scratch parent for temporary work
  WORK=$TMPDIR/open_visionbanana_goose_download
  PACKAGE_DIR=$WORK/packages            downloaded zip location
  EXTRACT_ROOT=$WORK/extracted          temporary unzip location
  GOOSE_EXTRACTED_ROOT=/path/to/root    existing folder containing goose/images/labels
  FORCE=0                               redownload/repack existing outputs
  KEEP_EXTRACTED=0                      keep $WORK after successful packing

The default URLs are the official GOOSE 2D train/val zips, not GOOSE-Ex.
Override GOOSE_TRAIN_URL / GOOSE_VAL_URL if the mirrors change.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

ROOT=${ROOT:-$HOME/open_visionbanana}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/sem_segm}
GOOSE_TAR=${GOOSE_TAR:-$ARCHIVE_DIR/goose.tar}
TMP_PARENT=${TMPDIR:-/tmp/${USER:-user}}
WORK=${WORK:-$TMP_PARENT/open_visionbanana_goose_download}
PACKAGE_DIR=${PACKAGE_DIR:-$WORK/packages}
EXTRACT_ROOT=${EXTRACT_ROOT:-$WORK/extracted}
GOOSE_BUILD_ROOT=${GOOSE_BUILD_ROOT:-$WORK/goose_build}
GOOSE_EXTRACTED_ROOT=${GOOSE_EXTRACTED_ROOT:-}
FORCE=${FORCE:-0}
KEEP_EXTRACTED=${KEEP_EXTRACTED:-0}

GOOSE_TRAIN_URL=${GOOSE_TRAIN_URL:-https://goose-dataset.de/storage/goose_2d_train.zip}
GOOSE_VAL_URL=${GOOSE_VAL_URL:-https://goose-dataset.de/storage/goose_2d_val.zip}

mkdir -p "$ARCHIVE_DIR"

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required tool not found in PATH: $1" >&2
    exit 1
  fi
}

require_tool tar

resolve_goose_dir() {
  local root=$1
  for candidate in "$root" "$root/goose" "$root/goose-dataset"; do
    if [[ -d "$candidate/images" && -d "$candidate/labels" && -f "$candidate/goose_label_mapping.csv" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "ERROR: could not find a normalized GOOSE root under $root (expected images/, labels/, goose_label_mapping.csv)." >&2
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
  wget -c -O "$tmp" "$url"
  mv -f "$tmp" "$out"
}

extract_zip() {
  local zip_path=$1
  local out_dir=$2
  require_tool unzip
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$out_dir"
  fi
  if [[ -d "$out_dir" ]]; then
    echo "== Already extracted: $out_dir =="
    return
  fi
  echo "== Extracting $zip_path to $out_dir =="
  mkdir -p "$out_dir"
  unzip -q -o "$zip_path" -d "$out_dir"
}

first_dir() {
  local root=$1
  local rel=$2
  find "$root" -type d -path "*/$rel" -print -quit
}

first_file() {
  local root=$1
  local name=$2
  find "$root" -type f -name "$name" -print -quit
}

move_tree_contents() {
  local src=$1
  local dst=$2
  mkdir -p "$dst"
  if [[ -d "$src" ]]; then
    find "$src" -mindepth 1 -maxdepth 1 -exec mv -t "$dst" {} +
  fi
}

normalize_downloads() {
  local train_extract=$1
  local val_extract=$2
  local out=$3
  local train_images train_labels val_images val_labels mapping license changelog

  rm -rf "$out"
  mkdir -p "$out/images/train" "$out/images/val" "$out/labels/train" "$out/labels/val"

  train_images=$(first_dir "$train_extract" "images/train")
  train_labels=$(first_dir "$train_extract" "labels/train")
  val_images=$(first_dir "$val_extract" "images/val")
  val_labels=$(first_dir "$val_extract" "labels/val")
  mapping=$(first_file "$train_extract" "goose_label_mapping.csv")
  [[ -n "$mapping" ]] || mapping=$(first_file "$val_extract" "goose_label_mapping.csv")
  license=$(first_file "$train_extract" "LICENSE")
  [[ -n "$license" ]] || license=$(first_file "$val_extract" "LICENSE")
  changelog=$(first_file "$train_extract" "CHANGELOG")
  [[ -n "$changelog" ]] || changelog=$(first_file "$val_extract" "CHANGELOG")

  for required in "$train_images" "$train_labels" "$val_images" "$val_labels" "$mapping"; do
    if [[ -z "$required" ]]; then
      echo "ERROR: extracted GOOSE zips do not contain the expected images/labels layout." >&2
      exit 1
    fi
  done

  move_tree_contents "$train_images" "$out/images/train"
  move_tree_contents "$train_labels" "$out/labels/train"
  move_tree_contents "$val_images" "$out/images/val"
  move_tree_contents "$val_labels" "$out/labels/val"
  cp "$mapping" "$out/goose_label_mapping.csv"
  [[ -n "$license" ]] && cp "$license" "$out/LICENSE"
  [[ -n "$changelog" ]] && cp "$changelog" "$out/CHANGELOG"
}

validate_goose_dir() {
  local dir=$1
  local missing=0
  for path in \
    "$dir/goose_label_mapping.csv" \
    "$dir/images/train" \
    "$dir/images/val" \
    "$dir/labels/train" \
    "$dir/labels/val"; do
    if [[ ! -e "$path" ]]; then
      echo "ERROR: missing required GOOSE path: $path" >&2
      missing=1
    fi
  done
  if [[ "$missing" != "0" ]]; then
    exit 1
  fi
}

if [[ "$FORCE" == "1" ]]; then
  rm -f "$GOOSE_TAR"
fi
if [[ -s "$GOOSE_TAR" && "$FORCE" != "1" ]]; then
  echo "== GOOSE tar already present: $GOOSE_TAR =="
  exit 0
fi

if [[ -n "$GOOSE_EXTRACTED_ROOT" ]]; then
  GOOSE_DIR=$(resolve_goose_dir "$GOOSE_EXTRACTED_ROOT")
else
  mkdir -p "$PACKAGE_DIR" "$EXTRACT_ROOT"
  download_file "$GOOSE_TRAIN_URL" "$PACKAGE_DIR/goose_2d_train.zip"
  download_file "$GOOSE_VAL_URL" "$PACKAGE_DIR/goose_2d_val.zip"
  extract_zip "$PACKAGE_DIR/goose_2d_train.zip" "$EXTRACT_ROOT/goose_2d_train"
  extract_zip "$PACKAGE_DIR/goose_2d_val.zip" "$EXTRACT_ROOT/goose_2d_val"
  normalize_downloads "$EXTRACT_ROOT/goose_2d_train" "$EXTRACT_ROOT/goose_2d_val" "$GOOSE_BUILD_ROOT/goose"
  GOOSE_DIR=$GOOSE_BUILD_ROOT/goose
fi

validate_goose_dir "$GOOSE_DIR"

echo "== Packing GOOSE tar =="
mkdir -p "$(dirname "$GOOSE_TAR")"
tar -C "$(dirname "$GOOSE_DIR")" -cf "$GOOSE_TAR.partial" "$(basename "$GOOSE_DIR")"
mv -f "$GOOSE_TAR.partial" "$GOOSE_TAR"
echo "   -> $GOOSE_TAR"

if [[ -z "$GOOSE_EXTRACTED_ROOT" && "$KEEP_EXTRACTED" != "1" ]]; then
  echo "== Cleaning temporary files: $WORK (set KEEP_EXTRACTED=1 to keep) =="
  rm -rf "$WORK"
fi

cat <<EOF

Done. Use with the full pool builder:
  GOOSE_TAR=$GOOSE_TAR \\
    bash scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh

Or run the small prep smoke test:
  GOOSE_TAR=$GOOSE_TAR bash scripts/dataset_prep/smoke_prepare_goose.sh
EOF
