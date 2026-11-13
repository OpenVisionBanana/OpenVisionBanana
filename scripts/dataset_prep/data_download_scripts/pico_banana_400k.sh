#!/usr/bin/env bash
set -euo pipefail

# Pico-Banana setup helper.
#
# This prepares the native Pico-Banana checkout and downloads. The trainer does
# not consume this layout directly; run dataset_prep/pico_banana/prepare_pico_banana.py
# afterwards to emit the canonical metadata.jsonl format.

show_help() {
  cat <<'EOF'
Prepare a native Pico-Banana checkout with metadata, extracted OpenImages
source splits, and optionally the edited target images.

Use case:
  Persistent-folder / simple setup.

Choose this helper when you do not mind storing the extracted files directly on
persistent storage. If you prefer to keep large edited-target collections as a
tar for later staging into TMPDIR, see:
  scripts/dataset_prep/data_download_scripts/download_pico_positive_edit.sh
  scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh

Configuration style:
  Set ROOT and related variables in the environment before running.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
ROOT=${ROOT:-${DATA_ROOT:-$REPO_ROOT/datasets_archives}/pico-banana-400k}
PYTHON=${PYTHON:-python3}
OPENIMAGES_DIR="$ROOT/openimages"
SOURCE_DIR="$OPENIMAGES_DIR/openimage_source_images"

if [[ ! -d "$ROOT/.git" ]]; then
  if [[ -d "$ROOT" && -n "$(find "$ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "ERROR: $ROOT exists but is not a pico-banana-400k git checkout." >&2
    echo "Set ROOT to an empty directory or an existing pico-banana-400k checkout." >&2
    exit 1
  fi
  git clone https://github.com/apple/pico-banana-400k.git "$ROOT"
fi

mkdir -p "$OPENIMAGES_DIR" "$SOURCE_DIR"

cd "$ROOT"

download_if_missing() {
  local url=$1
  local out=$2
  if [[ ! -e "$out" ]]; then
    wget -O "$out" "$url"
  fi
}

if [[ ! -e "$SOURCE_DIR/train_0.tar.gz" && ! -e "$SOURCE_DIR/train_0.tar" ]]; then
  download_if_missing \
    https://open-images-dataset.s3.amazonaws.com/tar/train_0.tar.gz \
    "$SOURCE_DIR/train_0.tar.gz"
fi
if [[ ! -e "$SOURCE_DIR/train_1.tar.gz" && ! -e "$SOURCE_DIR/train_1.tar" ]]; then
  download_if_missing \
    https://open-images-dataset.s3.amazonaws.com/tar/train_1.tar.gz \
    "$SOURCE_DIR/train_1.tar.gz"
fi
download_if_missing \
  https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv \
  "$OPENIMAGES_DIR/train-images-boxable-with-rotation.csv"
download_if_missing \
  https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl \
  "$OPENIMAGES_DIR/sft.jsonl"
download_if_missing \
  https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/preference.jsonl \
  "$OPENIMAGES_DIR/preference.jsonl"
download_if_missing \
  https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/multi-turn.jsonl \
  "$OPENIMAGES_DIR/multi-turn.jsonl"
download_if_missing \
  https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/manifest/sft_manifest.txt \
  "$OPENIMAGES_DIR/sft_manifest.txt"
download_if_missing \
  https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/manifest/preference_manifest.txt \
  "$OPENIMAGES_DIR/preference_manifest.txt"
download_if_missing \
  https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/manifest/multi_turn_manifest.txt \
  "$OPENIMAGES_DIR/multi_turn_manifest.txt"

extract_split() {
  local split=$1 marker="$SOURCE_DIR/.ovb-extracted.$1" archive
  [[ -f "$marker" ]] && return
  for archive in "$SOURCE_DIR/$split.tar.gz" "$SOURCE_DIR/$split.tar"; do
    [[ -f "$archive" ]] || continue
    echo "== Extracting $archive =="
    tar --skip-old-files -xf "$archive" -C "$SOURCE_DIR"
    touch "$marker"
    return
  done
}
extract_split train_0
extract_split train_1

# Download edited SFT target images (~258k files, ~400 GB) into
# $OPENIMAGES_DIR/images/positive-edit with parallel, resumable downloads
# (JOBS, default 32). Set MAKE_TAR=1 to also write openimages/positive-edit.tar.
if [[ "${SKIP_PICO_TARGET_DOWNLOAD:-0}" != "1" ]]; then
  ROOT="$ROOT" OPENIMAGES_DIR="$OPENIMAGES_DIR" MAKE_TAR="${MAKE_TAR:-0}" \
    bash "$SCRIPT_DIR/download_pico_positive_edit.sh"
fi

# The upstream script has path variables at the top. Run it from the Pico root
# after editing those variables if needed. The default checkout values match the
# layout produced above.
if [[ -f openimages/.ovb-mapped && -s openimages/sft_with_local_source_image_path.jsonl ]]; then
  echo "Source-image mapping already written: $ROOT/openimages/sft_with_local_source_image_path.jsonl"
else
  "$PYTHON" map_openimage_url_to_local.py
  touch openimages/.ovb-mapped
fi

echo "Native Pico-Banana data prepared at $ROOT"
echo "Next step:"
echo "  python <repo>/dataset_prep/pico_banana/prepare_pico_banana.py \\"
echo "    --pico-root $ROOT \\"
echo "    --metadata $OPENIMAGES_DIR/sft_with_local_source_image_path.jsonl \\"
echo "    --out-root /path/to/prepared/pico_banana_sft"
