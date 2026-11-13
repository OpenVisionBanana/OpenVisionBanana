#!/usr/bin/env bash
# Build a complete COCONut-XL installation from COCONut-B, COCONut-L, and XL.
#
# The resulting Detectron2-style layout is:
#   $TARGET_DIR/coco/{train2017,panoptic_train2017,annotations/panoptic_train2017.json}
#
# This follows the official COCONut preparation guide. It downloads the COCO
# train2017 + unlabeled2017 images and COCONut-B masks/JSON, then the four
# COCONut-L Objects365 patches and their release, then the seven XL-only patches
# and the COCONut-XL release. COCONut-XL annotations take precedence when they
# contain the COCONut-L images, so the final manifest and masks are truly XL.
set -euo pipefail

show_help() {
  cat <<'EOF'
Build a complete COCONut-XL tree (B -> L -> XL).

Usage:
  bash scripts/dataset_prep/data_download_scripts/download_datasets_coconut_xlarge.sh

Important configuration:
  ROOT=/path/to/open_visionbanana
  ARCHIVE_DIR=/path/to/datasets_archives/shared_seg
  TARGET_DIR=/path/to/coconut_xlarge
  COCO2017_ARCHIVE_DIR=/path/to/coco2017   Reuse train2017.zip / unlabeled2017.zip
                                      from download_datasets_coco.sh
                                      (default: $ARCHIVE_DIR/coco2017).
  DRY_RUN=1                           Print every stage without downloading.
  PYTHON=python3                      Python interpreter for COCONut-B and JSON merging.
  HF_BIN=hf                           Hugging Face CLI.
  WGET_BIN=wget                       Resumable downloader.
  OBJECTS365_TRAIN_URL=<URL>          Override the official Objects365 train base URL.

The complete output is $TARGET_DIR/coco/:
  train2017/                       COCO train + unlabeled images + Objects365 images
  panoptic_train2017/              COCONut-B + XL panoptic masks
  annotations/panoptic_train2017.json

The full download is very large. The script is resumable: existing archives,
Hugging Face snapshots, extracted files, links, and final JSON are reused.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$ROOT/datasets_archives/shared_seg}"
TARGET_DIR="${TARGET_DIR:-$ARCHIVE_DIR/coconut_xlarge}"
SOURCE_DIR="${SOURCE_DIR:-$TARGET_DIR/_sources}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$TARGET_DIR/_downloads}"
COCO2017_ARCHIVE_DIR="${COCO2017_ARCHIVE_DIR:-$ARCHIVE_DIR/coco2017}"
COCO_DIR="$TARGET_DIR/coco"
COCO_TRAIN_DIR="$COCO_DIR/train2017"
COCO_MASK_DIR="$COCO_DIR/panoptic_train2017"
COCO_ANNOTATION_DIR="$COCO_DIR/annotations"
FINAL_JSON="$COCO_ANNOTATION_DIR/panoptic_train2017.json"

B_DIR="$SOURCE_DIR/coconut_b"
B_JSON="$B_DIR/coconut_b.json"
B_MASK_DIR="$B_DIR/coconut_b"
LARGE_DIR="$SOURCE_DIR/coconut_large"
XL_DIR="$SOURCE_DIR/coconut_xlarge"
LARGE_EXTRACT_DIR="$SOURCE_DIR/coconut_large_extracted"
XL_EXTRACT_DIR="$SOURCE_DIR/coconut_xlarge_extracted"
OBJECTS365_ARCHIVE_DIR="$SOURCE_DIR/objects365_archives"
OBJECTS365_IMAGE_DIR="$SOURCE_DIR/objects365_images"
B_DOWNLOADER="$SOURCE_DIR/download_coconut.py"
B_RESUMABLE_DOWNLOADER="$SCRIPT_DIR/download_coconut_b_resumable.py"

HF_BIN="${HF_BIN:-hf}"
WGET_BIN="${WGET_BIN:-wget}"
PYTHON="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-0}"

LARGE_HF_REPO="${LARGE_HF_REPO:-xdeng77/coconut_large}"
XL_HF_REPO="${XL_HF_REPO:-xdeng77/coconut_xlarge}"
COCO_TRAIN_URL="${COCO_TRAIN_URL:-http://images.cocodataset.org/zips/train2017.zip}"
COCO_UNLABELED_URL="${COCO_UNLABELED_URL:-http://images.cocodataset.org/zips/unlabeled2017.zip}"
OBJECTS365_TRAIN_URL="${OBJECTS365_TRAIN_URL:-https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train}"
UPSTREAM_B_DOWNLOADER_URL="${UPSTREAM_B_DOWNLOADER_URL:-https://raw.githubusercontent.com/bytedance/coconut_cvpr2024/main/download_coconut.py}"

LARGE_PATCH_IDS=(32 35 40 50)
XL_PATCH_IDS=(17 23 25 28 38 42 44)
ALL_PATCH_IDS=("${LARGE_PATCH_IDS[@]}" "${XL_PATCH_IDS[@]}")

is_true() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" || "$1" == "yes" || "$1" == "YES" ]]
}

run() {
  if is_true "$DRY_RUN"; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command is not available: $1" >&2
    exit 1
  fi
}

has_files() {
  [[ -d "$1" ]] && find "$1" -mindepth 1 \( -type f -o -type l \) -print -quit | grep -q .
}

download_file_if_missing() {
  local output=$1
  local url=$2
  if [[ -s "$output" ]]; then
    echo "== Already present: $output =="
    return
  fi
  run mkdir -p "$(dirname -- "$output")"
  # The final name appears only after a complete download; reruns resume the .partial.
  run "$WGET_BIN" --continue --output-document="$output.partial" "$url"
  run mv -f "$output.partial" "$output"
}

download_hf_if_missing() {
  local repo=$1
  local output=$2
  if has_files "$output"; then
    echo "== Already present: $output =="
    return
  fi
  run mkdir -p "$output"
  run "$HF_BIN" download "$repo" --repo-type dataset --local-dir "$output"
}

extract_archives() {
  local source_root=$1
  local destination=$2
  local found=0
  run mkdir -p "$destination"

  if is_true "$DRY_RUN"; then
    echo "+ tar --skip-old-files -xf '<every .tar/.tar.gz under $source_root>' -C '$destination'"
    return
  fi

  local marker
  while IFS= read -r -d '' archive; do
    found=1
    marker="$destination/.ovb-extracted.$(basename -- "$archive")"
    if [[ -f "$marker" ]]; then
      echo "== Already extracted: $archive =="
      continue
    fi
    echo "== Extracting: $archive =="
    tar --skip-old-files -xf "$archive" -C "$destination"
    touch "$marker"
  done < <(find "$source_root" -type f \( -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' \) -print0)

  if [[ "$found" == "0" ]]; then
    echo "ERROR: no tar archives found under $source_root" >&2
    exit 1
  fi
}

link_files() {
  local source_root=$1
  local destination=$2
  local kind=$3
  local count=0
  local found=0

  run mkdir -p "$destination"
  if is_true "$DRY_RUN"; then
    echo "+ ln -s '<every .$kind file under $source_root>' '$destination/'"
    return
  fi

  while IFS= read -r -d '' source; do
    local name source_abs target
    found=1
    name=$(basename -- "$source")
    target="$destination/$name"
    if [[ -e "$target" || -L "$target" ]]; then
      continue
    fi
    source_abs="$(cd -- "$(dirname -- "$source")" && pwd)/$name"
    # Relative links keep the tree valid when the data root is moved.
    ln -sr "$source_abs" "$target"
    ((count += 1))
  done < <(find "$source_root" -type f -iname "*.$kind" -print0)

  if [[ "$found" == "0" ]]; then
    echo "ERROR: no .$kind files found under $source_root" >&2
    exit 1
  fi
  echo "== Linked $count .$kind files into $destination =="
}

# Link a COCO 2017 zip already fetched by download_datasets_coco.sh into
# DOWNLOAD_DIR, so download_file_if_missing does not fetch it a second time.
link_shared_coco_zip() {
  local name=$1
  local shared="$COCO2017_ARCHIVE_DIR/$name"
  local output="$DOWNLOAD_DIR/$name"
  if [[ -s "$output" || ! -s "$shared" ]]; then
    return
  fi
  run mkdir -p "$DOWNLOAD_DIR"
  run ln -sfnr "$(cd -- "$(dirname -- "$shared")" && pwd)/$name" "$output"
  echo "== Reusing shared $shared =="
}

prepare_coco_images() {
  local train_zip="$DOWNLOAD_DIR/train2017.zip"
  local unlabeled_zip="$DOWNLOAD_DIR/unlabeled2017.zip"
  local unlabeled_dir="$COCO_DIR/unlabeled2017"
  local marker="$COCO_DIR/.unlabeled2017_merged"

  if is_true "$DRY_RUN"; then
    local name
    for name in train2017.zip unlabeled2017.zip; do
      if [[ -s "$COCO2017_ARCHIVE_DIR/$name" ]]; then
        echo "+ ln -s $COCO2017_ARCHIVE_DIR/$name $DOWNLOAD_DIR/$name"
      fi
    done
    echo "+ download and extract COCO train2017.zip into $COCO_TRAIN_DIR"
    echo "+ download and merge COCO unlabeled2017.zip into $COCO_TRAIN_DIR"
    return
  fi

  if ! has_files "$COCO_TRAIN_DIR"; then
    link_shared_coco_zip train2017.zip
    download_file_if_missing "$train_zip" "$COCO_TRAIN_URL"
    run mkdir -p "$COCO_DIR"
    run unzip -q -n "$train_zip" -d "$COCO_DIR"
  fi

  if [[ ! -f "$marker" ]]; then
    link_shared_coco_zip unlabeled2017.zip
    download_file_if_missing "$unlabeled_zip" "$COCO_UNLABELED_URL"
    run unzip -q -n "$unlabeled_zip" -d "$COCO_DIR"
    if ! has_files "$unlabeled_dir"; then
      echo "ERROR: expected COCO unlabeled images at $unlabeled_dir" >&2
      exit 1
    fi
    if ! find "$unlabeled_dir" -maxdepth 1 -type f -print -quit | grep -q .; then
      echo "ERROR: no images found in $unlabeled_dir" >&2
      exit 1
    fi
    # -exec ... + batches paths below the kernel's argument-size limit.
    find "$unlabeled_dir" -maxdepth 1 -type f -exec mv -t "$COCO_TRAIN_DIR" {} +
    run rmdir "$unlabeled_dir"
    run touch "$marker"
  fi
}

prepare_coconut_b() {
  if is_true "$DRY_RUN"; then
    echo "+ download official COCONut-B converter: $UPSTREAM_B_DOWNLOADER_URL"
    echo "+ $PYTHON download_coconut_b_resumable.py --output-dir $B_DIR"
    return
  fi

  if [[ -s "$B_JSON" ]] && has_files "$B_MASK_DIR"; then
    echo "== COCONut-B masks and JSON already present =="
    return
  fi

  download_file_if_missing "$B_DOWNLOADER" "$UPSTREAM_B_DOWNLOADER_URL"
  "$PYTHON" "$B_RESUMABLE_DOWNLOADER" --output-dir "$B_DIR" --official-downloader "$B_DOWNLOADER"
  [[ -s "$B_JSON" ]] || { echo "ERROR: COCONut-B JSON was not created: $B_JSON" >&2; exit 1; }
  has_files "$B_MASK_DIR" || { echo "ERROR: COCONut-B masks were not created: $B_MASK_DIR" >&2; exit 1; }
}

merge_manifests() {
  if is_true "$DRY_RUN"; then
    echo "+ merge COCONut-B JSON with COCONut-L and COCONut-XL Objects365 manifests into $FINAL_JSON"
    return
  fi

  "$PYTHON" "$SCRIPT_DIR/merge_coconut_xlarge.py" \
    --base "$B_JSON" \
    --large-manifest "$LARGE_DIR/panseg_object365_train_v2.json" \
    --large-masks "$LARGE_EXTRACT_DIR/panoptic_object365" \
    --xlarge-root "$XL_EXTRACT_DIR" \
    --output "$FINAL_JSON"
}

# Written after the final checks below; a later run then has nothing to redo.
COMPLETE_MARKER="$TARGET_DIR/.ovb-complete"
if [[ -f "$COMPLETE_MARKER" ]] && ! is_true "$DRY_RUN"; then
  echo "== COCONut-XL tree already complete: $TARGET_DIR (delete $COMPLETE_MARKER to rebuild) =="
  exit 0
fi

if ! is_true "$DRY_RUN"; then
  require_tool "$HF_BIN"
  require_tool "$WGET_BIN"
  require_tool "$PYTHON"
  require_tool unzip
  require_tool tar
  "$PYTHON" -c 'import datasets, tqdm'
fi

run mkdir -p "$TARGET_DIR" "$SOURCE_DIR" "$DOWNLOAD_DIR" "$COCO_TRAIN_DIR" "$COCO_MASK_DIR" "$COCO_ANNOTATION_DIR"
prepare_coco_images
prepare_coconut_b
download_hf_if_missing "$LARGE_HF_REPO" "$LARGE_DIR"
download_hf_if_missing "$XL_HF_REPO" "$XL_DIR"

for patch_id in "${ALL_PATCH_IDS[@]}"; do
  download_file_if_missing "$OBJECTS365_ARCHIVE_DIR/patch${patch_id}.tar.gz" "${OBJECTS365_TRAIN_URL%/}/patch${patch_id}.tar.gz"
done

extract_archives "$LARGE_DIR" "$LARGE_EXTRACT_DIR"
extract_archives "$XL_DIR" "$XL_EXTRACT_DIR"
extract_archives "$OBJECTS365_ARCHIVE_DIR" "$OBJECTS365_IMAGE_DIR"

# XL masks are linked first so a complete XL release supersedes duplicate L masks.
link_files "$B_MASK_DIR" "$COCO_MASK_DIR" png
link_files "$XL_EXTRACT_DIR" "$COCO_MASK_DIR" png
link_files "$LARGE_EXTRACT_DIR" "$COCO_MASK_DIR" png
link_files "$OBJECTS365_IMAGE_DIR" "$COCO_TRAIN_DIR" jpg
merge_manifests

if ! is_true "$DRY_RUN"; then
  [[ -s "$FINAL_JSON" ]] || { echo "ERROR: missing final manifest: $FINAL_JSON" >&2; exit 1; }
  has_files "$COCO_TRAIN_DIR" || { echo "ERROR: no final training images found" >&2; exit 1; }
  has_files "$COCO_MASK_DIR" || { echo "ERROR: no final panoptic masks found" >&2; exit 1; }
  touch "$COMPLETE_MARKER"
fi

cat <<EOF

== Complete COCONut-XL tree ready ==
  images:      $COCO_TRAIN_DIR
  masks:       $COCO_MASK_DIR
  annotations: $FINAL_JSON
EOF
