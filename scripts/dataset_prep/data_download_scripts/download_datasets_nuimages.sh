#!/usr/bin/env bash
# Build datasets_archives/shared_seg/nuimages.tar from the official nuImages
# release (https://www.nuscenes.org/nuimages, registration required). The outer
# tar only bundles the two release archives, which staging extracts by name:
#
#   nuimages.tar -> nuimages-v1.0-all-metadata.tgz   tables + annotations
#                   nuimages-v1.0-all-samples.tgz    annotated key-camera JPEGs
#
# The sweeps archives (unannotated frames) are not needed. Semantic and instance
# segmentation both read this archive.
#
# Manual download:
#   1. Log in at https://www.nuscenes.org/nuimages and, under Downloads > nuImages,
#      get "Metadata" and "Samples" for the full v1.0 release (the two files above).
#   2. Put them in $NUIMAGES_DIR (default datasets_archives/shared_seg/nuimages/).
#   3. Run this script.
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_nuimages.sh
#   REMOVE_DOWNLOADS=1 ...   # delete the two downloads once bundled (saves ~17 GB)
#   FORCE=1 ...              # rebuild an existing nuimages.tar
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$ROOT/datasets_archives/shared_seg}"
NUIMAGES_DIR=${NUIMAGES_DIR:-$ARCHIVE_DIR/nuimages}
REMOVE_DOWNLOADS=${REMOVE_DOWNLOADS:-0}
FORCE=${FORCE:-0}

OUT_TAR=$ARCHIVE_DIR/nuimages.tar
MEMBERS=(nuimages-v1.0-all-metadata.tgz nuimages-v1.0-all-samples.tgz)

if [[ "$FORCE" != 1 && -s "$OUT_TAR" ]]; then
  echo "== Using existing $OUT_TAR (FORCE=1 rebuilds) =="
  exit 0
fi
mkdir -p "$NUIMAGES_DIR"

missing=()
for member in "${MEMBERS[@]}"; do
  [[ -s "$NUIMAGES_DIR/$member" ]] || missing+=("$member")
done
if (( ${#missing[@]} )); then
  cat >&2 <<EOF
nuImages archive(s) not found in $NUIMAGES_DIR: ${missing[*]}
Log in at https://www.nuscenes.org/nuimages, download the full v1.0 "Metadata"
and "Samples" archives into that folder (or set NUIMAGES_DIR), then re-run.
EOF
  exit 1
fi

echo "== Checking ${MEMBERS[0]} =="
# grep without -q reads the whole listing, so tar never sees SIGPIPE under pipefail.
tar -tzf "$NUIMAGES_DIR/${MEMBERS[0]}" | grep 'v1.0-train/category.json' > /dev/null || {
  echo "ERROR: ${MEMBERS[0]} has no v1.0-train/category.json." >&2
  exit 1
}

echo "== Bundling into $OUT_TAR =="
mkdir -p "$ARCHIVE_DIR"
tar -cf "$OUT_TAR.partial" -C "$NUIMAGES_DIR" "${MEMBERS[@]}"
mv -f "$OUT_TAR.partial" "$OUT_TAR"
if [[ "$REMOVE_DOWNLOADS" == 1 ]]; then
  for member in "${MEMBERS[@]}"; do rm -f "$NUIMAGES_DIR/$member"; done
fi
echo "Done: $OUT_TAR"
