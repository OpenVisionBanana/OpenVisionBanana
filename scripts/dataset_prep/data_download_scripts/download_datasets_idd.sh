#!/usr/bin/env bash
# Build datasets_archives/shared_seg/idd_1_2.tar from the two official India
# Driving Dataset segmentation releases (https://idd.insaan.iiit.ac.in/,
# registration required). The outer tar only bundles the two release archives,
# which staging extracts by name:
#
#   idd_1_2.tar -> idd-segmentation.tar.gz   IDD Segmentation (IDD 20k Part I)
#                  idd-20k-II.tar.gz         IDD Segmentation (IDD 20k Part II)
#
# Semantic and instance segmentation both read this archive.
#
# Manual download:
#   1. Log in at https://idd.insaan.iiit.ac.in/, accept the license, and under
#      Dataset > Download get both files named above.
#   2. Put them in $IDD_DIR (default datasets_archives/shared_seg/idd/).
#   3. Run this script.
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_idd.sh
#   REMOVE_DOWNLOADS=1 ...   # delete the two downloads once bundled (saves ~25 GB)
#   FORCE=1 ...              # rebuild an existing idd_1_2.tar
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$ROOT/datasets_archives/shared_seg}"
IDD_DIR=${IDD_DIR:-$ARCHIVE_DIR/idd}
REMOVE_DOWNLOADS=${REMOVE_DOWNLOADS:-0}
FORCE=${FORCE:-0}

OUT_TAR=$ARCHIVE_DIR/idd_1_2.tar
MEMBERS=(idd-segmentation.tar.gz idd-20k-II.tar.gz)

if [[ "$FORCE" != 1 && -s "$OUT_TAR" ]]; then
  echo "== Using existing $OUT_TAR (FORCE=1 rebuilds) =="
  exit 0
fi
mkdir -p "$IDD_DIR"

missing=()
for member in "${MEMBERS[@]}"; do
  [[ -s "$IDD_DIR/$member" ]] || missing+=("$member")
done
if (( ${#missing[@]} )); then
  cat >&2 <<EOF
IDD archive(s) not found in $IDD_DIR: ${missing[*]}
1. Register/log in and accept the license at https://idd.insaan.iiit.ac.in/
2. From Dataset > Download, get BOTH segmentation releases:
     IDD Segmentation (IDD 20k Part I)  -> idd-segmentation.tar.gz
     IDD Segmentation (IDD 20k Part II) -> idd-20k-II.tar.gz
3. Save them in $IDD_DIR (or set IDD_DIR) and re-run.
EOF
  exit 1
fi

for member in "${MEMBERS[@]}"; do
  echo "== Checking $member =="
  gzip -t "$IDD_DIR/$member"
done

echo "== Bundling into $OUT_TAR =="
mkdir -p "$ARCHIVE_DIR"
tar -cf "$OUT_TAR.partial" -C "$IDD_DIR" "${MEMBERS[@]}"
mv -f "$OUT_TAR.partial" "$OUT_TAR"
if [[ "$REMOVE_DOWNLOADS" == 1 ]]; then
  for member in "${MEMBERS[@]}"; do rm -f "$IDD_DIR/$member"; done
fi
echo "Done: $OUT_TAR"
