#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Download ScanNet v2 assets needed for depth and depth-derived normal evaluation.

You must have accepted the ScanNet Terms of Use before running this script.
The official downloader prompts for confirmation; this wrapper answers that
prompt after you opt in with ACCEPT_SCANNET_TOS=1.

Usage:
  ACCEPT_SCANNET_TOS=1 scripts/dataset_prep/download_scannet.sh

Environment:
  DATA_ROOT            Destination root. Default: /path/to/data/depth_est/scannet
  DOWNLOAD_SCRIPT      Required path to the official ScanNet downloader.
  ACCEPT_SCANNET_TOS   Must be 1/true/yes to run the official downloader.
  FORCE_SPLITS         1 to re-download split files from the official ScanNet GitHub repo.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT for this dataset}
DOWNLOAD_SCRIPT=${DOWNLOAD_SCRIPT:?Set DOWNLOAD_SCRIPT to the official ScanNet downloader}
ACCEPT_SCANNET_TOS=${ACCEPT_SCANNET_TOS:-0}
FORCE_SPLITS=${FORCE_SPLITS:-0}

if ! is_true "$ACCEPT_SCANNET_TOS"; then
  echo "ERROR: Set ACCEPT_SCANNET_TOS=1 after accepting the ScanNet Terms of Use." >&2
  exit 1
fi
if [[ ! -f "$DOWNLOAD_SCRIPT" ]]; then
  echo "ERROR: Missing official ScanNet downloader: $DOWNLOAD_SCRIPT" >&2
  exit 1
fi

mkdir -p "$DATA_ROOT/tasks"

if [[ ! -s "$DATA_ROOT/tasks/scannet_frames_25k.zip" ]]; then
  printf '\n' | python "$DOWNLOAD_SCRIPT" -o "$DATA_ROOT" --preprocessed_frames
else
  echo "Using existing ScanNet frames archive: $DATA_ROOT/tasks/scannet_frames_25k.zip"
fi

download_split() {
  local name=$1
  local out=$2
  local url="https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/$name"
  if [[ -s "$out" ]] && ! is_true "$FORCE_SPLITS"; then
    echo "Using existing split file: $out"
    return
  fi
  curl -L --fail -o "$out" "$url"
}

download_split scannetv2_train.txt "$DATA_ROOT/tasks/scannetv2_train.txt"
download_split scannetv2_val.txt "$DATA_ROOT/tasks/scannetv2_val.txt"
download_split scannetv2_test.txt "$DATA_ROOT/tasks/scannetv2_test.txt"

echo "ScanNet download assets are ready under: $DATA_ROOT"
