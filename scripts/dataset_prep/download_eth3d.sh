#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Download the ETH3D low-res two-view training archives used for monocular-input
depth and derived surface-normal evaluation.

Usage:
  scripts/dataset_prep/download_eth3d.sh

Environment:
  DATA_ROOT      Output directory. Default:
                 /path/to/data/depth_est/eth3d
  FORCE          1 to re-download existing archives. Default: 0.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT for this dataset}
FORCE=${FORCE:-0}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

download_one() {
  local url=$1
  local output=$2
  if [[ -s "$output" ]] && ! is_true "$FORCE"; then
    echo "Using existing archive: $output"
    return
  fi
  mkdir -p "$(dirname "$output")"
  echo "Downloading $url"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$output.partial" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$output.partial"
  else
    echo "ERROR: neither wget nor curl is available." >&2
    exit 1
  fi
  mv "$output.partial" "$output"
}

mkdir -p "$DATA_ROOT"
download_one "https://www.eth3d.net/data/two_view_training.7z" "$DATA_ROOT/two_view_training.7z"
download_one "https://www.eth3d.net/data/two_view_training_gt.7z" "$DATA_ROOT/two_view_training_gt.7z"

echo "ETH3D download finished: $DATA_ROOT"
