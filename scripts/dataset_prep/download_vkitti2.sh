#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Download the Virtual KITTI 2 archives needed for depth and derived-normal evaluation.

Usage:
  scripts/dataset_prep/download_vkitti2.sh

Environment:
  DATA_ROOT      Destination directory. Default: /path/to/data/depth_est/vkitti_2
  RGB_URL        RGB archive URL.
  DEPTH_URL      Depth archive URL.
  TEXTGT_URL     Text ground-truth archive URL.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT for this dataset}
RGB_URL=${RGB_URL:-https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar}
DEPTH_URL=${DEPTH_URL:-https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar}
TEXTGT_URL=${TEXTGT_URL:-https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_textgt.tar.gz}

mkdir -p "$DATA_ROOT"

download_if_needed() {
  local url=$1
  local filename=$2
  local output=$DATA_ROOT/$filename
  if [[ -s "$output" ]]; then
    echo "Using existing $output"
    return
  fi
  echo "Downloading $filename"
  wget -O "$output.partial" "$url"
  mv "$output.partial" "$output"
}

download_if_needed "$RGB_URL" vkitti_2.0.3_rgb.tar
download_if_needed "$DEPTH_URL" vkitti_2.0.3_depth.tar
download_if_needed "$TEXTGT_URL" vkitti_2.0.3_textgt.tar.gz

echo "Virtual KITTI 2 downloads are ready under $DATA_ROOT"
