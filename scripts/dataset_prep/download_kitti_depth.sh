#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Download KITTI depth benchmark archives used by the evaluation prep.

Usage:
  scripts/dataset_prep/download_kitti_depth.sh

Environment:
  DATA_ROOT          Destination root. Default: /path/to/data/depth_est/kitti
  FORCE_DOWNLOAD     1 to re-download existing archives. Default: 0.
  DOWNLOAD_FULL      1 to also download annotated and velodyne archives. Default: 0.

The KITTI website may require browser-based login for some users. If wget fails,
download the same archive names manually into DATA_ROOT and rerun preparation.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT for this dataset}
FORCE_DOWNLOAD=${FORCE_DOWNLOAD:-0}
DOWNLOAD_FULL=${DOWNLOAD_FULL:-0}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

download_if_needed() {
  local url=$1
  local out=$2
  if [[ -s "$out" ]] && ! is_true "$FORCE_DOWNLOAD"; then
    echo "Using existing $(basename "$out")"
    return
  fi
  wget -c -O "$out" "$url"
}

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

download_if_needed https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip data_depth_selection.zip

if is_true "$DOWNLOAD_FULL"; then
  download_if_needed https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip data_depth_annotated.zip
  download_if_needed https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip data_depth_velodyne.zip
fi

echo "KITTI depth archives are ready under: $DATA_ROOT"
