#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Download iBims-1 raw archives used for depth and derived surface-normal evaluation.

Usage:
  scripts/dataset_prep/download_ibims1.sh

Environment:
  DATA_ROOT     Output directory. Default: /path/to/data/depth_est/ibims1
  INCLUDE_AUX   1 to download aux_augmented, aux_mvs, and aux_patterns. Default: 1
  FORCE         1 to re-download existing files. Default: 0
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT for this dataset}
INCLUDE_AUX=${INCLUDE_AUX:-1}
FORCE=${FORCE:-0}
RSYNC_PASSWORD_VALUE=${RSYNC_PASSWORD_VALUE:-m1455541}
RSYNC_BASE=${RSYNC_BASE:-rsync://m1455541@dataserv.ub.tum.de/m1455541}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

download_one() {
  local name=$1
  if [[ -s "$DATA_ROOT/$name" ]] && ! is_true "$FORCE"; then
    echo "Using existing: $DATA_ROOT/$name"
    return
  fi
  echo "Downloading $name"
  RSYNC_PASSWORD="$RSYNC_PASSWORD_VALUE" rsync --progress "$RSYNC_BASE/$name" "$DATA_ROOT/"
}

mkdir -p "$DATA_ROOT"
download_one ibims1_core_mat.zip
download_one evaluation_scripts.zip
download_one imagelist.txt
download_one Ibims_Dataset.sha512

if is_true "$INCLUDE_AUX"; then
  download_one ibims1_aux_augmented.zip
  download_one ibims1_aux_mvs.zip
  download_one ibims1_aux_patterns.zip
fi

if [[ -f "$DATA_ROOT/Ibims_Dataset.sha512" ]]; then
  echo "Verifying downloaded files with sha512sum --ignore-missing"
  (cd "$DATA_ROOT" && sha512sum -c Ibims_Dataset.sha512 --ignore-missing)
fi

echo "iBims-1 download finished: $DATA_ROOT"
