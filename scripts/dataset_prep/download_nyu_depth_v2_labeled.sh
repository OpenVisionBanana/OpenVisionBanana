#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Download the NYU Depth V2 labeled MATLAB file and official train/test split.

By default this writes:
  /path/to/data/depth_est/nyu_depth_v2_labeled.mat
  /path/to/data/depth_est/splits.mat

Configuration style:
  Set DATA_ROOT, LABELED_MAT, SPLITS_MAT, FORCE, and downloader variables in
  the environment before running.

Example:
  scripts/dataset_prep/download_nyu_depth_v2_labeled.sh

The labeled file URL matches the direct command:
  wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT for this dataset}
LABELED_URL=${LABELED_URL:-http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat}
SPLITS_URL=${SPLITS_URL:-http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat}
LABELED_MAT=${LABELED_MAT:-$DATA_ROOT/nyu_depth_v2_labeled.mat}
SPLITS_MAT=${SPLITS_MAT:-$DATA_ROOT/splits.mat}
FORCE=${FORCE:-0}
DOWNLOAD_RETRIES=${DOWNLOAD_RETRIES:-4}

DOWNLOADER=
if command -v wget >/dev/null 2>&1; then
  DOWNLOADER=wget
elif command -v curl >/dev/null 2>&1; then
  DOWNLOADER=curl
else
  echo "ERROR: need either wget or curl in PATH." >&2
  exit 1
fi

download_one() {
  local url=$1
  local dst=$2
  mkdir -p "$(dirname "$dst")"
  if [[ "$FORCE" == "1" ]]; then
    rm -f "$dst"
  fi
  if [[ -s "$dst" ]]; then
    echo "Using existing file: $dst"
    return
  fi

  echo "Downloading $url"
  if [[ "$DOWNLOADER" == "wget" ]]; then
    wget -c \
      --tries="$DOWNLOAD_RETRIES" \
      --waitretry=2 \
      --retry-connrefused \
      -O "$dst" "$url"
  else
    curl -L --fail --retry "$DOWNLOAD_RETRIES" --retry-delay 2 -C - -o "$dst" "$url"
  fi
}

download_one "$LABELED_URL" "$LABELED_MAT"
download_one "$SPLITS_URL" "$SPLITS_MAT"

echo "NYU Depth V2 labeled download finished."
echo "  labeled mat: $LABELED_MAT"
echo "  splits mat:  $SPLITS_MAT"
