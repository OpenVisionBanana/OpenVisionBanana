#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Clone Apple's ml-hypersim tools and run an official Hypersim download command.

This helper intentionally keeps raw archives/extracted files in scratch-friendly
locations. For inode-limited persistent storage, run it inside a CPU job where
$TMPDIR is available, then convert to prepared tars with
scripts/dataset_prep/prepare_hypersim_depth_normals.sh.

Usage:
  scripts/dataset_prep/download_hypersim.sh

Important environment variables:
  DATA_ROOT              Persistent data root.
  HYPERSIM_ROOT          Persistent Hypersim root.
  HYPERSIM_REPO          Location for the ml-hypersim clone.
  DOWNLOADS_DIR          Where official archives are downloaded.
  DECOMPRESS_DIR         Where archives are extracted.
  DOWNLOAD_MODE          full, partial, or help. Default: help.
  PARTIAL_ARGS           Extra args for contrib/99991/download.py.
  DELETE_ARCHIVES        1 to pass --delete_archive_after_decompress in full mode.

Modes:
  DOWNLOAD_MODE=full
    Runs code/python/tools/dataset_download_images.py.

  DOWNLOAD_MODE=partial
    Runs contrib/99991/download.py with PARTIAL_ARGS. Use
    PARTIAL_ARGS="--list" first to inspect available files for your clone.

  DOWNLOAD_MODE=help
    Prints the official partial downloader help/list commands without
    downloading anything.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

ROOT=${ROOT:-$DEFAULT_ROOT}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT for this dataset}
HYPERSIM_ROOT=${HYPERSIM_ROOT:-$DATA_ROOT/depth_surfn_data/hypersim}
WORK_ROOT=${WORK_ROOT:-${TMPDIR:-$HYPERSIM_ROOT/work}}
HYPERSIM_REPO=${HYPERSIM_REPO:-$WORK_ROOT/ml-hypersim}
DOWNLOADS_DIR=${DOWNLOADS_DIR:-$WORK_ROOT/raw_downloads}
DECOMPRESS_DIR=${DECOMPRESS_DIR:-$WORK_ROOT/raw_extracted}
DOWNLOAD_MODE=${DOWNLOAD_MODE:-help}
DELETE_ARCHIVES=${DELETE_ARCHIVES:-0}
PARTIAL_ARGS=${PARTIAL_ARGS:-}
PYTHON=${PYTHON:-python3}

mkdir -p "$WORK_ROOT" "$DOWNLOADS_DIR" "$DECOMPRESS_DIR"

if [[ ! -d "$HYPERSIM_REPO/.git" ]]; then
  rm -rf "$HYPERSIM_REPO"
  git clone https://github.com/apple/ml-hypersim.git "$HYPERSIM_REPO"
else
  echo "Using existing ml-hypersim checkout: $HYPERSIM_REPO"
fi

case "${DOWNLOAD_MODE,,}" in
  full)
    args=(
      "$HYPERSIM_REPO/code/python/tools/dataset_download_images.py"
      --downloads_dir "$DOWNLOADS_DIR"
      --decompress_dir "$DECOMPRESS_DIR"
    )
    if [[ "$DELETE_ARCHIVES" == "1" ]]; then
      args+=(--delete_archive_after_decompress)
    fi
    "$PYTHON" "${args[@]}"
    ;;
  partial)
    cd "$HYPERSIM_REPO/contrib/99991"
    # shellcheck disable=SC2206
    args=($PARTIAL_ARGS)
    "$PYTHON" ./download.py "${args[@]}"
    ;;
  help)
    cat <<EOF
Hypersim tools are ready.

Repo:
  $HYPERSIM_REPO

Scratch download/extract roots:
  DOWNLOADS_DIR=$DOWNLOADS_DIR
  DECOMPRESS_DIR=$DECOMPRESS_DIR

Inspect partial downloader:
  cd "$HYPERSIM_REPO/contrib/99991"
  $PYTHON ./download.py --help
  $PYTHON ./download.py --list

Full image download:
  DOWNLOAD_MODE=full scripts/dataset_prep/download_hypersim.sh

Partial download example:
  DOWNLOAD_MODE=partial PARTIAL_ARGS="--list" scripts/dataset_prep/download_hypersim.sh
EOF
    ;;
  *)
    echo "ERROR: DOWNLOAD_MODE must be full, partial, or help. Got: $DOWNLOAD_MODE" >&2
    exit 1
    ;;
esac

echo "Hypersim download helper finished."
echo "  repo:          $HYPERSIM_REPO"
echo "  downloads:     $DOWNLOADS_DIR"
echo "  extracted raw: $DECOMPRESS_DIR"
