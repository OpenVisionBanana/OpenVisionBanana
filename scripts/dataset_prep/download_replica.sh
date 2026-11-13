#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Download Replica scene assets and archive them for inode-safe reuse.

The official Replica downloader extracts many files. This helper performs that
work in scratch, then writes one persistent tar:

  /path/to/data/3d_geo/replica/replica_v1_scene_assets.tar

Usage:
  scripts/dataset_prep/download_replica.sh

Important environment variables:
  DATA_ROOT             Persistent data root.
  REPLICA_ROOT          Persistent Replica root.
  WORK_ROOT             Scratch work root. Defaults under $TMPDIR.
  REPLICA_REPO          Replica-Dataset checkout path.
  RAW_EXTRACT_ROOT      Download/extract destination in scratch.
  SCENE_ASSETS_TAR      Final persistent raw scene-asset tar.
  FORCE                 1 to rebuild the checkout/download/tar.
  CLEAN_WORK_ON_EXIT    1 to remove WORK_ROOT after successful or failed run.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT for this dataset}
REPLICA_ROOT=${REPLICA_ROOT:-$DATA_ROOT/3d_geo/replica}
WORK_ROOT=${WORK_ROOT:-${TMPDIR:-/tmp/${USER:-$(id -un)}/replica_download_${SLURM_JOB_ID:-$$}}/replica_download}
REPLICA_REPO=${REPLICA_REPO:-$WORK_ROOT/Replica-Dataset}
RAW_EXTRACT_ROOT=${RAW_EXTRACT_ROOT:-$WORK_ROOT/replica_v1}
SCENE_ASSETS_TAR=${SCENE_ASSETS_TAR:-$REPLICA_ROOT/replica_v1_scene_assets.tar}
FORCE=${FORCE:-0}
CLEAN_WORK_ON_EXIT=${CLEAN_WORK_ON_EXIT:-0}

cleanup() {
  if [[ "$CLEAN_WORK_ON_EXIT" == "1" ]]; then
    rm -rf "$WORK_ROOT"
  fi
}
trap cleanup EXIT

mkdir -p "$REPLICA_ROOT" "$WORK_ROOT"

if [[ -s "$SCENE_ASSETS_TAR" && "$FORCE" != "1" ]]; then
  echo "Using existing Replica scene-asset tar: $SCENE_ASSETS_TAR"
  exit 0
fi

if [[ "$FORCE" == "1" ]]; then
  rm -rf "$REPLICA_REPO" "$RAW_EXTRACT_ROOT"
  rm -f "$SCENE_ASSETS_TAR" "$SCENE_ASSETS_TAR.partial"
fi

if [[ ! -d "$REPLICA_REPO/.git" ]]; then
  git clone https://github.com/facebookresearch/Replica-Dataset.git "$REPLICA_REPO"
else
  echo "Using existing Replica-Dataset checkout: $REPLICA_REPO"
fi

if [[ ! -d "$RAW_EXTRACT_ROOT" || "$FORCE" == "1" ]]; then
  mkdir -p "$RAW_EXTRACT_ROOT"
  (
    cd "$REPLICA_REPO"
    ./download.sh "$RAW_EXTRACT_ROOT"
  )
fi

if ! find "$RAW_EXTRACT_ROOT" -mindepth 2 -type f -print -quit | grep -q .; then
  echo "ERROR: Replica download produced no files under $RAW_EXTRACT_ROOT" >&2
  exit 1
fi

tmp_tar="$SCENE_ASSETS_TAR.partial"
rm -f "$tmp_tar"
echo "Archiving Replica scene assets to $SCENE_ASSETS_TAR"
tar -cf "$tmp_tar" -C "$(dirname "$RAW_EXTRACT_ROOT")" "$(basename "$RAW_EXTRACT_ROOT")"
mv -f "$tmp_tar" "$SCENE_ASSETS_TAR"

echo "Replica download finished."
echo "  raw scratch: $RAW_EXTRACT_ROOT"
echo "  raw tar:     $SCENE_ASSETS_TAR"
