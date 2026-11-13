#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Check or stage manually downloaded nuScenes archives.

nuScenes downloads require an official account and accepted terms, so this
script does not fetch data from the internet. Download the archive from:

  https://www.nuscenes.org/nuscenes#download

Then place it under DATA_ROOT or point NUSCENES_ARCHIVE at it.

Environment:
  DATA_ROOT          nuScenes storage root.
                     Default: /path/to/data/depth_est/nuScenes
  VERSION           v1.0-mini or v1.0-trainval. Default: v1.0-trainval.
  NUSCENES_ARCHIVE  Downloaded archive path. Default for mini: DATA_ROOT/v1.0-mini.tar
  META_ARCHIVE      Trainval metadata archive. By default, auto-detects
                    DATA_ROOT/v1.0-trainval_meta.tar or .tgz.
  BLOBS_GLOB        Trainval blob archive glob.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT for this dataset}
VERSION=${VERSION:-v1.0-trainval}

case "$VERSION" in
  v1.0-mini)
    NUSCENES_ARCHIVE=${NUSCENES_ARCHIVE:-$DATA_ROOT/v1.0-mini.tar}
    ;;
  v1.0-trainval)
    if [[ -z "${META_ARCHIVE:-}" ]]; then
      if [[ -f "$DATA_ROOT/v1.0-trainval_meta.tar" ]]; then
        META_ARCHIVE=$DATA_ROOT/v1.0-trainval_meta.tar
      else
        META_ARCHIVE=$DATA_ROOT/v1.0-trainval_meta.tgz
      fi
    fi
    BLOBS_GLOB=${BLOBS_GLOB:-$DATA_ROOT/v1.0-trainval??_blobs.tgz}
    ;;
  *)
    echo "ERROR: Unsupported VERSION=$VERSION" >&2
    exit 1
    ;;
esac

mkdir -p "$DATA_ROOT"
if [[ "$VERSION" == "v1.0-trainval" ]]; then
  if [[ ! -f "$META_ARCHIVE" ]]; then
    echo "ERROR: Missing nuScenes trainval metadata archive: $META_ARCHIVE" >&2
    exit 1
  fi
  mapfile -t blob_archives < <(compgen -G "$BLOBS_GLOB" | sort)
  if [[ ${#blob_archives[@]} -ne 10 ]]; then
    echo "ERROR: Expected 10 trainval blob archives matching $BLOBS_GLOB; found ${#blob_archives[@]}" >&2
    exit 1
  fi
  echo "Found nuScenes trainval metadata: $META_ARCHIVE"
  printf 'Found nuScenes trainval blob: %s\n' "${blob_archives[@]}"
  echo "DATA_ROOT=$DATA_ROOT"
  exit 0
fi

if [[ ! -f "$NUSCENES_ARCHIVE" ]]; then
  cat >&2 <<EOF
ERROR: Missing nuScenes archive: $NUSCENES_ARCHIVE

Download $VERSION from the official nuScenes page after logging in and accepting
the dataset terms, then place the file there or set NUSCENES_ARCHIVE.
EOF
  exit 1
fi

echo "Found nuScenes archive: $NUSCENES_ARCHIVE"
echo "DATA_ROOT=$DATA_ROOT"
