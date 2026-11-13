#!/usr/bin/env bash
# Download/repack VIPSeg for Open Vision Banana dataset prep.
#
# Output:
#   datasets_archives/sem_segm/vipseg.tar -> VIPSeg/{imgs,panomasks,VIPSeg_720P,...}
#
# The official archive is downloaded with gdown using the Google Drive id from
# the VIPSeg README. If you already have an extracted copy, set
# VIPSEG_EXTRACTED_ROOT=/path/to/VIPSeg to skip downloading and only validate +
# pack it.
set -euo pipefail

show_help() {
  cat <<'EOF'
Download/repack VIPSeg for Open Vision Banana dataset prep.

Outputs:
  $ARCHIVE_DIR/vipseg.tar

Configuration:
  ROOT=/path/to/open_visionbanana       repo/data root used for defaults
  ARCHIVE_DIR=$ROOT/datasets_archives/sem_segm
  VIPSEG_TAR=$ARCHIVE_DIR/vipseg.tar
  TMPDIR=/fast/scratch                  scratch parent for temporary work
  WORK=$TMPDIR/open_visionbanana_vipseg_download
  PACKAGE_DIR=$WORK/packages            downloaded tar location
  EXTRACT_ROOT=$WORK/extracted          temporary extract location
  VIPSEG_EXTRACTED_ROOT=/tmp/data/VIPSeg existing extracted VIPSeg directory
  FORCE=0                               redownload/repack existing outputs
  KEEP_EXTRACTED=0                      keep $WORK after successful packing

Download command used when VIPSEG_EXTRACTED_ROOT is not set:
  gdown "https://drive.google.com/uc?id=1B13QUiE82xf7N6nVHclb4ErN-Zuai-sZ"
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

ROOT=${ROOT:-$HOME/open_visionbanana}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/sem_segm}
VIPSEG_TAR=${VIPSEG_TAR:-$ARCHIVE_DIR/vipseg.tar}
TMP_PARENT=${TMPDIR:-/tmp/${USER:-user}}
WORK=${WORK:-$TMP_PARENT/open_visionbanana_vipseg_download}
PACKAGE_DIR=${PACKAGE_DIR:-$WORK/packages}
EXTRACT_ROOT=${EXTRACT_ROOT:-$WORK/extracted}
VIPSEG_EXTRACTED_ROOT=${VIPSEG_EXTRACTED_ROOT:-}
FORCE=${FORCE:-0}
KEEP_EXTRACTED=${KEEP_EXTRACTED:-0}

VIPSEG_DRIVE_ID=${VIPSEG_DRIVE_ID:-1B13QUiE82xf7N6nVHclb4ErN-Zuai-sZ}
VIPSEG_URL=${VIPSEG_URL:-https://drive.google.com/uc?id=$VIPSEG_DRIVE_ID}

mkdir -p "$ARCHIVE_DIR"

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required tool not found in PATH: $1" >&2
    exit 1
  fi
}

validate_vipseg_dir() {
  local dir=$1
  local missing=0
  for path in \
    "$dir/imgs" \
    "$dir/panomasks" \
    "$dir/train.txt" \
    "$dir/val.txt" \
    "$dir/VIPSeg_720P/panoVIPSeg_categories.json"; do
    if [[ ! -e "$path" ]]; then
      echo "ERROR: missing required VIPSeg path: $path" >&2
      missing=1
    fi
  done
  if [[ "$missing" != "0" ]]; then
    exit 1
  fi
}

resolve_vipseg_dir() {
  local root=$1
  for candidate in "$root" "$root/VIPSeg" "$root/vipseg"; do
    if [[ -d "$candidate/imgs" && -d "$candidate/panomasks" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "ERROR: could not find VIPSeg under $root (expected imgs/ and panomasks/)." >&2
  exit 1
}

download_file() {
  local out=$1
  if [[ "$FORCE" == "1" ]]; then
    rm -f "$out"
  fi
  if [[ -s "$out" ]]; then
    echo "== Already present: $out =="
    return
  fi
  require_tool gdown
  mkdir -p "$(dirname "$out")"
  echo "== Downloading VIPSeg =="
  echo "   gdown \"$VIPSEG_URL\""
  (cd "$(dirname "$out")" && gdown "$VIPSEG_URL")
  if [[ ! -s "$out" ]]; then
    local downloaded
    downloaded=$(find "$(dirname "$out")" -maxdepth 1 -type f -name 'VIPSeg*.tar' -print -quit)
    if [[ -n "$downloaded" && "$downloaded" != "$out" ]]; then
      mv -f "$downloaded" "$out"
    fi
  fi
  [[ -s "$out" ]] || { echo "ERROR: VIPSeg download did not produce $out" >&2; exit 1; }
}

extract_archive() {
  local archive=$1
  local out=$2
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$out"
  fi
  if [[ -d "$out" ]]; then
    echo "== Already extracted: $out =="
    return
  fi
  require_tool tar
  mkdir -p "$out"
  echo "== Extracting $archive to $out =="
  tar -xf "$archive" -C "$out"
}

if [[ "$FORCE" == "1" ]]; then
  rm -f "$VIPSEG_TAR"
fi
if [[ -s "$VIPSEG_TAR" && "$FORCE" != "1" ]]; then
  echo "== VIPSeg tar already present: $VIPSEG_TAR =="
  exit 0
fi

if [[ -n "$VIPSEG_EXTRACTED_ROOT" ]]; then
  VIPSEG_DIR=$(resolve_vipseg_dir "$VIPSEG_EXTRACTED_ROOT")
else
  package=$PACKAGE_DIR/VIPSeg.tar
  download_file "$package"
  extract_archive "$package" "$EXTRACT_ROOT"
  VIPSEG_DIR=$(resolve_vipseg_dir "$EXTRACT_ROOT")
fi

validate_vipseg_dir "$VIPSEG_DIR"

echo "== Packing VIPSeg tar =="
mkdir -p "$(dirname "$VIPSEG_TAR")"
tar -C "$(dirname "$VIPSEG_DIR")" -cf "$VIPSEG_TAR" "$(basename "$VIPSEG_DIR")"
echo "   -> $VIPSEG_TAR"

if [[ -z "$VIPSEG_EXTRACTED_ROOT" && "$KEEP_EXTRACTED" != "1" ]]; then
  echo "== Cleaning temporary files: $WORK (set KEEP_EXTRACTED=1 to keep) =="
  rm -rf "$WORK"
fi

cat <<EOF

Done. Use with the full pool builder:
  VIPSEG_TAR=$VIPSEG_TAR \\
    bash scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh

Or run the converters directly for a quick prep check:
  python dataset_prep/semseg/dataset_prep.py --vipseg-root "$VIPSEG_DIR" --out-root /tmp/vipseg_semseg --splits train,val --num-images 20
  python dataset_prep/instanceseg/dataset_prep.py --vipseg-root "$VIPSEG_DIR" --out-root /tmp/vipseg_instanceseg --splits train,val --limit 20
EOF
