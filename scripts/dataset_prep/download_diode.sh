#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Download and extract DIODE validation data for evaluation-only preparation.

Usage:
  scripts/dataset_prep/download_diode.sh

Environment:
  DATA_ROOT             Destination root. Default: /path/to/data/depth_surfn_data/diode
  FORCE_DOWNLOAD        1 to re-download archives.
  FORCE_EXTRACT         1 to re-extract val/, val_normal/, and data_list/.
  VERIFY_MD5            1 to verify official validation archive MD5s. Default: 1.

This downloads only validation RGB/depth/mask data, validation normals, data
lists, and the small official devkit. It does not download DIODE training data.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT for this dataset}
FORCE_DOWNLOAD=${FORCE_DOWNLOAD:-0}
FORCE_EXTRACT=${FORCE_EXTRACT:-0}
VERIFY_MD5=${VERIFY_MD5:-1}

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
  wget -O "$out" "$url"
}

verify_md5() {
  local expected=$1
  local file=$2
  local actual
  actual=$(md5sum "$file" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: MD5 mismatch for $file" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi
}

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

download_if_needed https://diode-dataset.s3.amazonaws.com/val.tar.gz val.tar.gz
download_if_needed https://diode-dataset.s3.amazonaws.com/val_normals.tar.gz val_normals.tar.gz
download_if_needed https://diode-1254389886.cos.ap-hongkong.myqcloud.com/data_list.zip data_list.zip

if is_true "$VERIFY_MD5"; then
  verify_md5 5c895d09201b88973c8fe4552a67dd85 val.tar.gz
  verify_md5 323ccaf60abebdb59705dcd8b529d709 val_normals.tar.gz
fi

if [[ ! -d val ]] || is_true "$FORCE_EXTRACT"; then
  rm -rf val
  tar -xzf val.tar.gz
fi

if [[ ! -d val_normal ]] || is_true "$FORCE_EXTRACT"; then
  rm -rf val_normal .diode-val-normals-extract
  mkdir -p .diode-val-normals-extract
  tar -xzf val_normals.tar.gz -C .diode-val-normals-extract
  mv .diode-val-normals-extract/val val_normal
  rm -rf .diode-val-normals-extract
fi

if [[ ! -d data_list ]] || is_true "$FORCE_EXTRACT"; then
  rm -rf data_list __MACOSX
  unzip -q data_list.zip
  rm -rf __MACOSX
fi

if [[ ! -d diode-devkit ]]; then
  git clone https://github.com/diode-dataset/diode-devkit.git
fi

echo "DIODE validation data is ready under: $DATA_ROOT"
