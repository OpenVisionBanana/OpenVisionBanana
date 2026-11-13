#!/usr/bin/env bash
set -euo pipefail

# Download all Pico-Banana edited target images from the SFT manifest into the
# layout expected by the prep/training scripts, then tar that folder for reuse.

show_help() {
  cat <<'EOF'
Download Pico-Banana edited target images and optionally bundle them into a tar
archive for later reuse.

Use case:
  Scratch/TMPDIR + archive/inode-aware setup.

Choose this helper when you want a reusable positive-edit tarball that can be
staged into local scratch by:
  scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh

It also works in the simple persistent-folder setup, but its biggest advantage
is when you care about inode pressure or want to avoid keeping the full edited
image tree permanently extracted.

Configuration style:
  Set ROOT, POSITIVE_EDIT_DIR, TAR_OUTPUT, JOBS, and related variables in the
  environment before running.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
ROOT=${ROOT:-${DATA_ROOT:-$REPO_ROOT/datasets_archives}/pico-banana-400k}
OPENIMAGES_DIR=${OPENIMAGES_DIR:-$ROOT/openimages}
MANIFEST=${MANIFEST:-$OPENIMAGES_DIR/sft_manifest.txt}
MANIFEST_URL=${MANIFEST_URL:-https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/manifest/sft_manifest.txt}
IMAGES_ROOT=${IMAGES_ROOT:-$OPENIMAGES_DIR/images}
POSITIVE_EDIT_DIR=${POSITIVE_EDIT_DIR:-$IMAGES_ROOT/positive-edit}
TAR_OUTPUT=${TAR_OUTPUT:-$OPENIMAGES_DIR/positive-edit.tar}
JOBS=${JOBS:-32}
FORCE=${FORCE:-0}
MAKE_TAR=${MAKE_TAR:-1}
DOWNLOAD_RETRIES=${DOWNLOAD_RETRIES:-4}
RETRY_ROUNDS=${RETRY_ROUNDS:-3}

if ! command -v xargs >/dev/null 2>&1; then
  echo "ERROR: xargs is required but was not found in PATH." >&2
  exit 1
fi

DOWNLOADER=
if command -v wget >/dev/null 2>&1; then
  DOWNLOADER=wget
elif command -v curl >/dev/null 2>&1; then
  DOWNLOADER=curl
else
  echo "ERROR: need either wget or curl in PATH." >&2
  exit 1
fi

mkdir -p "$OPENIMAGES_DIR" "$IMAGES_ROOT"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Downloading missing manifest to $MANIFEST"
  if [[ "$DOWNLOADER" == "wget" ]]; then
    wget -O "$MANIFEST" "$MANIFEST_URL"
  else
    curl -L --fail -o "$MANIFEST" "$MANIFEST_URL"
  fi
fi

if [[ "$FORCE" == "1" ]]; then
  rm -rf "$POSITIVE_EDIT_DIR"
  rm -f "$TAR_OUTPUT"
fi

mkdir -p "$POSITIVE_EDIT_DIR"

DONE_MARKER="$POSITIVE_EDIT_DIR/.ovb-download-complete"
if [[ -f "$DONE_MARKER" ]]; then
  echo "All Pico positive-edit images already downloaded (delete $DONE_MARKER to recheck)."
else
  tmp_dir=$(mktemp -d)
  tmp_manifest=$tmp_dir/manifest.txt
  trap 'rm -rf "$tmp_dir"' EXIT

  python3 - <<'PY' "$MANIFEST" "$tmp_manifest"
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
out = Path(sys.argv[2])
seen = set()

with manifest.open("r", encoding="utf-8") as handle, out.open("w", encoding="utf-8") as out_handle:
    for raw_line in handle:
        line = raw_line.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        out_handle.write(line + "\n")
PY

  url_count=$(wc -l < "$tmp_manifest")
  echo "Checking $url_count Pico positive-edit images; downloading missing ones with $JOBS parallel $DOWNLOADER jobs."

  export OPENIMAGES_DIR POSITIVE_EDIT_DIR DOWNLOADER DOWNLOAD_RETRIES

  download_one() {
    local url=$1
    local relative_path=${url#*images/}
    local dst=$OPENIMAGES_DIR/images/$relative_path
    local dst_dir
    dst_dir=$(dirname "$dst")
    mkdir -p "$dst_dir"

    if [[ -s "$dst" ]]; then
      return 0
    fi

    # The final name appears only after a complete download, so the -s check above
    # never accepts a file cut off by an interruption.
    if [[ "$DOWNLOADER" == "wget" ]]; then
      wget -q -c \
        --tries="$DOWNLOAD_RETRIES" \
        --waitretry=2 \
        --retry-connrefused \
        -O "$dst.partial" "$url"
    else
      curl -L --fail --retry "$DOWNLOAD_RETRIES" --retry-delay 2 -C - -sS -o "$dst.partial" "$url"
    fi
    mv -f "$dst.partial" "$dst"
  }

  export -f download_one

  run_download_round() {
    local input_manifest=$1
    local failure_manifest=$2
    : > "$failure_manifest"
    export FAILURE_MANIFEST="$failure_manifest"
    xargs -r -P "$JOBS" -n 1 bash -c '
      if ! download_one "$1"; then
        printf "%s\n" "$1" >> "$FAILURE_MANIFEST"
      fi
    ' _ < "$input_manifest"
  }

  current_manifest=$tmp_manifest
  attempt=1
  while true; do
    failure_manifest=$tmp_dir/failures_round_${attempt}.txt
    run_download_round "$current_manifest" "$failure_manifest"
    failed_count=$(wc -l < "$failure_manifest")
    if [[ "$failed_count" == "0" ]]; then
      rm -f "$failure_manifest"
      break
    fi
    if (( attempt >= RETRY_ROUNDS )); then
      echo "ERROR: $failed_count downloads still failed after $attempt round(s)." >&2
      echo "First failed URLs:" >&2
      sed -n '1,20p' "$failure_manifest" >&2
      rm -f "$failure_manifest"
      exit 1
    fi
    echo "Retrying $failed_count failed downloads (round $((attempt + 1))/$RETRY_ROUNDS)."
    current_manifest=$failure_manifest
    attempt=$((attempt + 1))
  done
  touch "$DONE_MARKER"
fi

if [[ ! -d "$POSITIVE_EDIT_DIR" ]] || ! find "$POSITIVE_EDIT_DIR" -type f -print -quit | grep -q .; then
  echo "ERROR: no files were downloaded into $POSITIVE_EDIT_DIR" >&2
  exit 1
fi

file_count=$(find "$POSITIVE_EDIT_DIR" -type f ! -name '.ovb-*' | wc -l)
echo "$file_count positive-edit images in $POSITIVE_EDIT_DIR"

if [[ "$MAKE_TAR" == "1" ]]; then
  mkdir -p "$(dirname "$TAR_OUTPUT")"
  tar -cf "$TAR_OUTPUT.partial" -C "$OPENIMAGES_DIR" images/positive-edit
  mv -f "$TAR_OUTPUT.partial" "$TAR_OUTPUT"
  echo "Wrote tar archive: $TAR_OUTPUT"
fi
