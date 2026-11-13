#!/usr/bin/env bash
# Download and extract Synscapes using the procedure published by its authors:
# https://synscapes.on.liu.se/download.html
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$ROOT/datasets_archives/shared_seg}"
TARGET_DIR="${TARGET_DIR:-$ARCHIVE_DIR/synscapes}"
CHUNKS_DIR="${CHUNKS_DIR:-$ARCHIVE_DIR/synscapes_chunks}"
SOURCE_URL="https://synscapes.on.liu.se/synscapes/"

# Reuse a complete original-layout TAR without extracting any source trees.
if [[ -s "$TARGET_DIR/synscapes.tar" ]]; then
  "${PYTHON:-python3}" - "$ROOT" "$TARGET_DIR/synscapes.tar" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from common.official_sources import complete_synscapes_raw_archive
if not complete_synscapes_raw_archive(Path(sys.argv[2])):
    raise SystemExit("Incomplete or unsafe packed Synscapes source. Original RGB, depth, class, instance and metadata frames are required.")
print("Verified packed Synscapes source; no extraction needed.")
PY
  exit 0
fi

if ! command -v wget >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
  echo "This script requires wget and tar." >&2
  exit 1
fi

cat <<EOF
Synscapes is approximately 337 GB and is licensed for non-commercial use.
By continuing, you agree to its license:
  https://synscapes.on.liu.se/download.html
EOF

MARKER="$TARGET_DIR/.ovb-extract-complete"

# 25,000 frames, each with a meta JSON and one file per image modality.
extracted_complete() {
  local root="$TARGET_DIR/Synscapes" sub
  [[ "$(find "$root/meta" -maxdepth 1 -name '[0-9]*.json' 2>/dev/null | wc -l)" -ge 25000 ]] || return 1
  for sub in class depth instance rgb rgb-2k; do
    [[ "$(find "$root/img/$sub" -maxdepth 1 -name '[0-9]*' 2>/dev/null | wc -l)" -ge 25000 ]] || return 1
  done
}

if [[ -f "$MARKER" ]] || extracted_complete; then
  touch "$MARKER"
  echo "Synscapes already extracted: $TARGET_DIR"
  exit 0
fi

mkdir -p "$CHUNKS_DIR" "$TARGET_DIR"
pushd "$CHUNKS_DIR" >/dev/null

# Synscapes is published as 41 five-GB chunks (synscapes.tar.gz.00 ... .40).
# --cut-dirs=1 stores them directly in $CHUNKS_DIR and -c resumes interruptions.
wget -c -r -np -nH --cut-dirs=1 -A 'synscapes.tar.gz.*' "$SOURCE_URL"

shopt -s nullglob
chunks=(synscapes.tar.gz.*)
if (( ${#chunks[@]} != 41 )); then
  echo "Expected 41 Synscapes chunks but found ${#chunks[@]}; refusing to extract." >&2
  exit 1
fi

LC_ALL=C cat "${chunks[@]}" | tar -xz -C "$TARGET_DIR"
popd >/dev/null
touch "$MARKER"

echo "Synscapes extracted to: $TARGET_DIR"
echo "Downloaded chunks retained in: $CHUNKS_DIR"
