#!/usr/bin/env bash
# Build datasets_archives/shared_seg/wd_public_v2p.tar from the official WildDash 2
# public panoptic release (https://www.wilddash.cc/download, registration
# required). Staging extracts it with tar and expects, at the archive root (or
# under one wd_public_v2p/ or wilddash2/ folder):
#
#   anonymized/  panoptic/  panoptic.json  wilddash2-meta.json  random_split/
#
# Semantic and instance segmentation both read this archive.
#
# Manual download:
#   1. Log in at https://www.wilddash.cc/download and download the public
#      panoptic ground truth, wd_public_v2p.zip.
#   2. Put it in $WILDDASH2_DIR (default datasets_archives/shared_seg/wilddash2/).
#   3. Run this script. It streams the ZIP into the TAR without extracting it.
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_wilddash2.sh
#   FORCE=1 ...   # rebuild an existing wd_public_v2p.tar
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$ROOT/datasets_archives/shared_seg}"
WILDDASH2_DIR=${WILDDASH2_DIR:-$ARCHIVE_DIR/wilddash2}
PYTHON=${PYTHON:-python3}
FORCE=${FORCE:-0}

OUT_TAR=$ARCHIVE_DIR/wd_public_v2p.tar

if [[ "$FORCE" != 1 && -s "$OUT_TAR" ]]; then
  echo "== Using existing $OUT_TAR (FORCE=1 rebuilds) =="
  exit 0
fi
mkdir -p "$WILDDASH2_DIR"

zip_path=$(find "$WILDDASH2_DIR" -maxdepth 1 -type f -iname 'wd_public_v2p*.zip' -print -quit)
if [[ -z "$zip_path" ]]; then
  cat >&2 <<EOF
WildDash 2 archive not found in $WILDDASH2_DIR.
Log in at https://www.wilddash.cc/download, download wd_public_v2p.zip into that
folder (or set WILDDASH2_DIR), then re-run.
EOF
  exit 1
fi

echo "== Converting $(basename -- "$zip_path") -> $OUT_TAR =="
mkdir -p "$ARCHIVE_DIR"
"$PYTHON" -I - "$zip_path" "$OUT_TAR.partial" <<'PY'
import sys
import tarfile
import time
import zipfile

source, target = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(source) as archive:
    names = archive.namelist()
    roots = {name.split("/", 1)[0] for name in names}
    prefix = ""
    if len(roots) == 1 and next(iter(roots)) in {"wd_public_v2p", "wilddash2"}:
        prefix = next(iter(roots)) + "/"
    required = ("anonymized/", "panoptic/", "panoptic.json", "wilddash2-meta.json", "random_split/")
    missing = [entry for entry in required
               if not any(name.startswith(prefix + entry) for name in names)]
    if missing:
        raise SystemExit(f"ERROR: {source} lacks {', '.join(missing)}; is it wd_public_v2p.zip?")
    with tarfile.open(target, "w", format=tarfile.PAX_FORMAT) as out:
        for info in archive.infolist():
            member = tarfile.TarInfo(info.filename.rstrip("/"))
            member.mtime = time.mktime(info.date_time + (0, 0, -1))
            if info.is_dir():
                member.type, member.mode = tarfile.DIRTYPE, 0o755
                out.addfile(member)
                continue
            member.size, member.mode = info.file_size, 0o644
            with archive.open(info) as stream:
                out.addfile(member, stream)
print(f"{len(names):,} entries written")
PY
mv -f "$OUT_TAR.partial" "$OUT_TAR"
echo "Done: $OUT_TAR"
