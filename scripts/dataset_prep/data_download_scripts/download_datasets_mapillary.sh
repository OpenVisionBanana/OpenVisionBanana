#!/usr/bin/env bash
# Build datasets_archives/shared_seg/mapillary.tar from the official Mapillary
# Vistas research-edition ZIP. The tar holds the release unchanged under one
# mapillary_vistasv2/ folder:
#
#   mapillary_vistasv2/config_v2.0.json
#   mapillary_vistasv2/{training,validation}/images/
#   mapillary_vistasv2/{training,validation}/v2.0/{labels,instances,...}/
#
# One archive serves all three tasks:
#   semseg       training|validation/{images,v2.0/labels}
#   instanceseg  training|validation/{images,v2.0/instances}
#   refseg       training|validation/images (SteerViT Mapillary rows)
# The staging launcher extracts only the subset each task needs.
#
# Manual download (license-gated):
#   1. Accept the Vistas license at https://www.mapillary.com/dataset/vistas and
#      download the "Research edition" ZIP (v1.2 + v2.0, ~25 GB).
#   2. Put it in $MAPILLARY_DIR (default datasets_archives/shared_seg/mapillary/).
#   3. Run this script. It streams the ZIP into the tar without extracting it.
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_mapillary.sh
#   REMOVE_DOWNLOADS=1 ...   # delete the ZIP once the tar is built (saves ~25 GB)
#   FORCE=1 ...              # rebuild an existing mapillary.tar
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
SHARED_SEG_ROOT=${SHARED_SEG_ROOT:-$ROOT/datasets_archives/shared_seg}
MAPILLARY_DIR=${MAPILLARY_DIR:-$SHARED_SEG_ROOT/mapillary}
PYTHON=${PYTHON:-python3}
REMOVE_DOWNLOADS=${REMOVE_DOWNLOADS:-0}
FORCE=${FORCE:-0}

OUT=$SHARED_SEG_ROOT/mapillary.tar

if [[ -s "$OUT" && "$FORCE" != 1 ]]; then
  echo "== Using existing $OUT (FORCE=1 rebuilds) =="
  exit 0
fi

mkdir -p "$MAPILLARY_DIR"
zips=()
while IFS= read -r -d '' path; do zips+=("$path"); done \
  < <(find "$MAPILLARY_DIR" -maxdepth 1 \( -type f -o -type l \) -iname '*.zip' -print0)
if (( ${#zips[@]} != 1 )); then
  cat >&2 <<EOF
Expected exactly one Mapillary Vistas ZIP in $MAPILLARY_DIR (found ${#zips[@]}).
Accept the license at https://www.mapillary.com/dataset/vistas, download the
"Research edition" ZIP into that folder (or set MAPILLARY_DIR), then re-run.
EOF
  exit 1
fi
zip_path=${zips[0]}

echo "== Converting $(basename -- "$zip_path") -> $OUT =="
"$PYTHON" -I - "$zip_path" "$OUT.partial" <<'PY'
import sys
import tarfile
import time
import zipfile

source, target = sys.argv[1], sys.argv[2]
prefix = "mapillary_vistasv2/"
with zipfile.ZipFile(source) as archive:
    names = archive.namelist()
    required = {
        "config_v2.0.json": 1,
        "training/images/": 18000,
        "training/v2.0/labels/": 18000,
        "training/v2.0/instances/": 18000,
        "validation/images/": 2000,
        "validation/v2.0/labels/": 2000,
        "validation/v2.0/instances/": 2000,
    }
    problems = []
    for entry, expected in required.items():
        found = sum(1 for name in names if name.startswith(entry) and not name.endswith("/"))
        print(f"   {entry}: {found}")
        if found < expected:
            problems.append(f"{entry} has {found} files, expected {expected}")
    if problems:
        raise SystemExit("ERROR: unexpected Mapillary ZIP layout (entries must sit at the "
                         "archive root):\n  " + "\n  ".join(problems))
    with tarfile.open(target, "w", format=tarfile.PAX_FORMAT) as out:
        for info in archive.infolist():
            member = tarfile.TarInfo(prefix + info.filename.rstrip("/"))
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
mv -f "$OUT.partial" "$OUT"
[[ "$REMOVE_DOWNLOADS" == 1 ]] && rm -f "$zip_path"
echo "Done: $OUT"
