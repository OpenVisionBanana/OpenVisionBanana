#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
source "$ROOT/scripts/lib/runtime.sh"

show_help() {
  cat <<'EOF'
Prepare the pinned DSINE/GeoNet NYUv2 test normals in canonical geometry-v2
format. Work is staged in TMPDIR and the completed TAR is published atomically.

Required:
  SOURCE_ZIP       Official dsine_eval ZIP.
  DEST_ARCHIVE     Destination nyu.tar-compatible archive path.

Optional runtime variables: ROOT, USE_CONTAINER, SIF, PYTHON, TMPDIR.
EOF
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then show_help; exit 0; fi

SOURCE_ZIP=${SOURCE_ZIP:?Set SOURCE_ZIP to the official DSINE evaluation ZIP}
DEST_ARCHIVE=${DEST_ARCHIVE:?Set DEST_ARCHIVE for the prepared NYUv2 normals TAR}
TMPDIR=${TMPDIR:-/tmp/${USER:-$(id -un)}/open_visionbanana}
STAGE_PARENT=${STAGE_PARENT:-$TMPDIR/prepare-dsine-nyu-${SLURM_JOB_ID:-manual-$$}}
STAGE_ROOT=$STAGE_PARENT/nyu_surface_normals
STAGE_ARCHIVE=$STAGE_PARENT/$(basename -- "$DEST_ARCHIVE")

[[ -f "$SOURCE_ZIP" ]] || { echo "ERROR: missing SOURCE_ZIP: $SOURCE_ZIP" >&2; exit 1; }
[[ ! -e "$DEST_ARCHIVE" ]] || { echo "ERROR: refusing to overwrite: $DEST_ARCHIVE" >&2; exit 1; }
mkdir -p "$STAGE_PARENT" "$(dirname -- "$DEST_ARCHIVE")"

ovb_run_python dataset_prep/surface_normals/prepare_dsine_nyu_eval.py \
  --source-zip "$SOURCE_ZIP" \
  --output-root "$STAGE_ROOT"
tar -cf "$STAGE_ARCHIVE" -C "$STAGE_PARENT" nyu_surface_normals
tar -tf "$STAGE_ARCHIVE" >/dev/null
mv "$STAGE_ARCHIVE" "$DEST_ARCHIVE"
echo "Prepared canonical NYUv2 normals: $DEST_ARCHIVE"
