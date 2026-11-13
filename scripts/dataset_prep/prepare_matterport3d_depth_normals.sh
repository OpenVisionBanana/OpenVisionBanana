#!/usr/bin/env bash
set -euo pipefail
show_help() {
  cat <<'EOF'
Prepare previously acquired Matterport3D scan archives for depth and normals.

Required environment: MATTERPORT3D_ROOT and SCANS_FILE.
Runtime: set USE_CONTAINER=1 and SIF for a container, or leave
USE_CONTAINER=0 to use PYTHON from the active environment.
EOF
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then show_help; exit 0; fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
source "$ROOT/scripts/lib/runtime.sh"
MATTERPORT3D_ROOT=${MATTERPORT3D_ROOT:?Set MATTERPORT3D_ROOT}
SOURCE_TAR_ROOT=${SOURCE_TAR_ROOT:-$MATTERPORT3D_ROOT/source_scene_tars}
PROFILE=${PROFILE:-standard}
profile_suffix=_${PROFILE}
[[ "$PROFILE" == standard ]] && profile_suffix=
SCRATCH_ROOT=${SCRATCH_ROOT:-${TMPDIR:-/tmp/${USER:-$(id -un)}}/matterport3d_prep_${SLURM_JOB_ID:-$$}}
SCANS_FILE=${SCANS_FILE:?Set SCANS_FILE to the authorized scan list}
DEPTH_PREPARED_TAR=${DEPTH_PREPARED_TAR:-$MATTERPORT3D_ROOT/matterport3d_depth${profile_suffix}.tar}
NORMALS_PREPARED_TAR=${NORMALS_PREPARED_TAR:-$MATTERPORT3D_ROOT/matterport3d_surface_normals${profile_suffix}.tar}
COMPLETION_MARKER=${COMPLETION_MARKER:-${DEPTH_PREPARED_TAR%.tar}.complete}
START_INDEX=${START_INDEX:-0}; MAX_SCENES=${MAX_SCENES:-}; SAMPLES_PER_SCENE=${SAMPLES_PER_SCENE:-}
sample_tag=${SAMPLES_PER_SCENE:-profile}
SHARD_ROOT=${SHARD_ROOT:-$MATTERPORT3D_ROOT/prepared_scene_shards/${PROFILE}_${sample_tag}}
SEED=${SEED:-42}; MIN_DEPTH=${MIN_DEPTH:-0.1}; MAX_DEPTH=${MAX_DEPTH:-20.0}
MIN_VALID_FRACTION=${MIN_VALID_FRACTION:-0.30}; FORCE_SCENES=${FORCE_SCENES:-0}; STRICT=${STRICT:-0}; STRICT_CALIBRATION=${STRICT_CALIBRATION:-0}
MIN_TMP_FREE_GB=${MIN_TMP_FREE_GB:-10}
WRITE_RAW_DEPTH=${WRITE_RAW_DEPTH:-0}; WRITE_RAW_NORMALS=${WRITE_RAW_NORMALS:-0}
NORMAL_ENCODING=${NORMAL_ENCODING:-vision_banana_x_left_red}

[[ -d "$SOURCE_TAR_ROOT" ]] || { echo "ERROR: missing source tars: $SOURCE_TAR_ROOT" >&2; exit 1; }
if [[ -s "$COMPLETION_MARKER" && -s "$DEPTH_PREPARED_TAR" && -s "$NORMALS_PREPARED_TAR" ]]; then
  echo "Matterport3D profile '$PROFILE' is already complete; skipping."
  echo "  depth:   $DEPTH_PREPARED_TAR"
  echo "  normals: $NORMALS_PREPARED_TAR"
  exit 0
fi
mkdir -p "$MATTERPORT3D_ROOT" "$SHARD_ROOT" "$SCRATCH_ROOT"
tmp_free_kb=$(df -Pk "$SCRATCH_ROOT" | awk 'NR==2 {print $4}')
(( tmp_free_kb >= MIN_TMP_FREE_GB * 1024 * 1024 )) || { echo "ERROR: $SCRATCH_ROOT has less than ${MIN_TMP_FREE_GB} GiB free." >&2; exit 1; }
args=(dataset_prep/matterport3d/prepare_matterport3d_depth_normals.py --source-tar-root "$SOURCE_TAR_ROOT" --scratch-root "$SCRATCH_ROOT" --shard-root "$SHARD_ROOT" --depth-tar "$DEPTH_PREPARED_TAR" --normals-tar "$NORMALS_PREPARED_TAR" --scans-file "$SCANS_FILE" --start-index "$START_INDEX" --profile "$PROFILE" --seed "$SEED" --min-depth "$MIN_DEPTH" --max-depth "$MAX_DEPTH" --min-valid-fraction "$MIN_VALID_FRACTION" --normal-encoding "$NORMAL_ENCODING")
[[ -n "$SAMPLES_PER_SCENE" ]] && args+=(--samples-per-scene "$SAMPLES_PER_SCENE")
[[ -n "$MAX_SCENES" ]] && args+=(--max-scenes "$MAX_SCENES")
[[ "$FORCE_SCENES" == 1 ]] && args+=(--force-scenes)
[[ "$STRICT" == 1 ]] && args+=(--strict)
[[ "$STRICT_CALIBRATION" == 1 ]] && args+=(--strict-calibration)
[[ "$WRITE_RAW_DEPTH" == 1 ]] && args+=(--write-raw-depth)
[[ "$WRITE_RAW_NORMALS" == 1 ]] && args+=(--write-raw-normals)
ovb_run_python "${args[@]}"
marker_partial="${COMPLETION_MARKER}.partial.${SLURM_JOB_ID:-$$}"
printf 'profile=%s\ndepth_tar=%s\nnormals_tar=%s\ncompleted_at=%s\n' \
  "$PROFILE" "$DEPTH_PREPARED_TAR" "$NORMALS_PREPARED_TAR" "$(date --iso-8601=seconds)" > "$marker_partial"
mv -f "$marker_partial" "$COMPLETION_MARKER"
