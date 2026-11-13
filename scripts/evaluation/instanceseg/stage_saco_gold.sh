#!/usr/bin/env bash
# Shared SA-Co/Gold tar-staging helper for the instance-seg eval wrapper and the
# Qwen gate. Mirrors the semseg cityscapes.tar flow: the canonical source is the
# archive (datasets_archives/instanceseg/SA-Co-Gold.tar), extracted once into a
# staging dir.
#
# Usage (source this file, then call the function):
#   . stage_saco_gold.sh
#   SACO_ROOT=$(stage_saco_gold "$SACO_ROOT" "$SACO_TAR" "$SACO_STAGE_DIR")
#
# Behaviour:
#   - If $SACO_ROOT already holds the extracted tree (gt-annotations/), it is
#     returned unchanged (no staging) -- so interactive runs pointed straight at an
#     extracted folder still work.
#   - Otherwise $SACO_TAR is extracted into $SACO_STAGE_DIR (the tar carries a
#     top-level SA-Co-Gold/ dir) and "$SACO_STAGE_DIR/SA-Co-Gold" is returned.
#     The extraction is idempotent: if the staged tree is already present it is
#     reused (point $SACO_STAGE_DIR at persistent scratch to extract once and reuse
#     across every checkpoint eval). Diagnostics go to stderr so stdout carries only
#     the resolved root for command substitution.

stage_saco_gold() {
  local saco_root=$1 saco_tar=$2 stage_dir=$3
  if [[ -d "$saco_root/gt-annotations" ]]; then
    printf '%s\n' "$saco_root"
    return 0
  fi
  if [[ ! -f "$saco_tar" ]]; then
    echo "ERROR: SA-Co/Gold not extracted at '$saco_root' and tar not found: '$saco_tar'" >&2
    return 1
  fi
  mkdir -p "$stage_dir"
  # Serialize concurrent stagers (array shards share $stage_dir): the lock winner
  # extracts, the rest block on flock then reuse the extracted tree -- avoids a
  # cold-cache race where many tasks untar into the same dir at once.
  (
    flock 200
    if [[ -d "$stage_dir/SA-Co-Gold/gt-annotations" ]]; then
      echo "Using existing staged SA-Co/Gold under: $stage_dir/SA-Co-Gold" >&2
    else
      echo "Staging SA-Co/Gold: extracting $saco_tar -> $stage_dir" >&2
      tar -xf "$saco_tar" -C "$stage_dir"
    fi
  ) 200>"$stage_dir/.saco_stage.lock"
  if [[ ! -d "$stage_dir/SA-Co-Gold/gt-annotations" ]]; then
    echo "ERROR: staged tree missing '$stage_dir/SA-Co-Gold/gt-annotations' after extract" >&2
    return 1
  fi
  printf '%s\n' "$stage_dir/SA-Co-Gold"
}
