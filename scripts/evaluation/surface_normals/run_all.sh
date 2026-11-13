#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
DATASETS=${DATASETS:-"nyu ibims1 diode kitti eth3d nuscenes vkitti2 scannet sintel"}
OUT_DIR=${OUT_DIR:?Set OUT_DIR to the persistent suite output directory}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}

is_true() { case "${1,,}" in 1|true|yes|on) return 0;; *) return 1;; esac; }
mkdir -p "$OUT_DIR"
failures=()

for dataset in $DATASETS; do
  dataset_out=$OUT_DIR/$dataset
  if is_true "$SKIP_COMPLETED" && [[ -s "$dataset_out/EVALUATION_COMPLETE" ]]; then
    echo "Skipping completed surface-normal dataset: $dataset"
    continue
  fi
  echo "=== Surface normals: $dataset ==="
  if ! env -u SOURCE_ARCHIVE -u DATA_ROOT \
      ROOT="$ROOT" DATASET="$dataset" OUT_DIR="$dataset_out" \
      "$SCRIPT_DIR/run_surface_normals.sh"; then
    failures+=("$dataset")
    is_true "$CONTINUE_ON_ERROR" || exit 1
  fi
done

find "$OUT_DIR" -mindepth 2 -maxdepth 2 -type f -name '*_results.json' -print | sort > "$OUT_DIR/result_files.txt"
if (( ${#failures[@]} )); then
  printf '%s\n' "${failures[@]}" > "$OUT_DIR/failed_datasets.txt"
  echo "Surface-normal suite completed with failures: ${failures[*]}" >&2
  exit 1
fi
rm -f "$OUT_DIR/failed_datasets.txt"
echo "Surface-normal suite complete: $OUT_DIR"
