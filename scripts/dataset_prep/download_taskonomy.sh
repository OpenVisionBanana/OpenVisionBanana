#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Download Taskonomy modality tars needed for Open Vision Banana depth and
surface-normal training.

This intentionally downloads only RGB, depth, and normal archives. It does not
extract them, so persistent storage keeps a low inode count.

Usage:
  scripts/dataset_prep/download_taskonomy.sh

Important environment variables:
  DATA_ROOT             Persistent data root.
  TASKONOMY_ROOT        Persistent Taskonomy root.
  RAW_TAR_ROOT          Where modality tars are stored.
  LINKS_FILE            Local Taskonomy links.txt path.
  LINKS_URL             Official Taskonomy links list.
  MODALITIES            Space-separated modalities. Default: "rgb depth_zbuffer normal".
  BUILDING_LIST         Optional file containing building names to download.
  NUM_BUILDINGS         "all" or integer sampled from sorted buildings.
  SEED                  Seed for NUM_BUILDINGS sampling.
  ESTIMATE_ONLY         1 to write URL sizes and exit.
  DOWNLOAD_JOBS         Concurrent file downloads. Default: 4.
  DOWNLOAD_ROUNDS       Failed-subset retry rounds in one job. Default: 6.
  ROUND_RETRY_DELAY     Seconds between failed-subset rounds. Default: 60.
  VERIFY_EXISTING_TARS  Validate uncached existing tars before skipping them.
                        Default: 1.
  CURL_RETRIES          Retries within each download round. Default: 5.
  CURL_RETRY_DELAY      Seconds between retries. Default: 5.
  FORCE_LINKS           1 to re-download links.txt.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

ROOT=${ROOT:-$DEFAULT_ROOT}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT for this dataset}
TASKONOMY_ROOT=${TASKONOMY_ROOT:-$DATA_ROOT/depth_surfn_data/taskonomy}
RAW_TAR_ROOT=${RAW_TAR_ROOT:-$TASKONOMY_ROOT/raw_tars}
MANIFEST_ROOT=${MANIFEST_ROOT:-$TASKONOMY_ROOT/manifests}
LINKS_URL=${LINKS_URL:-https://datasets.epfl.ch/taskonomy//links.txt}
LINKS_FILE=${LINKS_FILE:-$MANIFEST_ROOT/taskonomy_links.txt}
NEEDED_LINKS=${NEEDED_LINKS:-$MANIFEST_ROOT/taskonomy_ovb_needed_links.txt}
SIZE_FILE=${SIZE_FILE:-$MANIFEST_ROOT/taskonomy_ovb_sizes_bytes.txt}
VERIFIED_TARS_FILE=${VERIFIED_TARS_FILE:-$MANIFEST_ROOT/taskonomy_verified_tars.tsv}
FAILED_LINKS=${FAILED_LINKS:-$MANIFEST_ROOT/taskonomy_failed_links.txt}
REPAIRED_TARS_FILE=${REPAIRED_TARS_FILE:-$MANIFEST_ROOT/taskonomy_repaired_tars.jsonl}
MODALITIES=${MODALITIES:-rgb depth_zbuffer normal}
BUILDING_LIST=${BUILDING_LIST:-}
NUM_BUILDINGS=${NUM_BUILDINGS:-all}
SEED=${SEED:-42}
ESTIMATE_ONLY=${ESTIMATE_ONLY:-0}
DOWNLOAD_JOBS=${DOWNLOAD_JOBS:-4}
DOWNLOAD_ROUNDS=${DOWNLOAD_ROUNDS:-6}
ROUND_RETRY_DELAY=${ROUND_RETRY_DELAY:-60}
VERIFY_EXISTING_TARS=${VERIFY_EXISTING_TARS:-1}
CURL_RETRIES=${CURL_RETRIES:-5}
CURL_RETRY_DELAY=${CURL_RETRY_DELAY:-5}
FORCE_LINKS=${FORCE_LINKS:-0}
TASKONOMY_REPO=${TASKONOMY_REPO:-$TASKONOMY_ROOT/taskonomy_repo}
CLONE_REPO=${CLONE_REPO:-0}

mkdir -p "$TASKONOMY_ROOT" "$RAW_TAR_ROOT" "$MANIFEST_ROOT"

if [[ "$CLONE_REPO" == "1" ]]; then
  if [[ ! -d "$TASKONOMY_REPO/.git" ]]; then
    git clone https://github.com/StanfordVL/taskonomy.git "$TASKONOMY_REPO"
  else
    echo "Using existing Taskonomy checkout: $TASKONOMY_REPO"
  fi
fi

if [[ "$FORCE_LINKS" == "1" || ! -s "$LINKS_FILE" ]]; then
  echo "Downloading Taskonomy links list to $LINKS_FILE"
  curl -L "$LINKS_URL" -o "$LINKS_FILE"
else
  echo "Using existing links list: $LINKS_FILE"
fi

export LINKS_FILE NEEDED_LINKS MODALITIES BUILDING_LIST NUM_BUILDINGS SEED

python3 - <<'PY'
import os
import random
import re
from pathlib import Path

links_file = Path(os.environ["LINKS_FILE"])
needed_links = Path(os.environ["NEEDED_LINKS"])
modalities = os.environ["MODALITIES"].split()
building_list = os.environ.get("BUILDING_LIST", "").strip()
num_buildings = os.environ["NUM_BUILDINGS"]
seed = int(os.environ["SEED"])

pattern = re.compile(r"/([^/]+)_(" + "|".join(re.escape(m) for m in modalities) + r")\.tar$")
by_building = {}
for raw in links_file.read_text(encoding="utf-8").splitlines():
    url = raw.strip()
    if not url:
        continue
    match = pattern.search(url)
    if not match:
        continue
    building, modality = match.groups()
    by_building.setdefault(building, {})[modality] = url

complete = sorted(building for building, urls in by_building.items() if all(modality in urls for modality in modalities))
if building_list:
    requested = [
        line.strip()
        for line in Path(building_list).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    missing = sorted(set(requested) - set(complete))
    if missing:
        raise SystemExit(f"Missing complete modality links for buildings: {', '.join(missing[:20])}")
    selected = requested
elif num_buildings == "all":
    selected = complete
else:
    count = int(num_buildings)
    if count <= 0:
        raise SystemExit("NUM_BUILDINGS must be 'all' or a positive integer")
    rng = random.Random(seed)
    selected = sorted(rng.sample(complete, min(count, len(complete))))

rows = []
for building in selected:
    for modality in modalities:
        rows.append(by_building[building][modality])

needed_links.parent.mkdir(parents=True, exist_ok=True)
needed_links.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
print(f"complete buildings available: {len(complete)}")
print(f"selected buildings:           {len(selected)}")
print(f"download URLs:                {len(rows)}")
print(f"wrote:                        {needed_links}")
PY

if [[ "$ESTIMATE_ONLY" == "1" ]]; then
  echo "Estimating download sizes from HTTP headers."
  : > "$SIZE_FILE"
  while IFS= read -r url; do
    [[ -z "$url" ]] && continue
    size=$(curl -sIL "$url" | awk 'tolower($1)=="content-length:" {value=$2} END {gsub("\r","",value); print value}')
    printf '%s %s\n' "${size:-0}" "$url" >> "$SIZE_FILE"
  done < "$NEEDED_LINKS"
  awk '{sum += $1} END {printf "estimated total: %.2f GiB\n", sum/1024/1024/1024}' "$SIZE_FILE"
  echo "Wrote size estimate: $SIZE_FILE"
  exit 0
fi

if ! [[ "$DOWNLOAD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DOWNLOAD_JOBS must be a positive integer: $DOWNLOAD_JOBS" >&2
  exit 2
fi

touch "$VERIFIED_TARS_FILE"
: > "$FAILED_LINKS"

verification_record() {
  local tar_path=$1
  printf '%s\t%s\t%s' \
    "$(basename "$tar_path")" \
    "$(stat -c '%s' "$tar_path")" \
    "$(stat -c '%Y' "$tar_path")"
}

record_verified_tar() {
  local record
  record=$(verification_record "$1")
  grep -Fqx "$record" "$VERIFIED_TARS_FILE" || printf '%s\n' "$record" >> "$VERIFIED_TARS_FILE"
}

validate_tar() {
  local tar_path=$1
  local record

  [[ -s "$tar_path" ]] || return 1
  record=$(verification_record "$tar_path")
  if grep -Fqx "$record" "$VERIFIED_TARS_FILE"; then
    return 0
  fi
  tar -tf "$tar_path" >/dev/null 2>&1
}

repair_source_truncation() {
  local tar_path=$1
  local url=$2
  local remote_md5

  remote_md5=$(
    curl -fsSIL --retry 3 --retry-all-errors "$url" \
      | awk -F': ' 'tolower($1)=="x-amz-meta-md5chksum" {gsub("\r", "", $2); print $2; exit}'
  ) || return 1
  [[ -n "$remote_md5" ]] || return 1

  if python3 "$ROOT/dataset_prep/taskonomy/repair_truncated_tar.py" \
    "$tar_path" \
    --expected-md5-base64 "$remote_md5" \
    --manifest "$REPAIRED_TARS_FILE"; then
    echo "Repaired source-truncated final tar member: $tar_path"
    return 0
  fi
  return 1
}

download_one() {
  local url=$1
  local filename out partial curl_status

  filename=${url##*/}
  filename=${filename%%\?*}
  out="$RAW_TAR_ROOT/$filename"
  partial="$out.partial"

  if [[ -s "$out" ]]; then
    if [[ "$VERIFY_EXISTING_TARS" != "1" ]] || validate_tar "$out"; then
      [[ "$VERIFY_EXISTING_TARS" != "1" ]] || record_verified_tar "$out"
      echo "Using existing verified $out"
      return 0
    fi

    echo "Existing archive is incomplete; preserving it for resume: $out" >&2
    if [[ -e "$partial" ]]; then
      echo "Cannot recover $out: $partial already exists" >&2
      printf '%s\n' "$url" >> "$WORKER_FAILED_LINKS"
      return 1
    fi
    mv "$out" "$partial"
  fi

  if [[ -s "$partial" ]]; then
    if validate_tar "$partial" || repair_source_truncation "$partial" "$url"; then
      mv "$partial" "$out"
      record_verified_tar "$out"
      echo "Promoted complete partial archive: $out"
      return 0
    fi
  fi

  echo "Downloading $filename"
  curl_status=0
  curl -fL \
    --retry "$CURL_RETRIES" \
    --retry-all-errors \
    --retry-delay "$CURL_RETRY_DELAY" \
    --continue-at - \
    "$url" -o "$partial" || curl_status=$?

  if (( curl_status != 0 )); then
    echo "Download remains partial ($curl_status): $partial" >&2
    printf '%s\n' "$url" >> "$WORKER_FAILED_LINKS"
    return 1
  fi

  if ! validate_tar "$partial" && ! repair_source_truncation "$partial" "$url"; then
    echo "Downloaded archive failed tar validation: $partial" >&2
    printf '%s\n' "$url" >> "$WORKER_FAILED_LINKS"
    return 1
  fi

  mv "$partial" "$out"
  record_verified_tar "$out"
  echo "Completed and verified $out"
}

export ROOT RAW_TAR_ROOT VERIFIED_TARS_FILE REPAIRED_TARS_FILE VERIFY_EXISTING_TARS
export CURL_RETRIES CURL_RETRY_DELAY
export -f verification_record record_verified_tar validate_tar repair_source_truncation download_one

for value_name in DOWNLOAD_ROUNDS ROUND_RETRY_DELAY; do
  value=${!value_name}
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "$value_name must be a positive integer: $value" >&2
    exit 2
  fi
done

: > "$FAILED_LINKS"
pending_links=$NEEDED_LINKS
download_complete=0
round_files=()

for ((round = 1; round <= DOWNLOAD_ROUNDS; round++)); do
  WORKER_FAILED_LINKS="$MANIFEST_ROOT/.taskonomy_failed_${SLURM_JOB_ID:-$$}_$round.txt"
  round_files+=("$WORKER_FAILED_LINKS")
  : > "$WORKER_FAILED_LINKS"
  export WORKER_FAILED_LINKS

  pending_count=$(wc -l < "$pending_links")
  echo "Download round $round/$DOWNLOAD_ROUNDS: $pending_count URLs, $DOWNLOAD_JOBS workers."
  round_status=0
  tr '\n' '\0' < "$pending_links" \
    | xargs -0 -r -n 1 -P "$DOWNLOAD_JOBS" bash -c 'download_one "$1"' _ \
    || round_status=$?

  if (( round_status == 0 )) && [[ ! -s "$WORKER_FAILED_LINKS" ]]; then
    download_complete=1
    : > "$FAILED_LINKS"
    break
  fi

  cp "$WORKER_FAILED_LINKS" "$FAILED_LINKS"
  pending_links=$WORKER_FAILED_LINKS
  if (( round < DOWNLOAD_ROUNDS )); then
    failed_count=$(wc -l < "$pending_links")
    echo "Retrying $failed_count failed URLs in $ROUND_RETRY_DELAY seconds."
    sleep "$ROUND_RETRY_DELAY"
  fi
done

rm -f "${round_files[@]}"

if (( download_complete != 1 )); then
  echo "One or more Taskonomy downloads are incomplete." >&2
  echo "Re-run this job to resume them. Failed URLs: $FAILED_LINKS" >&2
  exit 1
fi

echo "Taskonomy download finished."
echo "  raw tars: $RAW_TAR_ROOT"
echo "  links:    $NEEDED_LINKS"
echo "  verified: $VERIFIED_TARS_FILE"
echo "  repaired: $REPAIRED_TARS_FILE"
