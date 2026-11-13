#!/usr/bin/env bash
set -euo pipefail

# Download only the ImgEdit benchmark assets needed for evaluation:
# - ImgEdit_Judge
# - Benchmark.tar
#
# Override ROOT or OUT_DIR from the environment instead of editing this file.

show_help() {
  cat <<'EOF'
Download the ImgEdit benchmark judge checkpoint and benchmark tarball from
Hugging Face repo sysuyy/ImgEdit.

Downloads only:
  - ImgEdit_Judge
  - Benchmark.tar

Configuration:
  ROOT=/path/to/data                 default: datasets/imgedit
  OUT_DIR=/path/to/output            default: $ROOT
  PYTHON=/path/to/python             default: first python/python3 with huggingface_hub, else python3
  HF_REPO_ID=sysuyy/ImgEdit          default: sysuyy/ImgEdit
  JUDGE_REPO_ID=sysuyy/ImgEdit       default: $HF_REPO_ID
  JUDGE_REPO_TYPE=dataset            default: dataset
  BENCHMARK_REPO_ID=sysuyy/ImgEdit   default: $HF_REPO_ID
  BENCHMARK_REPO_TYPE=dataset        default: dataset
  HF_REVISION=main                   default: main
  HF_TOKEN=...                       optional, passed through for gated/private access
  FORCE=1                            remove existing target files before downloading
  EXTRACT_BENCHMARK=1                extract Benchmark.tar into $OUT_DIR/Benchmark

Examples:
  scripts/dataset_prep/data_download_scripts/download_imgeditbench.sh
  ROOT=/scratch/$USER/imgedit scripts/dataset_prep/data_download_scripts/download_imgeditbench.sh
  PYTHON=/path/to/env/bin/python scripts/dataset_prep/data_download_scripts/download_imgeditbench.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

ROOT=${ROOT:-datasets/imgedit}
OUT_DIR=${OUT_DIR:-$ROOT}
HF_REPO_ID=${HF_REPO_ID:-sysuyy/ImgEdit}
JUDGE_REPO_ID=${JUDGE_REPO_ID:-$HF_REPO_ID}
JUDGE_REPO_TYPE=${JUDGE_REPO_TYPE:-dataset}
BENCHMARK_REPO_ID=${BENCHMARK_REPO_ID:-$HF_REPO_ID}
BENCHMARK_REPO_TYPE=${BENCHMARK_REPO_TYPE:-dataset}
HF_REVISION=${HF_REVISION:-main}
FORCE=${FORCE:-0}
EXTRACT_BENCHMARK=${EXTRACT_BENCHMARK:-0}

if [[ -z "${PYTHON:-}" ]]; then
  PYTHON=python3
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import huggingface_hub' >/dev/null 2>&1; then
      PYTHON=$candidate
      break
    fi
  done
fi

mkdir -p "$OUT_DIR"

if [[ "$FORCE" == "1" ]]; then
  rm -rf "$OUT_DIR/ImgEdit_Judge" "$OUT_DIR/Benchmark" "$OUT_DIR/Benchmark.tar"
fi

if [[ -e "$OUT_DIR/ImgEdit_Judge" && ! -d "$OUT_DIR/ImgEdit_Judge" ]]; then
  echo "Removing existing non-directory judge target: $OUT_DIR/ImgEdit_Judge"
  rm -f "$OUT_DIR/ImgEdit_Judge"
fi

download_with_huggingface_hub() {
  "$PYTHON" - <<'PY' "$JUDGE_REPO_ID" "$JUDGE_REPO_TYPE" "$BENCHMARK_REPO_ID" "$BENCHMARK_REPO_TYPE" "$HF_REVISION" "$OUT_DIR"
import os
import sys

from huggingface_hub import hf_hub_download, snapshot_download

judge_repo_id, judge_repo_type, benchmark_repo_id, benchmark_repo_type, revision, out_dir = sys.argv[1:7]
token = os.environ.get("HF_TOKEN") or None

snapshot_download(
    repo_id=judge_repo_id,
    repo_type=judge_repo_type,
    revision=revision,
    local_dir=out_dir,
    allow_patterns=["ImgEdit_Judge/**"],
    token=token,
)

hf_hub_download(
    repo_id=benchmark_repo_id,
    repo_type=benchmark_repo_type,
    filename="Benchmark.tar",
    revision=revision,
    local_dir=out_dir,
    token=token,
)
PY
}

repo_url_prefix() {
  local repo_type=$1
  if [[ "$repo_type" == "dataset" ]]; then
    printf "datasets/"
  fi
}

download_direct_file() {
  local repo_id=$1
  local repo_type=$2
  local path=$3
  local out=$4
  local prefix
  prefix=$(repo_url_prefix "$repo_type")
  local url="https://huggingface.co/${prefix}${repo_id}/resolve/${HF_REVISION}/${path}?download=true"

  if [[ -s "$out" ]]; then
    echo "Using existing file: $out"
    return 0
  fi

  mkdir -p "$(dirname "$out")"

  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$out" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail -C - -o "$out" "$url"
  else
    echo "ERROR: need python3 with huggingface_hub, wget, or curl in PATH." >&2
    exit 1
  fi
}

repo_api_prefix() {
  local repo_type=$1
  if [[ "$repo_type" == "dataset" ]]; then
    printf "datasets/"
  elif [[ "$repo_type" == "model" ]]; then
    printf "models/"
  else
    echo "ERROR: unsupported Hugging Face repo type: $repo_type" >&2
    exit 1
  fi
}

list_hf_directory_files() {
  local repo_id=$1
  local repo_type=$2
  local dir_path=$3
  local out_manifest=$4
  local api_prefix
  api_prefix=$(repo_api_prefix "$repo_type")

  "$PYTHON" - <<'PY' "$api_prefix" "$repo_id" "$HF_REVISION" "$dir_path" "$out_manifest"
import json
import os
import sys

try:
    from urllib.request import Request, urlopen
    from urllib.parse import quote
except ImportError:
    from urllib2 import Request, urlopen
    from urllib import quote

api_prefix, repo_id, revision, dir_path, out_manifest = sys.argv[1:6]
url = "https://huggingface.co/api/{repo_prefix}{repo_id}/tree/{revision}/{dir_path}?recursive=true".format(
    repo_prefix=api_prefix,
    repo_id=repo_id,
    revision=quote(revision, safe=""),
    dir_path=quote(dir_path.strip("/"), safe="/"),
)

headers = {}
token = os.environ.get("HF_TOKEN")
if token:
    headers["Authorization"] = "Bearer {}".format(token)

request = Request(url, headers=headers)
with urlopen(request) as response:
    entries = json.loads(response.read().decode("utf-8"))

paths = []
for entry in entries:
    if entry.get("type") == "file" and entry.get("path"):
        paths.append(entry["path"])

if not paths:
    raise SystemExit("No files found at {}".format(url))

with open(out_manifest, "w") as handle:
    for path in sorted(paths):
        handle.write(path + "\n")
PY
}

download_direct_directory() {
  local repo_id=$1
  local repo_type=$2
  local dir_path=$3
  local out_root=$4
  local tmp_manifest
  tmp_manifest=$(mktemp)
  trap 'rm -f "$tmp_manifest"' RETURN

  list_hf_directory_files "$repo_id" "$repo_type" "$dir_path" "$tmp_manifest"
  while IFS= read -r path; do
    download_direct_file "$repo_id" "$repo_type" "$path" "$out_root/$path"
  done < "$tmp_manifest"
}

if "$PYTHON" -c 'import huggingface_hub' >/dev/null 2>&1; then
  echo "Downloading selected files from Hugging Face to $OUT_DIR"
  echo "  judge:     $JUDGE_REPO_TYPE/$JUDGE_REPO_ID"
  echo "  benchmark: $BENCHMARK_REPO_TYPE/$BENCHMARK_REPO_ID"
  download_with_huggingface_hub
else
  echo "huggingface_hub is not installed for $PYTHON; falling back to direct Hugging Face downloads."
  echo "Tip: set PYTHON=/path/to/env/bin/python if you installed huggingface_hub in a different environment."
  download_direct_directory "$JUDGE_REPO_ID" "$JUDGE_REPO_TYPE" "ImgEdit_Judge" "$OUT_DIR"
  download_direct_file "$BENCHMARK_REPO_ID" "$BENCHMARK_REPO_TYPE" "Benchmark.tar" "$OUT_DIR/Benchmark.tar"
fi

if [[ ! -d "$OUT_DIR/ImgEdit_Judge" ]]; then
  echo "ERROR: missing downloaded judge checkpoint directory: $OUT_DIR/ImgEdit_Judge" >&2
  exit 1
fi

if [[ ! -s "$OUT_DIR/Benchmark.tar" ]]; then
  echo "ERROR: missing downloaded asset: $OUT_DIR/Benchmark.tar" >&2
  exit 1
fi

if [[ "$EXTRACT_BENCHMARK" == "1" ]]; then
  if [[ -f "$OUT_DIR/Benchmark/.ovb-extract-complete" ]]; then
    echo "Benchmark already extracted: $OUT_DIR/Benchmark"
  else
    # The release tar is rooted at Benchmark/; extract it beside the tar so the
    # evaluator finds $OUT_DIR/Benchmark/{singleturn,hard,multiturn}.
    # head closes the pipe, so tar may die of SIGPIPE; that is not a failure.
    first_member=$(tar -tf "$OUT_DIR/Benchmark.tar" | head -n 1) || true
    if [[ "$first_member" == Benchmark/* ]]; then
      tar -xf "$OUT_DIR/Benchmark.tar" -C "$OUT_DIR"
    else
      mkdir -p "$OUT_DIR/Benchmark"
      tar -xf "$OUT_DIR/Benchmark.tar" -C "$OUT_DIR/Benchmark"
    fi
    touch "$OUT_DIR/Benchmark/.ovb-extract-complete"
    echo "Extracted benchmark tarball to $OUT_DIR/Benchmark"
  fi
fi

echo "ImgEdit benchmark assets ready:"
echo "  judge:     $OUT_DIR/ImgEdit_Judge"
echo "  benchmark: $OUT_DIR/Benchmark.tar"
