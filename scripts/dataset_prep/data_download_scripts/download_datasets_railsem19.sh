#!/usr/bin/env bash
# Repack RailSem19 (rail scene semantic segmentation) into the normalized layout
# the Open Vision Banana semseg converter uses, and pack it as one tar:
#
#   datasets_archives/sem_segm/railsem19.tar -> railsem19/images/{train,val}/*.jpg
#                                               railsem19/labels/{train,val}/*.png
#                                               railsem19/{id2label,source,split}.json
#
# The tar is consumed by scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh.
# Layout matches FoodSeg103 / Trans10K-v2 (images/{split}/ + labels/{split}/ with
# uint8 class-id masks), so a reader is a near-copy of build_foodseg103_pairs.
#
# MANUAL DOWNLOAD. RailSem19 is gated behind a registration + licence agreement,
# so this script does NOT fetch anything. Get the archive yourself from
#
#   https://www.wilddash.cc/railsem19
#
# then point this script at the downloaded zip (or at an already-extracted tree):
#
#   RAILSEM19_ZIP=/path/to/rs19_val.zip \
#     bash scripts/dataset_prep/data_download_scripts/download_datasets_railsem19.sh
#
#   RAILSEM19_EXTRACTED_ROOT=/path/to/rs19_val \
#     bash scripts/dataset_prep/data_download_scripts/download_datasets_railsem19.sh
#
# RailSem19 ships 8,500 frames as ONE pool with no official train/val split, so
# this script makes a deterministic seeded split (default 10% val) and records it
# in split.json next to the images, keeping rebuilds reproducible.
set -euo pipefail

show_help() {
  cat <<'EOF'
Repack RailSem19 for Open Vision Banana dataset prep.

Download the data first (registration required):
  https://www.wilddash.cc/railsem19

Outputs:
  $ARCHIVE_DIR/railsem19.tar

Configuration:
  ROOT=/path/to/open_visionbanana        repo root (default: derived from this script)
  ARCHIVE_DIR=$ROOT/datasets_archives/sem_segm
  RAILSEM19_TAR=$ARCHIVE_DIR/railsem19.tar
  RAILSEM19_ZIP=/path/to/rs19_val.zip    downloaded archive to unpack
  RAILSEM19_EXTRACTED_ROOT=/path/to/tree already-extracted release (skips unzip)
  TMPDIR=/fast/scratch                   scratch parent for temporary work
  WORK=$TMPDIR/open_visionbanana_railsem19_download
  RAILSEM19_VAL_FRACTION=0.1             fraction of frames held out as val
  RAILSEM19_SEED=42                      seed for the deterministic split
  RAILSEM19_COPY_MODE=copy               copy | link (hardlink, saves space)
  FORCE=0                                repack even if the tar exists
  KEEP_EXTRACTED=0                       keep $WORK after successful packing

Examples:
  RAILSEM19_ZIP=~/downloads/rs19_val.zip bash scripts/dataset_prep/data_download_scripts/download_datasets_railsem19.sh
  RAILSEM19_EXTRACTED_ROOT=/scratch/rs19_val RAILSEM19_VAL_FRACTION=0.05 \
    bash scripts/dataset_prep/data_download_scripts/download_datasets_railsem19.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/sem_segm}
RAILSEM19_TAR=${RAILSEM19_TAR:-$ARCHIVE_DIR/railsem19.tar}
TMP_PARENT=${TMPDIR:-/tmp/${USER:-user}}
WORK=${WORK:-$TMP_PARENT/open_visionbanana_railsem19_download}
EXTRACT_ROOT=${EXTRACT_ROOT:-$WORK/extracted}
# Basename becomes the top-level directory inside the tar; keep it "railsem19".
BUILD_ROOT=${BUILD_ROOT:-$WORK/railsem19}
# Folder holding the manually downloaded rs19_val.zip (used when RAILSEM19_ZIP is unset).
RAILSEM19_DIR=${RAILSEM19_DIR:-$ARCHIVE_DIR}
RAILSEM19_ZIP=${RAILSEM19_ZIP:-}
RAILSEM19_EXTRACTED_ROOT=${RAILSEM19_EXTRACTED_ROOT:-}
RAILSEM19_VAL_FRACTION=${RAILSEM19_VAL_FRACTION:-0.1}
RAILSEM19_SEED=${RAILSEM19_SEED:-42}
RAILSEM19_COPY_MODE=${RAILSEM19_COPY_MODE:-copy}
PYTHON=${PYTHON:-python}
FORCE=${FORCE:-0}
KEEP_EXTRACTED=${KEEP_EXTRACTED:-0}

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required tool not found in PATH: $1" >&2
    exit 1
  fi
}

require_tool tar

if [[ "$FORCE" == "1" ]]; then
  rm -f "$RAILSEM19_TAR"
  rm -rf "$BUILD_ROOT"
fi

if [[ -s "$RAILSEM19_TAR" && "$FORCE" != "1" ]]; then
  echo "== RailSem19 tar already present: $RAILSEM19_TAR =="
  echo "   Set FORCE=1 to rebuild it."
  exit 0
fi

if [[ -z "$RAILSEM19_ZIP" && -z "$RAILSEM19_EXTRACTED_ROOT" && -d "$RAILSEM19_DIR" ]]; then
  RAILSEM19_ZIP=$(find "$RAILSEM19_DIR" -maxdepth 1 -type f -iname 'rs19_val*.zip' -print -quit)
fi
if [[ -z "$RAILSEM19_ZIP" && -z "$RAILSEM19_EXTRACTED_ROOT" ]]; then
  cat >&2 <<EOF
ERROR: RailSem19 must be downloaded manually first.

  1. Register and accept the licence at:  https://www.wilddash.cc/railsem19
  2. Download the RailSem19 archive (rs19_val.zip) into $RAILSEM19_DIR and rerun,
  3. or point this script at it:

       RAILSEM19_ZIP=/path/to/rs19_val.zip \\
         bash scripts/dataset_prep/data_download_scripts/download_datasets_railsem19.sh

     ...or, if you already unpacked it:

       RAILSEM19_EXTRACTED_ROOT=/path/to/rs19_val \\
         bash scripts/dataset_prep/data_download_scripts/download_datasets_railsem19.sh
EOF
  exit 1
fi

if [[ -n "$RAILSEM19_EXTRACTED_ROOT" ]]; then
  SRC_ROOT=$RAILSEM19_EXTRACTED_ROOT
  if [[ ! -d "$SRC_ROOT" ]]; then
    echo "ERROR: RAILSEM19_EXTRACTED_ROOT is not a directory: $SRC_ROOT" >&2
    exit 1
  fi
else
  if [[ ! -s "$RAILSEM19_ZIP" ]]; then
    echo "ERROR: RAILSEM19_ZIP does not exist or is empty: $RAILSEM19_ZIP" >&2
    exit 1
  fi
  require_tool unzip
  mkdir -p "$EXTRACT_ROOT"
  echo "== Extracting $RAILSEM19_ZIP -> $EXTRACT_ROOT =="
  unzip -q -o "$RAILSEM19_ZIP" -d "$EXTRACT_ROOT"
  SRC_ROOT=$EXTRACT_ROOT
fi

# The release nests its frames a level or two deep (e.g. rs19_val/jpgs/rs19_val/),
# and the exact nesting has varied between mirrors -- probe rather than assume.
echo "== Locating RailSem19 images + uint8 label masks under $SRC_ROOT =="
mkdir -p "$BUILD_ROOT"
env \
  SRC_ROOT="$SRC_ROOT" \
  BUILD_ROOT="$BUILD_ROOT" \
  VAL_FRACTION="$RAILSEM19_VAL_FRACTION" \
  SEED="$RAILSEM19_SEED" \
  COPY_MODE="$RAILSEM19_COPY_MODE" \
  "$PYTHON" - <<'PY'
import json
import os
import random
import shutil
import sys
from pathlib import Path

src = Path(os.environ["SRC_ROOT"])
out = Path(os.environ["BUILD_ROOT"])
val_fraction = float(os.environ["VAL_FRACTION"])
seed = int(os.environ["SEED"])
copy_mode = os.environ["COPY_MODE"]

if not 0.0 <= val_fraction < 1.0:
    sys.exit(f"ERROR: RAILSEM19_VAL_FRACTION must be in [0, 1), got {val_fraction}")


def _leaf_counts(root, suffixes):
    """Map directory -> number of direct children with one of `suffixes`."""
    counts = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            counts[p.parent] = counts.get(p.parent, 0) + 1
    return counts


def find_dir(names, suffixes):
    """Locate the directory actually holding the files.

    RailSem19 nests one level below the named folder (jpgs/rs19_val/*.jpg), so
    look for the fullest *leaf* directory, preferring one under a folder whose
    name matches, and falling back to a plain extension search.
    """
    counts = _leaf_counts(src, suffixes)
    if not counts:
        return None
    named = {d: n for d, n in counts.items()
             if any(part.lower() in names for part in d.parts)}
    pool = named or counts
    # most files wins; ties -> shallowest path for stability
    return sorted(pool.items(), key=lambda kv: (-kv[1], len(kv[0].parts)))[0][0]


img_dir = find_dir({"jpgs", "images", "rgb"}, {".jpg", ".jpeg", ".png"})
lbl_dir = find_dir({"uint8", "labels", "labelids"}, {".png"})
if img_dir is None or lbl_dir is None:
    sys.exit(
        "ERROR: could not locate RailSem19 images/labels under "
        f"{src}\n  images dir (jpgs/images/rgb): {img_dir}\n"
        f"  label dir (uint8/labels/labelids): {lbl_dir}\n"
        "  Point RAILSEM19_EXTRACTED_ROOT at the folder containing them."
    )
print(f"   images: {img_dir}")
print(f"   labels: {lbl_dir}")

images = {p.stem: p for p in sorted(img_dir.iterdir())
          if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file()}
labels = {p.stem: p for p in sorted(lbl_dir.iterdir())
          if p.suffix.lower() == ".png" and p.is_file()}
stems = sorted(set(images) & set(labels))
if not stems:
    sys.exit(f"ERROR: no image/label stem pairs matched between {img_dir} and {lbl_dir}")
dropped = len(images) - len(stems)
print(f"   paired frames: {len(stems):,} (images={len(images):,}, labels={len(labels):,},"
      f" unpaired images dropped={dropped:,})")

rng = random.Random(seed)
shuffled = list(stems)
rng.shuffle(shuffled)
n_val = int(round(len(shuffled) * val_fraction))
val = set(shuffled[:n_val])
split_of = {s: ("val" if s in val else "train") for s in stems}

for split in ("train", "val"):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)


def place(srcp: Path, dstp: Path):
    if dstp.exists():
        return
    if copy_mode == "link":
        try:
            os.link(srcp, dstp)
            return
        except OSError:
            pass
    shutil.copy2(srcp, dstp)


counts = {"train": 0, "val": 0}
for stem in stems:
    split = split_of[stem]
    place(images[stem], out / "images" / split / images[stem].name)
    place(labels[stem], out / "labels" / split / labels[stem].name)
    counts[split] += 1
print(f"   train={counts['train']:,}  val={counts['val']:,}")

# Carry the class definition through if the release ships one, so the taxonomy
# in common/semseg/taxonomy.py can be cross-checked the way Trans10K's is.
id2label = None
for cfg_name in ("rs19-config.json", "rs19_config.json", "config.json"):
    for cfg in src.rglob(cfg_name):
        try:
            cfg_data = json.loads(cfg.read_text())
        except Exception:
            continue
        labels_field = cfg_data.get("labels")
        if isinstance(labels_field, list) and labels_field:
            id2label = {}
            for entry in labels_field:
                if isinstance(entry, dict) and "name" in entry:
                    idx = entry.get("id", entry.get("trainId", len(id2label)))
                    id2label[str(idx)] = entry["name"]
            shutil.copy2(cfg, out / cfg.name)
            print(f"   class config: {cfg} -> {out / cfg.name} ({len(id2label)} classes)")
            break
    if id2label:
        break
if id2label is None:
    print("   NOTE: no rs19-config.json found; id2label.json not written.")
else:
    (out / "id2label.json").write_text(json.dumps(id2label, indent=2, sort_keys=True))

(out / "split.json").write_text(json.dumps(
    {"seed": seed, "val_fraction": val_fraction, "counts": counts, "split": split_of},
    indent=2, sort_keys=True))
(out / "source.json").write_text(json.dumps({
    "dataset": "railsem19",
    "url": "https://www.wilddash.cc/railsem19",
    "note": "Manual download (registration + licence agreement required).",
    "images_dir": str(img_dir),
    "labels_dir": str(lbl_dir),
    "frames": len(stems),
}, indent=2, sort_keys=True))
PY

echo "== Packing RailSem19 tar =="
mkdir -p "$(dirname "$RAILSEM19_TAR")"
tar -C "$(dirname "$BUILD_ROOT")" -cf "$RAILSEM19_TAR" "$(basename "$BUILD_ROOT")"
echo "   -> $RAILSEM19_TAR"

if [[ "$KEEP_EXTRACTED" != "1" ]]; then
  echo "== Cleaning temporary files: $WORK (set KEEP_EXTRACTED=1 to keep) =="
  rm -rf "$WORK"
fi

cat <<EOF

Done. RailSem19 is packed at:
  $RAILSEM19_TAR

NOTE: dataset_prep/semseg/dataset_prep.py has no --railsem19-root reader yet.
The packed layout (images/{train,val}/ + uint8 labels/{train,val}/) matches
FoodSeg103, so adding one means a build_railsem19_pairs modelled on
build_foodseg103_pairs plus a taxonomy entry in common/semseg/taxonomy.py.

Once that exists, wire it in through the pool builder:
  RAILSEM19_TAR=$RAILSEM19_TAR \\
    bash scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh
EOF
