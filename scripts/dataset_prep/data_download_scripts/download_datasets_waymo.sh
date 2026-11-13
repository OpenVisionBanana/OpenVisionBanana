#!/usr/bin/env bash
# Download the Waymo Open Dataset camera (2D panoptic video) segmentation labels
# and the camera images they annotate, and pack a tar the normal staging path can
# extract. The panoptic labels feed BOTH semantic and instance segmentation.
#
# Uses the v2 "modular" release (parquet, one file per sequence per component)
# rather than the v1.4.x tfrecords: only the camera_image and camera_segmentation
# components are needed, so this pulls ~2 of ~15 components instead of ~1 GB
# segment files that bundle lidar. It also means the conversion needs nothing but
# pyarrow + Pillow -- no TensorFlow, no waymo-open-dataset pip package.
#
# Only a subset of frames is labeled, and the official frame lists in the
# waymo-open-dataset repo name them, so only sequences that actually carry labels
# are downloaded. Every labeled time step is labeled on all 5 cameras:
#
#   training    13206 frames over 696 sequences  -> 66030 images
#               (tracked sets of 5 frames at t = 0/400/600/800/1200 ms)
#   validation   1930 frames over  20 sequences  ->  9650 images
#               (whole run segments at 5 Hz)
#   testing      3947 frames over  40 sequences  -> 19735 images, LABELS WITHHELD
#
# Only train+val labels are distributed; WAYMO_SPLITS accepts testing anyway and
# reports 0 images for it. The official frame lists are the authoritative labeled
# set (the bucket also carries empty camera_segmentation stubs).
#
# Sequences are processed one at a time (download the two parquets, convert,
# delete), so peak scratch stays at roughly one sequence (~200 MB) plus the output.
#
# ACCESS: the buckets reject anonymous callers. Accept the terms at
# https://waymo.com/open/terms with a Google account, then `gcloud auth login`
# (or set GOOGLE_APPLICATION_CREDENTIALS for a service account) before running.
#
# Output:
#   datasets_archives/sem_segm/waymo.tar -> waymo/images/{train,val}/*.jpg
#                                           waymo/panoptic/{train,val}/*.png
#                                           waymo/{id2label,things,source}.json
#
#   panoptic/*.png is uint16: raw_class_id * 1000 + instance_id, instance 0 = stuff.
#   Semantic seg reads the quotient, instance seg reads both halves.
#
# Usage:
#   bash scripts/dataset_prep/data_download_scripts/download_datasets_waymo.sh
#   WORK=$TMPDIR/waymo_download ...        # node-local scratch for the parquet staging
#   WAYMO_LIMIT=2 ...                      # smoke run: first N sequences per split, never packs a tar
#   WAYMO_SPLITS=validation ...            # validation is 20 sequences; train is 696
#   WAYMO_CAMERAS=1,2,3 ...                # camera ids to keep (default all 5; 1=FRONT)
#   WAYMO_MIN_FRAME_GAP_MS=1500 ...        # training: min ms between kept frames (0 = keep all)
#   WAYMO_VAL_MIN_FRAME_GAP_MS=5000 ...    # validation: sparser (it is a denser 5 Hz stream)
#   WAYMO_PARALLEL=4 ...                   # sequences downloaded/converted concurrently
#   WAYMO_WORKERS=8 ...                    # image/label encode threads per sequence
#   KEEP_EXTRACTED=1 ...                   # keep the materialized folder after packing
#   FORCE=1 ...                            # discard progress and rebuild from scratch
#
# CHUNKED / RESUMABLE download (for when you can't leave a job running):
#   WAYMO_CHUNK=25 bash scripts/dataset_prep/data_download_scripts/download_datasets_waymo.sh
#   ... re-run the SAME command; each run does the next 25 sequences and stops.
# Resume is automatic -- per-sequence markers live under
# $WAYMO_BUILD_ROOT/waymo/state/, which defaults to PERSISTENT storage (beside the
# archive, NOT $TMPDIR) so progress survives between logins/nodes. The tar is packed
# automatically only on the run that finishes the last sequence. To watch a slice:
#   for i in $(seq 1 28); do WAYMO_CHUNK=25 bash .../download_datasets_waymo.sh || break; done
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
ARCHIVE_DIR=${ARCHIVE_DIR:-$ROOT/datasets_archives/sem_segm}
TMP_PARENT=${TMPDIR:-$ARCHIVE_DIR}
WORK=${WORK:-$TMP_PARENT/open_visionbanana_waymo_download}
WAYMO_TAR=${WAYMO_TAR:-$ARCHIVE_DIR/waymo.tar}
# The accumulated images/labels default to PERSISTENT storage (beside the archive),
# NOT node-local $TMPDIR: a chunked download resumes across logins/nodes off the
# markers under here, so it must survive between sessions.
WAYMO_BUILD_ROOT=${WAYMO_BUILD_ROOT:-$ARCHIVE_DIR/waymo_build}
# The per-sequence parquet scratch is throwaway (deleted after each sequence), so
# it stays on fast node-local scratch.
WAYMO_STAGE=${WAYMO_STAGE:-$WORK/parquet_stage}
WAYMO_BUCKET=${WAYMO_BUCKET:-gs://waymo_open_dataset_v_2_0_1}
WAYMO_SPLITS=${WAYMO_SPLITS:-training,validation}
WAYMO_CAMERAS=${WAYMO_CAMERAS:-1,2,3,4,5}
# Temporal diversity: keep a labeled timestep only if it is at least this many ms
# after the previous kept one (greedy, per sequence). All 5 camera views at a kept
# timestep are kept -- this thins near-duplicate frames in time, not viewpoints.
#
# The camera runs at 10 Hz but neither split labels every frame, and the two splits
# are labeled on very different schemes, so they get separate gaps:
#   training  : bursts of 5 frames spanning 1200 ms, ~4 bursts per 20 s segment.
#               The 5 within a burst are near-identical, so 1500 ms collapses each
#               burst to one frame (~4 well-separated frames/segment, ~0.2 Hz).
#   validation: a continuous 5 Hz stream (~99 near-dup frames per 20 s segment) over
#               just 20 scenes, so it is far more redundant. The default 5000 ms
#               thins it to ~1 every 5 s (~4 frames/segment), matching training's
#               effective spacing instead of keeping ~13.
# Set either to 0 to keep every labeled frame for that split. NOTE: applied when a
# sequence is first converted, so changing it does not retro-apply to sequences a
# resumed run has already marked done.
WAYMO_MIN_FRAME_GAP_MS=${WAYMO_MIN_FRAME_GAP_MS:-1500}       # training (and any non-val split)
WAYMO_VAL_MIN_FRAME_GAP_MS=${WAYMO_VAL_MIN_FRAME_GAP_MS:-5000}  # validation, sampled more sparsely
WAYMO_LIMIT=${WAYMO_LIMIT:-}
# Chunked download: process at most this many NOT-YET-DONE sequences per run, then
# stop. Re-run the identical command to continue where it left off (resume is
# driven by per-sequence markers under $WAYMO_BUILD_ROOT/waymo/state/). Empty = no
# cap (download everything in one run). The tar is packed automatically only once
# every requested split is complete.
WAYMO_CHUNK=${WAYMO_CHUNK:-}
# Sequences processed concurrently. The bottleneck is the per-sequence parquet
# download (camera_image is hundreds of MB), which is network-bound, so running a
# few in parallel roughly multiplies throughput. Peak scratch scales with this:
# ~one camera_image parquet (~0.3 GB) per concurrent sequence. Resume/stats are
# per-sequence files, so parallelism stays crash-safe.
WAYMO_PARALLEL=${WAYMO_PARALLEL:-4}
WAYMO_WORKERS=${WAYMO_WORKERS:-8}
PYTHON=${PYTHON:-python}
GSUTIL=${GSUTIL:-gsutil}
FORCE=${FORCE:-0}
KEEP_EXTRACTED=${KEEP_EXTRACTED:-0}

# Frame lists pinned to a fixed waymo-open-dataset revision.
WOD_REPO_REF=${WOD_REPO_REF:-master}
WOD_FRAMES_BASE=${WOD_FRAMES_BASE:-https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/$WOD_REPO_REF/tutorial}

mkdir -p "$ARCHIVE_DIR" "$WORK" "$WAYMO_BUILD_ROOT" "$WAYMO_STAGE"

if [[ "$FORCE" == "1" ]]; then
  rm -rf "$WAYMO_BUILD_ROOT/waymo" "$WAYMO_TAR"
fi

if [[ -f "$WAYMO_TAR" && "$FORCE" != "1" ]]; then
  echo "Waymo tar already exists: $WAYMO_TAR"
  echo "Set FORCE=1 to rebuild it."
  exit 0
fi

# Resume only works if the accumulated tree outlives the session. Warn loudly if it
# was pointed at node-local scratch, where the next login may not find it.
case "$WAYMO_BUILD_ROOT" in
  /tmp/*|/scratch/*|/dev/shm/*)
    echo "WARNING: WAYMO_BUILD_ROOT=$WAYMO_BUILD_ROOT looks node-local. A chunked" >&2
    echo "         download resumes from markers there -- put it on shared storage" >&2
    echo "         (e.g. under $ARCHIVE_DIR) so re-runs find the earlier progress." >&2
    ;;
esac

# --------------------------------------------------------------------------- #
# 0. Access preflight. 'gcloud auth login' is interactive one-time setup, so    #
#    it is checked for and explained, never run here.                          #
# --------------------------------------------------------------------------- #
if ! command -v "$GSUTIL" >/dev/null 2>&1; then
  cat >&2 <<EOF2
ERROR: '$GSUTIL' not found on PATH.

The Waymo buckets reject anonymous callers, so this needs the Google Cloud CLI.
No root required -- it installs into your home directory:

  cd ~
  curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
  tar -xf google-cloud-cli-linux-x86_64.tar.gz
  ./google-cloud-sdk/install.sh --quiet
  export PATH=\$HOME/google-cloud-sdk/bin:\$PATH     # add to ~/.bashrc

Then do the one-time access setup:
  1. Accept the terms at https://waymo.com/open/terms (sign in with a Google account)
  2. gcloud auth login --no-launch-browser          # same account; prints a URL
                                                     # to open on your laptop
     ...or for an unattended host:
     gcloud auth activate-service-account --key-file=/path/to/key.json

Already installed somewhere off PATH? Set GSUTIL=/path/to/gsutil.
EOF2
  exit 1
fi

echo "== Checking access to $WAYMO_BUCKET =="
if ! "$GSUTIL" ls "$WAYMO_BUCKET" >/dev/null 2>&1; then
  cat >&2 <<EOF2
ERROR: cannot list $WAYMO_BUCKET.

'$GSUTIL' is installed, so this is credentials or dataset access. Either the
login is missing/expired, or the signed-in account has not accepted the Waymo
Open Dataset terms (accepting is what grants read access to the bucket):

  1. https://waymo.com/open/terms       sign in, accept
  2. gcloud auth login --no-launch-browser    with that SAME account
  3. $GSUTIL ls $WAYMO_BUCKET           should now list the split folders

Check who you are currently authenticated as:
  gcloud auth list

If a different release is wanted, override the bucket, e.g.
  WAYMO_BUCKET=gs://waymo_open_dataset_v_2_0_0 bash \$0
EOF2
  exit 1
fi

# --------------------------------------------------------------------------- #
# 1. Which sequences carry 2D panoptic labels.                                 #
# --------------------------------------------------------------------------- #
declare -A FRAME_LIST_FOR=( [training]=2d_pvps_training_frames.txt
                            [validation]=2d_pvps_validation_frames.txt
                            [testing]=2d_pvps_test_frames.txt )
IFS=',' read -r -a SPLIT_ARRAY <<< "$WAYMO_SPLITS"
for split in "${SPLIT_ARRAY[@]}"; do
  list=${FRAME_LIST_FOR[$split]:-}
  if [[ -z "$list" ]]; then
    echo "ERROR: unknown split '$split'. Use training, validation and/or testing." >&2
    exit 1
  fi
  if [[ ! -s "$WORK/$list" ]]; then
    echo "== Fetching the labeled-frame list for $split =="
    wget -nv -O "$WORK/$list" "$WOD_FRAMES_BASE/$list"
  fi
done

# --------------------------------------------------------------------------- #
# 2. Per sequence: fetch both components, convert, drop the parquet.           #
# --------------------------------------------------------------------------- #
env \
  ROOT="$ROOT" \
  WORK="$WORK" \
  WAYMO_OUT="$WAYMO_BUILD_ROOT/waymo" \
  WAYMO_STAGE="$WAYMO_STAGE" \
  WAYMO_BUCKET="$WAYMO_BUCKET" \
  WAYMO_SPLITS="$WAYMO_SPLITS" \
  WAYMO_CAMERAS="$WAYMO_CAMERAS" \
  WAYMO_LIMIT="$WAYMO_LIMIT" \
  WAYMO_CHUNK="$WAYMO_CHUNK" \
  WAYMO_MIN_FRAME_GAP_MS="$WAYMO_MIN_FRAME_GAP_MS" \
  WAYMO_VAL_MIN_FRAME_GAP_MS="$WAYMO_VAL_MIN_FRAME_GAP_MS" \
  WAYMO_PARALLEL="$WAYMO_PARALLEL" \
  WAYMO_WORKERS="$WAYMO_WORKERS" \
  GSUTIL="$GSUTIL" \
  "$PYTHON" - <<'PY'
import io
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

root = Path(os.environ["ROOT"])
sys.path.insert(0, str(root))
from common.semseg import taxonomy  # noqa: E402

work = Path(os.environ["WORK"])
out = Path(os.environ["WAYMO_OUT"])
stage = Path(os.environ["WAYMO_STAGE"])
bucket = os.environ["WAYMO_BUCKET"].rstrip("/")
gsutil = os.environ["GSUTIL"]
splits = [s.strip() for s in os.environ["WAYMO_SPLITS"].split(",") if s.strip()]
cameras = {int(c) for c in os.environ["WAYMO_CAMERAS"].split(",") if c.strip()}
limit_text = os.environ.get("WAYMO_LIMIT", "").strip()
limit = int(limit_text) if limit_text else None
chunk_text = os.environ.get("WAYMO_CHUNK", "").strip()
chunk = int(chunk_text) if chunk_text else None
default_gap_us = max(0, int(os.environ.get("WAYMO_MIN_FRAME_GAP_MS", "0") or "0")) * 1000
val_gap_us = max(0, int(os.environ.get("WAYMO_VAL_MIN_FRAME_GAP_MS", "0") or "0")) * 1000
parallel = max(1, int(os.environ.get("WAYMO_PARALLEL", "4") or "4"))
workers = max(1, int(os.environ.get("WAYMO_WORKERS", "8")))


def gap_us_for(split: str) -> int:
    """Validation is far more redundant (continuous 5 Hz over 20 scenes) so it gets
    its own, sparser gap."""
    return val_gap_us if split == "validation" else default_gap_us


def keep_timesteps(timestamps: set[int], gap_us: int) -> set[int]:
    """Greedy min-gap thinning: keep the earliest, then each frame at least
    ``gap_us`` after the last kept one. Deterministic, so a resumed run makes the
    same choice. gap_us == 0 keeps everything."""
    if gap_us <= 0:
        return set(timestamps)
    kept: set[int] = set()
    last = None
    for ts in sorted(timestamps):
        if last is None or ts - last >= gap_us:
            kept.add(ts)
            last = ts
    return kept

# --- resume + accumulated stats (per-sequence files) ------------------------- #
# A sequence is "done" once its own JSON exists under state/seq/<split>/; re-runs
# skip those. One file per sequence needs no lock under parallelism; it is written
# last, so a crash mid-sequence just redoes that sequence next run.
state_dir = out / "state"
seq_dir = state_dir / "seq"
seq_dir.mkdir(parents=True, exist_ok=True)

# Backward compat: older runs recorded "done" as empty markers under
# state/done/<split>/<context>.done plus one aggregate state/stats.json.
legacy_done_dir = state_dir / "done"
legacy_stats_path = state_dir / "stats.json"


def seq_file(split: str, context: str) -> Path:
    d = seq_dir / split
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{context}.json"


def is_done(split: str, context: str) -> bool:
    return (seq_file(split, context).exists()
            or (legacy_done_dir / split / f"{context}.done").exists())


def write_seq_stats(split: str, context: str, record: dict) -> None:
    path = seq_file(split, context)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record) + "\n", encoding="utf-8")
    tmp.replace(path)  # atomic: a torn file never looks "done"


def aggregate_stats() -> tuple[dict, set, Counter, int]:
    """Fold every per-sequence state file into the totals the metadata needs,
    seeding from the legacy aggregate for sequences finished before the per-file
    format. Cheap even at ~700 tiny files, and always consistent with disk."""
    written_per_split: dict[str, int] = {}
    thing_ids: set[int] = set()
    class_pixels: Counter = Counter()
    total_instances = 0
    if legacy_stats_path.exists():
        s = json.loads(legacy_stats_path.read_text(encoding="utf-8"))
        written_per_split = {k: int(v) for k, v in s.get("written", {}).items()}
        thing_ids = {int(v) for v in s.get("thing_ids", [])}
        class_pixels = Counter({int(k): int(v) for k, v in s.get("class_pixels", {}).items()})
        total_instances = int(s.get("total_instances", 0))
    for f in seq_dir.rglob("*.json"):
        rec = json.loads(f.read_text(encoding="utf-8"))
        ls = rec.get("local_split")
        written_per_split[ls] = written_per_split.get(ls, 0) + int(rec.get("written", 0))
        thing_ids.update(int(v) for v in rec.get("thing_ids", []))
        for k, v in rec.get("class_pixels", {}).items():
            class_pixels[int(k)] += int(v)
        total_instances += int(rec.get("n_instances", 0))
    return written_per_split, thing_ids, class_pixels, total_instances


DIVISOR = taxonomy.WAYMO_PANOPTIC_DIVISOR
N_RAW = len(taxonomy.WAYMO_RAW_CLASS_NAMES)
CAMERA_NAMES = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
SPLIT_ALIAS = {"training": "train", "validation": "val", "testing": "test"}
FRAME_LIST = {"training": "2d_pvps_training_frames.txt",
              "validation": "2d_pvps_validation_frames.txt",
              "testing": "2d_pvps_test_frames.txt"}

KEY_CONTEXT = "key.segment_context_name"
KEY_TIMESTAMP = "key.frame_timestamp_micros"
KEY_CAMERA = "key.camera_name"
COL_IMAGE = "[CameraImageComponent].image"
COL_PANOPTIC = "[CameraSegmentationLabelComponent].panoptic_label"
COL_DIVISOR = "[CameraSegmentationLabelComponent].panoptic_label_divisor"


def gs_copy(remote: str, local: Path) -> bool:
    """Fetch one object. Returns False when it simply is not there."""
    local.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([gsutil, "-q", "cp", remote, str(local)],
                            capture_output=True, text=True)
    if result.returncode == 0:
        return True
    if "No URLs matched" in result.stderr or "NotFound" in result.stderr:
        return False
    raise SystemExit(f"gsutil cp failed for {remote}:\n{result.stderr.strip()}")


def column(table, name: str) -> list:
    """Read a column by its v2 name, tolerating a renamed component prefix.

    Columns are '[<ComponentClassName>].<field>' (see _column_name in
    waymo_open_dataset/v2/component.py). If the component class is ever renamed the
    suffix still identifies the field, so match on that before giving up.
    """
    if name in table.column_names:
        return table.column(name).to_pylist()
    suffix = name.split("].")[-1]
    matches = [c for c in table.column_names if c.split("].")[-1] == suffix]
    if len(matches) == 1:
        return table.column(matches[0]).to_pylist()
    raise SystemExit(
        f"Column {name!r} is missing and {len(matches)} columns end in {suffix!r}.\n"
        f"Found: {table.column_names}\n"
        "The v2 schema may have changed -- check the component definitions in "
        "waymo-open-dataset/src/waymo_open_dataset/v2/perception/."
    )


def to_panoptic(png_bytes: bytes, src_divisor: int) -> tuple[np.ndarray, int]:
    """Release panoptic PNG -> uint16 raw_class*DIVISOR + compact instance id."""
    with Image.open(io.BytesIO(png_bytes)) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.int64)
    semantic, instance = np.divmod(arr, src_divisor)
    if int(semantic.max(initial=0)) >= N_RAW:
        raise SystemExit(
            f"Panoptic label holds class id {int(semantic.max())}, beyond the "
            f"{N_RAW} CameraSegmentation.Type values."
        )
    # Instance ids are only unique within an image, so renumber them densely to
    # 1..N per (image, class). That keeps the packed value inside uint16.
    out_arr = (semantic * DIVISOR).astype(np.int64)
    n_instances = 0
    for cls in np.unique(semantic[instance > 0]):
        cls_mask = semantic == cls
        ids = np.unique(instance[cls_mask & (instance > 0)])
        if len(ids) >= DIVISOR:
            raise SystemExit(f"{len(ids)} instances of class {int(cls)} exceeds the "
                             f"divisor {DIVISOR}; raise WAYMO_PANOPTIC_DIVISOR.")
        for compact, raw_id in enumerate(ids.tolist(), start=1):
            out_arr[cls_mask & (instance == raw_id)] = int(cls) * DIVISOR + compact
        n_instances += len(ids)
    if int(out_arr.max(initial=0)) > np.iinfo(np.uint16).max:
        raise SystemExit("Packed panoptic value overflows uint16.")
    return out_arr.astype(np.uint16), n_instances


def process_context(split: str, local_split: str, context: str, kept_ts: set[int],
                    image_dir: Path, label_dir: Path) -> dict:
    """Download one sequence's two parquets, convert its kept frames, write a
    per-sequence state file, and clean up. Runs in a worker thread: it touches only
    its own context-named parquets, output stems, and state file, so no locking is
    needed. The state file is written LAST -- until then the sequence is not "done".
    """
    seg_path = stage / f"{context}.camera_segmentation.parquet"
    img_path = stage / f"{context}.camera_image.parquet"
    record = {"local_split": local_split, "written": 0,
              "thing_ids": [], "class_pixels": {}, "n_instances": 0}
    try:
        if not gs_copy(f"{bucket}/{split}/camera_segmentation/{context}.parquet", seg_path):
            write_seq_stats(split, context, {**record, "status": "no_seg_shard"})
            return {"context": context, "written": 0, "status": "no camera_segmentation shard"}
        if not gs_copy(f"{bucket}/{split}/camera_image/{context}.parquet", img_path):
            write_seq_stats(split, context, {**record, "status": "no_img_shard"})
            return {"context": context, "written": 0, "status": "no camera_image shard"}

        seg_table = pq.read_table(seg_path)
        seg_by_key = {}
        for ts, cam, png, div in zip(column(seg_table, KEY_TIMESTAMP),
                                     column(seg_table, KEY_CAMERA),
                                     column(seg_table, COL_PANOPTIC),
                                     column(seg_table, COL_DIVISOR)):
            if int(cam) in cameras and int(ts) in kept_ts and png:
                seg_by_key[(int(ts), int(cam))] = (png, int(div))
        if not seg_by_key:
            write_seq_stats(split, context, {**record, "status": "no_labeled_frames"})
            return {"context": context, "written": 0, "status": "no labeled frames"}

        img_table = pq.read_table(img_path)
        jobs = []
        for ts, cam, jpeg in zip(column(img_table, KEY_TIMESTAMP),
                                 column(img_table, KEY_CAMERA),
                                 column(img_table, COL_IMAGE)):
            entry = seg_by_key.pop((int(ts), int(cam)), None)
            if entry is not None:
                jobs.append((int(ts), int(cam), jpeg, entry[0], entry[1]))
        dropped = len(seg_by_key)

        class_pixels: Counter = Counter()
        thing_ids: set[int] = set()
        n_instances = 0
        for ts, cam, jpeg, png, src_divisor in jobs:
            stem = f"{context}_{ts}_{CAMERA_NAMES.get(cam, cam)}"
            (image_dir / f"{stem}.jpg").write_bytes(jpeg)  # bytes untouched, no re-encode
            packed, n_inst = to_panoptic(png, src_divisor)
            Image.fromarray(packed).save(label_dir / f"{stem}.png")
            raw = packed.astype(np.int64) // DIVISOR
            values, counts = np.unique(raw, return_counts=True)
            class_pixels.update(dict(zip(values.tolist(), counts.tolist())))
            thing_ids.update(int(v) for v in np.unique(raw[(packed % DIVISOR) > 0]))
            n_instances += n_inst

        write_seq_stats(split, context, {
            "local_split": local_split, "written": len(jobs), "status": "ok",
            "thing_ids": sorted(thing_ids),
            "class_pixels": {str(k): int(v) for k, v in class_pixels.items()},
            "n_instances": int(n_instances),
        })
        return {"context": context, "written": len(jobs),
                "status": "ok" + (f" ({dropped} frames had no image, dropped)" if dropped else "")}
    finally:
        seg_path.unlink(missing_ok=True)
        img_path.unlink(missing_ok=True)


out.mkdir(parents=True, exist_ok=True)
remaining_total = 0
labeled_ts_total = 0   # timesteps named in the frame lists (all requested splits)
kept_ts_total = 0      # timesteps kept after min-gap thinning
gap_ms_by_split: dict[str, int] = {}
budget = chunk  # sequences this run may still start; None = unlimited

for split in splits:
    local_split = SPLIT_ALIAS.get(split, split)
    image_dir = out / "images" / local_split
    label_dir = out / "panoptic" / local_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # context name -> the frame timestamps that carry labels
    wanted: dict[str, set[int]] = defaultdict(set)
    for line in (work / FRAME_LIST[split]).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        context, timestamp = line.split(",", 1)
        wanted[context.strip()].add(int(timestamp))
    # Thin near-duplicate frames in time (all 5 camera views at a kept timestep stay).
    gap_us = gap_us_for(split)
    gap_ms_by_split[local_split] = gap_us // 1000
    kept = {c: keep_timesteps(ts, gap_us) for c, ts in wanted.items()}
    split_labeled = sum(len(ts) for ts in wanted.values())
    split_kept = sum(len(ts) for ts in kept.values())
    labeled_ts_total += split_labeled
    kept_ts_total += split_kept
    contexts = sorted(wanted)
    if limit is not None:
        contexts = contexts[:limit]

    todo = [c for c in contexts if not is_done(split, c)]
    done_already = len(contexts) - len(todo)
    thin = (f", thinned {split_labeled}->{split_kept} timesteps "
            f"(min gap {gap_us // 1000} ms)") if gap_us > 0 else ""
    print(f"== {split}: {len(contexts)} sequences "
          f"({done_already} already done, {len(todo)} to go){thin} ==", flush=True)

    # This run's slice, honoring the cross-split chunk budget.
    batch = todo if budget is None else todo[:budget]
    if budget is not None:
        budget -= len(batch)
    remaining_total += len(todo) - len(batch)
    if len(batch) < len(todo):
        print(f"  chunk budget reached; {len(todo) - len(batch)} left in {split}")

    written = 0
    done = 0
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(process_context, split, local_split, c, kept[c],
                             image_dir, label_dir): c for c in batch}
        for fut in as_completed(futures):
            res = fut.result()  # a hard error (bad schema/label) propagates and aborts
            written += res["written"]
            done += 1
            print(f"  [{done}/{len(batch)}] {res['context']}: {res['status']}"
                  + (f", +{res['written']} images" if res["written"] else ""), flush=True)
    if written == 0 and done_already == 0 and len(batch) > 0:
        print(f"  NOTE: {split} produced nothing so far. Waymo withholds the "
              f"camera_segmentation labels for the challenge test split -- the frame list "
              f"only names the frames you predict on -- so 0 images is expected for testing.")

# --- metadata (aggregated from every per-sequence state file) ---------------- #
written_per_split, thing_ids, class_pixels, total_instances = aggregate_stats()
names = taxonomy.WAYMO_RAW_CLASS_NAMES
(out / "id2label.json").write_text(
    json.dumps(
        {
            "classes": {str(i): name for i, name in enumerate(names)},
            "panoptic_label_divisor": DIVISOR,
            "ignore_raw_ids": list(taxonomy.WAYMO_IGNORE_RAW_IDS),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

official = set(taxonomy.WAYMO_THING_RAW_IDS)
# Union, not either alone: the official list keeps rare thing classes available even
# when a partial build never saw one, and the observed set catches any class that
# carries instances despite not being in deeplab2's list.
if thing_ids - official:
    print(f"NOTE: classes {sorted(thing_ids - official)} carry instance ids but are not in "
          f"deeplab2's thing list; keeping them.")
if official - thing_ids:
    print(f"NOTE: no instances seen for official thing classes {sorted(official - thing_ids)} "
          f"(expected on a partial build); keeping them anyway.")
(out / "things.json").write_text(
    json.dumps(
        {
            "thing_ids": sorted(thing_ids | official),
            "observed": sorted(thing_ids),
            "official_deeplab2": sorted(official),
            "source": (
                "thing_ids = union of deeplab2's _IS_THINGS and the raw class ids "
                "actually seen with a non-zero instance id"
            ),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

total_px = sum(class_pixels.values()) or 1
(out / "source.json").write_text(
    json.dumps(
        {
            "source": f"{bucket} (Waymo Open Dataset v2 modular format)",
            "upstream": "https://github.com/waymo-research/waymo-open-dataset",
            "components": ["camera_image", "camera_segmentation"],
            "splits": splits,
            "local_splits": written_per_split,
            "cameras": sorted(cameras),
            "min_frame_gap_ms": gap_ms_by_split,
            "timesteps_labeled": labeled_ts_total,
            "timesteps_kept": kept_ts_total,
            "temporal_sampling": (
                "labeled timesteps are thinned to at least min_frame_gap_ms[split] apart "
                "per sequence (greedy from the earliest); all kept cameras are emitted at "
                "each kept timestep. validation uses a sparser gap than training because "
                "it is a continuous 5 Hz stream over just 20 scenes. 0 = keep every frame."
            ),
            "num_raw_classes": len(names),
            "num_instances": total_instances,
            "class_pixel_fraction": {
                names[int(k)]: round(v / total_px, 6)
                for k, v in sorted(class_pixels.items()) if int(k) < len(names)
            },
            "class_source": (
                "CameraSegmentation.Type in dataset.proto, named per deeplab2's "
                "WOD-PVPS eval config (data/waymo_constants.py). Raw 0 (unknown) and "
                "raw 1 (sdc / ego vehicle) are read as ignore."
            ),
            "format": (
                "images/{split}/*.jpg are the release JPEGs byte-for-byte; "
                "panoptic/{split}/*.png are uint16 with pixel = raw_class_id * "
                f"{DIVISOR} + instance_id, instance 0 = stuff. Instance ids are "
                "renumbered densely per image and are NOT the release's global ids."
            ),
            "frame_lists": "tutorial/2d_pvps_{training,validation}_frames.txt in the WOD repo",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

# Tell the shell whether every requested split is fully converted, so it packs the
# tar only when the download is actually complete.
(state_dir / "remaining").write_text(str(remaining_total) + "\n", encoding="utf-8")
totals = ", ".join(f"{s}={n}" for s, n in sorted(written_per_split.items()))
print(f"\nImages on disk: {totals or '0'} | total instances: {total_instances}")
if remaining_total:
    print(f"{remaining_total} sequence(s) still to download -- re-run the same command to continue.")
else:
    print("All requested splits are complete.")
PY

# A WAYMO_LIMIT run converts only a couple of sequences and never packs, so a
# partial tar cannot be mistaken for a complete one.
if [[ -n "$WAYMO_LIMIT" ]]; then
  cat <<EOF2

Smoke run (WAYMO_LIMIT=$WAYMO_LIMIT): no tar was packed.
Inspect the partial output, then re-run without WAYMO_LIMIT to build for real:
  ls $WAYMO_BUILD_ROOT/waymo/panoptic/val | head
  python dataset_prep/semseg/dataset_prep.py --waymo-root $WAYMO_BUILD_ROOT/waymo \\
    --out-root /tmp/waymo_semseg --splits train,val --dry-run
EOF2
  exit 0
fi

# Pack only when the whole requested download is complete.
REMAINING=$(cat "$WAYMO_BUILD_ROOT/waymo/state/remaining" 2>/dev/null || echo 0)
if [[ "$REMAINING" != "0" ]]; then
  cat <<EOF2

Chunk done. $REMAINING sequence(s) remain -- the tar is NOT packed yet.
Re-run the SAME command to fetch the next chunk (resume is automatic):
  WAYMO_CHUNK=${WAYMO_CHUNK:-<n>} ... bash $0
Progress and stats so far:
  $WAYMO_BUILD_ROOT/waymo/state/stats.json
EOF2
  # The parquet scratch is throwaway, but the accumulated tree must persist for the
  # next chunk, so never remove the build root here.
  rm -rf "$WAYMO_STAGE"
  exit 0
fi

echo "Packing $WAYMO_TAR"
tar -C "$WAYMO_BUILD_ROOT" --exclude=waymo/state -cf "$WAYMO_TAR.new" waymo
mv -f "$WAYMO_TAR.new" "$WAYMO_TAR"

if [[ "$KEEP_EXTRACTED" != "1" ]]; then
  echo "== Removing the materialized tree ($WAYMO_BUILD_ROOT, $WORK) (set KEEP_EXTRACTED=1 to keep it) =="
  rm -rf "$WAYMO_BUILD_ROOT/waymo" "$WORK"
fi

cat <<EOF2

Done:
  $WAYMO_TAR

Use it through the normal staging launcher (feeds semseg AND instanceseg):
  WAYMO_TAR=$WAYMO_TAR bash scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh

Or run either converter directly (on the extracted tree, if kept):
  python dataset_prep/semseg/dataset_prep.py --waymo-root $WAYMO_BUILD_ROOT/waymo \\
    --out-root /tmp/waymo_semseg
  python dataset_prep/instanceseg/dataset_prep.py --waymo-root $WAYMO_BUILD_ROOT/waymo \\
    --out-root /tmp/waymo_instanceseg
EOF2