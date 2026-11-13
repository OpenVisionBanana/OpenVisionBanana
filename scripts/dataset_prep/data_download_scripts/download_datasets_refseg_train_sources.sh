#!/usr/bin/env bash
# Download the referring-seg TRAINING sources into OUT (default
# datasets_archives/refseg/, the REFSEG_SOURCE_ROOT the pool launcher reads).
# SteerViT (pulled from the HF hub by dataset_prep/refseg/prepare_steervit_refseg.py)
# provides the masks + expressions; the photos come from:
#
#   $OUT/visual_genome.tar  ->  VG_100K/<id>.jpg , VG_100K_2/<id>.jpg
#   shared_seg/mapillary.tar    official Mapillary Vistas ZIP, shared with semseg and
#                               instanceseg (download_datasets_mapillary.sh)
#   COCO 2014                   download_datasets_coco.sh
#
# It also fetches the two SteerViT annotation repos used by the non-zero-shot
# refseg_grefcoco / refseg_phrasecut members, pinned to the revisions the paper
# pool was built from:
#
#   $OUT/SteerViT-gRefCOCO/data/*.parquet           (GREFCOCO_DATASET)
#   $OUT/SteerViT-PhraseCutDataset/data/*.parquet   (PHRASECUT_DATASET)
#
# (EVAL datasets -- RefCOCOg/ReasonSeg -- are a different script: download_datasets_refseg.sh)
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
OUT=${OUT:-$ROOT/datasets_archives/refseg}
mkdir -p "$OUT"
OUT=$(cd -- "$OUT" && pwd)
STEERVIT_DIR=${STEERVIT_DIR:-$OUT}
MAPILLARY_TAR=${MAPILLARY_TAR:-$(dirname -- "$OUT")/shared_seg/mapillary.tar}
cd "$OUT"

SKIP_VISUAL_GENOME=${SKIP_VISUAL_GENOME:-0}
SKIP_STEERVIT_EXTRA=${SKIP_STEERVIT_EXTRA:-0}
SKIP_GREFCOCO=${SKIP_GREFCOCO:-$SKIP_STEERVIT_EXTRA}
SKIP_PHRASECUT=${SKIP_PHRASECUT:-$SKIP_STEERVIT_EXTRA}
GREFCOCO_REVISION=${GREFCOCO_REVISION:-93bc6f3eefa1ae21124ad62a7b8b1e373f9af49a}
PHRASECUT_REVISION=${PHRASECUT_REVISION:-53b323a133058743d02d117781f80bb2f2800158}

# ============================================================================ #
# 1. Visual Genome  (two image zips, ~15 GB total, directly downloadable)
# ============================================================================ #
if [[ "$SKIP_VISUAL_GENOME" == "1" ]]; then
  echo "Visual Genome skipped (SKIP_VISUAL_GENOME=1)."
elif [[ -s visual_genome.tar ]]; then
  echo "Using existing $OUT/visual_genome.tar"
else
  wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip    # -> VG_100K/
  wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip   # -> VG_100K_2/
  unzip -q -o images.zip
  unzip -q -o images2.zip
  tar cf visual_genome.tar.partial VG_100K VG_100K_2
  mv -f visual_genome.tar.partial visual_genome.tar
  rm -rf images.zip images2.zip VG_100K VG_100K_2
  echo "-> $OUT/visual_genome.tar  (VG_100K/, VG_100K_2/)"
fi

# ============================================================================ #
# 2. Mapillary Vistas v2.0  (license-gated; one archive shared with seg tasks)
# ============================================================================ #
# Refseg reads training|validation/images straight from the official ZIP at
# shared_seg/mapillary.tar, so no separate refseg copy is built here.
if [[ -s "$MAPILLARY_TAR" ]]; then
  echo "Using Mapillary images from $MAPILLARY_TAR"
else
  echo "Mapillary: $MAPILLARY_TAR is missing. Download the Vistas research edition and"
  echo "           run download_datasets_mapillary.sh (license-gated, manual)."
fi

# ============================================================================ #
# 3. SteerViT-gRefCOCO + SteerViT-PhraseCutDataset  (public HF dataset repos)
# ============================================================================ #
# Requires the Hugging Face CLI (pip install huggingface_hub). The repos hold only
# masks + expressions; gRefCOCO uses COCO train2014 photos (download_datasets_refseg.sh)
# and PhraseCut uses the Visual Genome photos above.
fetch_steervit() {
  local repo=$1 revision=$2
  local dest=$STEERVIT_DIR/${repo#*/}
  if compgen -G "$dest/data/train-*.parquet" > /dev/null; then
    echo "Using existing $dest"
    return
  fi
  local cli
  if command -v hf > /dev/null 2>&1; then
    cli=(hf download)
  elif command -v huggingface-cli > /dev/null 2>&1; then
    cli=(huggingface-cli download)
  else
    echo "ERROR: Hugging Face CLI not found (pip install huggingface_hub)." >&2
    exit 1
  fi
  "${cli[@]}" "$repo" --repo-type dataset --revision "$revision" --local-dir "$dest"
  compgen -G "$dest/data/train-*.parquet" > /dev/null || {
    echo "ERROR: $repo@$revision has no data/train-*.parquet under $dest" >&2
    exit 1
  }
  echo "-> $dest  ($repo@${revision:0:8})"
}

if [[ "$SKIP_GREFCOCO" == "1" ]]; then
  echo "SteerViT-gRefCOCO skipped (SKIP_GREFCOCO=1)."
else
  fetch_steervit JonaRuthardt/SteerViT-gRefCOCO "$GREFCOCO_REVISION"
fi
if [[ "$SKIP_PHRASECUT" == "1" ]]; then
  echo "SteerViT-PhraseCutDataset skipped (SKIP_PHRASECUT=1)."
else
  fetch_steervit JonaRuthardt/SteerViT-PhraseCutDataset "$PHRASECUT_REVISION"
fi

# ============================================================================ #
# Use them: the pool launcher stages both sources.
# By hand, extract them and point the prep at the dirs, e.g.
#   tar xf visual_genome.tar -C vg/
#   unzip -q shared_seg/mapillary.tar 'training/images/*' 'validation/images/*' \
#     -d map/mapillary_vistasv2
#   python dataset_prep/refseg/prepare_steervit_refseg.py --ignore-coco \
#     --visual-genome vg --mapillary map --out-root <prepared_refseg>
# ============================================================================ #
