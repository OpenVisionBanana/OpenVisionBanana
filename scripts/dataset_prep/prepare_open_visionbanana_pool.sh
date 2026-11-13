#!/usr/bin/env bash
set -euo pipefail

# Build canonical prepared datasets and a dataset_specs JSON for mixed training
# (the converters live in dataset_prep/ and are model-agnostic). Also used by
# stage_and_prepare_open_visionbanana_pool.sh after it stages data into scratch.
#
# Override paths with environment variables instead of editing this file.
#
# The specs JSON records the member paths and weights the trainer samples from
# and is checked by validate_dataset_specs.py.

show_help() {
  cat <<'EOF'
Build canonical prepared datasets and a weighted dataset-specs JSON for mixed
open-visionbanana-style training.

Use cases:
  1. Persistent-folder / simple setup
     Run this directly when your datasets already exist as extracted folders and
     you want to keep the prepared dataset pool in persistent storage.
  2. Scratch/TMPDIR + archive/inode-aware setup
     Usually called indirectly by
     scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh after it
     stages raw data into local scratch.

Configuration style:
  Set variables such as DATA_ROOT, PREPARED_ROOT, CITYSCAPES_ROOT,
  PICO_NATIVE_ROOT, and CONFIG_OUT in the environment before running.

Optional depth / surface-normal / image-generation members (all OFF unless
their weight is > 0; rebalance the other weights so the mix sums to 1.0):
  DEPTH_MEMBER_DIRS / DEPTH_WEIGHT        canonical depth member dirs; the task
  NORMALS_MEMBER_DIRS / NORMALS_WEIGHT    weight is split equally across dirs
  PD12M_WEIGHT / PD12M_SHARDS_DIR / PD12M_LIMIT / PD12M_MAX_SIDE / PD12M_OUT
                                          text-to-image member materialized from
                                          the PD12M WebDataset shards
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
[ -f "$ROOT/.venv/bin/activate" ] && source "$ROOT/.venv/bin/activate"
PYTHON=${PYTHON:-python}

DATA_ROOT=${DATA_ROOT:-$ROOT/datasets_archives}
PREPARED_ROOT=${PREPARED_ROOT:-$DATA_ROOT/open_visionbanana_prepared}

SEMSEG_OUT=${SEMSEG_OUT:-$PREPARED_ROOT/semseg}
# Pico checkout root: matches the default written by
# scripts/dataset_prep/data_download_scripts/pico_banana_400k.sh / download_pico_positive_edit.sh.
PICO_NATIVE_ROOT=${PICO_NATIVE_ROOT:-$DATA_ROOT/pico_banana/pico-banana-400k}
PICO_METADATA=${PICO_METADATA:-$PICO_NATIVE_ROOT/openimages/sft_with_local_source_image_path.jsonl}
PICO_OUT=${PICO_OUT:-$PREPARED_ROOT/pico_banana_sft_50k}
IDENTITY_OUT=${IDENTITY_OUT:-$PREPARED_ROOT/identity_10k}
REFSEG_OUT=${REFSEG_OUT:-$PREPARED_ROOT/refseg}
# Native ReasonSeg images + polygon annotations, stored as train.tar/test.tar.
REASONSEG_REFSEG_OUT=${REASONSEG_REFSEG_OUT:-$PREPARED_ROOT/refseg_reasonseg}
CONFIG_OUT=${CONFIG_OUT:-$PREPARED_ROOT/open_visionbanana_mix_v1.json}
# Dedicated task pools can omit Pico and its Pico-derived identity member.
PICO_ENABLE=${PICO_ENABLE:-1}
IDENTITY_ENABLE=${IDENTITY_ENABLE:-$PICO_ENABLE}

CITYSCAPES_ROOT=${CITYSCAPES_ROOT:-$DATA_ROOT/sem_segm/cityscapes}
# Keep Cityscapes validation held out for evaluation even when Cityscapes is
# explicitly enabled in the training pool. This does not affect other sources.
CITYSCAPES_SPLITS=${CITYSCAPES_SPLITS:-train}
# Synscapes shares its raw tree between semantic and instance segmentation.
# A set-but-empty value disables it.
SYNSCAPES_ROOT=${SYNSCAPES_ROOT-$DATA_ROOT/shared_seg/synscapes}
SYNSCAPES_SPLITS=${SYNSCAPES_SPLITS:-train,val}
# IDD carries polygon annotations shared by semantic and instance segmentation.
# The staging launcher resolves idd_1_2.tar to a parent holding both release parts.
IDD_ROOT=${IDD_ROOT-$DATA_ROOT/shared_seg/idd}
IDD_SPLITS=${IDD_SPLITS:-train,val}
# WildDash 2 public panoptic labels share one source tree between both tasks.
WILDDASH2_ROOT=${WILDDASH2_ROOT-$DATA_ROOT/shared_seg/wilddash2}
WILDDASH2_SPLITS=${WILDDASH2_SPLITS:-train,val}
# NuImages uses only annotated key-camera samples; temporal sweeps never enter
# the pool. The common discovery helper keeps one mask-bearing keyframe per
# driving log every 10 seconds by default.
NUIMAGES_ROOT=${NUIMAGES_ROOT-$DATA_ROOT/shared_seg/nuimages}
NUIMAGES_SPLITS=${NUIMAGES_SPLITS:-train,val}
NUIMAGES_SAMPLE_INTERVAL_SECONDS=${NUIMAGES_SAMPLE_INTERVAL_SECONDS:-10}
# COCONut-XL is already materialized under shared_seg. Its merged manifest has
# one logical training split and feeds both semantic and instance segmentation.
COCONUT_XL_ROOT=${COCONUT_XL_ROOT-$DATA_ROOT/shared_seg/coconut_xlarge/coco}
# Select merged COCONut-XL provenance groups (coco, objects365, or both).
COCONUT_XL_GROUPS=${COCONUT_XL_GROUPS:-coco,objects365}
# PACO-LVIS is instance-only. The archive-aware launcher extracts it to this
# path; direct users may instead point this at an extracted DatasetNinja tree.
PACO_LVIS_ROOT=${PACO_LVIS_ROOT-$DATA_ROOT/instanceseg/paco_lvis}
PACO_LVIS_SPLITS=${PACO_LVIS_SPLITS:-train,val,test}
# Official RefCOCOg UMD annotations define the held-out val images that must
# never be used through COCONut-XL or PACO-LVIS.
REFCOCOG_UMD_ROOT=${REFCOCOG_UMD_ROOT:-$DATA_ROOT/refseg/refcocog}
ACDC_ROOT=${ACDC_ROOT:-}
BDD_ROOT=${BDD_ROOT:-}
# ADEChallengeData2016 (150-class) + Pascal-VOC 2012 roots (see
# scripts/dataset_prep/data_download_scripts/download_datasets_semseg_ade_voc.sh).
ADE20K_ROOT=${ADE20K_ROOT:-}
VOC_ROOT=${VOC_ROOT:-}
# PASCAL-Context 59-class labels, packed inside voc.tar next to the VOC images by
# scripts/dataset_prep/data_download_scripts/download_datasets_pascal_context.sh. Points at the VOC
# root (the one carrying pascal_context_59classlabels/), so it defaults to VOC_ROOT.
PASCAL_CONTEXT_ROOT=${PASCAL_CONTEXT_ROOT:-}
# MUSES root with frame_camera/, gt_semantic/, and optionally gt_panoptic/.
MUSES_ROOT=${MUSES_ROOT:-}
# GOOSE 2D root with images/, labels/, and goose_label_mapping.csv.
GOOSE_ROOT=${GOOSE_ROOT:-}
# VIPSeg root with imgs/, panomasks/, and VIPSeg_720P/panoVIPSeg_categories.json.
VIPSEG_ROOT=${VIPSEG_ROOT:-}
# Mapillary Vistas root with config_v2.0.json and {training,validation}/images/ +
# {training,validation}/v2.0/labels/.
MAPILLARY_ROOT=${MAPILLARY_ROOT:-}
# FoodSeg103 root with images/{train,val}/ and labels/{train,val}/ materialized
# from the Hugging Face parquet dataset.
FOODSEG103_ROOT=${FOODSEG103_ROOT:-}
# Trans10K-v2 root with images/{train,val,test}/ and labels/{train,val,test}/
# materialized from the Hugging Face parquet dataset
# (scripts/dataset_prep/data_download_scripts/download_datasets_trans10k.sh).
TRANS10K_ROOT=${TRANS10K_ROOT:-}
# RailSem19 rail-scene semantic segmentation, native release layout (jpgs/ +
# uint8/, optionally nested under versions/<n>/); the reader resolves either.
# All 8,500 frames carry a mask and there is no official split, so the reader
# emits every one of them under "train" -- nothing is held out
# (scripts/dataset_prep/data_download_scripts/download_datasets_railsem19.sh).
RAILSEM19_ROOT=${RAILSEM19_ROOT:-}
# Waymo Open Dataset camera panoptic labels, materialized from the v2 parquet
# release (scripts/dataset_prep/data_download_scripts/download_datasets_waymo.sh). One tree feeds
# BOTH members: semseg reads the semantic half of the packed labels, instanceseg
# the instance half (WAYMO_INST_ROOT below).
WAYMO_ROOT=${WAYMO_ROOT:-}
# Hypersim photorealistic synthetic indoor scenes (NYU40), materialized from the
# release's per-frame HDF5 by dataset_prep/hypersim/prepare_hypersim_segmentation.py
# into the same packed-panoptic layout as Waymo. One tree feeds BOTH members:
# semseg reads the semantic half, instanceseg the instance half
# (HYPERSIM_SEG_INST_ROOT below).
HYPERSIM_SEG_ROOT=${HYPERSIM_SEG_ROOT:-}

SEMSEG_NUM_IMAGES=${SEMSEG_NUM_IMAGES:-}
SEMSEG_NUM_WORKERS=${SEMSEG_NUM_WORKERS:-1}


# Per-image diversification (on by default). When ON, each
# image yields SEMSEG_VARIANTS_PER_IMAGE randomized (prompt, mask) pairs (per-image
# class subset + random colors from the shared 200-color palette + randomized
# prompt). When OFF, it falls back to plain canonical pairs (full Cityscapes-palette
# mask + the fixed 19-class prompt; trainId datasets only).
SEMSEG_DIVERSIFY=${SEMSEG_DIVERSIFY:-1}
SEMSEG_VARIANTS_PER_IMAGE=${SEMSEG_VARIANTS_PER_IMAGE:-1}
# Variant composition by category (must sum to 1.0; per-variant random draw):
#   subset       -> random proper subset of present classes
#   full-set     -> all present classes
#   negative-set -> proper subset of present + absent classes named-but-unpainted
#   full-dataset -> every taxonomy class: present painted, all absent named-but-unpainted
SEMSEG_SUBSET_RATIO=${SEMSEG_SUBSET_RATIO:-0.7}
SEMSEG_FULL_SET_RATIO=${SEMSEG_FULL_SET_RATIO:-0.15}
SEMSEG_NEGATIVE_SET_RATIO=${SEMSEG_NEGATIVE_SET_RATIO:-0.15}
SEMSEG_FULL_DATASET_RATIO=${SEMSEG_FULL_DATASET_RATIO:-0}
SEMSEG_MAX_NEGATIVE_CLASSES=${SEMSEG_MAX_NEGATIVE_CLASSES:-30}  # cap on negatives per negative-set variant
SEMSEG_BACKGROUND_COLOR_PROB=${SEMSEG_BACKGROUND_COLOR_PROB:-0.0}
SEMSEG_INVERSE_FREQ_POWER=${SEMSEG_INVERSE_FREQ_POWER:-1.0}
SEMSEG_COLOR_MIN_DELTA_E=${SEMSEG_COLOR_MIN_DELTA_E:-25.0}
# Legend color draw: p200_randomnegdraw (legacy greedy over the full 200 palette),
# p100_randomnegdraw (same greedy over only the 100 most-separated colors),
# best64_maxnegdraw / best32_maxnegdraw (uniform from the 64/32 most-separated
# colors, rank-weighted farthest backfill on overflow), or bestdrawperimg
# (per-image pool sized by the classes present in the mask).
SEMSEG_COLOR_MAPPING=${SEMSEG_COLOR_MAPPING:-p200_randomnegdraw}
SEMSEG_MIN_CLASS_AREA=${SEMSEG_MIN_CLASS_AREA:-0.0}
SEMSEG_CANONICAL_ANCHOR=${SEMSEG_CANONICAL_ANCHOR:-0}  # also emit the canonical fixed pair (trainId datasets)
# Referring-segmentation training data (SteerViT, pulled from the HF hub). Emits
# the same prepared layout as the other members; the referred region is painted a
# per-row sampled named color (see dataset_prep/refseg/prepare_steervit_refseg.py,
# DEFAULT_COLOR_SPECS / --colors). Disable with REFSEG_ENABLE=0.
REFSEG_ENABLE=${REFSEG_ENABLE:-1}
REFSEG_SPLIT=${REFSEG_SPLIT:-train,validation}
REFSEG_MIN_WIDTH=${REFSEG_MIN_WIDTH:-336}
REFSEG_MIN_HEIGHT=${REFSEG_MIN_HEIGHT:-336}
REFSEG_IGNORE_COCO=${REFSEG_IGNORE_COCO:-1}
REFSEG_REFCOCOG_UMD_SPLITS=${REFSEG_REFCOCOG_UMD_SPLITS:-}  # e.g. train; empty = no native UMD filter
REFSEG_REFCOCOG_UMD_DIR=${REFSEG_REFCOCOG_UMD_DIR:-}        # dir containing refs(umd).p; empty = converter default
REFSEG_NUM_WORKERS=${REFSEG_NUM_WORKERS:-14}
REFSEG_COLORS=${REFSEG_COLORS:-}              # NAME:R,G,B;... (empty = script DEFAULT_COLOR_SPECS)
REFSEG_HF_DATASETS=${REFSEG_HF_DATASETS:-}     # empty = script default (JonaRuthardt/SteerViT)
# Masks of the listed `dataset` values come from a re-annotated release, e.g.
# REFSEG_MASK_OVERRIDE_HF_DATASET=JonaRuthardt/SteerViT-SAM3-bboxprompt with
# REFSEG_MASK_OVERRIDE_SOURCES="visual genome,refcoco,refcoco+,refcocog".
# Empty = every mask from the base release.
REFSEG_MASK_OVERRIDE_HF_DATASET=${REFSEG_MASK_OVERRIDE_HF_DATASET:-}
REFSEG_MASK_OVERRIDE_SOURCES=${REFSEG_MASK_OVERRIDE_SOURCES:-}
REFSEG_NO_RANDOM_COLORS=${REFSEG_NO_RANDOM_COLORS:-0}
REFSEG_EXTRA_ARGS=${REFSEG_EXTRA_ARGS:-}       # passthrough, e.g. --include-datasets / image roots
# Set REASONSEG_REFSEG_SPLITS=train, test, or train,test.  All released
# expressions are retained by default; use REASONSEG_REFSEG_ALL_EXPRESSIONS=0
# plus REASONSEG_REFSEG_EXPRESSIONS_PER_IMAGE to subsample per image.
REASONSEG_REFSEG_ENABLE=${REASONSEG_REFSEG_ENABLE:-0}
REASONSEG_REFSEG_ROOT=${REASONSEG_REFSEG_ROOT:-$DATA_ROOT/refseg/reasonseg_train_test}
REASONSEG_REFSEG_EXTRACT_ROOT=${REASONSEG_REFSEG_EXTRACT_ROOT:-}
REASONSEG_REFSEG_KEEP_EXTRACTED=${REASONSEG_REFSEG_KEEP_EXTRACTED:-0}
REASONSEG_REFSEG_SPLITS=${REASONSEG_REFSEG_SPLITS:-train}
REASONSEG_REFSEG_ALL_EXPRESSIONS=${REASONSEG_REFSEG_ALL_EXPRESSIONS:-1}
REASONSEG_REFSEG_EXPRESSIONS_PER_IMAGE=${REASONSEG_REFSEG_EXPRESSIONS_PER_IMAGE:-1}
REASONSEG_REFSEG_VAL_DIR=${REASONSEG_REFSEG_VAL_DIR:-$DATA_ROOT/refseg/reasonseg_val}
REASONSEG_REFSEG_EXCLUDE_VAL_OVERLAP=${REASONSEG_REFSEG_EXCLUDE_VAL_OVERLAP:-1}
REASONSEG_REFSEG_COLORS=${REASONSEG_REFSEG_COLORS:-}
REASONSEG_REFSEG_NO_RANDOM_COLORS=${REASONSEG_REFSEG_NO_RANDOM_COLORS:-0}
REASONSEG_REFSEG_WEIGHT=${REASONSEG_REFSEG_WEIGHT:-0.0}

# Depth + normals: DEPTH_MEMBER_DIRS / NORMALS_MEMBER_DIRS are space-separated
# canonical member dirs (metadata.jsonl + conditioning_images/ + target_images/),
# usually extracted from the prepare_*_depth_normals.sh tars by
# stage_and_prepare_open_visionbanana_pool.sh. Their task weight (below) is
# split EQUALLY across the member dirs.
DEPTH_MEMBER_DIRS=${DEPTH_MEMBER_DIRS:-}
NORMALS_MEMBER_DIRS=${NORMALS_MEMBER_DIRS:-}
# PD12M (image generation): materialized from the WebDataset shards into a
# canonical text-to-image member (target_images/ + captions, NO conditioning
# image) by scripts/dataset_prep/materialize_pd12m_canonical.py.
PD12M_SHARDS_DIR=${PD12M_SHARDS_DIR:-$DATA_ROOT/image_generation/raw/pd12m/500k/shards}

# --- Per-member row limits (seeded subsets; empty = full set) ----------------
# PICO/REFSEG use ${VAR-default} (no colon) so an explicitly empty value means
# "full set" instead of falling back to the default.
PICO_LIMIT=${PICO_LIMIT-50000}
IDENTITY_LIMIT=${IDENTITY_LIMIT:-30000}
REFSEG_LIMIT=${REFSEG_LIMIT-50000}
PD12M_LIMIT=${PD12M_LIMIT:-50000}

MAX_SIDE=${MAX_SIDE:-1024}
PICO_MAX_SIDE=${PICO_MAX_SIDE:-$MAX_SIDE}
IDENTITY_MAX_SIDE=${IDENTITY_MAX_SIDE:-$MAX_SIDE}
PD12M_MAX_SIDE=${PD12M_MAX_SIDE:-$MAX_SIDE}
PICO_COPY_MODE=${PICO_COPY_MODE:-copy}
IDENTITY_COPY_MODE=${IDENTITY_COPY_MODE:-copy}
PICO_NUM_WORKERS=${PICO_NUM_WORKERS:-1}
PD12M_OUT=${PD12M_OUT:-$PREPARED_ROOT/pd12m_${PD12M_LIMIT}}
SEED=${SEED:-42}
FORCE=${FORCE:-0}
VALIDATE=${VALIDATE:-1}

SAMPLES_PER_EPOCH=${SAMPLES_PER_EPOCH:-250048}
# --- Mixture weights (keep summing to 1.0) -----------------------------------
# Weights are relative, kept summing to 1.0 so they read directly as sampling
# fractions. The trainer normalizes them when it samples the pool.
# A member with weight 0 is dropped from the mix (and depth/normals/pd12m are
# then not staged/materialized at all).
SEMSEG_WEIGHT=${SEMSEG_WEIGHT:-0.20}
PICO_WEIGHT=${PICO_WEIGHT:-0.096}
IDENTITY_WEIGHT=${IDENTITY_WEIGHT:-0.064}
REFSEG_WEIGHT=${REFSEG_WEIGHT:-0.152}
INSTANCESEG_WEIGHT=${INSTANCESEG_WEIGHT:-0.2}
DEPTH_WEIGHT=${DEPTH_WEIGHT:-0.128}
NORMALS_WEIGHT=${NORMALS_WEIGHT:-0.064}
PD12M_WEIGHT=${PD12M_WEIGHT:-0.096}

# Instance segmentation member (ADE20K-instance + Mapillary Vistas + Pascal VOC).
# Colors each instance of one randomly-sampled class per image a distinct color;
# enabled when at least one source root is set (staged by stage_and_prepare). Roots
# are passed in by the staging script: ADE_INST_ROOT/MAPILLARY_INST_ROOT/VOC_INST_ROOT.
INSTANCESEG_ENABLE=${INSTANCESEG_ENABLE:-1}
INSTANCESEG_OUT=${INSTANCESEG_OUT:-$PREPARED_ROOT/instanceseg}
INSTANCESEG_SPLITS=${INSTANCESEG_SPLITS:-train,val}
INSTANCESEG_VARIANTS_PER_IMAGE=${INSTANCESEG_VARIANTS_PER_IMAGE:-1}
INSTANCESEG_NEGATIVE_RATIO=${INSTANCESEG_NEGATIVE_RATIO:-0.1}
INSTANCESEG_MAX_INSTANCES=${INSTANCESEG_MAX_INSTANCES:-}
INSTANCESEG_LIMIT=${INSTANCESEG_LIMIT-}  # debug cap on total pairs; empty = full set
INSTANCESEG_MIN_INSTANCE_AREA=${INSTANCESEG_MIN_INSTANCE_AREA:-}
# Prob. a positive prompt lists 1-4 extra unused legend colors (over-length
# legends, matching eval's fixed-length-legend behavior); 0 = exact count.
INSTANCESEG_LEGEND_SLACK_PROB=${INSTANCESEG_LEGEND_SLACK_PROB:-0.2}
# Color/prompt scheme: legend (default, exact-count legends) or a fixed-K draw
# mode (fixeddraw_32 / randomdraw_32 / fixeddraw_64 / randomdraw_64 /
# randomdraw_32_unlisted / fixeddraw_32_ordered_unlisted / fixeddraw_32_lr /
# fixeddraw_32_mixedrepresent). Draw modes ignore the slack prob and clamp
# max-instances to K.
INSTANCESEG_DRAW_MODE=${INSTANCESEG_DRAW_MODE:-legend}
INSTANCESEG_NUM_WORKERS=${INSTANCESEG_NUM_WORKERS:-${SEMSEG_NUM_WORKERS:-16}}

OVIS_FPS=${OVIS_FPS:-1.0}
OVIS_SOURCE_FPS=${OVIS_SOURCE_FPS:-5.0}
ADE_INST_ROOT=${ADE_INST_ROOT:-}
MAPILLARY_INST_ROOT=${MAPILLARY_INST_ROOT:-}
SYNSCAPES_INST_ROOT=${SYNSCAPES_INST_ROOT:-}
if [[ -z "$SYNSCAPES_INST_ROOT" && -n "$SYNSCAPES_ROOT" ]]; then
  if [[ -d "$SYNSCAPES_ROOT/img/instance" ||
        -d "$SYNSCAPES_ROOT/Synscapes/img/instance" ||
        -d "$SYNSCAPES_ROOT/synscapes/img/instance" ]]; then
    SYNSCAPES_INST_ROOT=$SYNSCAPES_ROOT
  fi
fi
IDD_INST_ROOT=${IDD_INST_ROOT:-}
if [[ -z "$IDD_INST_ROOT" && -n "$IDD_ROOT" ]]; then
  if [[ -d "$IDD_ROOT/gtFine" || -d "$IDD_ROOT/IDD_Segmentation/gtFine" ||
        -d "$IDD_ROOT/idd20kII/gtFine" ]]; then
    IDD_INST_ROOT=$IDD_ROOT
  fi
fi
WILDDASH2_INST_ROOT=${WILDDASH2_INST_ROOT:-}
if [[ -z "$WILDDASH2_INST_ROOT" && -n "$WILDDASH2_ROOT" && -d "$WILDDASH2_ROOT" ]]; then
  WILDDASH2_INST_ROOT=$WILDDASH2_ROOT
fi
NUIMAGES_INST_ROOT=${NUIMAGES_INST_ROOT:-}
if [[ -z "$NUIMAGES_INST_ROOT" && -n "$NUIMAGES_ROOT" && -d "$NUIMAGES_ROOT" ]]; then
  NUIMAGES_INST_ROOT=$NUIMAGES_ROOT
fi
COCONUT_XL_INST_ROOT=${COCONUT_XL_INST_ROOT:-}
if [[ -z "$COCONUT_XL_INST_ROOT" && -n "$COCONUT_XL_ROOT" &&
      -d "$COCONUT_XL_ROOT/train2017" && -d "$COCONUT_XL_ROOT/panoptic_train2017" &&
      -f "$COCONUT_XL_ROOT/annotations/panoptic_train2017.json" ]]; then
  COCONUT_XL_INST_ROOT=$COCONUT_XL_ROOT
fi
PACO_LVIS_INST_ROOT=${PACO_LVIS_INST_ROOT:-}
if [[ -z "$PACO_LVIS_INST_ROOT" && -n "$PACO_LVIS_ROOT" && -f "$PACO_LVIS_ROOT/meta.json" ]]; then
  PACO_LVIS_INST_ROOT=$PACO_LVIS_ROOT
fi
VOC_INST_ROOT=${VOC_INST_ROOT:-}
MUSES_INST_ROOT=${MUSES_INST_ROOT:-}
if [[ -z "$MUSES_INST_ROOT" && -n "$MUSES_ROOT" && -d "$MUSES_ROOT/gt_panoptic" ]]; then
  MUSES_INST_ROOT=$MUSES_ROOT
fi
# ACDC ships gt_panoptic/ alongside the semantic gt/, in the same COCO-panoptic
# format as MUSES, so the semseg root serves instanceseg too. Older acdc.tar
# builds have no gt_panoptic/ -- the -d guard keeps those semseg-only.
ACDC_INST_ROOT=${ACDC_INST_ROOT:-}
if [[ -z "$ACDC_INST_ROOT" && -n "$ACDC_ROOT" && -d "$ACDC_ROOT/gt_panoptic" ]]; then
  ACDC_INST_ROOT=$ACDC_ROOT
fi
GOOSE_INST_ROOT=${GOOSE_INST_ROOT:-}
if [[ -z "$GOOSE_INST_ROOT" && -n "$GOOSE_ROOT" && -d "$GOOSE_ROOT/labels" ]]; then
  GOOSE_INST_ROOT=$GOOSE_ROOT
fi
VIPSEG_INST_ROOT=${VIPSEG_INST_ROOT:-}
if [[ -z "$VIPSEG_INST_ROOT" && -n "$VIPSEG_ROOT" && -d "$VIPSEG_ROOT/panomasks" ]]; then
  VIPSEG_INST_ROOT=$VIPSEG_ROOT
fi
OVIS_INST_ROOT=${OVIS_INST_ROOT:-}
# Waymo panoptic labels carry instances too, so the same root serves instanceseg.
WAYMO_INST_ROOT=${WAYMO_INST_ROOT:-}
if [[ -z "$WAYMO_INST_ROOT" && -n "$WAYMO_ROOT" && -d "$WAYMO_ROOT/panoptic" ]]; then
  WAYMO_INST_ROOT=$WAYMO_ROOT
fi
# Hypersim likewise: same packed tree, both members.
HYPERSIM_SEG_INST_ROOT=${HYPERSIM_SEG_INST_ROOT:-}
if [[ -z "$HYPERSIM_SEG_INST_ROOT" && -n "$HYPERSIM_SEG_ROOT" && -d "$HYPERSIM_SEG_ROOT/panoptic" ]]; then
  HYPERSIM_SEG_INST_ROOT=$HYPERSIM_SEG_ROOT
fi

is_true() { case "${1,,}" in 1|true|yes|y|on) return 0 ;; *) return 1 ;; esac; }

# True when a mix weight opts a member in (handles floats like "0.15").
weight_positive() { awk -v w="${1:-0}" 'BEGIN { exit !(w + 0 > 0) }'; }

# Run from $ROOT so the converters' repo-relative paths resolve.
run_in_env() {
  (cd "$ROOT" && "$@")
}

has_nonempty_metadata() {
  local metadata_path=$1
  [[ -s "$metadata_path" ]]
}

mkdir -p "$PREPARED_ROOT"

if [[ "$FORCE" == "1" ]] || ! has_nonempty_metadata "$SEMSEG_OUT/metadata.jsonl"; then
  semseg_args=(dataset_prep/semseg/dataset_prep.py --out-root "$SEMSEG_OUT" --seed "$SEED")
  has_semseg_root=0
  if [[ -n "$SEMSEG_NUM_IMAGES" ]]; then
    semseg_args+=(--num-images "$SEMSEG_NUM_IMAGES")
  fi
  if [[ "$SEMSEG_NUM_WORKERS" != "1" ]]; then
    semseg_args+=(--num-workers "$SEMSEG_NUM_WORKERS")
  fi
  case "${SEMSEG_DIVERSIFY,,}" in
    1|true|yes|y|on)
      semseg_args+=(
        --variants-per-image "$SEMSEG_VARIANTS_PER_IMAGE"
        --subset-ratio "$SEMSEG_SUBSET_RATIO"
        --full-set-ratio "$SEMSEG_FULL_SET_RATIO"
        --negative-set-ratio "$SEMSEG_NEGATIVE_SET_RATIO"
        --full-dataset-ratio "$SEMSEG_FULL_DATASET_RATIO"
        --max-negative-classes "$SEMSEG_MAX_NEGATIVE_CLASSES"
        --background-color-prob "$SEMSEG_BACKGROUND_COLOR_PROB"
        --inverse-freq-power "$SEMSEG_INVERSE_FREQ_POWER"
        --color-min-delta-e "$SEMSEG_COLOR_MIN_DELTA_E"
        --color-mapping "$SEMSEG_COLOR_MAPPING"
        --min-class-area "$SEMSEG_MIN_CLASS_AREA"
      )
      case "${SEMSEG_CANONICAL_ANCHOR,,}" in 1|true|yes|y|on) semseg_args+=(--canonical-anchor) ;; esac
      ;;
    *)
      # Plain canonical pairs only (full palette mask + fixed 19-class prompt).
      semseg_args+=(--variants-per-image 0 --canonical-anchor --inverse-freq-power 0)
      ;;
  esac
  if [[ -d "$CITYSCAPES_ROOT" ]]; then
    semseg_args+=(
      --cityscapes-root "$CITYSCAPES_ROOT"
      --cityscapes-splits "$CITYSCAPES_SPLITS"
    )
    has_semseg_root=1
  fi
  if [[ -n "$SYNSCAPES_ROOT" && -d "$SYNSCAPES_ROOT" ]]; then
    semseg_args+=(--synscapes-root "$SYNSCAPES_ROOT" --synscapes-splits "$SYNSCAPES_SPLITS")
    has_semseg_root=1
  fi
  if [[ -n "$IDD_ROOT" && -d "$IDD_ROOT" ]]; then
    semseg_args+=(--idd-root "$IDD_ROOT" --idd-splits "$IDD_SPLITS")
    has_semseg_root=1
  fi
  if [[ -n "$WILDDASH2_ROOT" && -d "$WILDDASH2_ROOT" ]]; then
    semseg_args+=(--wilddash2-root "$WILDDASH2_ROOT" --wilddash2-splits "$WILDDASH2_SPLITS")
    has_semseg_root=1
  fi
  if [[ -n "$NUIMAGES_ROOT" && -d "$NUIMAGES_ROOT" ]]; then
    semseg_args+=(--nuimages-root "$NUIMAGES_ROOT" --nuimages-splits "$NUIMAGES_SPLITS" --nuimages-sample-interval-seconds "$NUIMAGES_SAMPLE_INTERVAL_SECONDS")
    has_semseg_root=1
  fi
  if [[ -n "$COCONUT_XL_ROOT" && -d "$COCONUT_XL_ROOT/train2017" &&
        -d "$COCONUT_XL_ROOT/panoptic_train2017" &&
        -f "$COCONUT_XL_ROOT/annotations/panoptic_train2017.json" ]]; then
    semseg_args+=(--coconut-xlarge-root "$COCONUT_XL_ROOT" --coconut-xlarge-groups "$COCONUT_XL_GROUPS" --refcocog-umd-root "$REFCOCOG_UMD_ROOT")
    has_semseg_root=1
  fi
  if [[ -n "$ACDC_ROOT" && -d "$ACDC_ROOT" ]]; then
    semseg_args+=(--acdc-root "$ACDC_ROOT")
    has_semseg_root=1
  fi
  if [[ -n "$BDD_ROOT" && -d "$BDD_ROOT" ]]; then
    semseg_args+=(--bdd-root "$BDD_ROOT")
    has_semseg_root=1
  fi
  if [[ -n "$ADE20K_ROOT" && -d "$ADE20K_ROOT" ]]; then
    semseg_args+=(--ade20k-root "$ADE20K_ROOT")
    has_semseg_root=1
  fi
  if [[ -n "$VOC_ROOT" && -d "$VOC_ROOT" ]]; then
    semseg_args+=(--voc-root "$VOC_ROOT")
    has_semseg_root=1
  fi
  if [[ -n "$MUSES_ROOT" && -d "$MUSES_ROOT" ]]; then
    semseg_args+=(--muses-root "$MUSES_ROOT")
    has_semseg_root=1
  fi
  if [[ -n "$GOOSE_ROOT" && -d "$GOOSE_ROOT" ]]; then
    semseg_args+=(--goose-root "$GOOSE_ROOT")
    has_semseg_root=1
  fi
  if [[ -n "$VIPSEG_ROOT" && -d "$VIPSEG_ROOT" ]]; then
    semseg_args+=(--vipseg-root "$VIPSEG_ROOT")
    has_semseg_root=1
  fi
  # Falls back to VOC_ROOT: the context labels ride along inside voc.tar.
  pascal_context_root=${PASCAL_CONTEXT_ROOT:-$VOC_ROOT}
  if [[ -n "$pascal_context_root" && -d "$pascal_context_root/pascal_context_59classlabels" ]]; then
    semseg_args+=(--pascal-context-root "$pascal_context_root")
    has_semseg_root=1
  fi
  if [[ -n "$MAPILLARY_ROOT" && -d "$MAPILLARY_ROOT" ]]; then
    semseg_args+=(--mapillary-root "$MAPILLARY_ROOT")
    has_semseg_root=1
  fi
  if [[ -n "$FOODSEG103_ROOT" && -d "$FOODSEG103_ROOT" ]]; then
    semseg_args+=(--foodseg103-root "$FOODSEG103_ROOT")
    has_semseg_root=1
  fi
  if [[ -n "$TRANS10K_ROOT" && -d "$TRANS10K_ROOT" ]]; then
    semseg_args+=(--trans10k-root "$TRANS10K_ROOT")
    has_semseg_root=1
  fi
  if [[ -n "$RAILSEM19_ROOT" && -d "$RAILSEM19_ROOT" ]]; then
    semseg_args+=(--railsem19-root "$RAILSEM19_ROOT")
    has_semseg_root=1
  fi
  if [[ -n "$WAYMO_ROOT" && -d "$WAYMO_ROOT" ]]; then
    semseg_args+=(--waymo-root "$WAYMO_ROOT")
    has_semseg_root=1
  fi
  if [[ -n "$HYPERSIM_SEG_ROOT" && -d "$HYPERSIM_SEG_ROOT" ]]; then
    semseg_args+=(--hypersim-root "$HYPERSIM_SEG_ROOT")
    has_semseg_root=1
  fi
  if [[ "$has_semseg_root" == "0" ]]; then
    echo "ERROR: no semantic-segmentation roots found. Set CITYSCAPES_ROOT, SYNSCAPES_ROOT, IDD_ROOT, WILDDASH2_ROOT, COCONUT_XL_ROOT, ACDC_ROOT, BDD_ROOT, MUSES_ROOT, GOOSE_ROOT, VIPSEG_ROOT, MAPILLARY_ROOT, ADE20K_ROOT, VOC_ROOT, PASCAL_CONTEXT_ROOT, FOODSEG103_ROOT, TRANS10K_ROOT, RAILSEM19_ROOT, WAYMO_ROOT, and/or HYPERSIM_SEG_ROOT." >&2
    exit 1
  fi
  run_in_env "$PYTHON" "${semseg_args[@]}"
else
  echo "Using existing semantic-segmentation prepared dataset: $SEMSEG_OUT"
fi

if is_true "$PICO_ENABLE" && { [[ "$FORCE" == "1" ]] || ! has_nonempty_metadata "$PICO_OUT/metadata.jsonl"; }; then
  if [[ ! -f "$PICO_METADATA" ]]; then
    echo "ERROR: Pico mapped metadata is missing: $PICO_METADATA" >&2
    echo "Run scripts/dataset_downloads/pico_banana_400k.sh first, or set PICO_METADATA." >&2
    exit 1
  fi
  pico_args=(
    dataset_prep/pico_banana/prepare_pico_banana.py
    --pico-root "$PICO_NATIVE_ROOT"
    --metadata "$PICO_METADATA"
    --out-root "$PICO_OUT"
    --seed "$SEED"
    --max-side "$PICO_MAX_SIDE"
    --copy-mode "$PICO_COPY_MODE"
    --num-workers "$PICO_NUM_WORKERS"
  )
  # Conditional like refseg below: an empty PICO_LIMIT means no cap.
  [[ -n "$PICO_LIMIT" ]] && pico_args+=(--limit "$PICO_LIMIT")
  run_in_env "$PYTHON" "${pico_args[@]}"
elif is_true "$PICO_ENABLE"; then
  echo "Using existing Pico-Banana prepared dataset: $PICO_OUT"
else
  echo "PICO_ENABLE is off; skipping Pico-Banana preparation."
fi

if is_true "$IDENTITY_ENABLE" && { [[ "$FORCE" == "1" ]] || ! has_nonempty_metadata "$IDENTITY_OUT/metadata.jsonl"; }; then
  identity_args=(
    dataset_prep/identity/prepare_identity.py
    --image-root "$PICO_OUT/conditioning_images"
    --out-root "$IDENTITY_OUT"
    --limit "$IDENTITY_LIMIT"
    --seed "$SEED"
    --max-side "$IDENTITY_MAX_SIDE"
    --copy-mode "$IDENTITY_COPY_MODE"
    --source-dataset identity_from_pico_banana
  )
  run_in_env "$PYTHON" "${identity_args[@]}"
elif is_true "$IDENTITY_ENABLE"; then
  echo "Using existing identity prepared dataset: $IDENTITY_OUT"
else
  echo "IDENTITY_ENABLE is off; skipping identity preparation."
fi

if is_true "$REFSEG_ENABLE"; then
  if [[ "$FORCE" == "1" ]] || ! has_nonempty_metadata "$REFSEG_OUT/metadata.jsonl"; then
    refseg_args=(
      dataset_prep/refseg/prepare_steervit_refseg.py
      --out-root "$REFSEG_OUT"
      --refcocog-umd-dir "$REFCOCOG_UMD_ROOT"
      --split "$REFSEG_SPLIT"
      --num-workers "$REFSEG_NUM_WORKERS"
      --seed "$SEED"
    )
    is_true "$REFSEG_IGNORE_COCO" && refseg_args+=(--ignore-coco)
    [[ -n "$REFSEG_MIN_WIDTH" ]] && refseg_args+=(--min-width "$REFSEG_MIN_WIDTH")
    [[ -n "$REFSEG_MIN_HEIGHT" ]] && refseg_args+=(--min-height "$REFSEG_MIN_HEIGHT")
    [[ -n "$REFSEG_REFCOCOG_UMD_SPLITS" ]] && refseg_args+=(--refcocog-umd-splits "$REFSEG_REFCOCOG_UMD_SPLITS")
    [[ -n "$REFSEG_REFCOCOG_UMD_DIR" ]] && refseg_args+=(--refcocog-umd-dir "$REFSEG_REFCOCOG_UMD_DIR")
    [[ -n "$REFSEG_COLORS" ]] && refseg_args+=(--colors "$REFSEG_COLORS")
    [[ -n "$REFSEG_HF_DATASETS" ]] && refseg_args+=(--hf-datasets "$REFSEG_HF_DATASETS")
    [[ -n "$REFSEG_MASK_OVERRIDE_HF_DATASET" ]] && refseg_args+=(--mask-override-hf-dataset "$REFSEG_MASK_OVERRIDE_HF_DATASET")
    [[ -n "$REFSEG_MASK_OVERRIDE_SOURCES" ]] && refseg_args+=(--mask-override-sources "$REFSEG_MASK_OVERRIDE_SOURCES")
    [[ -n "$REFSEG_LIMIT" ]] && refseg_args+=(--limit "$REFSEG_LIMIT")
    is_true "$REFSEG_NO_RANDOM_COLORS" && refseg_args+=(--no-random-colors)
    # shellcheck disable=SC2206  -- intentional word-splitting of the passthrough
    [[ -n "$REFSEG_EXTRA_ARGS" ]] && refseg_args+=($REFSEG_EXTRA_ARGS)
    run_in_env "$PYTHON" "${refseg_args[@]}"
  else
    echo "Using existing refseg prepared dataset: $REFSEG_OUT"
  fi
fi


if is_true "$INSTANCESEG_ENABLE" && [[ -n "$ADE_INST_ROOT$MAPILLARY_INST_ROOT$SYNSCAPES_INST_ROOT$IDD_INST_ROOT$WILDDASH2_INST_ROOT$NUIMAGES_INST_ROOT$COCONUT_XL_INST_ROOT$PACO_LVIS_INST_ROOT$VOC_INST_ROOT$MUSES_INST_ROOT$ACDC_INST_ROOT$GOOSE_INST_ROOT$VIPSEG_INST_ROOT$OVIS_INST_ROOT$WAYMO_INST_ROOT$HYPERSIM_SEG_INST_ROOT" ]]; then
  if [[ "$FORCE" == "1" ]] || ! has_nonempty_metadata "$INSTANCESEG_OUT/metadata.jsonl"; then
    iseg_args=(
      dataset_prep/instanceseg/dataset_prep.py
      --out-root "$INSTANCESEG_OUT"
      --splits "$INSTANCESEG_SPLITS"
      --variants-per-image "$INSTANCESEG_VARIANTS_PER_IMAGE"
      --negative-ratio "$INSTANCESEG_NEGATIVE_RATIO"
      --num-workers "$INSTANCESEG_NUM_WORKERS"
      --seed "$SEED"
    )
    [[ -n "$ADE_INST_ROOT" ]] && iseg_args+=(--ade-root "$ADE_INST_ROOT")
    if [[ -n "$COCONUT_XL_INST_ROOT$PACO_LVIS_INST_ROOT" ]]; then
      iseg_args+=(--refcocog-umd-root "$REFCOCOG_UMD_ROOT")
    fi
    [[ -n "$MAPILLARY_INST_ROOT" ]] && iseg_args+=(--mapillary-root "$MAPILLARY_INST_ROOT")
    [[ -n "$SYNSCAPES_INST_ROOT" ]] && iseg_args+=(--synscapes-root "$SYNSCAPES_INST_ROOT" --synscapes-splits "$SYNSCAPES_SPLITS")
    [[ -n "$IDD_INST_ROOT" ]] && iseg_args+=(--idd-root "$IDD_INST_ROOT" --idd-splits "$IDD_SPLITS")
    [[ -n "$WILDDASH2_INST_ROOT" ]] && iseg_args+=(--wilddash2-root "$WILDDASH2_INST_ROOT" --wilddash2-splits "$WILDDASH2_SPLITS")
    [[ -n "$NUIMAGES_INST_ROOT" ]] && iseg_args+=(--nuimages-root "$NUIMAGES_INST_ROOT" --nuimages-splits "$NUIMAGES_SPLITS" --nuimages-sample-interval-seconds "$NUIMAGES_SAMPLE_INTERVAL_SECONDS")
    [[ -n "$VOC_INST_ROOT" ]] && iseg_args+=(--voc-root "$VOC_INST_ROOT")
    [[ -n "$COCONUT_XL_INST_ROOT" ]] && iseg_args+=(--coconut-xlarge-root "$COCONUT_XL_INST_ROOT" --coconut-xlarge-groups "$COCONUT_XL_GROUPS")
    [[ -n "$PACO_LVIS_INST_ROOT" ]] && iseg_args+=(--paco-lvis-root "$PACO_LVIS_INST_ROOT" --paco-lvis-splits "$PACO_LVIS_SPLITS")
    [[ -n "$MUSES_INST_ROOT" && -d "$MUSES_INST_ROOT" ]] && iseg_args+=(--muses-root "$MUSES_INST_ROOT")
    [[ -n "$ACDC_INST_ROOT" && -d "$ACDC_INST_ROOT" ]] && iseg_args+=(--acdc-root "$ACDC_INST_ROOT")
    [[ -n "$GOOSE_INST_ROOT" && -d "$GOOSE_INST_ROOT" ]] && iseg_args+=(--goose-root "$GOOSE_INST_ROOT")
    [[ -n "$VIPSEG_INST_ROOT" && -d "$VIPSEG_INST_ROOT" ]] && iseg_args+=(--vipseg-root "$VIPSEG_INST_ROOT")
    [[ -n "$OVIS_INST_ROOT" && -d "$OVIS_INST_ROOT" ]] && iseg_args+=(--ovis-root "$OVIS_INST_ROOT" --ovis-fps "$OVIS_FPS" --ovis-source-fps "$OVIS_SOURCE_FPS")
    [[ -n "$WAYMO_INST_ROOT" && -d "$WAYMO_INST_ROOT" ]] && iseg_args+=(--waymo-root "$WAYMO_INST_ROOT")
    [[ -n "$HYPERSIM_SEG_INST_ROOT" && -d "$HYPERSIM_SEG_INST_ROOT" ]] && iseg_args+=(--hypersim-root "$HYPERSIM_SEG_INST_ROOT")
    [[ -n "$INSTANCESEG_LIMIT" ]] && iseg_args+=(--limit "$INSTANCESEG_LIMIT")
    [[ -n "$INSTANCESEG_MAX_INSTANCES" ]] && iseg_args+=(--max-instances "$INSTANCESEG_MAX_INSTANCES")
    [[ -n "$INSTANCESEG_MIN_INSTANCE_AREA" ]] && iseg_args+=(--min-instance-area "$INSTANCESEG_MIN_INSTANCE_AREA")
    [[ -n "$INSTANCESEG_LEGEND_SLACK_PROB" ]] && iseg_args+=(--legend-slack-prob "$INSTANCESEG_LEGEND_SLACK_PROB")
    [[ -n "$INSTANCESEG_DRAW_MODE" ]] && iseg_args+=(--draw-mode "$INSTANCESEG_DRAW_MODE")
    run_in_env "$PYTHON" "${iseg_args[@]}"
  else
    echo "Using existing instanceseg prepared dataset: $INSTANCESEG_OUT"
  fi
elif is_true "$INSTANCESEG_ENABLE"; then
  echo "instanceseg enabled but no source roots set (including SYNSCAPES_INST_ROOT, IDD_INST_ROOT, WILDDASH2_INST_ROOT, NUIMAGES_INST_ROOT, COCONUT_XL_INST_ROOT, or PACO_LVIS_INST_ROOT); skipping."
fi

if is_true "$REASONSEG_REFSEG_ENABLE"; then
  if [[ "$FORCE" == "1" ]] || ! has_nonempty_metadata "$REASONSEG_REFSEG_OUT/metadata.jsonl"; then
    reasonseg_args=(
      dataset_prep/refseg/prepare_reasonseg.py
      --reasonseg-root "$REASONSEG_REFSEG_ROOT"
      --out-root "$REASONSEG_REFSEG_OUT"
      --splits "$REASONSEG_REFSEG_SPLITS"
      --expressions-per-image "$REASONSEG_REFSEG_EXPRESSIONS_PER_IMAGE"
      --reasonseg-val-dir "$REASONSEG_REFSEG_VAL_DIR"
      --seed "$SEED"
    )
    [[ -n "$REASONSEG_REFSEG_EXTRACT_ROOT" ]] && reasonseg_args+=(--extract-root "$REASONSEG_REFSEG_EXTRACT_ROOT")
    is_true "$REASONSEG_REFSEG_KEEP_EXTRACTED" && reasonseg_args+=(--keep-extracted)
    is_true "$REASONSEG_REFSEG_ALL_EXPRESSIONS" || reasonseg_args+=(--no-all-expressions)
    is_true "$REASONSEG_REFSEG_EXCLUDE_VAL_OVERLAP" || reasonseg_args+=(--no-exclude-val-overlap)
    [[ -n "$REASONSEG_REFSEG_COLORS" ]] && reasonseg_args+=(--colors "$REASONSEG_REFSEG_COLORS")
    is_true "$REASONSEG_REFSEG_NO_RANDOM_COLORS" && reasonseg_args+=(--no-random-colors)
    run_in_env "$PYTHON" "${reasonseg_args[@]}"
  else
    echo "Using existing ReasonSeg prepared dataset: $REASONSEG_REFSEG_OUT"
  fi
fi

# --- Depth / surface-normal members (already-canonical dirs, no conversion) --
# Verify each opted-in member dir actually holds canonical data; drop empties so
# the specs never point at a hollow member.
check_member_dirs() {
  local task=$1 dirs=$2 kept=() d
  for d in $dirs; do
    if has_nonempty_metadata "$d/metadata.jsonl"; then
      kept+=("$d")
    else
      echo "WARNING: $task member dir has no metadata.jsonl, skipping: $d" >&2
    fi
  done
  echo "${kept[@]:-}"
}
if weight_positive "$DEPTH_WEIGHT"; then
  DEPTH_MEMBER_DIRS=$(check_member_dirs depth "$DEPTH_MEMBER_DIRS")
  [[ -z "$DEPTH_MEMBER_DIRS" ]] && echo "WARNING: DEPTH_WEIGHT set but no usable DEPTH_MEMBER_DIRS; depth is dropped from the mix." >&2
fi
if weight_positive "$NORMALS_WEIGHT"; then
  NORMALS_MEMBER_DIRS=$(check_member_dirs normals "$NORMALS_MEMBER_DIRS")
  [[ -z "$NORMALS_MEMBER_DIRS" ]] && echo "WARNING: NORMALS_WEIGHT set but no usable NORMALS_MEMBER_DIRS; normals are dropped from the mix." >&2
fi

# --- PD12M image-generation member (materialized text-to-image subset) -------
if weight_positive "$PD12M_WEIGHT"; then
  if [[ "$FORCE" == "1" ]] || ! has_nonempty_metadata "$PD12M_OUT/metadata.jsonl"; then
    if [[ ! -d "$PD12M_SHARDS_DIR" ]]; then
      echo "ERROR: PD12M_WEIGHT is set but PD12M_SHARDS_DIR not found: $PD12M_SHARDS_DIR" >&2
      exit 1
    fi
    echo "Materializing PD12M subset ($PD12M_LIMIT rows) into $PD12M_OUT"
    [[ "$FORCE" == "1" ]] && rm -rf "$PD12M_OUT"
    run_in_env "$PYTHON" scripts/dataset_prep/materialize_pd12m_canonical.py \
      --shards-dir "$PD12M_SHARDS_DIR" \
      --out "$PD12M_OUT" \
      --limit "$PD12M_LIMIT" \
      --require-count \
      --seed "$SEED" \
      --max-side "$PD12M_MAX_SIDE"
  else
    echo "Using existing materialized PD12M member: $PD12M_OUT"
  fi
fi

CONFIG_OUT="$CONFIG_OUT" \
PICO_ENABLE="$PICO_ENABLE" \
IDENTITY_ENABLE="$IDENTITY_ENABLE" \
REFSEG_ENABLE="$REFSEG_ENABLE" \
REFSEG_OUT="$REFSEG_OUT" \
REFSEG_WEIGHT="$REFSEG_WEIGHT" \
REASONSEG_REFSEG_ENABLE="$REASONSEG_REFSEG_ENABLE" \
REASONSEG_REFSEG_OUT="$REASONSEG_REFSEG_OUT" \
REASONSEG_REFSEG_WEIGHT="$REASONSEG_REFSEG_WEIGHT" \
INSTANCESEG_ENABLE="$INSTANCESEG_ENABLE" \
INSTANCESEG_OUT="$INSTANCESEG_OUT" \
INSTANCESEG_WEIGHT="$INSTANCESEG_WEIGHT" \
SAMPLES_PER_EPOCH="$SAMPLES_PER_EPOCH" \
SEED="$SEED" \
SEMSEG_OUT="$SEMSEG_OUT" \
PICO_OUT="$PICO_OUT" \
IDENTITY_OUT="$IDENTITY_OUT" \
SEMSEG_WEIGHT="$SEMSEG_WEIGHT" \
PICO_WEIGHT="$PICO_WEIGHT" \
IDENTITY_WEIGHT="$IDENTITY_WEIGHT" \
DEPTH_MEMBER_DIRS="$DEPTH_MEMBER_DIRS" \
DEPTH_WEIGHT="$DEPTH_WEIGHT" \
NORMALS_MEMBER_DIRS="$NORMALS_MEMBER_DIRS" \
NORMALS_WEIGHT="$NORMALS_WEIGHT" \
PD12M_OUT="$PD12M_OUT" \
PD12M_WEIGHT="$PD12M_WEIGHT" \
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

config = {
    "samples_per_epoch": int(os.environ["SAMPLES_PER_EPOCH"]),
    "seed": int(os.environ["SEED"]),
    "shuffle": True,
    "datasets": [
        {
            "name": "semseg",
            "path": os.environ["SEMSEG_OUT"],
            "weight": float(os.environ["SEMSEG_WEIGHT"]),
        },
    ],
}

def _enabled(var):
    return str(os.environ.get(var, "0")).lower() in ("1", "true", "yes", "y", "on")


def _has_rows(path):
    p = Path(path) / "metadata.jsonl"
    return p.exists() and p.stat().st_size > 0


if _enabled("PICO_ENABLE") and _has_rows(os.environ.get("PICO_OUT", "")):
    config["datasets"].append({
        "name": "pico_banana",
        "path": os.environ["PICO_OUT"],
        "weight": float(os.environ["PICO_WEIGHT"]),
    })

if _enabled("IDENTITY_ENABLE") and _has_rows(os.environ.get("IDENTITY_OUT", "")):
    config["datasets"].append({
        "name": "identity",
        "path": os.environ["IDENTITY_OUT"],
        "weight": float(os.environ["IDENTITY_WEIGHT"]),
    })

if _enabled("REFSEG_ENABLE"):
    config["datasets"].append({
        "name": "refseg",
        "path": os.environ["REFSEG_OUT"],
        "weight": float(os.environ["REFSEG_WEIGHT"]),
    })

if _enabled("REASONSEG_REFSEG_ENABLE") and _has_rows(os.environ.get("REASONSEG_REFSEG_OUT", "")):
    config["datasets"].append({
        "name": "refseg_reasonseg",
        "path": os.environ["REASONSEG_REFSEG_OUT"],
        "weight": float(os.environ["REASONSEG_REFSEG_WEIGHT"]),
    })

# instanceseg is appended only when it actually produced data (it is skipped when no
# source roots are staged), so the specs never point at an empty member.
if _enabled("INSTANCESEG_ENABLE") and _has_rows(os.environ.get("INSTANCESEG_OUT", "")):
    config["datasets"].append({
        "name": "instanceseg",
        "path": os.environ["INSTANCESEG_OUT"],
        "weight": float(os.environ["INSTANCESEG_WEIGHT"]),
    })

# Depth / surface-normal members: one member per source dir, the TASK weight
# split equally across them (member name = dir basename, e.g. hypersim_depth).
for task in ("DEPTH", "NORMALS"):
    weight = float(os.environ.get(f"{task}_WEIGHT", 0) or 0)
    dirs = [d for d in os.environ.get(f"{task}_MEMBER_DIRS", "").split() if _has_rows(d)]
    if weight > 0 and dirs:
        for d in dirs:
            config["datasets"].append({
                "name": Path(d).name,
                "path": d,
                "weight": weight / len(dirs),
            })

# PD12M image-generation member (text-to-image, no conditioning images).
pd12m_weight = float(os.environ.get("PD12M_WEIGHT", 0) or 0)
if pd12m_weight > 0 and _has_rows(os.environ.get("PD12M_OUT", "")):
    config["datasets"].append({
        "name": "pd12m",
        "path": os.environ["PD12M_OUT"],
        "weight": pd12m_weight,
    })

total_weight = sum(d["weight"] for d in config["datasets"])
if abs(total_weight - 1.0) > 1e-6:
    print(f"WARNING: mix weights sum to {total_weight:.4f}, not 1.0 — "
          "the trainer normalizes, but rebalance for clarity.")

out = Path(os.environ["CONFIG_OUT"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Wrote dataset specs: {out}")
PY

echo "Prepared dataset pool:"
echo "  semseg:   $SEMSEG_OUT"
is_true "$PICO_ENABLE" && echo "  pico:     $PICO_OUT"
is_true "$IDENTITY_ENABLE" && echo "  identity: $IDENTITY_OUT"
is_true "$REFSEG_ENABLE" && echo "  refseg:   $REFSEG_OUT"
is_true "$REASONSEG_REFSEG_ENABLE" && echo "  refseg_reasonseg: $REASONSEG_REFSEG_OUT (weight $REASONSEG_REFSEG_WEIGHT)"
{ is_true "$INSTANCESEG_ENABLE" && [[ -n "$ADE_INST_ROOT$MAPILLARY_INST_ROOT$SYNSCAPES_INST_ROOT$IDD_INST_ROOT$WILDDASH2_INST_ROOT$NUIMAGES_INST_ROOT$COCONUT_XL_INST_ROOT$PACO_LVIS_INST_ROOT$VOC_INST_ROOT$MUSES_INST_ROOT$ACDC_INST_ROOT$GOOSE_INST_ROOT$VIPSEG_INST_ROOT$OVIS_INST_ROOT$WAYMO_INST_ROOT$HYPERSIM_SEG_INST_ROOT" ]]; } && echo "  instanceseg: $INSTANCESEG_OUT"
{ weight_positive "$DEPTH_WEIGHT" && [[ -n "$DEPTH_MEMBER_DIRS" ]]; } && echo "  depth:    $DEPTH_MEMBER_DIRS (task weight $DEPTH_WEIGHT)"
{ weight_positive "$NORMALS_WEIGHT" && [[ -n "$NORMALS_MEMBER_DIRS" ]]; } && echo "  normals:  $NORMALS_MEMBER_DIRS (task weight $NORMALS_WEIGHT)"
weight_positive "$PD12M_WEIGHT" && echo "  pd12m:    $PD12M_OUT (weight $PD12M_WEIGHT)"
echo "  specs:    $CONFIG_OUT"

if [[ "$VALIDATE" == "1" ]]; then
  "$PYTHON" "$ROOT/scripts/dataset_prep/validate_dataset_specs.py" "$CONFIG_OUT"
fi

