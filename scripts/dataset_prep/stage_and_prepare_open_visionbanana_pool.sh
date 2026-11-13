#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

show_help() {
  cat <<'EOF'
Build the mixed open-visionbanana-style dataset pool: stage raw archives into
local scratch (for example $TMPDIR / node-local NVMe), extract them, and rebuild
the canonical prepared pool. This is a DATA-PREP step only; it does not train.

Use case:
  Scratch/TMPDIR + archive/inode-aware setup.

Use this when the raw data is kept as archives/tars (to respect home quota and
filesystem inode/per-directory limits) and must be extracted into local scratch
before use.

Usage:
  scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh

What the script does:
  1. Extracts semseg archives into a staging filesystem (node-local scratch or
     persistent fscratch).
  2. Copies the Pico-Banana checkout into local scratch.
  3. Extracts OpenImages source tarballs and Pico edited target images there.
  4. Rebuilds the canonical prepared dataset pool and dataset-specs JSON.

Important staging variables:
  ROOT                         Repo root. Defaults to the current checkout.
  TMPDIR                       Scratch parent directory.
  STAGING_ROOT                 Root of all staged files.
  DATA_ROOT                    Persistent data root used to derive defaults.
  SEMSEG_SOURCE_ROOT           Persistent semseg archive directory.
  PICO_SOURCE_ROOT             Persistent Pico-Banana checkout.
  CITYSCAPES_TAR / ACDC_TAR / BDD_TAR
  SYNSCAPES_SOURCE_ROOT        Extracted Synscapes tree (default shared_seg/synscapes).
  SYNSCAPES_TAR                Optional tar fallback when no extracted tree exists.
  SYNSCAPES_ENABLE             0 to exclude Synscapes (default: 1).
  IDD_SOURCE_ROOT              Optional extracted IDD parent (default shared_seg/idd).
  IDD_TAR                      IDD two-part outer archive (default shared_seg/idd_1_2.tar).
  IDD_ENABLE                   0 to exclude IDD (default: 1); both tasks use train,val.
  WILDDASH2_SOURCE_ROOT         Optional extracted WildDash 2 tree (default shared_seg/wilddash2).
  WILDDASH2_TAR                 WildDash 2 public panoptic archive (default shared_seg/wd_public_v2p.tar).
  WILDDASH2_ENABLE              0 to exclude WildDash 2 (default: 1); both tasks use train,val.
  NUIMAGES_SOURCE_ROOT          Optional extracted NuImages root (default shared_seg/nuimages).
  NUIMAGES_TAR                  NuImages nested release archive (default shared_seg/nuimages.tar).
  NUIMAGES_ENABLE               0 to exclude NuImages (default: 1); both tasks use train,val.
  NUIMAGES_SAMPLE_INTERVAL_SECONDS
                               Per-log mask-bearing keyframe interval (default: 10 seconds).
  COCONUT_XL_SOURCE_ROOT        Materialized COCONut-XL coco/ tree (default shared_seg/coconut_xlarge/coco).
  COCONUT_XL_ENABLE             0 to exclude COCONut-XL (default: 1); it is used in place by both tasks.
  PACO_LVIS_SOURCE_ROOT        Optional extracted PACO-LVIS root (default instanceseg/paco_lvis).
  PACO_LVIS_TAR                PACO-LVIS DatasetNinja archive (default instanceseg/paco-lvis-DatasetNinja.tar).
  PACO_LVIS_ENABLE             0 to exclude PACO-LVIS (default: 1); train,val,test by default.
  PACO_LVIS_SPLITS             PACO-LVIS splits to stage/use (default: train,val,test).
  REFCOCOG_UMD_ROOT             Official RefCOCOg UMD annotations; required to exclude its val images from COCONut-XL/PACO-LVIS.
  MUSES_TAR                    Optional MUSES archive built by download_datasets_muses.sh.
  GOOSE_TAR                    Optional GOOSE 2D archive built by download_datasets_goose.sh.
  VIPSEG_TAR                   Optional VIPSeg archive built by download_datasets_vipseg.sh.
  OVIS_TAR                     Optional OVIS archive built by download_datasets_ovis.sh.
  FOODSEG103_TAR               Optional FoodSeg103 archive built by download_datasets_foodseg103.sh.
  TRANS10K_TAR                 Optional Trans10K-v2 archive built by download_datasets_trans10k.sh.
  RAILSEM19_TAR                Optional RailSem19 archive in its native release layout
                               (semseg only; the release ships no instance ids).
  WAYMO_TAR                    Optional Waymo camera-panoptic archive built by download_datasets_waymo.sh
                               (feeds both the semseg and instanceseg members).
  HYPERSIM_SEG_TAR             Optional Hypersim NYU40 archive built by
                               dataset_prep/hypersim/prepare_hypersim_segmentation.py --prepared-tar
                               (feeds both the semseg and instanceseg members). This is the
                               MATERIALIZED tree, not the raw HDF5 release.
  MAPILLARY_SEMSEG_TAR         Mapillary Vistas archive used for semseg (defaults to MAPILLARY_INST_TAR).
  MAPILLARY_SEMSEG_ENABLE      0 to skip staging Mapillary v2.0 labels for semseg.
  INSTANCESEG_V2_BASELINE_ONLY 1 to stage only the nine instance sources used by
                               the v2 baseline (for controlled prompt/mask rebuilds).
  PICO_ENABLE                  0 to skip Pico/OpenImages staging when its prepared
                               member already exists.
  REFSEG_ENABLE                0 to skip referring-seg source staging when its
                               prepared member already exists.
  PICO_POSITIVE_EDIT_TAR       Optional tar containing edited Pico targets.
  PICO_POSITIVE_EDIT_TAR_FALLBACK
  GEOMETRY_SOURCE_ROOT         Dir with prepared depth/normals member tars
                               (default datasets_archives/depth_surfacen/prepared/training).
  DEPTH_TARS / NORMALS_TARS    Explicit tar lists (default: auto-discovered).
  DEPTH_WEIGHT / NORMALS_WEIGHT
                               Task mix weights; 0 (default) skips the task,
                               > 0 extracts the tars into PREPARED_ROOT and
                               splits the weight equally across sources.
  PD12M_WEIGHT                 > 0 materializes a PD12M text-to-image member
                               (PD12M_SHARDS_DIR, PD12M_LIMIT rows).
  FORCE                        1 to rebuild staged extracts.
  ARCHIVE_SEGMENTATION_MEMBERS  1 to publish semseg.tar + instanceseg.tar with
                               indexed sidecars after preparation (default: 0).
  CLEAN_STAGING_AFTER_SUCCESS   1 to remove STAGING_ROOT after successful
                               archive publication (default: 0).

Important output variables:
  PREPARED_ROOT                Prepared-dataset directory (the pool).
  CONFIG_OUT                   Dataset-specs JSON documenting the mix.
  SEMSEG_OUT / PICO_OUT / IDENTITY_OUT
                               Per-subset output directories.

Composition knobs (weights, limits, semseg/refseg/instanceseg options) all
live in prepare_open_visionbanana_pool.sh with overridable defaults; every
exported env var is forwarded to it.

Typical example:
  STAGING_ROOT=/tmp/$USER/open_visionbanana_data \
  PREPARED_ROOT=/path/to/persistent/open_visionbanana_prepared \
  scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh

EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_path() {
  local path=$1
  local label=$2
  if [[ ! -e "$path" ]]; then
    echo "ERROR: $label not found: $path" >&2
    exit 1
  fi
}

dir_has_files() {
  local dir=$1
  [[ -d "$dir" ]] && find "$dir" -type f -print -quit | grep -q .
}

require_nonempty_dir() {
  local dir=$1
  local label=$2
  if ! dir_has_files "$dir"; then
    echo "ERROR: $label is missing or empty: $dir" >&2
    exit 1
  fi
}

ROOT=${ROOT:-$DEFAULT_ROOT}
[ -f "$ROOT/.venv/bin/activate" ] && source "$ROOT/.venv/bin/activate"
PYTHON=${PYTHON:-python}

DATA_ROOT=${DATA_ROOT:-$ROOT/datasets_archives}
REFCOCOG_UMD_ROOT=${REFCOCOG_UMD_ROOT:-$DATA_ROOT/refseg/refcocog}
SEMSEG_SOURCE_ROOT=${SEMSEG_SOURCE_ROOT:-$DATA_ROOT/sem_segm}
# Pico checkout: matches the default written by scripts/dataset_prep/data_download_scripts/
# pico_banana_400k.sh / download_pico_positive_edit.sh.
PICO_SOURCE_ROOT=${PICO_SOURCE_ROOT:-$DATA_ROOT/pico_banana/pico-banana-400k}
PICO_SOURCE_OPENIMAGES_DIR=${PICO_SOURCE_OPENIMAGES_DIR:-$PICO_SOURCE_ROOT/openimages}

TMPDIR=${TMPDIR:-/tmp/${USER:-$(id -un)}/open_visionbanana}
STAGING_ROOT=${STAGING_ROOT:-$TMPDIR/open_visionbanana_data}
SEMSEG_STAGE=${SEMSEG_STAGE:-$STAGING_ROOT/sem_segm}
PICO_STAGE_ROOT=${PICO_STAGE_ROOT:-$STAGING_ROOT/pico-banana-400k}
PICO_STAGE_OPENIMAGES_DIR=${PICO_STAGE_OPENIMAGES_DIR:-$PICO_STAGE_ROOT/openimages}
REFSEG_STAGE=${REFSEG_STAGE:-$STAGING_ROOT/refseg_sources}
INSTANCESEG_STAGE=${INSTANCESEG_STAGE:-$STAGING_ROOT/instanceseg}
PREPARED_ROOT=${PREPARED_ROOT:-$STAGING_ROOT/open_visionbanana_prepared}

CITYSCAPES_TAR=${CITYSCAPES_TAR:-$SEMSEG_SOURCE_ROOT/cityscapes.tar}
ACDC_TAR=${ACDC_TAR:-$SEMSEG_SOURCE_ROOT/acdc.tar}
BDD_TAR=${BDD_TAR:-$SEMSEG_SOURCE_ROOT/bdd_10k.tar}
# ADE20k + Pascal-VOC are optional training sources: staged only if their tar exists
# (built by scripts/dataset_prep/data_download_scripts/download_datasets_semseg_ade_voc.sh).
ADE20K_TAR=${ADE20K_TAR:-$SEMSEG_SOURCE_ROOT/ade20k.tar}
VOC_TAR=${VOC_TAR:-$SEMSEG_SOURCE_ROOT/voc.tar}
# MUSES contains frame_camera + gt_semantic for semseg and gt_panoptic for
# instance segmentation (see scripts/dataset_prep/data_download_scripts/download_datasets_muses.sh).
MUSES_TAR=${MUSES_TAR:-$SEMSEG_SOURCE_ROOT/muses.tar}
# GOOSE contains images + labels for both semantic and instance segmentation.
GOOSE_TAR=${GOOSE_TAR:-$SEMSEG_SOURCE_ROOT/goose.tar}
# VIPSeg contains imgs + panomasks (+ VIPSeg_720P categories JSON).
VIPSEG_TAR=${VIPSEG_TAR:-$SEMSEG_SOURCE_ROOT/vipseg.tar}
# OVIS contains video frames + YouTube-VOS/COCO-video annotations for instance segmentation.
OVIS_TAR=${OVIS_TAR:-$SEMSEG_SOURCE_ROOT/ovis.tar}
# FoodSeg103 images/labels materialized from the Hugging Face parquet dataset.
FOODSEG103_TAR=${FOODSEG103_TAR:-$SEMSEG_SOURCE_ROOT/foodseg103.tar}
# Trans10K-v2 transparent-object images/labels, likewise materialized from parquet.
TRANS10K_TAR=${TRANS10K_TAR:-$SEMSEG_SOURCE_ROOT/trans10k.tar}
# RailSem19 rail-scene semantic segmentation, packed in its native release layout
# (railsem19/versions/<n>/{jpgs,uint8,jsons}/rs19_val/). The semseg reader resolves
# the jpgs/+uint8/ parent itself, so the nesting depth does not matter here.
RAILSEM19_TAR=${RAILSEM19_TAR:-$SEMSEG_SOURCE_ROOT/railsem19.tar}
# Waymo camera panoptic: one tree serving both semseg (semantic half of the packed
# labels) and instanceseg (instance half), so it stages once under SEMSEG_STAGE.
WAYMO_TAR=${WAYMO_TAR:-$SEMSEG_SOURCE_ROOT/waymo.tar}

# Instance-segmentation sources: ADE20K-instance + Mapillary Vistas v2.0 + Pascal
# VOC 2012 (each staged onto node-local scratch only if its tar exists). ADE's class
# name table (instanceInfo100_train.txt) sits beside its tar and is copied next to
# the extract so the converter's --ade-root has it. INSTANCESEG_ENABLE gates all of it.
INSTANCESEG_ENABLE=${INSTANCESEG_ENABLE:-1}
# When enabled, staging extracts only the nine sources of the baseline
# instance-seg member (ADE-instance, Mapillary-instance, VOC-instance, MUSES,
# ACDC, GOOSE, VIPSeg, OVIS, Waymo), skipping semseg-only sources and Hypersim.
INSTANCESEG_V2_BASELINE_ONLY=${INSTANCESEG_V2_BASELINE_ONLY:-0}
INSTANCESEG_SOURCE_ROOT=${INSTANCESEG_SOURCE_ROOT:-$DATA_ROOT/instanceseg}
ADE_INST_TAR=${ADE_INST_TAR:-$INSTANCESEG_SOURCE_ROOT/ade20k_instanceseg/ade20k_instanceseg.tar}
ADE_INST_INFO_DIR=${ADE_INST_INFO_DIR:-$INSTANCESEG_SOURCE_ROOT/ade20k_instanceseg}
# Mapillary Vistas lives in shared_seg/: one archive shared by the semseg,
# instance-seg and refseg members (hence the folder name). It is the official
# download -- a ZIP despite the .tar name, with entries at the archive root
# (config_v2.0.json, training/..., validation/...) and no mapillary_vistasv2/
# prefix; extract_mapillary_subset below normalizes both facts.
SHARED_SEG_ROOT=${SHARED_SEG_ROOT:-$DATA_ROOT/shared_seg}
SYNSCAPES_ENABLE=${SYNSCAPES_ENABLE:-1}
SYNSCAPES_SOURCE_ROOT=${SYNSCAPES_SOURCE_ROOT:-$SHARED_SEG_ROOT/synscapes}
SYNSCAPES_TAR=${SYNSCAPES_TAR:-$SHARED_SEG_ROOT/synscapes.tar}
# Include both logical splits in both pool members by default.
SYNSCAPES_SPLITS=${SYNSCAPES_SPLITS:-train,val}
# The IDD outer archive holds the two release parts used by semantic and instance prep.
IDD_ENABLE=${IDD_ENABLE:-1}
IDD_SOURCE_ROOT=${IDD_SOURCE_ROOT:-$SHARED_SEG_ROOT/idd}
IDD_TAR=${IDD_TAR:-$SHARED_SEG_ROOT/idd_1_2.tar}
IDD_SPLITS=${IDD_SPLITS:-train,val}
# WildDash 2 is one flat panoptic archive shared by semantic and instance prep.
WILDDASH2_ENABLE=${WILDDASH2_ENABLE:-1}
WILDDASH2_SOURCE_ROOT=${WILDDASH2_SOURCE_ROOT:-$SHARED_SEG_ROOT/wilddash2}
WILDDASH2_TAR=${WILDDASH2_TAR:-$SHARED_SEG_ROOT/wd_public_v2p.tar}
WILDDASH2_SPLITS=${WILDDASH2_SPLITS:-train,val}
# NuImages has nested metadata/samples tarballs. Staging extracts only the
# labelled key-camera JPEGs in samples/, never the unlabelled sweeps.
NUIMAGES_ENABLE=${NUIMAGES_ENABLE:-1}
NUIMAGES_SOURCE_ROOT=${NUIMAGES_SOURCE_ROOT:-$SHARED_SEG_ROOT/nuimages}
NUIMAGES_TAR=${NUIMAGES_TAR:-$SHARED_SEG_ROOT/nuimages.tar}
NUIMAGES_SPLITS=${NUIMAGES_SPLITS:-train,val}
NUIMAGES_SAMPLE_INTERVAL_SECONDS=${NUIMAGES_SAMPLE_INTERVAL_SECONDS:-10}
# COCONut-XL is already materialized (roughly 587k panoptic pairs), so it is
# validated and used in place rather than copied to node-local scratch.
COCONUT_XL_ENABLE=${COCONUT_XL_ENABLE:-1}
COCONUT_XL_SOURCE_ROOT=${COCONUT_XL_SOURCE_ROOT:-$SHARED_SEG_ROOT/coconut_xlarge/coco}
COCONUT_XL_ROOT=
COCONUT_XL_GROUPS=${COCONUT_XL_GROUPS:-coco,objects365}
# PACO-LVIS is an instance-only DatasetNinja/Supervisely export. All three
# released splits have labels, so train, val, and test are included by default.
PACO_LVIS_ENABLE=${PACO_LVIS_ENABLE:-1}
PACO_LVIS_SOURCE_ROOT=${PACO_LVIS_SOURCE_ROOT:-$INSTANCESEG_SOURCE_ROOT/paco_lvis}
PACO_LVIS_TAR=${PACO_LVIS_TAR:-$INSTANCESEG_SOURCE_ROOT/paco-lvis-DatasetNinja.tar}
PACO_LVIS_SPLITS=${PACO_LVIS_SPLITS:-train,val,test}
PACO_LVIS_ROOT=${PACO_LVIS_ROOT:-}
MAPILLARY_TAR=${MAPILLARY_TAR:-$SHARED_SEG_ROOT/mapillary.tar}
MAPILLARY_INST_TAR=${MAPILLARY_INST_TAR:-$MAPILLARY_TAR}
# Hypersim sits in shared_seg/ for the same reason: one tree feeds both the semseg
# and instance-seg members (packed panoptic, exactly like Waymo).
#
# Two different archives live here, and only one of them is stageable:
#   hypersim.tar      the raw release (per-frame HDF5, HDR colour); no converter reads it.
#   hypersim_seg.tar  the materialized tree, built once by
#                     dataset_prep/hypersim/prepare_hypersim_segmentation.py --prepared-tar.
# Staging does not run that conversion; assert_hypersim_seg_materialized below
# fails if HYPERSIM_SEG_TAR points at the raw archive.
HYPERSIM_RAW_TAR=${HYPERSIM_RAW_TAR:-$SHARED_SEG_ROOT/hypersim.tar}
HYPERSIM_SEG_TAR=${HYPERSIM_SEG_TAR:-$SHARED_SEG_ROOT/hypersim_seg.tar}
VOC_INST_TAR=${VOC_INST_TAR:-$INSTANCESEG_SOURCE_ROOT/VOC2012.tar}
# The same Mapillary tar also feeds semantic segmentation (v2.0/labels/). Labels are
# staged into the instance-seg tree when that exists so images/config are shared.
MAPILLARY_SEMSEG_TAR=${MAPILLARY_SEMSEG_TAR:-$MAPILLARY_INST_TAR}
MAPILLARY_SEMSEG_ENABLE=${MAPILLARY_SEMSEG_ENABLE:-1}
# Cityscapes is excluded from the training pool by default (ACDC+BDD cover the
# same 19-class driving taxonomy). Set INCLUDE_CITYSCAPES=1 to add it back.
INCLUDE_CITYSCAPES=${INCLUDE_CITYSCAPES:-0}
# When it is included, use training images only so Cityscapes val stays clean
# for semantic-segmentation evaluation.
CITYSCAPES_SPLITS=${CITYSCAPES_SPLITS:-train}

# Depth / surface-normal member tars (already canonical: metadata.jsonl +
# conditioning_images/ + target_images/ under one top-level dir), extracted
# straight into PREPARED_ROOT whenever tars are present (presence-based, like the
# optional semseg tars). The prepare script's DEPTH_WEIGHT/NORMALS_WEIGHT decide
# mix inclusion; all weight/PD12M knobs default there.
GEOMETRY_SOURCE_ROOT=${GEOMETRY_SOURCE_ROOT:-$DATA_ROOT/depth_surfacen/prepared/training}
# Space-separated tar lists; default = auto-discover in GEOMETRY_SOURCE_ROOT.
DEPTH_TARS=${DEPTH_TARS:-}
NORMALS_TARS=${NORMALS_TARS:-}

PICO_POSITIVE_EDIT_TAR=${PICO_POSITIVE_EDIT_TAR:-$PICO_SOURCE_OPENIMAGES_DIR/images/positive-edit.tar}
PICO_POSITIVE_EDIT_TAR_FALLBACK=${PICO_POSITIVE_EDIT_TAR_FALLBACK:-$PICO_SOURCE_OPENIMAGES_DIR/positive-edit.tar}
# Resolve the downloader from either supported location.
if [[ -z "${PICO_POSITIVE_EDIT_DOWNLOADER:-}" ]]; then
  for _candidate in \
    "$ROOT/scripts/dataset_downloads/download_pico_positive_edit.sh" \
    "$ROOT/scripts/dataset_prep/data_download_scripts/download_pico_positive_edit.sh"; do
    [[ -f "$_candidate" ]] && { PICO_POSITIVE_EDIT_DOWNLOADER=$_candidate; break; }
  done
fi
PICO_POSITIVE_EDIT_DOWNLOADER=${PICO_POSITIVE_EDIT_DOWNLOADER:-$ROOT/scripts/dataset_downloads/download_pico_positive_edit.sh}
PICO_TARGET_DOWNLOAD_JOBS=${PICO_TARGET_DOWNLOAD_JOBS:-16}

# Referring-seg source images (SteerViT carries masks + metadata from the HF hub,
# but NOT the source RGB). Like semseg/pico, the tars are extracted onto node-local
# scratch and the converter reads them there; REFSEG_EXTRA_ARGS is set to the
# node-local roots. Skipped if REFSEG_ENABLE=0 or if REFSEG_EXTRA_ARGS already
# supplies image roots (caller is managing them).
REFSEG_ENABLE=${REFSEG_ENABLE:-1}
# Pico staging (checkout + OpenImages splits + target downloads + mapped metadata)
# is the most expensive non-semseg step. PICO_ENABLE=0 skips its staging and
# preparation entirely, which is useful for task-specific pools.
PICO_ENABLE=${PICO_ENABLE:-1}
IDENTITY_ENABLE=${IDENTITY_ENABLE:-$PICO_ENABLE}
REFSEG_SOURCE_ROOT=${REFSEG_SOURCE_ROOT:-$DATA_ROOT/refseg}
REFSEG_VISUAL_GENOME_TAR=${REFSEG_VISUAL_GENOME_TAR:-$REFSEG_SOURCE_ROOT/visual_genome.tar}
REFSEG_MAPILLARY_TAR=${REFSEG_MAPILLARY_TAR:-$MAPILLARY_TAR}
REFSEG_EXTRA_ARGS=${REFSEG_EXTRA_ARGS:-}

SEMSEG_NUM_WORKERS=${SEMSEG_NUM_WORKERS:-16}
PICO_NUM_WORKERS=${PICO_NUM_WORKERS:-16}
PICO_MAX_SIDE=${PICO_MAX_SIDE:-0}
PICO_COPY_MODE=${PICO_COPY_MODE:-hardlink}
IDENTITY_MAX_SIDE=${IDENTITY_MAX_SIDE:-0}
IDENTITY_COPY_MODE=${IDENTITY_COPY_MODE:-hardlink}

CONFIG_OUT=${CONFIG_OUT:-$PREPARED_ROOT/open_visionbanana_mix_v1_tmpdir.json}
SEMSEG_OUT=${SEMSEG_OUT:-}
PICO_OUT=${PICO_OUT:-}
IDENTITY_OUT=${IDENTITY_OUT:-}

SKIP_PICO_TARGET_DOWNLOAD=${SKIP_PICO_TARGET_DOWNLOAD:-0}
FORCE=${FORCE:-0}
ARCHIVE_SEGMENTATION_MEMBERS=${ARCHIVE_SEGMENTATION_MEMBERS:-0}
CLEAN_STAGING_AFTER_SUCCESS=${CLEAN_STAGING_AFTER_SUCCESS:-0}

run_in_workdir() {
  local workdir=$1
  shift
  (cd "$workdir" && "$@")
}

has_any_dir() {
  local parent=$1
  shift
  local candidate
  for candidate in "$@"; do
    if [[ -d "$parent/$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

extract_tar_if_needed() {
  local archive=$1
  local out_dir=$2
  shift 2
  local marker archive_name candidate
  # An absent archive means the source is excluded from this run (already
  # published, disabled, or not staged): its rows are skipped downstream.
  if [[ ! -f "$archive" ]]; then
    echo "Skipping absent archive: $archive"
    return 0
  fi
  require_path "$archive" "archive"
  mkdir -p "$out_dir"
  archive_name=$(basename -- "$archive")
  marker="$out_dir/.ovb-extracted-${archive_name}.complete"
  if is_true "$FORCE"; then
    for candidate in "$@"; do
      rm -rf "$out_dir/$candidate"
    done
    rm -f "$marker"
  fi
  if [[ -f "$marker" ]] && has_any_dir "$out_dir" "$@"; then
    echo "Using existing extracted data under: $out_dir"
    return
  fi
  # Directory existence alone is not a completion signal: a previous job may
  # have died during extraction. Recreate only this archive's known top-level
  # outputs before retrying it.
  if has_any_dir "$out_dir" "$@"; then
    echo "Discarding incomplete extraction of $archive from $out_dir"
    for candidate in "$@"; do
      rm -rf "$out_dir/$candidate"
    done
  fi
  rm -f "$marker"
  echo "Extracting $archive into $out_dir"
  tar -xf "$archive" -C "$out_dir"
  if ! has_any_dir "$out_dir" "$@"; then
    echo "ERROR: extraction of $archive did not create expected directories: $*" >&2
    exit 1
  fi
  touch "$marker"
}

normalize_synscapes_root() {
  local root=$1 candidate
  for candidate in "$root" "$root/Synscapes" "$root/synscapes"; do
    if [[ -d "$candidate/img/rgb" && -d "$candidate/img/class" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

stage_synscapes_if_needed() {
  SYNSCAPES_ROOT=
  is_true "$SYNSCAPES_ENABLE" || {
    echo "SYNSCAPES_ENABLE off; skipping Synscapes."
    return
  }
  is_true "$INSTANCESEG_V2_BASELINE_ONLY" && {
    echo "Fixed instance baseline profile; skipping Synscapes."
    return
  }

  local resolved
  if resolved=$(normalize_synscapes_root "$SYNSCAPES_SOURCE_ROOT"); then
    SYNSCAPES_ROOT=$resolved
    echo "Using extracted Synscapes tree in place: $SYNSCAPES_ROOT"
    return
  fi
  if [[ ! -f "$SYNSCAPES_TAR" ]]; then
    echo "WARNING: Synscapes tree/tar not found; Synscapes rows will be skipped." >&2
    return
  fi
  extract_tar_if_needed "$SYNSCAPES_TAR" "$SEMSEG_STAGE" Synscapes synscapes
  if ! resolved=$(normalize_synscapes_root "$SEMSEG_STAGE"); then
    echo "ERROR: staged Synscapes archive has no img/{rgb,class} tree." >&2
    exit 1
  fi
  SYNSCAPES_ROOT=$resolved
}

normalize_idd_root() {
  local root=$1
  # A parent containing either/both extracted release parts is the preferred
  # converter input. A direct source root is also useful for a one-part setup.
  if [[ -d "$root/IDD_Segmentation/gtFine" || -d "$root/idd20kII/gtFine" ||
        -d "$root/gtFine" ]]; then
    printf '%s\n' "$root"
    return 0
  fi
  return 1
}

stage_idd_if_needed() {
  IDD_ROOT=
  is_true "$IDD_ENABLE" || {
    echo "IDD_ENABLE off; skipping IDD."
    return
  }
  is_true "$INSTANCESEG_V2_BASELINE_ONLY" && {
    echo "Fixed instance baseline profile; skipping IDD."
    return
  }

  local resolved stage_root member
  if resolved=$(normalize_idd_root "$IDD_SOURCE_ROOT"); then
    IDD_ROOT=$resolved
    echo "Using extracted IDD tree in place: $IDD_ROOT"
    return
  fi
  if [[ ! -f "$IDD_TAR" ]]; then
    echo "WARNING: IDD tree/tar not found; IDD rows will be skipped." >&2
    return
  fi

  stage_root="$SEMSEG_STAGE/idd"
  if is_true "$FORCE"; then
    rm -rf "$stage_root"
  fi
  if [[ ! -d "$stage_root/IDD_Segmentation" || ! -d "$stage_root/idd20kII" ]]; then
    mkdir -p "$stage_root"
    for member in idd-segmentation.tar.gz idd-20k-II.tar.gz; do
      if ! tar -tf "$IDD_TAR" -- "$member" >/dev/null; then
        echo "ERROR: $IDD_TAR does not contain expected member: $member" >&2
        exit 1
      fi
      echo "Extracting IDD member $member into $stage_root"
      tar -xOf "$IDD_TAR" "$member" | tar -xzf - -C "$stage_root"
    done
  else
    echo "Using existing extracted IDD data under: $stage_root"
  fi
  if ! resolved=$(normalize_idd_root "$stage_root"); then
    echo "ERROR: staged IDD archive has no gtFine tree." >&2
    exit 1
  fi
  IDD_ROOT=$resolved
}


normalize_wilddash2_root() {
  local root=$1 candidate
  for candidate in "$root" "$root/wilddash2" "$root/wd_public_v2p"; do
    if [[ -d "$candidate/anonymized" && -d "$candidate/panoptic" &&
          -f "$candidate/panoptic.json" && -f "$candidate/wilddash2-meta.json" &&
          -d "$candidate/random_split" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}


stage_wilddash2_if_needed() {
  WILDDASH2_ROOT=
  is_true "$WILDDASH2_ENABLE" || {
    echo "WILDDASH2_ENABLE off; skipping WildDash 2."
    return
  }
  is_true "$INSTANCESEG_V2_BASELINE_ONLY" && {
    echo "Fixed instance baseline profile; skipping WildDash 2."
    return
  }

  local resolved stage_root
  if resolved=$(normalize_wilddash2_root "$WILDDASH2_SOURCE_ROOT"); then
    WILDDASH2_ROOT=$resolved
    echo "Using extracted WildDash 2 tree in place: $WILDDASH2_ROOT"
    return
  fi
  if [[ ! -f "$WILDDASH2_TAR" ]]; then
    echo "WARNING: WildDash 2 tree/tar not found; WildDash 2 rows will be skipped." >&2
    return
  fi

  stage_root="$SEMSEG_STAGE/wilddash2"
  if is_true "$FORCE"; then
    rm -rf "$stage_root"
  fi
  if ! resolved=$(normalize_wilddash2_root "$stage_root"); then
    mkdir -p "$stage_root"
    echo "Extracting WildDash 2 archive into $stage_root"
    tar -xf "$WILDDASH2_TAR" -C "$stage_root"
    if ! resolved=$(normalize_wilddash2_root "$stage_root"); then
      echo "ERROR: staged WildDash 2 archive lacks anonymized/, panoptic/, or metadata." >&2
      exit 1
    fi
  else
    echo "Using existing extracted WildDash 2 data under: $resolved"
  fi
  WILDDASH2_ROOT=$resolved
}


normalize_nuimages_root() {
  local root=$1 candidate
  for candidate in "$root" "$root/nuimages"; do
    if [[ -d "$candidate/samples" && -f "$candidate/v1.0-train/category.json" &&
          -f "$candidate/v1.0-train/sample.json" && -f "$candidate/v1.0-train/sample_data.json" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}


stage_nuimages_if_needed() {
  NUIMAGES_ROOT=
  is_true "$NUIMAGES_ENABLE" || {
    echo "NUIMAGES_ENABLE off; skipping NuImages."
    return
  }
  is_true "$INSTANCESEG_V2_BASELINE_ONLY" && {
    echo "Fixed instance baseline profile; skipping NuImages."
    return
  }

  local resolved stage_root member
  if resolved=$(normalize_nuimages_root "$NUIMAGES_SOURCE_ROOT"); then
    NUIMAGES_ROOT=$resolved
    echo "Using extracted NuImages tree in place: $NUIMAGES_ROOT"
    return
  fi
  if [[ ! -f "$NUIMAGES_TAR" ]]; then
    echo "WARNING: NuImages tree/tar not found; NuImages rows will be skipped." >&2
    return
  fi

  stage_root="$SEMSEG_STAGE/nuimages"
  if is_true "$FORCE"; then
    rm -rf "$stage_root"
  fi
  if ! resolved=$(normalize_nuimages_root "$stage_root"); then
    mkdir -p "$stage_root"
    # The outer archive contains two nested tgz members. The samples member
    # contains only annotated key-camera JPEGs; deliberately do not extract
    # the unannotated temporal sweeps package.
    for member in nuimages-v1.0-all-metadata.tgz nuimages-v1.0-all-samples.tgz; do
      if ! tar -tf "$NUIMAGES_TAR" -- "$member" >/dev/null; then
        echo "ERROR: $NUIMAGES_TAR does not contain expected member: $member" >&2
        exit 1
      fi
      echo "Extracting NuImages member $member into $stage_root"
      tar -xOf "$NUIMAGES_TAR" "$member" | tar -xzf - -C "$stage_root"
    done
    if ! resolved=$(normalize_nuimages_root "$stage_root"); then
      echo "ERROR: staged NuImages archive lacks samples/ or v1.0-train metadata." >&2
      exit 1
    fi
  else
    echo "Using existing extracted NuImages data under: $resolved"
  fi
  NUIMAGES_ROOT=$resolved
}


# A complete COCONut-XL root carries the expensive panoptic PNGs. Its compact
# source form instead keeps those masks in panoptic_train2017.tar beside the
# RGB images and manifest. The latter is staged below when needed.
resolve_coconut_xlarge_assets_root() {
  local root=$1 candidate
  for candidate in "$root" "$root/coco" "$root/coconut_xlarge/coco"; do
    if [[ -d "$candidate/train2017" &&
          -f "$candidate/annotations/panoptic_train2017.json" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

normalize_coconut_xlarge_root() {
  local root=$1 candidate
  for candidate in "$root" "$root/coco" "$root/coconut_xlarge/coco"; do
    if [[ -d "$candidate/train2017" && -d "$candidate/panoptic_train2017" &&
          -f "$candidate/annotations/panoptic_train2017.json" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

stage_coconut_xlarge_if_needed() {
  COCONUT_XL_ROOT=
  is_true "$COCONUT_XL_ENABLE" || {
    echo "COCONUT_XL_ENABLE off; skipping COCONut-XL."
    return
  }
  is_true "$INSTANCESEG_V2_BASELINE_ONLY" && {
    echo "Fixed instance baseline profile; skipping COCONut-XL."
    return
  }

  local resolved assets archive stage_base staged_root marker
  if resolved=$(normalize_coconut_xlarge_root "$COCONUT_XL_SOURCE_ROOT"); then
    COCONUT_XL_ROOT=$resolved
    echo "Using materialized COCONut-XL tree in place: $COCONUT_XL_ROOT"
    return
  fi
  if ! assets=$(resolve_coconut_xlarge_assets_root "$COCONUT_XL_SOURCE_ROOT"); then
    echo "WARNING: COCONut-XL images/manifest not found; rows will be skipped. Expected $COCONUT_XL_SOURCE_ROOT" >&2
    return
  fi
  archive="$assets/panoptic_train2017.tar"
  if [[ ! -f "$archive" ]]; then
    echo "WARNING: COCONut-XL panoptic masks are absent and no staging TAR exists: $archive" >&2
    return
  fi

  # Preserve the persistent RGB/JSON assets and stage only the 587k mask PNGs.
  # The enclosing STAGING_ROOT is removed after both converters finish.
  stage_base="$SEMSEG_STAGE/coconut_xlarge"
  staged_root="$stage_base/coco"
  marker="$stage_base/.ovb-extracted-panoptic_train2017.tar.complete"
  if is_true "$FORCE"; then
    rm -rf "$stage_base"
  fi
  mkdir -p "$staged_root"
  ln -sfn "$assets/train2017" "$staged_root/train2017"
  ln -sfn "$assets/annotations" "$staged_root/annotations"
  if [[ ! -f "$marker" || ! -d "$staged_root/panoptic_train2017" ]]; then
    rm -rf "$staged_root/panoptic_train2017"
    rm -f "$marker"
    echo "Extracting temporary COCONut-XL panoptic masks into $staged_root"
    tar -xf "$archive" -C "$staged_root"
    [[ -d "$staged_root/panoptic_train2017" ]] || {
      echo "ERROR: $archive did not create panoptic_train2017/." >&2
      exit 1
    }
    touch "$marker"
  else
    echo "Using staged COCONut-XL panoptic masks: $staged_root/panoptic_train2017"
  fi
  COCONUT_XL_ROOT=$staged_root
}

# PACO-LVIS is a flat DatasetNinja/Supervisely export. Unlike the other shared
# segmentation archives it is instance-only, so stage only its requested split
# directories under INSTANCESEG_STAGE.
normalize_paco_lvis_root() {
  local root=$1 candidate
  for candidate in "$root" "$root/paco_lvis" "$root/paco-lvis-DatasetNinja"; do
    if [[ -f "$candidate/meta.json" ]] &&
       { [[ -d "$candidate/train2017/ann" ]] || [[ -d "$candidate/val2017/ann" ]] || [[ -d "$candidate/test2017/ann" ]]; }; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

paco_lvis_split_dirs() {
  local raw normalized
  local -a selected
  IFS=',' read -r -a selected <<< "$PACO_LVIS_SPLITS"
  for raw in "${selected[@]}"; do
    case "${raw,,}" in
      train|training) normalized=train2017 ;;
      val|valid|validation) normalized=val2017 ;;
      test|testing) normalized=test2017 ;;
      *) echo "ERROR: PACO_LVIS_SPLITS must contain train, val, and/or test (got: $raw)" >&2; return 1 ;;
    esac
    printf '%s\n' "$normalized"
  done
}

paco_lvis_root_has_requested_splits() {
  local root=$1 split
  [[ -f "$root/meta.json" ]] || return 1
  while IFS= read -r split; do
    [[ -d "$root/$split/ann" && -d "$root/$split/img" ]] || return 1
  done < <(paco_lvis_split_dirs)
}

stage_paco_lvis_if_needed() {
  PACO_LVIS_ROOT=
  is_true "$PACO_LVIS_ENABLE" || {
    echo "PACO_LVIS_ENABLE off; skipping PACO-LVIS."
    return
  }
  is_true "$INSTANCESEG_V2_BASELINE_ONLY" && {
    echo "Fixed instance baseline profile; skipping PACO-LVIS."
    return
  }

  # Validate this before accepting an existing tree or beginning a large extract.
  paco_lvis_split_dirs >/dev/null
  local resolved stage_root
  local -a split_dirs
  if resolved=$(normalize_paco_lvis_root "$PACO_LVIS_SOURCE_ROOT"); then
    if paco_lvis_root_has_requested_splits "$resolved"; then
      PACO_LVIS_ROOT=$resolved
      echo "Using extracted PACO-LVIS tree in place: $PACO_LVIS_ROOT"
      return
    fi
    echo "PACO-LVIS source tree lacks one or more requested splits; using archive fallback."
  fi
  if [[ ! -f "$PACO_LVIS_TAR" ]]; then
    echo "WARNING: PACO-LVIS tree/tar not found; PACO-LVIS rows will be skipped." >&2
    return
  fi

  stage_root="$INSTANCESEG_STAGE/paco_lvis"
  if is_true "$FORCE"; then
    rm -rf "$stage_root"
  fi
  if ! paco_lvis_root_has_requested_splits "$stage_root"; then
    mkdir -p "$stage_root"
    echo "Extracting PACO-LVIS metadata and requested split(s) into $stage_root"
    # Tar treats a selected directory prefix as all its member files even though
    # this DatasetNinja tar has no explicit directory entries.
    mapfile -t split_dirs < <(paco_lvis_split_dirs)
    tar -xf "$PACO_LVIS_TAR" -C "$stage_root" meta.json "${split_dirs[@]}"
    if ! paco_lvis_root_has_requested_splits "$stage_root"; then
      echo "ERROR: staged PACO-LVIS archive lacks meta.json or requested img/ann split trees." >&2
      exit 1
    fi
  else
    echo "Using existing extracted PACO-LVIS data under: $stage_root"
  fi
  PACO_LVIS_ROOT=$stage_root
}


# Extract the requested Mapillary subpaths into <dest>/mapillary_vistasv2/.
# Patterns are given WITHOUT that prefix (e.g. 'training/images/*'). The shipped
# archive is a ZIP whose entries sit at the archive root, so unzip writes them
# under the prefix directory; a genuine tar (which carries the prefix already)
# is extracted directly. Either way the converters see the tree they expect.
extract_mapillary_subset() {
  local archive=$1 dest=$2
  shift 2
  local root=$dest/mapillary_vistasv2
  if [[ "$(head -c 2 "$archive" 2>/dev/null)" == "PK" ]]; then
    mkdir -p "$root"
    if command -v unzip >/dev/null 2>&1; then
      # -n: never overwrite, so re-runs top up an interrupted extract cheaply.
      unzip -qq -n "$archive" "$@" -d "$root"
    else
      # Fallback for nodes without unzip. Same semantics as -n: existing files
      # are left alone, so a rerun tops up a partial extract.
      local pattern_list
      pattern_list=$(printf '%s\n' "$@")
      ZIP_ARCHIVE="$archive" ZIP_DEST="$root" ZIP_PATTERNS="$pattern_list" "$PYTHON" - <<'PY'
import fnmatch, os, zipfile
from pathlib import Path

archive = os.environ["ZIP_ARCHIVE"]
dest = Path(os.environ["ZIP_DEST"]).resolve()
patterns = [p for p in os.environ["ZIP_PATTERNS"].splitlines() if p]
written = skipped = 0
with zipfile.ZipFile(archive) as zf:
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if patterns and not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        target = (dest / name).resolve()
        if not str(target).startswith(str(dest) + os.sep):
            raise SystemExit(f"ERROR: zip entry escapes the destination: {name}")
        if target.exists():
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write via a temp name so an interrupted run leaves no half file that
        # the exists() check above would then treat as complete.
        tmp = target.with_name(target.name + ".partial")
        with zf.open(info) as src, open(tmp, "wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)
        tmp.replace(target)
        written += 1
print(f"zipfile extract: {written:,} written, {skipped:,} already present")
PY
    fi
  else
    local patterns=() pattern
    for pattern in "$@"; do
      patterns+=("mapillary_vistasv2/$pattern")
    done
    mkdir -p "$dest"
    tar -xf "$archive" -C "$dest" --wildcards "${patterns[@]}"
  fi
}

stage_pico_checkout_if_needed() {
  require_path "$PICO_SOURCE_ROOT" "Pico-Banana source root"
  mkdir -p "$(dirname "$PICO_STAGE_ROOT")"
  if is_true "$FORCE"; then
    rm -rf "$PICO_STAGE_ROOT"
  fi
  if [[ -d "$PICO_STAGE_ROOT" ]]; then
    echo "Using existing staged Pico checkout: $PICO_STAGE_ROOT"
    return
  fi
  echo "Copying Pico-Banana checkout to $PICO_STAGE_ROOT"
  cp -a "$PICO_SOURCE_ROOT" "$PICO_STAGE_ROOT"
}

extract_openimages_split_if_needed() {
  local split_name=$1
  local source_dir=$PICO_STAGE_OPENIMAGES_DIR/openimage_source_images
  local extracted_dir=$source_dir/$split_name
  local tar_path=$source_dir/$split_name.tar
  local tgz_path=$source_dir/$split_name.tar.gz

  if is_true "$FORCE"; then
    rm -rf "$extracted_dir"
  fi
  if [[ -d "$extracted_dir" ]]; then
    echo "Using existing extracted OpenImages split: $extracted_dir"
    return
  fi
  if [[ -f "$tar_path" ]]; then
    echo "Extracting $tar_path"
    tar -xf "$tar_path" -C "$source_dir"
    return
  fi
  if [[ -f "$tgz_path" ]]; then
    echo "Extracting $tgz_path"
    tar -xzf "$tgz_path" -C "$source_dir"
    return
  fi
  echo "ERROR: neither $tar_path nor $tgz_path exists." >&2
  exit 1
}

extract_pico_targets_from_tar_if_possible() {
  local targets_dir=$PICO_STAGE_OPENIMAGES_DIR/images/positive-edit
  local staged_tar=$PICO_STAGE_OPENIMAGES_DIR/images/positive-edit.tar
  local staged_tar_fallback=$PICO_STAGE_OPENIMAGES_DIR/positive-edit.tar
  local tar_candidate

  if dir_has_files "$targets_dir"; then
    echo "Using existing Pico target images: $targets_dir"
    return 0
  fi

  for tar_candidate in \
    "$staged_tar" \
    "$staged_tar_fallback" \
    "$PICO_POSITIVE_EDIT_TAR" \
    "$PICO_POSITIVE_EDIT_TAR_FALLBACK"; do
    if [[ -f "$tar_candidate" ]]; then
      echo "Extracting Pico target images from tar: $tar_candidate"
      # Two layouts ship in the wild: the upstream tar roots its members at
      # positive-edit/, the one built by download_pico_positive_edit.sh roots them
      # at images/positive-edit/. Extracting the former at the openimages root
      # drops the payload beside images/ instead of inside it, and the check below
      # then reports no targets -- which sends the caller to the network download.
      # Pick the root that lands it in $targets_dir either way. head closes the
      # pipe, so tar dies of SIGPIPE under pipefail; that status is not a failure.
      local first_member=""
      local extract_root=$PICO_STAGE_OPENIMAGES_DIR
      first_member=$(tar -tf "$tar_candidate" 2>/dev/null | head -n 1) || true
      if [[ "$first_member" == positive-edit/* ]]; then
        extract_root=$PICO_STAGE_OPENIMAGES_DIR/images
      fi
      mkdir -p "$extract_root"
      tar -xf "$tar_candidate" -C "$extract_root"
      if dir_has_files "$targets_dir"; then
        return 0
      fi
    fi
  done

  return 1
}

download_pico_targets_if_needed() {
  local targets_dir=$PICO_STAGE_OPENIMAGES_DIR/images/positive-edit

  if dir_has_files "$targets_dir"; then
    echo "Using existing Pico target images: $targets_dir"
    return
  fi
  if extract_pico_targets_from_tar_if_possible; then
    return
  fi
  if is_true "$SKIP_PICO_TARGET_DOWNLOAD"; then
    echo "ERROR: Pico target images are missing under $targets_dir" >&2
    echo "Checked both extracted directories and the following tar locations:" >&2
    echo "  $PICO_STAGE_OPENIMAGES_DIR/images/positive-edit.tar" >&2
    echo "  $PICO_STAGE_OPENIMAGES_DIR/positive-edit.tar" >&2
    echo "  $PICO_POSITIVE_EDIT_TAR" >&2
    echo "  $PICO_POSITIVE_EDIT_TAR_FALLBACK" >&2
    exit 1
  fi

  echo "Downloading Pico target images into $targets_dir"
  env \
    ROOT="$PICO_STAGE_ROOT" \
    OPENIMAGES_DIR="$PICO_STAGE_OPENIMAGES_DIR" \
    MANIFEST="$PICO_STAGE_OPENIMAGES_DIR/sft_manifest.txt" \
    POSITIVE_EDIT_DIR="$targets_dir" \
    TAR_OUTPUT="$PICO_STAGE_OPENIMAGES_DIR/positive-edit.tar" \
    JOBS="$PICO_TARGET_DOWNLOAD_JOBS" \
    MAKE_TAR=0 \
    FORCE="$FORCE" \
    "$PICO_POSITIVE_EDIT_DOWNLOADER"
}

resolve_dir() {
  local parent=$1
  shift
  local candidate
  for candidate in "$@"; do
    if [[ -d "$parent/$candidate" ]]; then
      printf '%s\n' "$parent/$candidate"
      return
    fi
  done
  echo "ERROR: could not find extracted dataset directory under $parent (tried: $*)" >&2
  exit 1
}

# vipseg.tar double-nests (VIPSeg/VIPSeg/{imgs,panomasks,VIPSeg_720P}), so the
# resolved dir can sit one level above the real dataset root. Descend to the dir
# that actually holds VIPSeg_720P/panomasks (this dir, else one level in). Both
# the semseg --vipseg-root and the instanceseg VIPSEG_INST_ROOT check rely on it.
normalize_vipseg_root() {
  local dir=$1 nested
  if [[ -d "$dir/VIPSeg_720P" || -d "$dir/panomasks" ]]; then
    printf '%s\n' "$dir"
    return
  fi
  for nested in "$dir"/*/; do
    [[ -d "$nested" ]] || continue
    if [[ -d "${nested}VIPSeg_720P" || -d "${nested}panomasks" ]]; then
      printf '%s\n' "${nested%/}"
      return
    fi
  done
  printf '%s\n' "$dir"
}

# acdc.tar double-nests (acdc/ACDC/{rgb_anon,gt}), so the resolved dir can sit one
# level above the real dataset root -- and build_acdc_pairs then finds no
# rgb_anon/<condition>/<split> and silently yields zero pairs. Descend to the dir
# that actually holds rgb_anon/gt (this dir, else one level in).
normalize_acdc_root() {
  local dir=$1 nested
  if [[ -d "$dir/rgb_anon" || -d "$dir/gt" ]]; then
    printf '%s\n' "$dir"
    return
  fi
  for nested in "$dir"/*/; do
    [[ -d "$nested" ]] || continue
    if [[ -d "${nested}rgb_anon" || -d "${nested}gt" ]]; then
      printf '%s\n' "${nested%/}"
      return
    fi
  done
  printf '%s\n' "$dir"
}

# Stage SteerViT source images onto node-local scratch (parity with semseg/pico)
# and point REFSEG_EXTRA_ARGS at the node-local roots. No flattening: VG's tar
# layout (VG_100K/) matches the loader, and the converter resolves Mapillary in its
# native mapillary_vistasv2/<split>/images/ tree (get_mapillary_path). Skipped if
# refseg is off or the caller already supplied image roots in REFSEG_EXTRA_ARGS.
stage_refseg_sources_if_needed() {
  if ! is_true "$REFSEG_ENABLE"; then
    echo "REFSEG_ENABLE is off; skipping refseg source staging."
    return
  fi
  if [[ "$REFSEG_EXTRA_ARGS" == *--visual-genome* || "$REFSEG_EXTRA_ARGS" == *--mapillary* ]]; then
    echo "REFSEG_EXTRA_ARGS already supplies image roots; skipping refseg staging."
    return
  fi

  local image_args=""

  # Visual Genome -> <stage>/visual_genome/VG_100K[_2]/ (matches --visual-genome).
  local vg_dir=$REFSEG_STAGE/visual_genome
  if [[ -f "$REFSEG_VISUAL_GENOME_TAR" ]]; then
    if is_true "$FORCE"; then rm -rf "$vg_dir"; fi
    if has_any_dir "$vg_dir" VG_100K VG_100K_2; then
      echo "Using existing staged Visual Genome images: $vg_dir"
    else
      echo "Extracting Visual Genome images into $vg_dir"
      mkdir -p "$vg_dir"
      tar -xf "$REFSEG_VISUAL_GENOME_TAR" -C "$vg_dir"
    fi
    require_nonempty_dir "$vg_dir" "Visual Genome staged images"
    image_args+=" --visual-genome $vg_dir"
  else
    echo "WARNING: Visual Genome tar not found ($REFSEG_VISUAL_GENOME_TAR); VG rows will be skipped." >&2
  fi

  # Mapillary Vistas -> <stage>/mapillary/mapillary_vistasv2/<split>/images/
  # (matches the --mapillary default + get_mapillary_path). Extract only the JPGs
  # to skip labels/panoptic/instances; the native split layout is preserved.
  local map_dir=$REFSEG_STAGE/mapillary
  if [[ -f "$REFSEG_MAPILLARY_TAR" ]]; then
    if is_true "$FORCE"; then rm -rf "$map_dir"; fi
    if has_any_dir "$map_dir" mapillary_vistasv2; then
      echo "Using existing staged Mapillary images: $map_dir"
    else
      echo "Extracting Mapillary images into $map_dir"
      extract_mapillary_subset "$REFSEG_MAPILLARY_TAR" "$map_dir" \
        'training/images/*.jpg' 'validation/images/*.jpg'
    fi
    require_nonempty_dir "$map_dir" "Mapillary staged images"
    image_args+=" --mapillary $map_dir"
  else
    echo "WARNING: Mapillary tar not found ($REFSEG_MAPILLARY_TAR); Mapillary rows will be skipped." >&2
  fi

  REFSEG_EXTRA_ARGS="${image_args# }${REFSEG_EXTRA_ARGS:+ $REFSEG_EXTRA_ARGS}"
  export REFSEG_EXTRA_ARGS
  echo "REFSEG_EXTRA_ARGS: ${REFSEG_EXTRA_ARGS:-<none>}"
}

# Stage the instance-seg sources onto node-local scratch (parity with semseg/pico).
# Mapillary is large, so only the bits the converter needs are extracted: images +
# v2.0 instance PNGs + the class config. Skipped entirely if INSTANCESEG_ENABLE=0.
stage_instanceseg_sources_if_needed() {
  is_true "$INSTANCESEG_ENABLE" || { echo "INSTANCESEG_ENABLE off; skipping instance-seg staging."; return; }
  mkdir -p "$INSTANCESEG_STAGE"
  stage_paco_lvis_if_needed
  if [[ -f "$ADE_INST_TAR" ]]; then
    extract_tar_if_needed "$ADE_INST_TAR" "$INSTANCESEG_STAGE" ade20k_instanceseg
    # The name tables are NOT inside the tar; drop them beside the extract.
    for f in instanceInfo100_train.txt instanceInfo100_val.txt categoryMapping.txt; do
      [[ -f "$ADE_INST_INFO_DIR/$f" ]] && cp -f "$ADE_INST_INFO_DIR/$f" "$INSTANCESEG_STAGE/ade20k_instanceseg/" || true
    done
  fi
  if [[ -f "$MAPILLARY_INST_TAR" && ! -f "$INSTANCESEG_STAGE/mapillary_vistasv2/config_v2.0.json" ]]; then
    echo "Extracting Mapillary v2.0 (images + instances + config) -> $INSTANCESEG_STAGE"
    extract_mapillary_subset "$MAPILLARY_INST_TAR" "$INSTANCESEG_STAGE" \
      'config_v2.0.json' \
      'training/images/*' 'validation/images/*' \
      'training/v2.0/instances/*' 'validation/v2.0/instances/*'
  fi
  [[ -f "$VOC_INST_TAR" ]] && extract_tar_if_needed "$VOC_INST_TAR" "$INSTANCESEG_STAGE" VOCdevkit
}

# Extract the depth / surface-normal member tars straight into PREPARED_ROOT.
# Each tar holds one canonical member under a single top-level dir; the dir name
# is read from the tar (it does not always match the tar basename, e.g.
# synscapes_depth.tar -> depth_synscapes/). Fills DEPTH_MEMBER_DIRS /
# NORMALS_MEMBER_DIRS for prepare_open_visionbanana_pool.sh.
DEPTH_MEMBER_DIRS=${DEPTH_MEMBER_DIRS:-}
NORMALS_MEMBER_DIRS=${NORMALS_MEMBER_DIRS:-}

extract_geometry_member() {
  local archive=$1
  local member_name
  member_name=$(tar -tf "$archive" | head -1 | cut -d/ -f1)
  if [[ -z "$member_name" ]]; then
    echo "ERROR: cannot determine member dir inside $archive" >&2
    exit 1
  fi
  local member_dir=$PREPARED_ROOT/$member_name
  if is_true "$FORCE"; then
    rm -rf "$member_dir"
  fi
  # progress goes to stderr: stdout is this function's return value (the
  # member dir path), captured with $(...) by the caller.
  if [[ -s "$member_dir/metadata.jsonl" ]]; then
    echo "Using existing geometry member: $member_dir" >&2
  else
    echo "Extracting $archive -> $member_dir" >&2
    tar -xf "$archive" -C "$PREPARED_ROOT"
    require_nonempty_dir "$member_dir" "geometry member $member_name"
    # Older geometry tars carry the file_name/edited_file_name/prompt dialect;
    # rewrite it to the canonical schema (no-op when already canonical).
    (cd "$ROOT" && "$PYTHON" scripts/dataset_prep/normalize_member_metadata.py "$member_dir") >&2
  fi
  printf '%s\n' "$member_dir"
}

# Presence-based, like the optional semseg tars: stage whatever geometry tars
# exist; the prepare script's DEPTH_WEIGHT/NORMALS_WEIGHT decide mix inclusion
# (their defaults live there, so this gate must not read them). Point
# GEOMETRY_SOURCE_ROOT away (or clear DEPTH_TARS/NORMALS_TARS) to skip extraction.
stage_geometry_members_if_needed() {
  local task=$1 tars=$2 pattern=$3
  if [[ -z "$tars" ]]; then
    tars=$(ls "$GEOMETRY_SOURCE_ROOT"/$pattern 2>/dev/null || true)
  fi
  if [[ -z "$tars" ]]; then
    echo "No $task tars found under $GEOMETRY_SOURCE_ROOT ($pattern); skipping $task staging."
    return
  fi
  local tar_path dirs=()
  for tar_path in $tars; do
    dirs+=("$(extract_geometry_member "$tar_path")")
  done
  if [[ "$task" == depth ]]; then
    DEPTH_MEMBER_DIRS="${dirs[*]}"
  else
    NORMALS_MEMBER_DIRS="${dirs[*]}"
  fi
}

# Mapillary Vistas v2.0 for SEMANTIC segmentation: config + images + v2.0/labels.
# Reuses the instance-seg extract when it is already there (images and config are the
# same files) so the 29G tar is only unpacked once. Sets MAPILLARY_ROOT.
stage_mapillary_semseg_if_needed() {
  is_true "$MAPILLARY_SEMSEG_ENABLE" || { echo "MAPILLARY_SEMSEG_ENABLE off; skipping Mapillary semseg staging."; return; }
  if [[ ! -f "$MAPILLARY_SEMSEG_TAR" ]]; then
    echo "WARNING: Mapillary tar not found ($MAPILLARY_SEMSEG_TAR); Mapillary semseg rows will be skipped." >&2
    return
  fi
  local dest=$SEMSEG_STAGE
  [[ -d "$INSTANCESEG_STAGE/mapillary_vistasv2" ]] && dest=$INSTANCESEG_STAGE
  MAPILLARY_ROOT=$dest/mapillary_vistasv2
  if ! is_true "$FORCE" && [[ -f "$MAPILLARY_ROOT/config_v2.0.json" && -d "$MAPILLARY_ROOT/training/v2.0/labels" ]]; then
    echo "Using existing staged Mapillary semseg tree: $MAPILLARY_ROOT"
    return
  fi
  echo "Extracting Mapillary v2.0 (images + labels + config) -> $dest"
  extract_mapillary_subset "$MAPILLARY_SEMSEG_TAR" "$dest" \
    'config_v2.0.json' \
    'training/images/*' 'validation/images/*' \
    'training/v2.0/labels/*' 'validation/v2.0/labels/*'
  require_nonempty_dir "$MAPILLARY_ROOT/training/v2.0/labels" "Mapillary v2.0 semantic labels"
}

mkdir -p "$STAGING_ROOT" "$SEMSEG_STAGE" "$PREPARED_ROOT"

if is_true "$INSTANCESEG_V2_BASELINE_ONLY"; then
  is_true "$INSTANCESEG_ENABLE" || {
    echo "ERROR: INSTANCESEG_V2_BASELINE_ONLY=1 requires INSTANCESEG_ENABLE=1" >&2
    exit 1
  }
  # Fail before a long partial build if any source needed to reproduce the
  # baseline is unavailable.
  for archive in "$ADE_INST_TAR" "$MAPILLARY_INST_TAR" "$VOC_INST_TAR" \
                 "$ACDC_TAR" "$MUSES_TAR" "$GOOSE_TAR" "$VIPSEG_TAR" \
                 "$OVIS_TAR" "$WAYMO_TAR"; do
    require_path "$archive" "v2 baseline instance-seg archive"
  done
  extract_tar_if_needed "$ACDC_TAR" "$SEMSEG_STAGE" acdc ACDC
  extract_tar_if_needed "$MUSES_TAR" "$SEMSEG_STAGE" muses MUSES
  extract_tar_if_needed "$GOOSE_TAR" "$SEMSEG_STAGE" goose goose-dataset
  extract_tar_if_needed "$VIPSEG_TAR" "$SEMSEG_STAGE" VIPSeg vipseg
  extract_tar_if_needed "$OVIS_TAR" "$SEMSEG_STAGE" OVIS ovis
  extract_tar_if_needed "$WAYMO_TAR" "$SEMSEG_STAGE" waymo Waymo
else
  if is_true "$INCLUDE_CITYSCAPES"; then
    extract_tar_if_needed "$CITYSCAPES_TAR" "$SEMSEG_STAGE" cityscapes Cityscapes
  fi
  extract_tar_if_needed "$ACDC_TAR" "$SEMSEG_STAGE" acdc ACDC
  extract_tar_if_needed "$BDD_TAR" "$SEMSEG_STAGE" bdd_10k BDD bdd
  [[ -f "$ADE20K_TAR" ]] && extract_tar_if_needed "$ADE20K_TAR" "$SEMSEG_STAGE" ADEChallengeData2016 ade20k
  [[ -f "$VOC_TAR" ]] && extract_tar_if_needed "$VOC_TAR" "$SEMSEG_STAGE" voc VOC
  [[ -f "$MUSES_TAR" ]] && extract_tar_if_needed "$MUSES_TAR" "$SEMSEG_STAGE" muses MUSES
  [[ -f "$GOOSE_TAR" ]] && extract_tar_if_needed "$GOOSE_TAR" "$SEMSEG_STAGE" goose goose-dataset
  [[ -f "$VIPSEG_TAR" ]] && extract_tar_if_needed "$VIPSEG_TAR" "$SEMSEG_STAGE" VIPSeg vipseg
  [[ -f "$OVIS_TAR" ]] && extract_tar_if_needed "$OVIS_TAR" "$SEMSEG_STAGE" OVIS ovis
  [[ -f "$FOODSEG103_TAR" ]] && extract_tar_if_needed "$FOODSEG103_TAR" "$SEMSEG_STAGE" foodseg103 FoodSeg103
  [[ -f "$TRANS10K_TAR" ]] && extract_tar_if_needed "$TRANS10K_TAR" "$SEMSEG_STAGE" trans10k Trans10K
  [[ -f "$RAILSEM19_TAR" ]] && extract_tar_if_needed "$RAILSEM19_TAR" "$SEMSEG_STAGE" railsem19 RailSem19
  [[ -f "$WAYMO_TAR" ]] && extract_tar_if_needed "$WAYMO_TAR" "$SEMSEG_STAGE" waymo Waymo
  [[ -f "$HYPERSIM_SEG_TAR" ]] && extract_tar_if_needed "$HYPERSIM_SEG_TAR" "$SEMSEG_STAGE" hypersim_seg hypersim
fi
stage_synscapes_if_needed
stage_idd_if_needed
stage_wilddash2_if_needed
stage_nuimages_if_needed
. "$ROOT/scripts/dataset_prep/coconut_xlarge_full_recovery.sh"
stage_coconut_xlarge_with_coco_and_objects365_recovery
if is_true "$PICO_ENABLE"; then
  stage_pico_checkout_if_needed

  extract_openimages_split_if_needed train_0
  extract_openimages_split_if_needed train_1
  download_pico_targets_if_needed
else
  echo "PICO_ENABLE is off; skipping Pico/OpenImages staging."
fi

stage_refseg_sources_if_needed
stage_instanceseg_sources_if_needed
MAPILLARY_ROOT=
stage_mapillary_semseg_if_needed
if is_true "$INSTANCESEG_V2_BASELINE_ONLY"; then
  echo "INSTANCESEG_V2_BASELINE_ONLY is on; skipping geometry staging."
else
  stage_geometry_members_if_needed depth "$DEPTH_TARS" '*_depth*.tar'
  stage_geometry_members_if_needed normals "$NORMALS_TARS" '*_surface_normals*.tar'
fi

PICO_METADATA=$PICO_STAGE_OPENIMAGES_DIR/sft_with_local_source_image_path.jsonl
if is_true "$PICO_ENABLE"; then
  require_nonempty_dir "$PICO_STAGE_OPENIMAGES_DIR/openimage_source_images/train_0" "OpenImages train_0 extraction"
  require_nonempty_dir "$PICO_STAGE_OPENIMAGES_DIR/openimage_source_images/train_1" "OpenImages train_1 extraction"
  require_nonempty_dir "$PICO_STAGE_OPENIMAGES_DIR/images/positive-edit" "Pico target images"

  if is_true "$FORCE"; then
    rm -f "$PICO_METADATA"
  fi
  if [[ ! -f "$PICO_METADATA" ]]; then
    require_path "$PICO_STAGE_ROOT/map_openimage_url_to_local.py" "Pico mapping script"
    run_in_workdir "$PICO_STAGE_ROOT" "$PYTHON" map_openimage_url_to_local.py
  fi
  require_path "$PICO_METADATA" "mapped Pico metadata"
fi

CITYSCAPES_ROOT=
BDD_ROOT=
ADE20K_ROOT=
VOC_ROOT=
if is_true "$INSTANCESEG_V2_BASELINE_ONLY"; then
  ACDC_ROOT=$(normalize_acdc_root "$(resolve_dir "$SEMSEG_STAGE" acdc ACDC)")
else
  if is_true "$INCLUDE_CITYSCAPES"; then
    CITYSCAPES_ROOT=$(resolve_dir "$SEMSEG_STAGE" cityscapes Cityscapes)
  fi
  ACDC_ROOT=$(normalize_acdc_root "$(resolve_dir "$SEMSEG_STAGE" acdc ACDC)")
  BDD_ROOT=$(resolve_dir "$SEMSEG_STAGE" bdd_10k BDD bdd)
  [[ -f "$ADE20K_TAR" ]] && ADE20K_ROOT=$(resolve_dir "$SEMSEG_STAGE" ADEChallengeData2016 ade20k)
  [[ -f "$VOC_TAR" ]] && VOC_ROOT=$(resolve_dir "$SEMSEG_STAGE" voc VOC)
fi
# PASCAL-Context labels ship inside voc.tar (download_datasets_pascal_context.sh),
# so they need no staging of their own -- just point at the same extracted root.
PASCAL_CONTEXT_ROOT=
[[ -n "$VOC_ROOT" && -d "$VOC_ROOT/pascal_context_59classlabels" ]] && PASCAL_CONTEXT_ROOT=$VOC_ROOT
MUSES_ROOT=
[[ -f "$MUSES_TAR" ]] && MUSES_ROOT=$(resolve_dir "$SEMSEG_STAGE" muses MUSES)
GOOSE_ROOT=
[[ -f "$GOOSE_TAR" ]] && GOOSE_ROOT=$(resolve_dir "$SEMSEG_STAGE" goose goose-dataset)
VIPSEG_ROOT=
[[ -f "$VIPSEG_TAR" ]] && VIPSEG_ROOT=$(normalize_vipseg_root "$(resolve_dir "$SEMSEG_STAGE" VIPSeg vipseg)")
OVIS_ROOT=
[[ -f "$OVIS_TAR" ]] && OVIS_ROOT=$(resolve_dir "$SEMSEG_STAGE" OVIS ovis)
FOODSEG103_ROOT=
if ! is_true "$INSTANCESEG_V2_BASELINE_ONLY" && [[ -f "$FOODSEG103_TAR" ]]; then
  FOODSEG103_ROOT=$(resolve_dir "$SEMSEG_STAGE" foodseg103 FoodSeg103)
fi
TRANS10K_ROOT=
if ! is_true "$INSTANCESEG_V2_BASELINE_ONLY" && [[ -f "$TRANS10K_TAR" ]]; then
  TRANS10K_ROOT=$(resolve_dir "$SEMSEG_STAGE" trans10k Trans10K)
fi
RAILSEM19_ROOT=
if ! is_true "$INSTANCESEG_V2_BASELINE_ONLY" && [[ -f "$RAILSEM19_TAR" ]]; then
  RAILSEM19_ROOT=$(resolve_dir "$SEMSEG_STAGE" railsem19 RailSem19)
fi
WAYMO_ROOT=
[[ -f "$WAYMO_TAR" ]] && WAYMO_ROOT=$(resolve_dir "$SEMSEG_STAGE" waymo Waymo)
HYPERSIM_SEG_ROOT=
if ! is_true "$INSTANCESEG_V2_BASELINE_ONLY" && [[ -f "$HYPERSIM_SEG_TAR" ]]; then
  HYPERSIM_SEG_ROOT=$(resolve_dir "$SEMSEG_STAGE" hypersim_seg hypersim)
  # The raw release sits next to the materialized one under the same shared_seg/
  # directory with a near-identical name, so a mis-set HYPERSIM_SEG_TAR is easy and
  # would otherwise surface as "0 hypersim pairs" a long way downstream.
  if [[ -n "$HYPERSIM_SEG_ROOT" && ! -f "$HYPERSIM_SEG_ROOT/id2label.json" ]]; then
    echo "ERROR: $HYPERSIM_SEG_TAR staged to $HYPERSIM_SEG_ROOT, which has no id2label.json." >&2
    if [[ -d "$HYPERSIM_SEG_ROOT/work/raw_extracted" || -d "$HYPERSIM_SEG_ROOT/ml-hypersim" ]]; then
      echo "       That looks like the RAW HDF5 release, which neither converter can read." >&2
    fi
    echo "       Build the materialized tree first, then point HYPERSIM_SEG_TAR at it:" >&2
    echo "         python -m dataset_prep.hypersim.prepare_hypersim_segmentation \\" >&2
    echo "             --raw-root <extracted>/work/raw_extracted \\" >&2
    echo "             --out-root <staging>/hypersim_seg --all-frames \\" >&2
    echo "             --prepared-tar $HYPERSIM_SEG_TAR" >&2
    exit 1
  fi
fi

SEMSEG_OUT=${SEMSEG_OUT:-$PREPARED_ROOT/semseg}
PICO_OUT=${PICO_OUT:-$PREPARED_ROOT/pico_banana_sft_50k}
IDENTITY_OUT=${IDENTITY_OUT:-$PREPARED_ROOT/identity_10k}
# Referring-seg training data is pulled from the HF hub (no tar to stage); only
# its output path is resolved here. The REFSEG_* knobs inherit via env below.
REFSEG_OUT=${REFSEG_OUT:-$PREPARED_ROOT/refseg}

# Instance-seg roots: only set for sources that actually staged, so the converter
# (and the specs append) include exactly what is present.
ADE_INST_ROOT=
MAPILLARY_INST_ROOT=
VOC_INST_ROOT=
if is_true "$INSTANCESEG_ENABLE"; then
  [[ -d "$INSTANCESEG_STAGE/ade20k_instanceseg" ]] && ADE_INST_ROOT=$INSTANCESEG_STAGE/ade20k_instanceseg
  [[ -d "$INSTANCESEG_STAGE/mapillary_vistasv2" ]] && MAPILLARY_INST_ROOT=$INSTANCESEG_STAGE/mapillary_vistasv2
  [[ -d "$INSTANCESEG_STAGE/VOCdevkit/VOC2012" ]] && VOC_INST_ROOT=$INSTANCESEG_STAGE/VOCdevkit/VOC2012
fi
SYNSCAPES_INST_ROOT=
if is_true "$INSTANCESEG_ENABLE" && [[ -n "$SYNSCAPES_ROOT" && -d "$SYNSCAPES_ROOT/img/instance" ]]; then
  SYNSCAPES_INST_ROOT=$SYNSCAPES_ROOT
fi
IDD_INST_ROOT=
if is_true "$INSTANCESEG_ENABLE" && [[ -n "$IDD_ROOT" ]] && normalize_idd_root "$IDD_ROOT" >/dev/null; then
  IDD_INST_ROOT=$IDD_ROOT
fi
WILDDASH2_INST_ROOT=
if is_true "$INSTANCESEG_ENABLE" && [[ -n "$WILDDASH2_ROOT" ]] && normalize_wilddash2_root "$WILDDASH2_ROOT" >/dev/null; then
  WILDDASH2_INST_ROOT=$WILDDASH2_ROOT
fi
NUIMAGES_INST_ROOT=
COCONUT_XL_INST_ROOT=
if is_true "$INSTANCESEG_ENABLE" && [[ -n "$COCONUT_XL_ROOT" ]] && normalize_coconut_xlarge_root "$COCONUT_XL_ROOT" >/dev/null; then
  COCONUT_XL_INST_ROOT=$COCONUT_XL_ROOT
fi
PACO_LVIS_INST_ROOT=
if is_true "$INSTANCESEG_ENABLE" && [[ -n "$PACO_LVIS_ROOT" ]] && paco_lvis_root_has_requested_splits "$PACO_LVIS_ROOT"; then
  PACO_LVIS_INST_ROOT=$PACO_LVIS_ROOT
fi
if is_true "$INSTANCESEG_ENABLE" && [[ -n "$NUIMAGES_ROOT" ]] && normalize_nuimages_root "$NUIMAGES_ROOT" >/dev/null; then
  NUIMAGES_INST_ROOT=$NUIMAGES_ROOT
fi
MUSES_INST_ROOT=
if is_true "$INSTANCESEG_ENABLE" && [[ -n "$MUSES_ROOT" && -d "$MUSES_ROOT/gt_panoptic" ]]; then
  MUSES_INST_ROOT=$MUSES_ROOT
fi
# ACDC ships gt_panoptic/ next to the semantic gt/ (COCO panoptic, as MUSES), so
# one staged tree feeds both members; archives without it stay semseg-only.
ACDC_INST_ROOT=
if is_true "$INSTANCESEG_ENABLE" && [[ -n "$ACDC_ROOT" && -d "$ACDC_ROOT/gt_panoptic" ]]; then
  ACDC_INST_ROOT=$ACDC_ROOT
fi
GOOSE_INST_ROOT=
if is_true "$INSTANCESEG_ENABLE" && [[ -n "$GOOSE_ROOT" && -d "$GOOSE_ROOT/labels" ]]; then
  GOOSE_INST_ROOT=$GOOSE_ROOT
fi
VIPSEG_INST_ROOT=
if is_true "$INSTANCESEG_ENABLE" && [[ -n "$VIPSEG_ROOT" && -d "$VIPSEG_ROOT/panomasks" ]]; then
  VIPSEG_INST_ROOT=$VIPSEG_ROOT
fi
OVIS_INST_ROOT=
if is_true "$INSTANCESEG_ENABLE" && [[ -n "$OVIS_ROOT" && -f "$OVIS_ROOT/annotations_train.json" ]]; then
  OVIS_INST_ROOT=$OVIS_ROOT
fi
# Waymo's packed labels carry instances as well, so the staged semseg tree is
# handed to the instance-seg converter unchanged -- no second extract.
WAYMO_INST_ROOT=
if is_true "$INSTANCESEG_ENABLE" && [[ -n "$WAYMO_ROOT" && -d "$WAYMO_ROOT/panoptic" ]]; then
  WAYMO_INST_ROOT=$WAYMO_ROOT
fi
# Hypersim likewise: the same packed tree serves both members.
HYPERSIM_SEG_INST_ROOT=
if is_true "$INSTANCESEG_ENABLE" && [[ -n "$HYPERSIM_SEG_ROOT" && -d "$HYPERSIM_SEG_ROOT/panoptic" ]]; then
  HYPERSIM_SEG_INST_ROOT=$HYPERSIM_SEG_ROOT
fi
INSTANCESEG_OUT=${INSTANCESEG_OUT:-$PREPARED_ROOT/instanceseg}

echo "Preparing canonical dataset pool under $PREPARED_ROOT"
env \
  PICO_ENABLE="$PICO_ENABLE" \
  IDENTITY_ENABLE="$IDENTITY_ENABLE" \
  REFSEG_ENABLE="$REFSEG_ENABLE" \
  REASONSEG_REFSEG_ENABLE="${REASONSEG_REFSEG_ENABLE:-}" \
  REASONSEG_REFSEG_OUT="${REASONSEG_REFSEG_OUT:-}" \
  REASONSEG_REFSEG_ROOT="${REASONSEG_REFSEG_ROOT:-}" \
  REASONSEG_REFSEG_EXTRACT_ROOT="${REASONSEG_REFSEG_EXTRACT_ROOT:-}" \
  REASONSEG_REFSEG_KEEP_EXTRACTED="${REASONSEG_REFSEG_KEEP_EXTRACTED:-}" \
  REASONSEG_REFSEG_SPLITS="${REASONSEG_REFSEG_SPLITS:-}" \
  REASONSEG_REFSEG_ALL_EXPRESSIONS="${REASONSEG_REFSEG_ALL_EXPRESSIONS:-}" \
  REASONSEG_REFSEG_EXPRESSIONS_PER_IMAGE="${REASONSEG_REFSEG_EXPRESSIONS_PER_IMAGE:-}" \
  REASONSEG_REFSEG_VAL_DIR="${REASONSEG_REFSEG_VAL_DIR:-}" \
  REASONSEG_REFSEG_EXCLUDE_VAL_OVERLAP="${REASONSEG_REFSEG_EXCLUDE_VAL_OVERLAP:-}" \
  REASONSEG_REFSEG_COLORS="${REASONSEG_REFSEG_COLORS:-}" \
  REASONSEG_REFSEG_NO_RANDOM_COLORS="${REASONSEG_REFSEG_NO_RANDOM_COLORS:-}" \
  REASONSEG_REFSEG_WEIGHT="${REASONSEG_REFSEG_WEIGHT:-}" \
  ROOT="$ROOT" \
  INSTANCESEG_ENABLE="$INSTANCESEG_ENABLE" \
  INSTANCESEG_OUT="$INSTANCESEG_OUT" \
  ADE_INST_ROOT="$ADE_INST_ROOT" \
  MAPILLARY_INST_ROOT="$MAPILLARY_INST_ROOT" \
  SYNSCAPES_INST_ROOT="$SYNSCAPES_INST_ROOT" \
  IDD_INST_ROOT="$IDD_INST_ROOT" \
  WILDDASH2_INST_ROOT="$WILDDASH2_INST_ROOT" \
  NUIMAGES_INST_ROOT="$NUIMAGES_INST_ROOT" \
  COCONUT_XL_INST_ROOT="$COCONUT_XL_INST_ROOT" \
  PACO_LVIS_INST_ROOT="$PACO_LVIS_INST_ROOT" \
  MUSES_INST_ROOT="$MUSES_INST_ROOT" \
  VOC_INST_ROOT="$VOC_INST_ROOT" \
  ACDC_INST_ROOT="$ACDC_INST_ROOT" \
  GOOSE_INST_ROOT="$GOOSE_INST_ROOT" \
  VIPSEG_INST_ROOT="$VIPSEG_INST_ROOT" \
  OVIS_INST_ROOT="$OVIS_INST_ROOT" \
  WAYMO_INST_ROOT="$WAYMO_INST_ROOT" \
  HYPERSIM_SEG_INST_ROOT="$HYPERSIM_SEG_INST_ROOT" \
  OVIS_FPS="${OVIS_FPS:-}" \
  OVIS_SOURCE_FPS="${OVIS_SOURCE_FPS:-}" \
  INSTANCESEG_WEIGHT="${INSTANCESEG_WEIGHT:-}" \
  INSTANCESEG_NEGATIVE_RATIO="${INSTANCESEG_NEGATIVE_RATIO:-}" \
  INSTANCESEG_VARIANTS_PER_IMAGE="${INSTANCESEG_VARIANTS_PER_IMAGE:-}" \
  INSTANCESEG_MAX_INSTANCES="${INSTANCESEG_MAX_INSTANCES:-}" \
  INSTANCESEG_MIN_INSTANCE_AREA="${INSTANCESEG_MIN_INSTANCE_AREA:-}" \
  INSTANCESEG_LEGEND_SLACK_PROB="${INSTANCESEG_LEGEND_SLACK_PROB:-}" \
  INSTANCESEG_DRAW_MODE="${INSTANCESEG_DRAW_MODE:-}" \
  INSTANCESEG_SPLITS="${INSTANCESEG_SPLITS:-}" \
  INSTANCESEG_NUM_WORKERS="${INSTANCESEG_NUM_WORKERS:-}" \
  PYTHON="$PYTHON" \
  PREPARED_ROOT="$PREPARED_ROOT" \
  SEMSEG_OUT="$SEMSEG_OUT" \
  PICO_OUT="$PICO_OUT" \
  IDENTITY_OUT="$IDENTITY_OUT" \
  REFSEG_OUT="$REFSEG_OUT" \
  REFSEG_EXTRA_ARGS="$REFSEG_EXTRA_ARGS" \
  CONFIG_OUT="$CONFIG_OUT" \
  SEMSEG_NUM_WORKERS="$SEMSEG_NUM_WORKERS" \
  PICO_NUM_WORKERS="$PICO_NUM_WORKERS" \
  PICO_MAX_SIDE="$PICO_MAX_SIDE" \
  PICO_COPY_MODE="$PICO_COPY_MODE" \
  IDENTITY_MAX_SIDE="$IDENTITY_MAX_SIDE" \
  IDENTITY_COPY_MODE="$IDENTITY_COPY_MODE" \
  CITYSCAPES_ROOT="$CITYSCAPES_ROOT" \
  CITYSCAPES_SPLITS="$CITYSCAPES_SPLITS" \
  SYNSCAPES_ROOT="$SYNSCAPES_ROOT" \
  SYNSCAPES_SPLITS="$SYNSCAPES_SPLITS" \
  IDD_ROOT="$IDD_ROOT" \
  IDD_SPLITS="$IDD_SPLITS" \
  WILDDASH2_ROOT="$WILDDASH2_ROOT" \
  WILDDASH2_SPLITS="$WILDDASH2_SPLITS" \
  REFCOCOG_UMD_ROOT="$REFCOCOG_UMD_ROOT" \
  NUIMAGES_ROOT="$NUIMAGES_ROOT" \
  COCONUT_XL_ROOT="$COCONUT_XL_ROOT" \
  COCONUT_XL_GROUPS="$COCONUT_XL_GROUPS" \
  NUIMAGES_SPLITS="$NUIMAGES_SPLITS" \
  NUIMAGES_SAMPLE_INTERVAL_SECONDS="$NUIMAGES_SAMPLE_INTERVAL_SECONDS" \
  PACO_LVIS_ROOT="$PACO_LVIS_ROOT" \
  PACO_LVIS_SPLITS="$PACO_LVIS_SPLITS" \
  ACDC_ROOT="$ACDC_ROOT" \
  BDD_ROOT="$BDD_ROOT" \
  ADE20K_ROOT="$ADE20K_ROOT" \
  VOC_ROOT="$VOC_ROOT" \
  PASCAL_CONTEXT_ROOT="$PASCAL_CONTEXT_ROOT" \
  MUSES_ROOT="$MUSES_ROOT" \
  GOOSE_ROOT="$GOOSE_ROOT" \
  VIPSEG_ROOT="$VIPSEG_ROOT" \
  MAPILLARY_ROOT="$MAPILLARY_ROOT" \
  FOODSEG103_ROOT="$FOODSEG103_ROOT" \
  TRANS10K_ROOT="$TRANS10K_ROOT" \
  RAILSEM19_ROOT="$RAILSEM19_ROOT" \
  WAYMO_ROOT="$WAYMO_ROOT" \
  HYPERSIM_SEG_ROOT="$HYPERSIM_SEG_ROOT" \
  PICO_NATIVE_ROOT="$PICO_STAGE_ROOT" \
  PICO_METADATA="$PICO_METADATA" \
  DEPTH_MEMBER_DIRS="$DEPTH_MEMBER_DIRS" \
  DEPTH_WEIGHT="${DEPTH_WEIGHT:-}" \
  NORMALS_MEMBER_DIRS="$NORMALS_MEMBER_DIRS" \
  NORMALS_WEIGHT="${NORMALS_WEIGHT:-}" \
  PD12M_WEIGHT="${PD12M_WEIGHT:-}" \
  PD12M_SHARDS_DIR="${PD12M_SHARDS_DIR:-}" \
  PD12M_LIMIT="${PD12M_LIMIT:-}" \
  PD12M_OUT="${PD12M_OUT:-}" \
  PD12M_MAX_SIDE="${PD12M_MAX_SIDE:-}" \
  FORCE="$FORCE" \
  "$ROOT/scripts/dataset_prep/prepare_open_visionbanana_pool.sh"
# Persist source-level composition before optional archive publication removes
# the member folders. This report is the durable record of image/mask counts.
report_args=(--out "$PREPARED_ROOT/dataset_composition.json")
[[ -s "$SEMSEG_OUT/metadata.jsonl" ]] && report_args+=(--member "semseg=$SEMSEG_OUT")
[[ -n "${INSTANCESEG_OUT:-}" && -s "$INSTANCESEG_OUT/metadata.jsonl" ]] && report_args+=(--member "instanceseg=$INSTANCESEG_OUT")
if (( ${#report_args[@]} > 2 )); then
  "$PYTHON" "$ROOT/scripts/dataset_prep/write_prepared_dataset_report.py" "${report_args[@]}"
fi

# After preparation no converter needs the raw extracted sources; remove them
# before tar creation to free scratch space and inodes.
if is_true "$CLEAN_STAGING_AFTER_SUCCESS"; then
  stage_real=$(realpath -m -- "$STAGING_ROOT")
  prepared_real=$(realpath -m -- "$PREPARED_ROOT")
  case "$stage_real" in
    /|"$prepared_real"|"$prepared_real"/*)
      echo "ERROR: refusing to clean unsafe STAGING_ROOT: $stage_real" >&2
      exit 2
      ;;
  esac
  echo "Removing completed source staging tree: $stage_real"
  rm -rf -- "$stage_real"
fi

# Archive only after the ordinary folder build and config validation succeed.
# The resulting TARs are intentionally uncompressed: indexed_prepared_tar uses
# byte offsets for random reads. After each archive is indexed and validated,
# its source member folder is removed to release its image inodes.
if is_true "$ARCHIVE_SEGMENTATION_MEMBERS"; then
  echo "Publishing indexed semantic- and instance-segmentation TARs"
  "$PYTHON" "$ROOT/scripts/dataset_prep/archive_prepared_member.py" \
    --member "$SEMSEG_OUT" --specs "$CONFIG_OUT" --name semseg --remove-member
  "$PYTHON" "$ROOT/scripts/dataset_prep/archive_prepared_member.py" \
    --member "$INSTANCESEG_OUT" --specs "$CONFIG_OUT" --name instanceseg --remove-member
  echo "Published: ${SEMSEG_OUT}.tar and ${INSTANCESEG_OUT}.tar"
fi
echo "Prepared dataset pool is ready."
echo "  prepared root: $PREPARED_ROOT"
echo "  dataset specs: $CONFIG_OUT"

# The assembled COCONut tree can contain dangling Objects365 symlinks after its
# transient downloader sources are removed. Materialize only that provenance
# from the retained source archives; non-Objects365 paths retain normal staging.
stage_coconut_xlarge_with_objects365_recovery() {
  if ! is_true "$COCONUT_XL_ENABLE" || [[ ",$COCONUT_XL_GROUPS," != *,objects365,* ]]; then
    stage_coconut_xlarge_if_needed
    return
  fi
  local assets source_root large_tar xl_tar archives_dir stage_base staged_root marker image_marker mask_marker archive
  if ! assets=$(resolve_coconut_xlarge_assets_root "$COCONUT_XL_SOURCE_ROOT"); then
    stage_coconut_xlarge_if_needed
    return
  fi
  source_root=$(dirname -- "$assets")/_sources
  large_tar=$source_root/coconut_large/panoptic_object365.tar
  xl_tar=$source_root/coconut_xlarge/coconuts_xlarge.tar
  archives_dir=$source_root/objects365_archives
  # A real sample means the source is already usable and needs no recovery.
  if [[ -f "$assets/train2017/objects365_v2_00940000.jpg" && -f "$assets/panoptic_train2017/objects365_v2_00940000.png" ]]; then
    stage_coconut_xlarge_if_needed
    return
  fi
  if [[ ! -f "$large_tar" || ! -f "$xl_tar" || ! -d "$archives_dir" ]]; then
    echo "WARNING: Objects365 recovery archives are unavailable; using normal COCONut staging." >&2
    stage_coconut_xlarge_if_needed
    return
  fi
  stage_base="$SEMSEG_STAGE/coconut_xlarge"
  staged_root="$stage_base/coco"
  marker="$stage_base/.ovb-objects365-recovery-v1.complete"
  image_marker="$stage_base/.ovb-objects365-images.complete"
  mask_marker="$stage_base/.ovb-objects365-masks.complete"
  if is_true "$FORCE"; then rm -rf "$stage_base"; fi
  mkdir -p "$staged_root/train2017" "$staged_root/panoptic_train2017"
  ln -sfn "$assets/annotations" "$staged_root/annotations"
  # Hard-link regular COCO RGBs; dangling Objects365 links are removed below.
  if [[ ! -f "$marker" ]]; then cp -al "$assets/train2017/." "$staged_root/train2017/"; fi
  if [[ ! -f "$image_marker" ]]; then
    find "$staged_root/train2017" -maxdepth 1 -type l -name 'objects365_v2_*.jpg' -delete
    echo "Recovering Objects365 RGB patches into $staged_root/train2017"
    while IFS= read -r archive; do tar --skip-old-files --strip-components=1 -xzf "$archive" -C "$staged_root/train2017"; done < <(find "$archives_dir" -maxdepth 1 -type f -name 'patch*.tar.gz' -printf '%p\n' | sort)
    touch "$image_marker"
  fi
  if [[ ! -f "$mask_marker" ]]; then
    archive="$assets/panoptic_train2017.tar"
    [[ -f "$archive" ]] && tar -xf "$archive" -C "$staged_root"
    mkdir -p "$staged_root/panoptic_train2017"
    find "$staged_root/panoptic_train2017" -maxdepth 1 -type l -name 'objects365_v2_*.png' -delete
    echo "Recovering Objects365 panoptic masks into $staged_root/panoptic_train2017"
    tar --wildcards --strip-components=2 -xf "$xl_tar" -C "$staged_root/panoptic_train2017" 'coconuts_xlarge/panseg/*.png'
    tar --skip-old-files --strip-components=1 -xf "$large_tar" -C "$staged_root/panoptic_train2017"
    [[ -f "$staged_root/panoptic_train2017/objects365_v2_00940000.png" ]] || { echo "ERROR: Objects365 mask recovery failed" >&2; exit 1; }
    touch "$mask_marker"
  fi
  touch "$marker"
  COCONUT_XL_ROOT=$staged_root
}

