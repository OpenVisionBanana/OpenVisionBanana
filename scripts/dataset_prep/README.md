# Dataset preparation internals

Start with [DATASET.md](../../DATASET.md) for download and preparation commands.
This page describes the catalog, converters and archive helpers used by
`scripts/setup_data.py`. To run its stages separately, use
`scripts/download_data.py` and `scripts/prepare_data.py`.

## On this page

| Section | What you'll find |
|---|---|
| [Downloading](#how-downloading-works) | Catalog, YAML controls and [gated sources](#gated-sources) |
| [Preparation](#how-preparation-works) | Pool construction, converters and publication safety |
| [Geometry](#geometry-data) | Current route, indexing and the legacy local-package installer |
| [Archive layout](#prepared-archive-layout) | Metadata, sidecars and custom training data |
| [Converter details](#converter-details) | Semantic variants, instance targets and referring evaluation masks |
| [Storage and downloads](#storage-and-download-details) | Packed inputs, temporary space, retries and source exceptions |
| ↳ [Release checks](#checking-the-release-files) | Build and verify an isolated release candidate |
| [Normal maintenance](#normal-maintenance-tools) | Repair utilities and reference-target builders |
| [Legacy creative inputs](#legacy-creative-inputs) | WebDataset wrappers and Pico bulk archives |
| [Manual benchmark acquisition](#manual-author-benchmark-acquisition) | Alternative author download and extraction commands |
| [Individual CLI examples](#individual-download-and-pool-examples) | Pool-only builds, subsets and worker controls |

## How downloading works

The catalog in [common/release_data.py](../../common/release_data.py) and its
[official-source adapters](../../common/official_sources.py) define raw folders,
acquisition helpers and required files. Pool selections expand into source datasets.
Use the [public commands](../../DATASET.md#how-to-download-the-data) for acquisition.

In YAML, `acquisition` controls downloads and `preparation` controls conversion;
CLI options take precedence. `configs/data.yaml` is a small raw-conversion example.
Downloads need network access; preparation works offline once all assets and
dependencies are present. Missing restricted inputs return exit code 2 after
independent downloads finish; combined setup stops before preparation.

### Gated sources

| Dataset | Official download | Put the files in | Builds there |
|---|---|---|---|
| Cityscapes | [cityscapes-dataset.com](https://www.cityscapes-dataset.com/downloads/): `gtFine_trainvaltest.zip`, `leftImg8bit_trainvaltest.zip` | `Semseg/cityscapes/` | `cityscapes.tar` with `gtFine/`, `leftImg8bit/` |
| ACDC | [acdc.vision.ee.ethz.ch](https://acdc.vision.ee.ethz.ch/download): `rgb_anon_trainvaltest.zip`, `gt_trainval.zip`, `gt_panoptic_trainval.zip` | `Shared_seg/acdc/` | `acdc.tar` |
| BDD100K 10K seg | [bdd-data.berkeley.edu](http://bdd-data.berkeley.edu/): `bdd100k_seg.zip` (legacy, used for the paper) or `bdd100k_images_10k.zip` + `bdd100k_sem_seg_labels_trainval.zip` | `Semseg/bdd100k/` | `bdd_10k.tar` |
| Mapillary Vistas | [mapillary.com/dataset/vistas](https://www.mapillary.com/dataset/vistas): research-edition ZIP | `Shared_seg/mapillary_vistas/` | `mapillary.tar` |
| IDD | [idd.insaan.iiit.ac.in](https://idd.insaan.iiit.ac.in/): `idd-segmentation.tar.gz`, `idd-20k-II.tar.gz` | `Shared_seg/idd/` | `idd_1_2.tar` |
| nuImages | [nuscenes.org/nuimages](https://www.nuscenes.org/nuimages): `nuimages-v1.0-all-metadata.tgz`, `nuimages-v1.0-all-samples.tgz` | `Shared_seg/nuimages/` | `nuimages.tar` |
| WildDash 2 | [wilddash.cc](https://www.wilddash.cc/download): `wd_public_v2p.zip` | `Shared_seg/wilddash2/` | `wd_public_v2p.tar` |
| RailSem19 | [wilddash.cc/railsem19](https://www.wilddash.cc/railsem19): `rs19_val.zip` | `Semseg/railsem19/` | `railsem19.tar` |
| Waymo | accept [waymo.com/open/terms](https://waymo.com/open/terms), then `gcloud auth login` | (downloaded by the script) | `Shared_seg/waymo/waymo.tar` |
| ReasonSeg | [LISA Google Drive](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy): `train.zip`, `test.zip`, `val.zip` | `Refseg/reasonseg/reasonseg_train_test/` (train, test), `Refseg/reasonseg/reasonseg_val/` (val) | `{train,test}.tar`; val extracted in place |
| SA-Co/Gold | from its authors | `Instanceseg/saco_gold/` (`gt-annotations/`, `metaclip-images/`, `sa1b-images/`) | — |

Cityscapes uses `val` for evaluation and only `train` in the non-zero-shot pool.
Current setup can keep its raw TAR packed; see the main guide for accepted
layouts. Geometry sources go under `Shared_geo/`. Their required inputs are in
[DATASET.md](../../DATASET.md#geometry-source-requirements).

## How preparation works

`scripts/prepare_data.py` (invoked by `setup_data.py`) writes
`prepared/V1_prepared_seg_geo/`: benchmark layouts under `eval/<Task>/` via
[`prepare_task_eval.py`](prepare_task_eval.py), geometry tars via the
per-source depth/normal converters, and the training pools via
[`build_training_pool.py`](build_training_pool.py), which drives the pool
pipeline with the paper settings:

1. `build_training_pool.py` runs the converters in
   [`dataset_prep/`](../../dataset_prep/) with the selected pool settings.
   Sources that need extraction use a temporary workspace selected by
   `--staging-root` or the system default. Pico editing reads packed raw shards;
   identity reads original conditioning images from the indexed Pico archive.
   PD12M uses the public preparation adapter. Missing required inputs prevent
   a complete pool. Reruns reuse existing members; `--overwrite` rebuilds them.
2. Every member is published as one uncompressed per-source tar with a SQLite
   byte-offset index ([`archive_prepared_member.py`](archive_prepared_member.py) /
   [`build_prepared_tar_index.py`](build_prepared_tar_index.py)), validated by
   [`validate_dataset_specs.py`](validate_dataset_specs.py): semseg/instanceseg
   are split per source
   ([`split_prepared_member_by_source.py`](split_prepared_member_by_source.py)),
   refseg is built per annotation source, and the editing/identity/generation
   members are single tars.
3. A pool is only a manifest (`manifests/<pool>_pool.json`) selecting member
   tars, each weighted by its task's paper mass split by prepared row count.
   Members are shared, so another pool, including a custom one from
   `--pool <name> --sources …` (see `--list-sources`), converts only the
   sources no earlier pool built. Weights and `samples_per_epoch` are
   training-time sampling parameters, not baked into the images, so a pool can
   be re-weighted without rebuilding.

Paired members carry `conditioning_image`, `image` and `text`, with task/source/ID
metadata and masks where needed. Generation rows use an image/caption.
The [archive layout](#prepared-archive-layout) defines stored paths and sidecars.

Creative outputs are built and indexed before replacement. Conversion/index
failures preserve old outputs and the manifest; handled replacement failures
roll back. Stop readers and allow room for replacements. Forced termination
can interrupt multi-file publication.

**Converter behaviour** (the paper settings are fixed by
`build_training_pool.py`): semseg diversifies each image into randomized
(prompt, mask) variants — class subsets, full sets and negative sets, colours
sampled from the committed palettes in
[`common/semseg/diversify.py`](../../common/semseg/diversify.py) with CIELAB
separation. Instanceseg paints every instance of one sampled noun phrase a
distinct colour from the committed ordered palette with an explicit legend,
sharing its prompt builder with the evaluator
([`common/instanceseg/prompts.py`](../../common/instanceseg/prompts.py)).
Refseg prepares SteerViT rows (the images come from Visual Genome, Mapillary
and COCO-2014); evaluation-holdout images (RefCOCOg UMD val, COCO val2017)
are excluded from every COCO-backed member. Each converter's flags are
documented in its own `--help`.

**Previewing a dataset tar:** every large raw archive has one
`preview_<dataset>_tar.py` helper (Synscapes, IDD, nuImages, COCONut-XL,
PACO-LVIS, WildDash 2) with the same interface — it streams a handful of
frames out of the raw archive, runs the production converters on just those,
and writes an HTML gallery (conditioning image, prompt, prepared target). A
cheap sanity check before a full pool build; `selected_samples.json` records
the picks, and `--sample-ids`/`--sample-keys` reproduces them:

```bash
python scripts/dataset_prep/preview_<dataset>_tar.py \
  --<dataset>-tar /data/ovb/raw/<Group>/<dataset>/<archive>.tar \
  --out-root /tmp/<dataset>_preview --num-samples 4
```

## Geometry data

Use the ZS or NZS configuration in [DATASET.md](../../DATASET.md#complete-zs-and-nzs-recipes)
to acquire original geometry sources and prepare training and evaluation data.
Both model sizes share the same data. Training filters and masks are applied
during preparation; evaluation preserves the author benchmark targets.
The base-clean filter rejects insufficient valid coverage, undefined normals
marked valid, low-information maps and highly irregular maps. The loader does
no additional dataset filtering. Metadata may reference `samples/<id>/` TAR paths.
The old `--config configs/data_published.yaml --pool zeroshot|base|added_only`
route is retired and refuses acquisition.

<details>
<summary>Legacy local geometry-package installer</summary>

The installer below supports existing local geometry packages. It does not
download the old test uploads. Point `GEOMETRY_RELEASE_ROOT` at your existing
directory containing `RELEASE_MANIFEST.json` and its ZIPs. Choose a separate
`PREPARED_ROOT` for this older layout:

```bash
python scripts/dataset_prep/install_geometry_release.py \
  --release-root "$GEOMETRY_RELEASE_ROOT" \
  --destination-root "$PREPARED_ROOT" --profile all --plan
python scripts/dataset_prep/install_geometry_release.py \
  --release-root "$GEOMETRY_RELEASE_ROOT" \
  --destination-root "$PREPARED_ROOT" --profile all
```

It checks the release manifest, ZIP CRCs, package manifests, byte counts and
SHA-256 records before atomically publishing the inner tars, and writes
`geometry_release_installation.json` (`--verify-only` re-checks an install;
`--force` only when intentionally replacing a mismatched asset). Profiles:
`full` (all benchmarks under `{depth,surface_normals}/`, including the official
iBims-1 archives and depth-only KITTI Eigen), `ablation` (deterministic sample
manifests under `profiles/ablation/`), `training` (paired geometry tars,
shipped indexes and `geometry_mix.json` under `training/`), and `all`. The old
`ibims1.tar` names remain only for explicit legacy `nyu_10m` protocol runs;
KITTI Eigen is intentionally not a surface-normal benchmark.

</details>

Training reads uncompressed prepared TARs without extraction. Setup builds
their indexes automatically. To index and register one yourself:

```bash
python scripts/dataset_prep/build_prepared_tar_index.py /prepared/depth.tar
python scripts/training/add_dataset_spec.py \
  --config /prepared/mix.json --name depth --weight 1 \
  --type indexed_prepared_tar --path /prepared/depth.tar \
  --index-path /prepared/depth.tar.ovb-index.sqlite
```

The default index includes SHA-256 verification, so copying the tar while
changing only its timestamp does not invalidate it. Compressed tars cannot be
indexed and should be extracted normally.

Model access is separate from dataset setup. See
[`scripts/models/README.md`](../models/README.md).

Detailed converter notes: [depth](../../dataset_prep/depth/README.md),
[surface normals](../../dataset_prep/surface_normals/README.md), and
[PD12M](../../dataset_prep/pd12m/README.md).

## Prepared archive layout

Training TARs contain `metadata.jsonl` and have a byte-offset
`.ovb-index.sqlite` sidecar. Optional `dataset_info.json` records provenance
and encoding. Image folders vary; paired segmentation data commonly uses:

```text
train/Semseg/
├── ade20k.tar   ade20k.tar.ovb-index.sqlite   ade20k.tar.sha256
├── goose.tar    goose.tar.ovb-index.sqlite    goose.tar.sha256
└── ...

ade20k.tar
└── ade20k/
    ├── metadata.jsonl               # one JSON row per training pair
    ├── dataset_info.json            # provenance record (optional; not needed for training)
    ├── conditioning_images/*.png    # input images
    └── target_images/*.png          # coloured segmentation targets
```

`.sha256` verifies one TAR; the version's `SHA256SUMS` lists all train/eval TARs.
Manifest paths are relative, so a prepared tree moves as a whole. Pools share
member archives; building another converts only missing sources. For custom
fine-tuning, index each source TAR, list it in a weighted mix and follow
[TRAIN.md](../../TRAIN.md).

## Converter details

The converters under `dataset_prep/` turn each native source into one pool
member with `metadata.jsonl`, `conditioning_images/` and `target_images/`.
Three of them have behaviour that the YAML recipes do not expose.

### Semantic segmentation variants

[`dataset_prep/semseg/dataset_prep.py`](../../dataset_prep/semseg/dataset_prep.py)
writes `SEMSEG_VARIANTS_PER_IMAGE` randomized prompt and mask pairs per image.
Each variant falls into one of three categories, drawn per variant with the
ratios `--subset-ratio`, `--full-set-ratio` and `--negative-set-ratio`, which
must sum to one. A subset variant lists a random proper subset of the classes
present, weighted towards rare classes. A full-set variant lists all classes
present. A negative-set variant adds `K` absent classes from the dataset
vocabulary, with `K` uniform between 1 and `--max-negative-classes`, which are
named in the prompt with a colour but never painted. Only the selected present
classes are painted and the rest of the mask stays black, unless
`SEMSEG_BACKGROUND_COLOR_PROB` recolours the background. The category is
recorded in the metadata as `variant_type`. `SEMSEG_CANONICAL_ANCHOR=1` also
writes the canonical pair, the full 19-class mask in the standard Cityscapes
palette with the fixed prompt, for datasets that use Cityscapes train ids. The
chosen settings are stored next to the member in `diversify_config.json`.

| Variable | Default | Meaning |
|---|---|---|
| `SEMSEG_VARIANTS_PER_IMAGE` | `1` | randomized prompt and mask variants per image |
| `SEMSEG_CANONICAL_ANCHOR` | `0` | also write the canonical fixed-palette pair |
| `SEMSEG_SUBSET_RATIO` | `1.0` | fraction of variants with a proper subset of the present classes |
| `SEMSEG_FULL_SET_RATIO` | `0.0` | fraction of variants with all present classes |
| `SEMSEG_NEGATIVE_SET_RATIO` | `0.0` | fraction of variants that add absent classes |
| `SEMSEG_MAX_NEGATIVE_CLASSES` | `30` | cap on absent classes per negative-set variant |
| `SEMSEG_BACKGROUND_COLOR_PROB` | `0.0` | probability that the background gets a random colour instead of black |
| `SEMSEG_INVERSE_FREQ_POWER` | `1.0` | rare-class sampling boost (0 is uniform) |
| `SEMSEG_COLOR_MIN_DELTA_E` | `25.0` | minimum CIELAB distance between the colours of one mask |
| `SEMSEG_MIN_CLASS_AREA` | `0.0` | drop classes below this fraction of the image area |

Colours and notation come from
[`common/semseg/diversify.py`](../../common/semseg/diversify.py), which holds two
committed palettes. `palette_200.json` provides 200 well-separated colours for the
`rgb` and `hex` notations, and `palette_named.json` provides 37 common colour
names for the `name` notation, which is used only when at most 37 colours are
needed. Each prompt uses one notation throughout, and the colours of one mask are
sampled with a greedy CIELAB separation of at least `SEMSEG_COLOR_MIN_DELTA_E`.
The semantic segmentation evaluator draws its random-draw legends from the same
200-colour palette.

### Instance segmentation targets

[`dataset_prep/instanceseg/dataset_prep.py`](../../dataset_prep/instanceseg/dataset_prep.py)
keeps one randomly sampled class per image and paints every instance of that
class in its own colour on a black background. Colours are chosen by
farthest-point sampling in CIELAB
([`common/instanceseg/colors.py`](../../common/instanceseg/colors.py)) so that
they remain separable when the evaluator decodes the output. A fraction of the
variants, `INSTANCESEG_NEGATIVE_RATIO` (default 0.2), name a class without
instances in the image and have an all-black target, which teaches the model to
paint nothing for an absent category. Prompts use the templates in
[`common/instanceseg/prompts.py`](../../common/instanceseg/prompts.py), the same
ones the evaluator uses. Source decoders and name tables are in
[`dataset_prep/instanceseg/instance_taxonomy.py`](../../dataset_prep/instanceseg/instance_taxonomy.py).
Further settings are `INSTANCESEG_VARIANTS_PER_IMAGE` (default 1),
`INSTANCESEG_MAX_INSTANCES` and `INSTANCESEG_MIN_INSTANCE_AREA`. Pascal VOC
instances need the `SegmentationObject` masks of the full `VOCdevkit/VOC2012`
release, which the semantic segmentation archive does not contain.

### Referring segmentation evaluation masks

[`dataset_prep/refseg/prepare_refseg_eval.py`](../../dataset_prep/refseg/prepare_refseg_eval.py)
rasterizes the two held-out benchmarks with their canonical tools, so the masks
match the published ground truth exactly.

| Dataset | Split | Images | Referred objects | Expressions | Mask values | Rasterizer |
|---|---|---|---|---|---|---|
| [ReasonSeg](https://github.com/dvlab-research/LISA) | val | 200 | 340 | 344 (113 long reasoning queries, 87 short phrases) | `{0, 1, 255}` | `cv2.polylines` and `fillPoly`, as in LISA |
| [RefCOCOg](https://github.com/lichengunc/refer) (UMD split) | val | 1,300 | 2,573 | 4,896 sentences | `{0, 1}` | pycocotools `frPyObjects` and `decode` |

ReasonSeg masks use 0 for background, 1 for the target and 255 for ignore
regions, which appear in 76 of the 200 images. The evaluator excludes value 255
from both intersection and union. Output goes to `reasonseg_val/masks/<stem>.png`.
RefCOCOg UMD val is read from `refs(umd).p` and `instances.json` with images
from `train2014/`. One binary mask is written per referring expression, keyed by
`ref_id`, together with `refcocog_val_umd.json`, which records each reference's
`ann_id`, `image_id`, image path, size, `category_id` and sentences. Scores are
reported over the 4,896 sentences. Output goes to
`refcocog/val_umd/{masks/<ref_id>.png, refcocog_val_umd.json}`.

```bash
python dataset_prep/refseg/prepare_refseg_eval.py reasonseg
python dataset_prep/refseg/prepare_refseg_eval.py refcocog
```

## Storage and download details

Hypersim acquisition retains scene ZIPs. Both its geometry reader and its
segmentation reader load individual HDF5 channels without scene extraction.
Use either archived or extracted inputs for a frame; duplicate copies cause
an error.

Preparation writes packed training archives and indexes to the destination.
Interrupted stages can rebuild temporary work while reusing completed stages.
Fresh Replica and Synscapes downloads still extract into their raw directories.
Replica retains both the download parts and their combined archive, so allow
space for those copies as well as the extracted assets.

Evaluation archives use a locked cache in the system temporary directory.
The launcher does not extract them into the result folder. Temporary cache
files can persist between invocations until the operating system removes them.

Checks reject insufficient reported bytes or entries before known-size archive
extraction. General conversion starts with a 64 MiB free-space reserve. Set
`OVB_SCRATCH_RESERVE_BYTES` only if you want to increase that optional reserve.
This check does not estimate the full output size or reserve space across
workers. Try a small conversion before increasing concurrency.

Existing original-layout `raw/Shared_geo/synscapes/synscapes.tar` inputs can remain
packed. The acquisition helper checks matching RGB, depth, class, instance and
metadata frame sets. The geometry adapter selects its existing TAR readers.
The pool stages that TAR in its temporary working directory. Target algorithms
and split settings do not change. These are header and adapter checks, not a
fresh full-source acceptance run.

RefCOCOg acquisition now retains the original COCO `train2014.zip` instead of
extracting photos onto persistent storage. It reuses the shared COCO ZIP through
a relative link between persistent input folders. It retains the small UMD
annotation files and checks the complete official validation-image ID hash.
Preparation extracts only the validation photos into its temporary workspace, invokes the
existing mask converter, then publishes `eval/Refseg/refcocog.tar` with real
image bytes, masks and the index. No prepared link points into temporary files.
The public evaluator supplies the extracted image root instead of the old paths
stored in that index. Existing loose raw layouts remain supported and unchanged.
The training pool uses temporary COCO staging, with the RefCOCOg
ZIP as a fallback when the shared train-image ZIP is absent.

Use the original annotation ZIP from the refer project. The downloader uses the
raw Wayback snapshot linked by that project's README issue because the original
university server is unavailable at this check. A normal Wayback viewer URL can
return HTML instead of ZIP bytes. `REFCOCOG_ZIP_URL` can select another original
archive URL. Do not use an OVB-prepared substitute or untrusted pickle files.
An HTTP success code alone does not establish a valid source ZIP.
The photo ZIP default uses
`https://s3.amazonaws.com/images.cocodataset.org/zips/train2014.zip`.
It addresses the same public COCO bucket with a valid HTTPS certificate.
The `https://images.cocodataset.org` hostname failed certificate hostname
verification in the 28 October probe. Do not disable certificate checks.
The official validation check passed with all 1,300 RefCOCOg UMD validation photos, 2,573 masks and 4,896
expressions. It checks exact converter output and a moved-TAR evaluator dry-run,
not model predictions or complete COCO training acquisition.

Do not replace existing inputs with persistent links into temporary directories.

The shell layer uses `OVB_DOWNLOAD_ATTEMPTS` (default 3) and
`OVB_DOWNLOAD_RETRY_DELAY` (default 60 seconds). The shared Python downloader
makes up to three attempts for temporary network
failures and resumes partial downloads. Delays start at one second and follow
`Retry-After` up to 60 seconds. Authentication, missing files, certificate and
validation errors fail immediately. External download tools keep their own
retry settings.

If a container inherits an invalid certificate path, set `SSL_CERT_FILE` to
its CA bundle. Keep TLS verification enabled.

The standalone PD12M materializer uses a maximum count unless `--require-count`
is set. Public setup requires the requested number of complete examples.
Pico raw shards contain at most 256 rows, image hashes and completion receipts.

NZS ScanNet training downloads only small pinned exclusion lists under
`raw/Shared_geo/geometry_splits/data_split/`, not benchmark image archives.
Matterport3D accepts official same-view RGB/depth at twice the normal dimensions;
metric depth is resized with nearest-neighbor sampling. Unexplained mismatches fail.

### Checking the release files

`DATA_RELEASE_FILES.json` lists the changes and additional files required by
this workflow, including the shared geometry and benchmark helpers. Build an
isolated candidate with:

```bash
python scripts/dataset_prep/check_data_release.py --destination ./release_candidate
```

Choose a destination that does not exist. The command combines the Git base
with the listed files, then checks the CLIs and offline tests from the candidate.
It records the candidate file hashes in `RELEASE_SNAPSHOT.json`. It does not
stage or commit files in the source repository. Review the inventory when
preparing the Git release.

## Normal maintenance tools

<details>
<summary>Maintainer utilities and reference-target builders</summary>

NYUv2 and Synscapes retain standalone preparation wrappers. The
`repair_surface_normals_unit_archive.py` utility repairs a known non-unit-vector
scene by writing a new archive and checksum; it refuses existing outputs and is
not part of normal setup.

`scripts/analysis/build_reference_normal_subsets.py` builds canonical float-XYZ
references for NYUv2, iBims-1, DIODE, ScanNet, VKITTI2 and Sintel. Use `--limit 0`
for the complete upstream split.

Existing geometry-v6 packages install official DSINE/GeoNet NYUv2 test data as
`surface_normals/nyu.tar` and official iBims-1 as
`surface_normals/ibims1_official.tar`. Full and ablation profiles are described
above; KITTI Eigen has no normal counterpart.

</details>

## Legacy creative inputs

<details>
<summary>Legacy WebDataset and bulk-archive inputs</summary>

1. `download_pd12m.sh` selects/downloads source rows and creates resumable
   WebDataset shards under `RAW_ROOT/pd12m/<preset>`.
2. `prepare_pd12m.sh` works offline, validates those shards, and publishes a
   dataset-specs JSON under `PREPARED_ROOT/pd12m/<preset>`.

```bash
RAW_ROOT=/persistent/raw PRESET=500k scripts/dataset_prep/download_pd12m.sh
RAW_ROOT=/persistent/raw PREPARED_ROOT=/persistent/prepared PRESET=500k \
  scripts/dataset_prep/prepare_pd12m.sh
```

These wrappers validate and reference WebDataset shards without copying the
images into a prepared TAR. `PRESET`, `WORKERS` and optional `METADATA_JSONL`
control their acquisition. For `scripts/setup_data.py` (`--datasets pd12m`, or
as a source of `zeroshot_pool`), use the public workflow described above.

### Pico bulk-archive converter

The older Pico bulk-archive converter reads `openimages/sft_with_local_source_image_path.jsonl`, with
257,730 rows resolving `sft.jsonl` URLs to local paths. Its input photos come
from `train_0` (156,070) and `train_1` (101,660), with `positive-edit` targets.
For that route, `PICO_SOURCE_ROOT` selects an extracted checkout and
`PICO_LIMIT=50000` selects a seed-42 subset. Increasing it toward 257k uses more
of the same SFT data. Preference and multi-turn training require their own images.


| Earlier local subset | Examples | Metadata | Image assets |
|---|---|---|---|
| Single-turn SFT | ~257,730 | `sft.jsonl` | `images/positive-edit.tar` (~366 GiB) and source shards |
| Preference | ~56,473 | `preference.jsonl` | Metadata only in the older setup |
| Multi-turn SFT | ~77,250 | `multi-turn.jsonl` | Metadata only in the older setup |

</details>

## Manual author benchmark acquisition

<details>
<summary>Alternative: acquire and extract author benchmarks manually</summary>

Public setup fetches these assets automatically. The alternative below follows
the author instructions for an extracted benchmark root.

Install `requirements/geometry.txt` for the PPD scorer. Obtain the benchmark
data and pin the [upstream split lists](https://github.com/huawei-bayerlab/marigold-v2/tree/18466672fb8661152434a0efe1184939164cda19/evaluation/data_split):

```bash
python -m pip install -r requirements/geometry.txt
git clone https://github.com/huawei-bayerlab/marigold-v2.git /repos/marigold-v2
git -C /repos/marigold-v2 checkout 18466672fb8661152434a0efe1184939164cda19
mkdir -p /data/marigold_eval
(
  set -e
  cd /data/marigold_eval
  wget -r -np -nH --cut-dirs=4 -R 'index.html*' \
    https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/
  find . -type f -name '*.tar' -exec sh -c 'tar -xf "$1" -C "$(dirname "$1")"' sh {} \;
)
```

Choose writable locations in place of `/repos` and `/data`. The extracted root
must contain `nyuv2/`, `kitti/`, `eth3d/`, `scannet/` and `diode/`, with members
matching the split lists. Counts are respectively **654, 652, 454, 800, 771**.
The download follows the [upstream evaluation instructions](https://github.com/huawei-bayerlab/marigold-v2/blob/18466672fb8661152434a0efe1184939164cda19/README.md#evaluation);
dataset access/use conditions still apply.

</details>

## Individual download and pool examples

The [main guide](../../DATASET.md) covers source listing, status and descriptions.
Use `--datasets all` for catalogue-wide acquisition or a pool name for its sources.
The following builds only the other-task pool, without geometry:

```bash
python scripts/download_data.py --destination ./data/raw \
  --datasets zeroshot_pool --dry-run
python scripts/prepare_data.py --raw-root ./data/raw \
  --destination ./data/prepared --datasets zeroshot_pool --splits train --seed 42
```

Replace `zeroshot_pool` with `nonzeroshot_pool` for the other pool. Members are
reused; the respective outputs are `manifests/zeroshot_pool.json` and
`manifests/nonzeroshot_pool.json`. Remove `--dry-run` to download.
A bounded pool check uses a separate destination:

```bash
python scripts/prepare_data.py --raw-root ./data/raw \
  --destination ./data/prepared_smoke --datasets zeroshot_pool --splits train \
  --seed 42 --set zeroshot_pool.limit=64 --set zeroshot_pool.semseg-num-images=64
```

Override per-source settings and converter concurrency with:

```bash
python scripts/prepare_data.py --raw-root ./data/raw \
  --destination ./data/prepared --datasets hypersim,taskonomy \
  --splits train --seed 42 --max-samples 100000 \
  --set hypersim.samples-per-scene=50 --dataset-workers 3 --workers 32
```

To build a custom other-task subset:

```bash
python scripts/dataset_prep/build_training_pool.py --list-sources
python scripts/dataset_prep/build_training_pool.py --pool my_pool \
  --sources goose,ovis,vg_mapillary,pico_banana,identity,pd12m \
  --raw-root ./data/raw --root ./data/prepared/custom --staging-root ./work
```
