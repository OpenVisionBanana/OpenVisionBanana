# Datasets

Use `scripts/setup_data.py` to download and prepare data. The ZS and NZS
recipes are shared by the 4B and 9B models. Run commands from the repository
in the environment described in [MAIN.md](MAIN.md#installation).

## On this page

| Section | What you'll find |
|---|---|
| [ZS/NZS setup](#complete-zs-and-nzs-recipes) | Complete recipes and training manifests |
| [Sources](#dataset-source-policy) | Download policy and [dataset catalog](#datasets-at-a-glance) |
| [Download and prepare](#how-to-download-the-data) | Manual inputs and [conversion options](#how-to-prepare-the-data) |
| [Geometry](#geometry-preparation) | Sources, targets and filtering |
| ↳ [Restricted access](#manual-geometry-downloads) | Matterport3D and ScanNet files and destinations |
| ↳ [Replica dependencies](#replica-ptex-renderer) | Renderer installation |
| [Generation and editing](#image-generation-and-editing) | PD12M, Pico-Banana and identity data |
| [Outputs and recovery](#training-and-evaluation-data-layout) | File layout, [checksums](#validating-prepared-data), [resume](#resuming-preparation) and [storage](#archives-and-temporary-storage) |
| [Evaluation data](#evaluation-benchmarks) | Benchmark acquisition and preparation |
| [Small tests](#small-smoke-tests) | Bounded downloads and CPU loader checks |
| [Optional workflows](#optional-workflows-and-reference) | Standalone conversion, existing installations and storage controls |

## Complete ZS and NZS recipes

Before a complete run, obtain the [restricted originals](#how-to-download-the-data)
and install [Replica's renderer dependencies](#replica-ptex-renderer).
Choose a writable data root; ZS and NZS share raw downloads but have separate outputs.

```bash
# Inspect prerequisites without downloading or preparing data.
python scripts/setup_data.py --config configs/data_zs.yaml --data-root ./data --dry-run

# Download and prepare either recipe.
python scripts/setup_data.py --config configs/data_zs.yaml --data-root ./data
python scripts/setup_data.py --config configs/data_nzs.yaml --data-root ./data
```

Missing restricted inputs cause setup to print required files and absolute
destinations, download independent public sources, then stop before preparation.
Obtain access, supply the originals and rerun the same command. The scripts do
not accept agreements for you. Exit code 2 means acquisition is incomplete.

Use the complete manifest for training:

| Recipe | Prepared directory below `./data/prepared/` | Training manifest within it |
|---|---|---|
| ZS | `V1_prepared_seg_geo/` | `manifests/zs_training.json` |
| NZS | `V1_prepared_seg_geo_nzs/` | `manifests/nzs_training.json` |

These manifests include geometry; `zeroshot_pool.json` and
`nonzeroshot_pool.json` contain only the other tasks. Sampling weights and
training commands are in [TRAIN.md](TRAIN.md).

<details>
<summary>Optional: download and prepare in separate steps</summary>

To run acquisition and preparation separately:

```bash
python scripts/download_data.py --config configs/data_zs.yaml --data-root ./data
python scripts/prepare_data.py --config configs/data_zs.yaml --data-root ./data
```

</details>

CLI options override YAML values; relative paths resolve from your working
directory. See the [configuration reference](scripts/dataset_prep/README.md#how-downloading-works)
for acquisition/conversion controls and smaller examples.

## Dataset source policy

Datasets come from their authors or independent public distributions, including
[Spawning/PD12M](https://huggingface.co/datasets/Spawning/PD12M),
[Apple's Pico-Banana release](https://github.com/apple-aiml-research/pico-banana-400k)
and author-maintained Marigold benchmarks. Temporary OVB test uploads are
excluded. Project model downloads are separate from this dataset policy.
Dataset licenses apply to automatic downloads too.

## Datasets at a glance

<details>
<summary>Full dataset catalog and task-specific exceptions</summary>

`†` marks a dataset that is also a reported evaluation benchmark, and an
<u>underlined</u> dataset trains only the non-zero-shot pool. A linked
dataset needs a login or license on its official page; everything else
downloads automatically. `--describe <dataset>` prints the exact files and
target folder for any entry.

| Used for | Automatic download | Manual download (link = official page) |
|---|---|---|
| Training — both pools | ADE20K, PASCAL VOC 2012 + Context, GOOSE 2D, MUSES, VIPSeg, FoodSeg103, Trans10K-v2, OVIS, Visual Genome, Pico-Banana-400K, PD12M | [ACDC](https://acdc.vision.ee.ethz.ch/download), [BDD100K](http://bdd-data.berkeley.edu/), [Mapillary Vistas v2.0](https://www.mapillary.com/dataset/vistas), [Waymo Open camera-panoptic](https://waymo.com/open/terms), [RailSem19](https://www.wilddash.cc/railsem19) |
| Training — added for the non-zero-shot pool | <u>COCO 2014</u>, <u>COCO 2017</u>, <u>COCONut-XL</u>, Synscapes, <u>PACO-LVIS</u>, <u>RefCOCO</u>, <u>RefCOCO+</u>, <u>RefCOCOg</u> †, <u>gRefCOCO</u>, <u>LVIS</u> (COCO-2014 images), <u>PhraseCut</u> (Visual Genome images) | <u>[Cityscapes](https://www.cityscapes-dataset.com/downloads/)</u> †, <u>[IDD Segmentation](https://idd.insaan.iiit.ac.in/)</u>, <u>[nuImages](https://www.nuscenes.org/nuimages)</u>, <u>[WildDash 2](https://www.wilddash.cc/download)</u>, <u>[ReasonSeg](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy)</u> † |
| Geometry training — both pools | Hypersim, Replica, Synscapes, Taskonomy | [Matterport3D](https://niessner.github.io/Matterport/) |
| Geometry training — added for the non-zero-shot pool | NYUv2, DIODE, Virtual KITTI 2 | [ScanNet v2](https://github.com/ScanNet/ScanNet#scannet-data) |
| Evaluation only | COCO-Object-80, ImgEdit-Bench, GenAI-Bench, Marigold depth/normal benchmarks, DSINE VKITTI targets, DIODE normals | [SA-Co/Gold](https://huggingface.co/datasets/facebook/SACo-Gold), [nuScenes](https://www.nuscenes.org/nuscenes#download) (optional) |

The default geometry evaluation uses the author-distributed benchmark assets.
The separate raw KITTI, ETH3D and iBims-1 entries remain available for manual
conversion, but are not prerequisites for that evaluation route.

Exceptions encoded in the catalog: COCO-image rows are excluded from Visual Genome during zero-shot training. Hypersim feeds segmentation in both pools as well as geometry. Synscapes feeds segmentation
only in the non-zero-shot pool but geometry in both. RefCOCOg is downloaded for
the zero-shot build too, yet only its `refs(umd).p` evaluation list is
read there — no RefCOCOg rows enter the zero-shot training pool.

</details>

## How to download the data

The combined recipe above is the recommended route. For individual sources,
use dataset names or aliases (`nyu`, `bdd`, `mapillary`, `saco`, `coconut`,
`pico`). A pool name expands into its sources.

```bash
python scripts/download_data.py --list
python scripts/download_data.py --describe matterport3d
python scripts/download_data.py --destination ./data/raw --datasets pd12m,pico_banana
python scripts/download_data.py --destination ./data/raw --datasets all --status
```

Downloads resume and reuse completed sources. Several sources require
registration or author approval before download, among them Cityscapes,
Mapillary Vistas, Waymo Open, nuImages, Matterport3D and ScanNet. Follow the
official page linked in the table below, place the originals below `./data/` as
listed and rerun. Matterport3D and ScanNet require user-approved access from
their authors, and their [access instructions and file layouts](#manual-geometry-downloads)
are given further down. When originals are missing, the setup scripts print the
missing inputs and their destinations, download the independent public sources,
and stop before preparation until the originals are supplied.
The [CLI reference](scripts/dataset_prep/README.md#individual-download-and-pool-examples)
includes catalogue-wide downloads, dry runs and pool-only preparation.

<details>
<summary>Manual downloads: required files and destinations</summary>

| Dataset (link = official page) | What to download | Put the files in |
|---|---|---|
| [ACDC](https://acdc.vision.ee.ethz.ch/download) | `rgb_anon_trainvaltest.zip`, `gt_trainval.zip`, `gt_panoptic_trainval.zip` | `raw/Shared_seg/acdc/` |
| [BDD100K](http://bdd-data.berkeley.edu/) | the 10K images and semantic-segmentation label zips | `raw/Semseg/bdd100k/` |
| [Mapillary Vistas v2.0](https://www.mapillary.com/dataset/vistas) | the research-edition ZIP | `raw/Shared_seg/mapillary_vistas/` |
| [Waymo Open camera-panoptic](https://waymo.com/open/terms) | accept the terms and `gcloud auth login`; the script then downloads for you | `raw/Shared_seg/waymo/` |
| [RailSem19](https://www.wilddash.cc/railsem19) | `rs19_val.zip` | `raw/Semseg/railsem19/` |
| [Cityscapes](https://www.cityscapes-dataset.com/downloads/) | `gtFine_trainvaltest.zip`, `leftImg8bit_trainvaltest.zip` | `raw/Semseg/cityscapes/` |
| [IDD Segmentation](https://idd.insaan.iiit.ac.in/) | `idd-segmentation.tar.gz`, `idd-20k-II.tar.gz` | `raw/Shared_seg/idd/` |
| [nuImages](https://www.nuscenes.org/nuimages) | `nuimages-v1.0-all-metadata.tgz`, `nuimages-v1.0-all-samples.tgz` | `raw/Shared_seg/nuimages/` |
| [WildDash 2](https://www.wilddash.cc/download) | `wd_public_v2p.zip` | `raw/Shared_seg/wilddash2/` |
| [ReasonSeg](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy) | `train.zip` and `test.zip` into `reasonseg_train_test/`, `val.zip` into `reasonseg_val/` | `raw/Refseg/reasonseg/` |
| [SA-Co/Gold](https://huggingface.co/datasets/facebook/SACo-Gold) | the annotation JSONs into `gt-annotations/`; the MetaCLIP images ([Roboflow](https://universe.roboflow.com/sa-co-gold)) into `metaclip-images/`; the SA-1B images (`sa_co_gold.tar` via [Meta's download page](https://ai.meta.com/datasets/segment-anything-downloads/)) into `sa1b-images/` | `raw/Instanceseg/saco_gold/` |
| [Matterport3D](https://niessner.github.io/Matterport/) | four ZIPs per scan: `undistorted_color_images.zip`, `undistorted_depth_images.zip`, `undistorted_normal_images.zip`, `undistorted_camera_parameters.zip` | `raw/Shared_geo/matterport3d/` |
| [ScanNet v2](https://github.com/ScanNet/ScanNet#scannet-data) | `scannet_frames_25k.zip`, `scannetv2_train.txt`, `scannetv2_val.txt` | `raw/Shared_geo/scannet/` |
| [nuScenes](https://www.nuscenes.org/nuscenes#download) | the trainval metadata, `samples/` and `sweeps/`, extracted | `raw/Shared_geo/nuscenes/` |
| [iBims-1](https://www.ibims1.org/) | `ibims1_core_mat.zip` and `imagelist.txt`; Marigold/DSINE normals are separate derived targets, not official normal GT | `raw/Shared_geo/ibims1/` |
| [KITTI depth](https://www.cvlibs.net/datasets/kitti/eval_depth.php) | `depth_selection.zip`; Eigen-652 evaluation instead uses the prepared Marigold benchmark | `raw/Shared_geo/kitti/` |
| [ETH3D](https://www.eth3d.net/datasets) | `two_view_training.7z` and `two_view_training_gt.7z` (legacy low-resolution adapter, not the Marigold 454-image benchmark) | `raw/Shared_geo/eth3d/` |

</details>

## How to prepare the data

Use the complete recipes above to prepare geometry and the other tasks together.

### Geometry preparation

ZS uses Hypersim, Matterport3D, Replica, Synscapes and Taskonomy. NZS adds
NYUv2 training IDs, DIODE training data, ScanNet training locations and non-clone
Virtual KITTI 2 variants. ScanNet excludes physical locations in either pinned
depth or normal benchmark list, including other captures of those locations.

Preparation writes targets and binary validity masks. The base-clean policy
trims unreliable borders and rejects low-coverage or unreliable normal maps.
Shared ZS sources use paired depth/normal crops; NZS additions are filtered
separately so a rejected normal does not discard valid depth. Crops update
intrinsics. Training filters do not alter evaluation targets.

Archives store metric RGB depth; training resizes valid metric fields before
encoding relative inverse grayscale. Normal targets use camera-facing
`vision_banana_x_left_red`. [TRAIN.md](TRAIN.md) defines the encoding and mask
flags. Source validity is required even when loss masking is disabled.

Hypersim uses all complete frames before filtering; Taskonomy selects 25,000
frames with seed 42; Matterport3D uses its balanced standard profile. Replica
limits poses, near-duplicate views and scene contributions. YAML settings and
filtering reports record selections, exclusions and crops. Rebuilt counts can
differ from historical subsets.

### Geometry source requirements

<details>
<summary>Source files, annotation types and NYUv2 labels</summary>

Paths below are relative to `<data-root>/raw/Shared_geo/`. The download command
prints each source's accepted files with `--describe DATASET`.

| Source | Acquisition | Raw directory | Training normals |
|---|---|---|---|
| Hypersim | Automatic original scene ZIPs | `hypersim/` | Supplied geometry annotations |
| Matterport3D | User-approved access and original modality ZIPs | `matterport3d/<scan_id>/` | Supplied same-view annotations and calibration |
| Replica | Automatic original release, renderer dependencies required | `replica/` | Derived from rendered depth |
| Synscapes | Automatic public release | `synscapes/` | Supplied geometry annotations |
| Taskonomy | Automatic author modality archives | `taskonomy/` | Supplied geometry annotations |
| NYUv2 | Automatic original MAT, official split and author DSINE ZIP | `nyuv2/`, `dsine/` | Supplied GeoNet labels and paired DSINE RGB; official train IDs only |
| DIODE | Automatic original training archives | `diode/` | Supplied float arrays; reproduced identity convention |
| ScanNet | User-approved access and original frame archive | `scannet/` | Derived from depth with source validity |
| Virtual KITTI 2 | Automatic original modality archives | `vkitti2/` | Derived from depth and calibration |

Metadata records the representation, camera convention and supplied/derived
status. Derived normals are not independent annotations. Evaluation sources
are documented in [EVALUATION.md](EVALUATION.md).

NYUv2 uses the 795 GeoNet training labels and paired RGB from the
[DSINE author bundle](https://github.com/baegwangbin/DSINE#readme), separate from
the 654 test images. Setup verifies the ZIP's pinned checksum without fetching
model checkpoints. If Drive throttles access, supply the author's `dsine_eval.zip`
under `raw/Shared_geo/dsine/` and rerun.

</details>

### Manual geometry downloads

<details>
<summary>Matterport3D and ScanNet: access, archives and file layouts</summary>

For Matterport3D, follow the [official access procedure](https://niessner.github.io/Matterport/).
Sign and send the Terms of Use agreement to `matterport3d@googlegroups.com`.
After approval, use the authors' downloader and keep these ZIPs for each scan:

```text
data/raw/Shared_geo/matterport3d/<scan_id>/
  undistorted_color_images.zip
  undistorted_depth_images.zip
  undistorted_normal_images.zip
  undistorted_camera_parameters.zip
```

Original per-scan TARs containing these four ZIPs are also accepted.

For ScanNet, follow the [official access procedure](https://github.com/ScanNet/ScanNet#scannet-data).
Submit the agreement with your institutional email to `scannet@googlegroups.com`.
Download the frame ZIP after approval and obtain both split lists from
`Tasks/Benchmark/` in the official repository:

```text
data/raw/Shared_geo/scannet/
  scannet_frames_25k.zip
  scannetv2_train.txt
  scannetv2_val.txt
```

Keep the frame ZIP packed; raw `.sens` scans alone are insufficient. Setup
downloads the small benchmark-exclusion lists automatically. After staging the
originals, rerun your complete recipe above. To prepare just these sources:

```bash
python scripts/setup_data.py --config configs/data_nzs.yaml --data-root ./data \
  --datasets matterport3d,scannet --splits train
```

A `--datasets` override selects standalone preparation, not a complete joint mix.

</details>

### Replica Ptex renderer

<details>
<summary>Install and configure the Replica renderer</summary>

Replica needs Habitat-Sim for camera sampling and geometry, plus the official
Replica SDK for textured RGB. Install Habitat-Sim, CMake, Make, a C++ compiler,
Eigen, EGL/OpenGL, GLEW, and JPEG/PNG development files in your environment.
The build script preserves caller `CXXFLAGS`, including a custom sysroot.

```bash
bash scripts/dataset_prep/build_replica_ptex.sh ./data/tools/replica
python scripts/setup_data.py --config configs/data_zs.yaml --data-root ./data
```

The renderer uses the pinned Replica SDK and checks RGB/depth registration
before exporting each view. The default recipe requires Ptex RGB; older
vertex-colour rendering is unsupported.
The default executable is under `<data-root>/tools/replica/renderer/ReplicaSDK/ReplicaRenderer`.
An absolute `replica.ptex-renderer` override selects that executable directly.
A relative override resolves from the selected data root.

</details>

### Image generation and editing

| Task | Source | Default training examples | Raw directory below `./data/` |
|---|---|---|---|
| Generation | PD12M author metadata and public source images | 250,000 complete image/caption pairs | `raw/ImgGeneration/pd12m/` |
| Editing | Apple Pico-Banana single-turn SFT | Full SFT subset | `raw/ImgEdit/pico_banana/` |
| Identity | Original conditioning images from the prepared Pico archive | 50,000 unique images, seed 42 | No additional download |

Both sources download automatically. Pico stops if a selected image is
unavailable. PD12M records download failures, but preparation fails if fewer
than the requested number of complete examples remain. Set
`preparation.pd12m.limit` for a smaller recipe.

Pico shards are resumable and read without full extraction; reruns verify
completed shards and rebuild interrupted ones. Keep smoke/full inputs separate.

Generation rows contain an image/caption; editing rows contain conditioning and
target images plus an instruction. Indexed TARs appear under
`train/ImgGeneration/` and `train/ImgEdit/`. Try the [small tests](#small-smoke-tests) first.

## Training and evaluation data layout

Training reads indexed TARs without extraction. Evaluation reads task-specific
benchmark directories or supported TARs. ZS setup writes:

```text
data/
├── raw/<Group>/<dataset>/
└── prepared/V1_prepared_seg_geo/
    ├── train/
    │   ├── Semseg/<source>.tar
    │   ├── Instanceseg/<source>.tar
    │   ├── Refseg/refseg_<source>.tar
    │   ├── ImgEdit/pico_banana_sft_full.tar
    │   ├── ImgEdit/identity_50000.tar
    │   ├── ImgGeneration/pd12m_250000.tar
    │   ├── Depth/<dataset>.tar
    │   └── Normals/<dataset>.tar
    ├── eval/<Task>/<dataset>[.tar]
    ├── manifests/zs_training.json
    └── SHA256SUMS
```

Groups follow the task names above; shared raw inputs use `Shared_seg` or
`Shared_geo`. NZS writes the separate version listed in the recipe table.
Each training TAR has metadata, an index and checksum. Relative manifest paths
let a prepared version move as a whole. See the
[archive format](scripts/dataset_prep/README.md#prepared-archive-layout) for custom data.

### Validating prepared data

Verify checksums from the prepared version, using its actual directory:

```bash
(cd ./data/prepared/V1_prepared_seg_geo && sha256sum -c SHA256SUMS)
```

Then run the [CPU loader checks](#small-smoke-tests). Setup records provenance in
`manifests/preparation_manifest.json`; geometry also writes a 10:20 template in
`manifests/geometry_training_specs.json`. Metadata declares depth/normal
representation, camera convention, validity policy, split and preparation version.

### Resuming preparation

Rerun the same command to reuse stages with matching sources, settings, converter
versions and checksums. Receipts are in `manifests/preparation_stages/`; missing
indexes are rebuilt. Checksum verification may still read large archives.

Changed inputs/settings or older outputs without receipts require `--overwrite`
or a new YAML `version`. Stop training readers and allow space for replacements.
Handled creative conversion/indexing failures preserve old outputs; multi-file
replacement has rollback, but forced termination can interrupt it.
See [recovery details](scripts/dataset_prep/README.md#how-preparation-works).

### Evaluation benchmarks

One command downloads and materializes the reported benchmarks under
`prepared/V1_prepared_seg_geo/eval/<Task>/<dataset>[.tar]`, in the layout each
`scripts/evaluate_*.py` front end reads:

```bash
python scripts/setup_data.py --splits eval --data-root ./data --datasets \
  cityscapes,coco_object_80,saco_gold,refcocog,reasonseg,imgedit_bench,genai_bench,marigold_depth,marigold_normals,dsine,diode_normals
```

Add `nuscenes` to the list for optional geometry evaluation. For the
Marigold-protocol depth benchmarks and author-distributed normal targets, follow
[EVALUATION.md](EVALUATION.md#marigold-v2-depth-and-corrected-normal-targets).

## Small smoke tests

Use separate directories for these bounded original-source checks:

```bash
# Training geometry: three Hypersim frames; also fetches GenAI-Bench prompts.
python scripts/setup_data.py --config configs/data_smoke.yaml --data-root ./data_smoke

# Evaluation data only: two NYUv2 depth examples and two per normal benchmark.
python scripts/setup_data.py --config configs/data_depth_smoke.yaml --data-root ./depth_smoke
python scripts/setup_data.py --config configs/data_normals_smoke.yaml --data-root ./normals_smoke

# Editing training data: three original/edited Pico pairs.
python scripts/setup_data.py --config configs/data_editing_smoke.yaml --data-root ./editing_smoke

# Generation training data: three PD12M image/caption pairs.
python scripts/setup_data.py --config configs/data_generation_smoke.yaml --data-root ./generation_smoke
```

The normal recipe samples NYUv2, ScanNet, iBims-1 and Sintel author benchmarks.
A [Taskonomy recipe](configs/data_taskonomy_smoke.yaml) also checks matched
RGB/depth/normal members from one building. HTTP range downloads may transfer
more bytes than the retained sample size; provenance marks sampled inputs.

The CPU loader needs training dependencies, not model weights. It tests both
mask settings and samples up to 16 rows per archive across scenes/positions,
reporting selections and missing scene metadata. Add `--rows-per-archive 0`
to check all rows and `--resolution 512 512` for that training resolution.

These checks do not measure model quality or certify a full recipe. Test finite losses, checkpoint saving/resume
and actual inference in your intended GPU environment before long runs.

## Optional workflows and reference

<details>
<summary>Optional: standalone conversion and worker controls</summary>

For standalone conversion, inspect supported splits and accepted overrides:

```bash
python scripts/prepare_data.py --list
python scripts/prepare_data.py --describe nyuv2
```

Unsupported dataset/split pairs fail before conversion; `--allow-unsupported`
explicitly selects the supported subset. Unknown `--set DATASET.KEY=VALUE`
keys fail immediately. `--max-samples` applies only where a converter supports a
deterministic global cap.

`--dataset-workers` runs different converters concurrently; `--workers`
parallelizes each converter. Never run two processes that write the same
`<split>/<Task>/<dataset>.tar`. Different datasets can share a destination;
bookkeeping uses a filesystem lock. See the
[pool-only examples](scripts/dataset_prep/README.md#individual-download-and-pool-examples)
for preparation of the established other tasks without geometry.

</details>

<details>
<summary>Existing Cityscapes installations</summary>

For existing Cityscapes installations, the downloader also accepts
`raw/Semseg/cityscapes/cityscapes.tar` in its established raw layout.
Preflight requires 2,975 training and 500 validation RGB/label pairs, with
raw IDs, rendered train IDs and polygon annotations. The downloader still
validates the packed source and completes its acquisition step. It does not
restore a persistent extracted tree. Evaluation preparation retains a packed
TAR, and the evaluation launcher uses a temporary cache.
A raw archive does not attest license approval or the original download history.
Prepared OVB training TARs are not substitutes for these raw inputs.

</details>

### Archives and temporary storage

<details>
<summary>Temporary workspace, packed inputs and storage requirements</summary>

Temporary files use the system default. Select another writable disk if needed:

```bash
mkdir -p ./work
python scripts/setup_data.py --config configs/data_zs.yaml \
  --data-root ./data --work-dir ./work
```

All three setup commands accept `--work-dir` and clean their temporary
subdirectories after success or handled failure. Forced termination can leave
files behind; raw inputs and completed outputs are preserved.

Hypersim ZIPs and DIODE training archives stay packed, although DIODE needs
temporary spool space. Fresh Replica/Synscapes acquisition extracts raw assets;
Replica also retains download parts and a combined archive. Evaluation uses a
separate temporary cache. Capacity checks do not estimate all output/concurrent
needs. See [storage details](scripts/dataset_prep/README.md#storage-and-download-details).

</details>

## Custom other-task pools

<details>
<summary>Custom subsets and sampling weights</summary>

For subsets or custom weights, use the
[pool-builder reference](scripts/dataset_prep/README.md#individual-download-and-pool-examples).

Complete joint training uses the ZS/NZS YAML recipes above.

</details>

We are working on making the data easier to download and prepare.
