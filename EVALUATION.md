# Evaluation

## On this page

| Section | What you'll find |
|---|---|
| [Common interface](#common-interface) | Model paths, options and dry runs |
| [Task commands](#task-commands) | Launchers, data caching and multi-GPU execution |
| [Paper reproduction](#reproducing-the-paper) | Model settings and task-specific protocols |
| ↳ [SA-Co/Gold decoding](#sa-cogold-decoding) | Instance decoding and scoring controls |
| ↳ [ReasonSeg rewrites](#reasonseg-query-rewrites) | Query preprocessing |
| ↳ [Data layout](#data-layout) | Accepted benchmark directories |
| [Geometry contracts](#geometry-contracts) | Depth alignment and normal conventions |
| ↳ [Author benchmarks](#marigold-v2-depth-and-corrected-normal-targets) | Setup, evaluation commands and startup checks |
| [Other task metrics](#other-tasks) | Segmentation metric definitions |
| [Editing and generation](#image-editing-and-generation) | Evaluation data, judges and result files |

## Common interface

Every task launcher accepts a FLUX.2-Klein base model or fine-tuned transformer and one, several, or all datasets:

```bash
python scripts/evaluate_depth.py --data-root /data/ovb/prepared \
  --output-dir outputs/depth --datasets nyu,diode --model /models/klein \
  --transformer-path outputs/final/checkpoint-50000 --num-gpus 4 \
  --steps 28 --guidance 3.5 --seed 0
```

Common options:

- Model and data: `--model`, `--transformer-path`, `--datasets`, `--data-root`, `--output-dir`.
- Inference: `--num-gpus`, `--steps`, `--guidance`, `--seed`, `--limit`.
- Inspection and overrides: `--dry-run` resolves samples without loading the model; repeated `--extra-arg` passes options to the low-level evaluator.
- Output packing: when a dataset finishes, its image artifacts are packed into `artifacts.tar` inside the output directory, while JSON results, `logs/`, `scores/` and resume caches stay extracted. Pass `--no-archive-outputs` to keep everything extracted. Depth and normals outputs are never packed.

Task-specific options:

- Image-conditioned tasks expose `--native-resolution`. Depth and normals enable it by default, as in the paper, and `--no-native-resolution` turns it off.
- Depth, normals, and semantic segmentation expose `--resume`.
- Model offload is available only where the underlying pipeline supports it.

Use each script's `--help` for exact options. Low-level evaluators under `evaluation/` remain directly callable.

Depth and normal frontends use the training example's detailed inverse-gray and concise canonical-normal prompts by default. Use `--prompt` to override either contract for a matched experiment.

## Task commands

[examples/jobs/evaluate_release.job](examples/jobs/evaluate_release.job) wraps the commands
below for a Slurm cluster. It reads `MODEL`, `CKPT` and `DATA_ROOT` from the environment and takes the task name as its first argument.

```bash
python scripts/evaluate_semseg.py      --data-root DATA --output-dir OUT --num-gpus 4   # cityscapes,coco_object_80
python scripts/evaluate_instanceseg.py --data-root DATA --output-dir OUT --num-gpus 4   # SA-Co/Gold 14k
python scripts/evaluate_refseg.py       --data-root DATA --output-dir OUT --num-gpus 4   # refcocog,reasonseg
python scripts/evaluate_depth.py        --data-root DATA --output-dir OUT --datasets all --num-gpus 4
python scripts/evaluate_normals.py      --data-root DATA --output-dir OUT --datasets all --num-gpus 4
python scripts/evaluate_imgedit.py      --data-root DATA --output-dir OUT --num-gpus 4   # basic,uge x 3 runs + judge
python scripts/evaluate_generation.py   --data-root DATA --output-dir OUT --num-gpus 4   # GenAI-Bench, seeds 0-2 + judge
```

The front ends accept extracted directories or compatible versioned `.tar`
outputs. `prepare_data.py` currently creates those archives for the geometry
dataset/split pairs marked `yes` by its `--list`. Cityscapes can also remain
packed. Other non-geometry evaluation data uses the official layouts below.
Archives use a locked cache in the system temporary directory. No environment
variable is required. Each machine maintains its own cache. The operating
system can remove temporary caches between runs. Result folders do not hold
extracted dataset copies. Ensure sufficient free space for the selected benchmark.
Multi-GPU execution assigns one independent shard to each
visible GPU, uses deterministic round-robin sample assignment, and shares
aggregation state. A sample receives the same generation seed regardless of GPU
count. Arguments for the low-level evaluator that start with `--` are passed as
`--extra-arg=--flag`.

The segmentation, image-editing and image-generation front ends also span nodes.
Start the same command once per node with `--num-nodes N`; shards are then spread
over N × `--num-gpus` GPUs, steps that need every shard (judging, scoring,
summaries) run on node 0, and `--output-dir` must be on a filesystem all nodes
share. Pass each machine's rank and a common run identifier explicitly.
For example, run the first command on one machine and the second on another:

```bash
python scripts/evaluate_imgedit.py --data-root DATA --output-dir OUT \
  --num-gpus 1 --num-nodes 2 --node-rank 0 --run-id example
python scripts/evaluate_imgedit.py --data-root DATA --output-dir OUT \
  --num-gpus 1 --num-nodes 2 --node-rank 1 --run-id example
```

## Reproducing the paper

The task-specific defaults below describe the selected paper protocols. The resulting scores are listed in the [README results table](README.md#-results). For
geometry, also select the benchmark lane and target release described below.
Pass the base model the checkpoint was trained from and the fine-tuned transformer:

```bash
# 9B models (FunAILab/Open-Vision-Banana, final_models/9B/)
MODEL=black-forest-labs/FLUX.2-klein-base-9B
CKPT=/path/to/final_models/9B/zeroshot/checkpoint-coco86750-w25/transformer
CKPT=/path/to/final_models/9B/nonzeroshot/checkpoint-avg82500-85000-87500-92500-97500_uniform/transformer
# 4B models (final_models/4B/)
MODEL=black-forest-labs/FLUX.2-klein-base-4B
CKPT=/path/to/final_models/4B/zeroshot/checkpoint-87500-95000-w25/transformer
CKPT=/path/to/final_models/4B/nonzeroshot/checkpoint-avg90000-92500-97500-100000_uniform/transformer

for task in semseg instanceseg refseg imgedit generation; do
  python scripts/evaluate_$task.py --model "$MODEL" --transformer-path "$CKPT" \
    --data-root DATA --output-dir OUT --num-gpus 4
done
```

All tasks use 28 steps, guidance 3.5 and generation seed 0 unless stated.

| Task | Benchmark | Metric | Protocol (default) |
|---|---|---|---|
| Semantic seg. | Cityscapes val (500) | mIoU | one random-draw pass, best-64 colour pool |
| Semantic seg. | COCO-Object-80 val (5000) | mIoU | one pass, 3 grouped legends of 27 classes (RGB), 3 per FLUX call |
| Instance seg. | SA-Co/Gold (167,522 images) | cgF1 | bundled Qwen3-VL-8B gate, fixed-draw 32 colours, refined decode (below) |
| Referring seg. | RefCOCOg UMD val (4896) | cIoU |  |
| Reasoning seg. | ReasonSeg val (200) | gIoU | bundled Gemma 4 31B query rewrites |
| Image editing | ImgEdit-Bench Basic | judge score | 3 runs (seeds 1001-1003), 16 logical shards, ImgEdit judge |
| Image editing | ImgEdit-Bench UGE | judge score | as above |
| Text-to-image | GenAI-Bench (1600) | VQAScore | seeds 0-2, CLIP-FlanT5-XXL judge |

The front ends run one pass per image by default and, for SA-Co/Gold, score the
14k seed-0 subset used by the design-choice analyses of the paper. The README
table reports the final models on the complete evaluation sets with legend
ensembling, which the following flags select:

```bash
python scripts/evaluate_semseg.py      --datasets cityscapes     --repeats 20 ...  # Cityscapes mIoU of the README table
python scripts/evaluate_semseg.py      --datasets coco_object_80 --repeats 90 ...  # COCO-Object protocol of the paper
python scripts/evaluate_instanceseg.py --full --repeats 11 ...                     # SA-Co/Gold cgF1, Qwen3-VL-8B-Instruct gate
python scripts/evaluate_instanceseg.py --full --repeats 11 --full-gate qwen3vl30ba3b ...  # starred 9B non-zero-shot score
```

Optional protocols:

| Option | Effect |
|---|---|
| `evaluate_semseg.py --datasets cityscapes --repeats` | 5 colour draws, majority vote |
| `evaluate_semseg.py --datasets coco_object_80 --repeats` | 10 colour draws, majority vote |
| `evaluate_refseg.py --datasets reasonseg --no-rewrite` | raw ReasonSeg queries (gIoU) |
| `evaluate_refseg.py --datasets reasonseg --reasonseg-rewrites qwen3vl32bthink` | bundled Qwen3-VL-32B-Thinking query rewrites (gIoU) |
| `evaluate_instanceseg.py --repeats 16` | 16 random palette draws, refined per draw, inclusive-majority mask vote (cgF1) |

`--repeats N` sets any other count; the vote over every prefix 1..N is written as
`*_results_1toK.json`. For semantic segmentation the two protocols are:

```bash
python scripts/evaluate_semseg.py --repeats 1 ...                            # single pass (default)
python scripts/evaluate_semseg.py --mode legend-ensembling --repeats 20 ...  # 20 colour draws, majority vote
```

`evaluate_generation.py --compare-to OUT_BASE/genai_bench`
adds a paired comparison against the base model evaluated the same way.

VLM outputs are bundled because regenerating them is not deterministic:
`evaluation/assets/saco_gold_gate_decisions_qwen3vl8b.json` is the presence gate
for the 14k subset. The image ids of that subset are listed per SA-Co/Gold
subset in `evaluation/assets/saco_gold_14k_seed0_subset.json`. Two gates over
all 167,522 images are bundled for `--full`:
`saco_gold_gate_decisions_qwen3vl8b_full.json` is used by default and underlies
the README scores, and `saco_gold_gate_decisions_qwen3vl30ba3b_full.json`,
selected with `--full-gate qwen3vl30ba3b`, gives the starred score of the 9B
non-zero-shot model. New gates can be computed with
`evaluation/instanceseg/gate_saco_gold_qwen.py`. The ReasonSeg query rewrites are
described below.

Batch sizes are part of the protocol where they change the FLUX calls: SA-Co/Gold
uses 4 queries per call for 9B and 8 for 4B (chosen from the model name;
`--batch-size` overrides), and ImgEdit seeds one generator per shard, so its 16
shards are kept as a logical layout and queued over the available GPUs. SA-Co/Gold
batches the queries of each shard, so its scores move slightly with the total
shard count; the paper used 16 shards (4 nodes × 4 GPUs).

### SA-Co/Gold decoding

Each generated output is snapped to the 32 prompted colours plus black at generation
resolution, resized to the GT size (nearest neighbour) and decoded with tau 58,
gamma 1, theta_size 8e-4 and theta_erosion 0.1. Painted pixels that the decoder left
unassigned, judged on the unsnapped output, join the nearest instance within 3 px.
Each instance is then refined with OpenCV GrabCut on the input photograph
(2 iterations, fixed 3 px band, kept only if at least 90% of the mask survives), with
the OpenCV RNG seeded from the subset, query id and pass index. These settings were
tuned with the 4B zero-shot model on the 14k seed-0 subset.
On a set of saved 9B outputs the masks are identical with OpenCV 4.10, 4.13 and 5.0,
NumPy 2.2.6 and 2.4.4, and SciPy 1.15.3 and 1.18.0.

`--repeats N` generates each gate-positive query with N random palette draws,
refines every draw, matches instances across draws at IoU 0.5 and keeps instances
and pixels supported by at least half the draws. Results go to `saco_gold_xN/`, with
each draw scored alone (`saco_gold_results_repeatNN.json`) and every prefix voted
(`saco_gold_results_1toNN.json`); `repeat_analysis_results.json` adds the mean over
single draws.

`--raw-rgb-root RUN` re-decodes and scores the saved raw outputs of an earlier run
(`RUN/<subset>/pred_rgb_raw/`, written by passing `--extra-arg=--save_rgb`) on CPU,
without loading the model. Pass the same subset, gate and `--repeats` settings as
that run and a new `--output-dir`; a missing raw output is an error.
Exact decode/refinement parameters and their switches are listed by
`python evaluation/instanceseg/inference_flux2_klein_instanceseg.py --help`.

### ReasonSeg query rewrites

ReasonSeg queries are implicit ("something used for playing videos or music"), so a
VLM first rewrites the first query of each image into one explicit referring
expression, which FLUX then segments. Both rewrite caches behind the scores above
ship with the repository:

| `evaluate_refseg.py` option | Bundled cache | Rewriter |
|---|---|---|
| default (`--reasonseg-rewrites gemma4_31b`) | `evaluation/assets/reasonseg_val_rewrites_gemma4_31b.json` | Gemma 4 31B instruct, greedy |
| `--reasonseg-rewrites qwen3vl32bthink` | `evaluation/assets/reasonseg_val_rewrites_qwen3vl32bthink.json` | Qwen3-VL-32B-Thinking, sampled at temperature 0.6 |

To generate a cache yourself, run `evaluation/refseg/rewrite_reasonseg_qwen.py`.
`--rewriter` selects the model and decoding settings used for the bundled cache.
Gemma 4 needs transformers 5 while `requirements.txt` pins 4.57.6, so install it
into a separate directory for this pass only:

```bash
python -m pip install --target /envs/tf5 -r requirements/reasonseg-rewrite-gemma.txt
PYTHONPATH=/envs/tf5 python evaluation/refseg/rewrite_reasonseg_qwen.py --rewriter gemma4_31b \
  --data_dir DATA/refseg/reasonseg_val --out REWRITES/reasonseg_rewrites_gemma4_31b.json
python scripts/evaluate_refseg.py --datasets reasonseg \
  --reasonseg-rewrite-file REWRITES/reasonseg_rewrites_gemma4_31b.json \
  --model "$MODEL" --transformer-path "$CKPT" --data-root DATA --output-dir OUT_OWN --num-gpus 4
```

Other VLMs use the same script. For Qwen3-VL-8B-Instruct, replace `--rewriter gemma4_31b`
with `--qwen_model Qwen/Qwen3-VL-8B-Instruct --temperature 0 --max_new_tokens 256`
and drop the `PYTHONPATH`; it runs with `requirements.txt` alone.

The rewrite pass runs on one GPU with room for the 62 GB of bf16 weights. It exits
non-zero if any rewrite is unusable, lists those in `<out>.failed.json`, and
`--repair` retries only them. A regenerated cache can differ from the bundled one in
a few rewrites because bf16 GPU kernels are not bit-exact across runs, so use the
bundled cache to reproduce the tables.

### Data layout

After `scripts/setup_data.py --splits eval` ([DATASET.md](DATASET.md)), pass its
data root: every benchmark below is then found under
`prepared/V1_prepared_seg_geo/eval/<Task>/<dataset>/`. `--data-root` may also point
at the dataset directory itself or at a parent holding:

| Dataset | Directory | Source |
|---|---|---|
| Cityscapes | `cityscapes/` or `sem_segm/cityscapes/` (`leftImg8bit/`, `gtFine/` with `*_labelTrainIds.png`) | cityscapes-dataset.com |
| COCO-Object-80 | `coco/` (`images/val2017/`, `annotations/val2017/` stuffthingmaps PNGs, `annotations/panoptic_annotations/panoptic_val2017.json` for category ids) | cocodataset.org 2017 + COCO-Stuff `stuffthingmaps_trainval2017` |
| SA-Co/Gold | `instanceseg/SA-Co-Gold/` (`gt-annotations/`, `metaclip-images/`, `sa1b-images/`) | facebook/SACo-Gold |
| RefCOCOg | `refseg/refcocog/` (`val_umd/refcocog_val_umd.json`, `train2014/`) | `dataset_prep/refseg/prepare_refseg_eval.py refcocog` |
| ReasonSeg | `refseg/reasonseg_val/` (`*.jpg` + `*.json`) | LISA ReasonSeg val |
| ImgEdit-Bench | `image_editing/imgedit_bench/Benchmark/` or `imgedit/Benchmark/` (`singleturn/`, `hard/`), judge in `ImgEdit_Judge/` next to it or `--judge-model` | huggingface.co/datasets/sysuyy/ImgEdit |
| GenAI-Bench | `image_generation/prepared/genai-bench/prompts.jsonl` or `--prompts-jsonl` | `evaluation/image_generation/genai_bench/prepare_prompts.py` |

## Geometry contracts

Depth supports NYUv2, iBims-1, DIODE, KITTI depth selection, KITTI Eigen,
ETH3D, nuScenes, Virtual KITTI 2 and ScanNet. The default representation is
relative inverse grayscale, matching the training recipe. Use the checkpoint's
actual encoding when evaluating it.

Depth results distinguish these scoring methods:

- Direct metric RGB: decode into camera-z meters, resize the scalar field,
  clip to the dataset range and score valid pixels.
- Disparity-affine: fit `a p + b` to inverse ground-truth depth per image,
  convert back to depth, clip and score.
- Depth-affine: fit `a p + b` directly to metric depth.

A matching held-out calibration record can select the affine family for the
primary relative-depth score. Keep the other methods labeled separately.
Lower AbsRel is better; `delta_1` is the fraction within a factor of 1.25.

Normals supports NYUv2, iBims-1, DIODE, KITTI, ETH3D, nuScenes, Virtual KITTI 2,
ScanNet and Sintel. Predictions default to camera-space
`vision_banana_x_left_red`. Ground-truth encoding defaults to `auto`: float-XYZ
arrays are read as vectors, while RGB targets must declare their coordinate
convention. The evaluator normalizes vectors and applies the recorded validity
masks before computing angular errors.

Use the declared convention for primary normal scores. Optional axis or SO(3)
calibration is fitted on separate calibration images and frozen before benchmark
evaluation. Test-five convention search uses benchmark ground truth and must
be reported as a diagnostic.

### Marigold V2 depth and corrected normal targets

Install `requirements/geometry.txt`, then prepare the author benchmark assets
and pinned split lists with the shared setup command:

```bash
python -m pip install -r requirements/geometry.txt
python scripts/setup_data.py --config configs/data_zs.yaml \
  --splits eval --data-root ./data

python scripts/evaluate_depth.py --model /models/klein \
  --transformer-path /checkpoints/final/transformer --data-root ./data \
  --output-dir outputs/depth_ppd --num-gpus 4

python scripts/evaluate_normals.py --model /models/klein \
  --transformer-path /checkpoints/final/transformer --data-root ./data \
  --output-dir outputs/normals --num-gpus 4
```

The defaults are the paper settings. Depth uses the Marigold V2 PPD protocol
(`--benchmark-protocol marigold_v2_ppd`) with relative inverse grayscale
encoding, native resolution, 28 steps, guidance 3.5 and seed 0 over NYUv2,
KITTI-Eigen, ETH3D, ScanNet and DIODE. The split lists are found under
`<data-root>/prepared/V1_prepared_seg_geo*/manifests/benchmark_splits`, or can
be given with `--benchmark-split-root`. Normals run at native resolution over
NYUv2, iBims-1, DIODE, ScanNet, Virtual KITTI 2 and Sintel. Replace the model
and checkpoint paths with your own.

Depth `all` selects NYUv2-654, KITTI-Eigen-652, ETH3D-454, ScanNet-800
and DIODE-771, using the author distributions and split lists pinned to Marigold
V2 commit `18466672fb8661152434a0efe1184939164cda19`.

Match the checkpoint's encoding before scoring. Relative inverse grayscale
uses disparity-affine alignment with `--ppd-primary-alignment auto`; fixed
metric RGB/Barron encodings use log-depth RANSAC after metric decoding.
Read `primary_metrics` and `depth_evaluation.primary_lane` together.
Use matching alignment methods, splits and masks when comparing scores.
See [alignment and diagnostic fields](evaluation/depth/README.md#alignment-and-diagnostics).

The normal adapters preserve NYUv2 GeoNet, iBims-1 DSINE, DIODE float,
ScanNet FrameNet, VKITTI2 DSINE and Sintel targets and masks. The frontend
records `geometry-eval-v6`. Compare matching sources, splits, crops, masks,
aggregation and inference settings. Report calibration and convention-search
results separately from fixed-convention scores.

For a startup check, select one dataset and add `--limit 2 --dry-run`.
This checks discovery without loading weights. Remove `--dry-run` for a
two-image GPU check, then remove the limit for the full benchmark.
Use a new output directory when changing the protocol, or explicitly rescore
retained predictions.

The commands specify native resolution, 28 steps,
guidance 3.5, seed 0, and the detailed inverse-depth/concise-normal prompts.
Calibration records bind the model/checkpoint, prompt hash, steps, guidance,
seed mode, encoding, calibration split and fitted transform. Calibration images
must be separate from test images.

See [depth details](evaluation/depth/README.md),
[normal metrics](evaluation/surface_normals/README.md), or the
[manual benchmark acquisition reference](scripts/dataset_prep/README.md#manual-author-benchmark-acquisition).

## Other tasks

Semantic segmentation decodes each output with the image's exact legend and
scores mIoU (`aAcc`/`mAcc` are also reported). Instance segmentation reports cgF1,
IL_MCC and pmF1 per subset and their mean. RefCOCOg reports cumulative IoU;
ReasonSeg reports generalized IoU.

## Image editing and generation

Prepare their evaluation inputs with:

```bash
python scripts/setup_data.py --data-root ./data --splits eval \
  --datasets imgedit_bench,genai_bench
```

Use the task commands above with your base model and transformer path.
Editing runs Basic and UGE with seeds 1001-1003 and 16 logical generation
shards, then scores with the local ImgEdit_Judge (Qwen2.5-VL).
`--judge-model` selects its directory; `--skip-judge` generates without scoring.
Run averages are written to `<suite>/average_across_runs.json`.

Generation runs 1,600 GenAI-Bench prompts with seeds 0-2 and scores them with
the CLIP-FlanT5-XXL VQAScore judge. `--evaluator qwen2.5-vl-7b` selects the
Qwen judge instead. Install the [pinned scorer](evaluation/image_generation/README.md)
first. `--score-python` selects a separate scoring interpreter,
`--score-pythonpath` prepends the CLIP-FlanT5 compatibility tree, and
`--score-model-path` selects a local Qwen judge snapshot.
`--skip-score` generates without scoring. Results go to
`genai_bench/summary.json`; `--compare-to OUT_BASE/genai_bench` adds a paired
comparison against a base model evaluated with matching settings.
