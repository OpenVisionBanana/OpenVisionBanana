# 🟣 Evaluation

## On this page

| Section | What you'll find |
|---|---|
| [Run addressing](#-how-runs-are-addressed) | Model, checkpoint, output paths and common options |
| [Semantic segmentation](#-semantic-segmentation-cityscapes--coco-thing--coco-stuff-val) | Launching, [prompt protocols](#prompt-protocols-prompt_style) and [sharded scoring](#sharded-scoring) |
| [Image editing](#image-editing-imgedit-bench-basic--uge) | Basic/UGE generation and judge commands |
| [Referring segmentation](#-referring-segmentation-refcocog--reasonseg) | RefCOCOg and ReasonSeg launch settings |
| [Instance segmentation](#-instance-segmentation-sa-cogold) | SA-Co/Gold presence gating and evaluation |
| [Legacy geometry wrappers](#legacy-geometry-shell-wrappers) | Archive layouts, suites and resume controls |
| ↳ [Depth details](#legacy-depth-details) | Older defaults and metric lanes |
| ↳ [Normal-axis overrides](#legacy-normal-axis-overrides) | Explicit historical diagnostic settings |

Evaluates the base model or a fine-tuned transformer on four downstream suites:

| Suite | Entrypoint | Benchmark | Score |
|---|---|---|---|
| Semantic segmentation | [`semseg/inference_partial_flux2_klein_img2img.py`](../../evaluation/semseg/inference_partial_flux2_klein_img2img.py) | Cityscapes **val** · COCO-Thing / COCO-Stuff **val** (supercategories) | mIoU / aAcc / mAcc |
| Image editing | [`imgedit/inference_flux2_klein_imgedit.py`](../../evaluation/imgedit/inference_flux2_klein_imgedit.py) + a VLM judge | ImgEdit-Bench **Basic** + **UGE** | 1–5 judge score |
| Referring segmentation | [`refseg/inference_flux2_klein_refseg.py`](../../evaluation/refseg/inference_flux2_klein_refseg.py) | RefCOCOg UMD **val** · ReasonSeg **val** | **cIoU** (RefCOCOg) · **gIoU** (ReasonSeg) — LISA metric |
| Instance segmentation | [`instanceseg/inference_flux2_klein_instanceseg.py`](../../evaluation/instanceseg/inference_flux2_klein_instanceseg.py) | SA-Co/Gold **test** (7 subsets) | **cgF1** / IL_MCC / pmF1 — SAM3 CGF1 metric |

The shell launchers in `scripts/evaluation/…` take environment variables and
call the Python evaluators. Run them in your installed environment. For the
paper protocol, use the `scripts/evaluate_*.py` commands in
[`EVALUATION.md`](../../EVALUATION.md).

## 🔷 How runs are addressed

Pick a model with `MODEL_TAG` + `CKPT`. The launchers resolve the fine-tuned
weights at `…/checkpoints/…/$MODEL_TAG/$CKPT/transformer` (via `CKPT_ROOT`) and the
base model from your local snapshot (`LOCAL_MODEL`). Outputs go to
`…/eval_outputs/…/$MODEL_TAG/$CKPT/<suite>/` (via `OUT_DIR`). For a **base-model**
run, pass an empty `TRANSFORMER_PATH=`.

Common knobs: `STEPS` (diffusion steps, default 28), `GUIDANCE_SCALE` (4.0),
`NUM_PROCESSES` (GPUs; the evaluators shard one process per GPU, rank 0 aggregates),
`LIMIT` (first N samples for a quick pass).

## 🔷 Semantic segmentation (Cityscapes / COCO-Thing / COCO-Stuff val)

Launcher: [`scripts/evaluation/semseg/run_cityscapes_flux2_klein.sh`](semseg/run_cityscapes_flux2_klein.sh)
→ the Python evaluator. It reads the dataset from an extracted `DATA_ROOT` (or
stages a `*.tar`), runs sharded inference, decodes predictions to class ids, and
scores mIoU.

Three benchmarks via `DATASET`: `cityscapes` (19 trainId classes) and the two
COCO evaluations, which score the **val** split on **supercategories only** —
`coco-thing` maps every COCO *thing* category to its supercategory (12 classes:
accessory, animal, …; all non-thing pixels are ignore), and `coco-stuff` maps
the COCO-Stuff categories to their supercategories (15 classes; the `other`
supercategory is excluded as ignore). The class↔color tables live in
[`coco_thing_label_color_mapping.csv`](../../evaluation/semseg/coco_thing_label_color_mapping.csv) /
[`coco_stuff_label_color_mapping.csv`](../../evaluation/semseg/coco_stuff_label_color_mapping.csv).

### Prompt protocols (`PROMPT_STYLE`)

The model paints a colored mask; the evaluator decodes pixels back to classes by
nearest-color snapping. **The prompt protocol decides which colors are in play**,
and each writes to its own `OUT_DIR` suffix so they never collide:

| `PROMPT_STYLE` | colors | decode targets | OUT_DIR suffix |
|---|---|---|---|
| `fixed` | the fixed class-palette prompt (e.g. 19-class Cityscapes palette) | standard palette | *(none)* |
| `randomdraw` | same fixed prompt, but per-image random colors from the 200-color palette (`LEGEND_SEED`) | the per-image drawn colors | `_randomdraw` |
| `set` | per-image class **subset** + random colors + randomized prompt — the training-style diversification (`SUBSET_SEED`) | per-image legend over the selected classes | `_set` |

- **`fixed`** — classic single-prompt, standard-palette task; the in-distribution
  check for models trained with the canonical pair.
- **`randomdraw`** — same classes and prompt, recolored per image: tests whether the
  model *reads* the legend colors instead of memorizing the palette.
- **`set`** — the eval analogue of training diversification: each image gets a random
  subset of present classes, random colors, and a randomized prompt, scored only over
  the selected classes. Reproducible via `SUBSET_SEED`; a saved `eval_set.json` can be
  replayed with `EVAL_SET`.

`fixed`/`randomdraw` run the full-class evaluator
([`inference_partial_flux2_klein_img2img.py`](../../evaluation/semseg/inference_partial_flux2_klein_img2img.py));
`set` runs the dynamic-subset evaluator ([`set_semseg.py`](../../evaluation/semseg/set_semseg.py)).
Every results JSON records its own `prompt_style`. Per-image legends are
deterministic (seeded) and archived under `legends/`; the evaluators also write
side-by-side **panels** (`input | ground truth | raw prediction | snapped
prediction`) under `panels/` for quick visual inspection.

```bash
# one protocol
MODEL_TAG=<run_id> CKPT=checkpoint-20000 PROMPT_STYLE=set \
  bash scripts/evaluation/semseg/run_cityscapes_flux2_klein.sh

# COCO instead of Cityscapes: point DATASET + the data root at your COCO layout
MODEL_TAG=<run_id> CKPT=checkpoint-20000 DATASET=coco-stuff PROMPT_STYLE=fixed \
  DATA_ROOT=/path/to/coco bash scripts/evaluation/semseg/run_cityscapes_flux2_klein.sh
```

Useful knobs: `LIMIT` (first N val images; empty for the full set), `STEPS` (28),
`GUIDANCE_SCALE` (4.0), `DATA_ROOT` (extracted dataset), `NUM_PROCESSES` (GPUs).

### Sharded scoring

For a big val set, the evaluator supports cross-process/-job sharding
(`--num_shards` / `--shard_index`): each shard scores `samples[rank::N]` and writes
`shard_metrics/`, then
[`aggregate_semseg_shards.py`](../../evaluation/semseg/aggregate_semseg_shards.py) merges them into
the final mIoU. Spread N shards across N GPUs (or N separate runs) and aggregate once.

## Image editing (ImgEdit-Bench: Basic + UGE)

Launcher: [`scripts/evaluation/imgedit/run_imgedit_flux2_klein.sh`](imgedit/run_imgedit_flux2_klein.sh).
It generates Basic and UGE edits and scores them from 1 to 5 with a local VLM
judge. Generation and judging run separately, so saved edits can be re-scored:

1. [`imgedit/inference_flux2_klein_imgedit.py`](../../evaluation/imgedit/inference_flux2_klein_imgedit.py)
   edits each benchmark input. It shards work across GPUs with
   `samples[shard::NUM_SHARDS]` and skips existing outputs on rerun.
2. Compare the edited and original images against the instruction with the
   local Qwen judge
   [`imgedit/judge_imgedit_qwen.py`](../../evaluation/imgedit/judge_imgedit_qwen.py)
   (Qwen2.5-VL; uses `flash_attention_2` if available, pass `--no_flash_attention`
   for SDPA).

```bash
# generate + Qwen judge
MODEL_TAG=<run_id> CKPT=checkpoint-20000 \
  bash scripts/evaluation/imgedit/run_imgedit_flux2_klein.sh
```

The launcher writes `imgedit_basic_*/` and `imgedit_uge_*/` with predictions and
`score_summary.json`. `EVAL_STEPS` (28) and `GUIDANCE_SCALE` (3.5) match the semseg
defaults.

## 🔷 Referring segmentation (RefCOCOg / ReasonSeg)

Launcher: [`scripts/evaluation/refseg/run_refseg_flux2_klein.sh`](refseg/run_refseg_flux2_klein.sh)
(`DATASET=refcocog|reasonseg`; `NUM_PROCESSES` for torchrun sharding). Prompts
the model with a referring expression to paint a binary mask (referred region
white, rest black), decodes the output by black/white nearest-color snap, and
scores with LISA's metric: **cIoU** (cumulative Σ fg-intersection / Σ fg-union —
the RefCOCOg headline) and **gIoU** (mean per-sample fg IoU — the ReasonSeg
headline); both are always reported. ReasonSeg `255` (ignore) pixels are
excluded from both intersection and union.

Unlike the semseg eval (which stages a `cityscapes.tar`), the refseg eval reads
its data as **extracted folders** under `datasets_archives/refseg/` — no tar
staging: RefCOCOg is the `refcocog/val_umd/refcocog_val_umd.json` index + its
`masks/` beside the extracted COCO `train2014/` images (`INDEX` /
`IMAGES_ROOT`), and ReasonSeg is the `reasonseg_val/` folder of
`<stem>.jpg`/`<stem>.json` pairs + `masks/` (`DATA_DIR`). The set is small
enough (a few thousand files) to live on the shared filesystem.

Rasterize the eval masks once with
`scripts/dataset_prep/prepare_refseg_eval.sh` (see
the [converter details](../dataset_prep/README.md#referring-segmentation-evaluation-masks)), then run

```bash
DATASET=refcocog  bash scripts/evaluation/refseg/run_refseg_flux2_klein.sh   # cIoU, 4,896 per-sentence samples
DATASET=reasonseg bash scripts/evaluation/refseg/run_refseg_flux2_klein.sh   # gIoU, 200 one-per-image samples
```

As elsewhere, pick a fine-tuned transformer with `MODEL_TAG`/`CKPT` (or
`TRANSFORMER_PATH`), or leave it unset for the base model. The default
white-on-black prompt is overridable via `PROMPT_TEMPLATE` (must contain
`{expr}`) — the repo bakes no referring-seg prompt into training, so this is a
genuine protocol choice.

ReasonSeg's queries are **implicit** ("the source of power for the ship"), so
they are rewritten into explicit referring expressions by default
(`REASONSEG_REWRITE=1`; `0` = raw-query baseline — the two are not comparable),
read from a cache produced **once** by the checkpoint-independent
[`refseg/rewrite_reasonseg_qwen.py`](../../evaluation/refseg/rewrite_reasonseg_qwen.py)
(Qwen3-VL-32B-Thinking, single-turn `image + query → {"target": …}`) and reused
by every eval. The eval fails loud if the cache (`REASONSEG_REWRITE_FILE`) is
missing, and the output folder encodes the rewriter tag so different rewriters
never collide.

## 🔷 Instance segmentation (SA-Co/Gold)

Launcher: [`scripts/evaluation/instanceseg/run_saco_gold_flux2_klein.sh`](instanceseg/run_saco_gold_flux2_klein.sh).
Given (image, noun-phrase) the model paints each instance a distinct color on
black; predictions are cluster-decoded back to instance masks
([`instanceseg/decode.py`](../../evaluation/instanceseg/decode.py)) and scored with SAM3's CGF1
metric (**cgF1** / IL_MCC / pmF1;
[`instanceseg/cgf1_eval.py`](../../evaluation/instanceseg/cgf1_eval.py)) over the 7 SA-Co/Gold
test subsets. Unlike the refseg eval, the data is staged from a **tar**
(`datasets_archives/instanceseg/SA-Co-Gold.tar`) into temporary storage by
[`stage_saco_gold.sh`](instanceseg/stage_saco_gold.sh)
(an already-extracted `SACO_ROOT` is used as-is).

Two passes, in order:

```bash
# 1. image-level presence gate — checkpoint-INDEPENDENT Qwen3-VL pass, computed
#    once and reused by every checkpoint (writes gate_decisions_qwen.json):
python evaluation/instanceseg/gate_saco_gold_qwen.py --gate_file datasets_archives/instanceseg/gate_decisions_qwen.json

# 2. generation + eval (GATE=file reads the cached decisions)
MODEL_TAG=<run_id> CKPT=checkpoint-20000 \
  bash scripts/evaluation/instanceseg/run_saco_gold_flux2_klein.sh
```

`PROMPT_SET` picks the prompt scheme the checkpoint was trained on: `legend`
(default — explicit per-instance color assignments; tune with `LEGEND_LEN`,
default 10, and `NOTATION=mix|name|rgb`), `ordered` (v1 fixed-palette-order
wording), or `legacy` (pre-fix). Prompts come from
[`common/instanceseg/prompts.py`](../../common/instanceseg/prompts.py), shared
with the training converter so train/eval phrasing can't drift. The gate mode
is part of the score's meaning: `GATE=file` (cached Qwen decisions),
`oracle` (GT presence — debug upper bound), or `none` (generate for all).

The decode defaults are the tuned SA-Co/Gold settings: palette snap, τ=58,
γ=1, θ_size=8e-4, θ_erosion=0.1, gap fill and GrabCut
([`instanceseg/postprocess.py`](../../evaluation/instanceseg/postprocess.py)).
Changing them changes metric comparability. Sharding works like the other evaluators
(`--num_shards`/`--shard_index` + a `--merge_only` aggregation pass).

## Legacy geometry shell wrappers

For current commands, use [EVALUATION.md](../../EVALUATION.md).
This reference covers older local packages and shell wrappers. They share these model options:

- `MODEL_FAMILY=klein` selects the FLUX.2-Klein runtime.
- `MODEL` points to the complete local base snapshot.
- `TRANSFORMER_PATH` loads a full-rank/transformer artifact.
- `LORA_PATH` loads a standard Diffusers LoRA adapter.
  Do not set it together with
  `TRANSFORMER_PATH`.

Run one geometry dataset with `depth/run_depth.sh` or
`surface_normals/run_surface_normals.sh`. Run every supported dataset with the
corresponding `run_all.sh`. A dataset is complete only after its launcher
atomically writes a non-empty `EVALUATION_COMPLETE` sentinel; `run_all.sh`
checks that sentinel rather than treating a partial result JSON as success.
GenAI-Bench uses `image_generation/run_genai_bench.sh`, with prompt preparation,
generation and scoring selectable independently.

See the focused [depth](../../evaluation/depth/README.md),
[surface-normal](../../evaluation/surface_normals/README.md), and
[GenAI-Bench](../../evaluation/image_generation/README.md) guides. Temporary
storage uses the system default without requiring environment variables.

After sourcing `configs/environment.example`, the suite launcher
derives models, archives, and output locations from `OVB_STORAGE_ROOT`:

```bash
scripts/evaluation/run_suite.sh --suite depth --model-family klein
scripts/evaluation/run_suite.sh --suite normals --model-family klein
scripts/evaluation/run_suite.sh --suite genai --model-family klein
```

The local package layout has two data profiles. `full` uses
`$PREPARED_ROOT/{depth,surface_normals}` and evaluates depth datasets `nyu`,
`ibims1`, `eth3d`, `diode`, `kitti`, `nuscenes`, `vkitti2`, `scannet`, and
`kitti_eigen`, plus normal datasets `nyu`, `ibims1`, `diode`, `kitti`,
`eth3d`, `nuscenes`, `vkitti2`, `scannet` and `sintel`. `ablation` uses the deterministic
nuScenes/ScanNet/VKITTI2 subsets below
`$PREPARED_ROOT/profiles/ablation`:

```bash
scripts/evaluation/run_suite.sh --suite depth \
  --geometry-profile full --model-family klein
scripts/evaluation/run_suite.sh --suite normals \
  --geometry-profile ablation --model-family klein
```

The suite launcher separates profile outputs under
`$OUTPUT_ROOT/evaluation/<suite>/<profile>/<model-family>`. Set
`GEOMETRY_EVAL_ROOT` to change the geometry result root without changing the
GenAI-Bench output location.

Both geometry shell launchers default to 28 steps, guidance scale 1.0 and a
fixed per-sample seed. The public Python frontends default to guidance 3.5.
Depth reports AbsRel, SqRel, RMSE, RMSE-log, log10, SILog and
delta-accuracy thresholds, retaining direct and affine diagnostic lanes in the
result JSON. Surface normals report signed camera-space angular mean, median,
RMSE, and fractions below 11.25, 22.5, and 30 degrees; invalid predictions stay
in the primary denominator at 90 degrees, while valid-only and bidirectional
scores remain labeled diagnostics. The official iBIMS protocol is the default
and reads `ibims1_official.tar`; select `IBIMS1_DEPTH_PROTOCOL=nyu_10m` or
`IBIMS1_NORMALS_PROTOCOL=nyu_10m` only to reproduce the legacy archives. KITTI
Eigen is depth-only and is labeled `affine_diagnostic_only`.

Current normal evaluation reads encoding and axes from archive metadata.
Float-XYZ targets retain their declared axes. Ambiguous older RGB archives
require an explicit legacy diagnostic configuration; do not add sign overrides
to current targets based on the dataset name. See the surface normal guide
for that older interface.

Interrupted geometry runs resume from `<dataset-output>/.resume_v1` by default,
committing every 50 samples. `EVAL_NUM_GPUS=N` enables deterministic sample
sharding within one dataset. Use one process per GPU when scheduling separate
datasets in parallel to avoid nested sharding. A successful run stages
its final artifacts, writes `EVALUATION_COMPLETE`, and removes its resume state.

Without a sourced configuration, pass `--root /path/to/storage`; the wrapper
derives its model, prepared-archive, and output locations from that one path.

Sourcing the configuration also derives `CITYSCAPES_TAR`, `COCO_TAR`,
`REFSEG_ROOT`, `REASONSEG_REWRITE_FILE`, `IMGEDIT_DATA_ROOT`, `SACO_TAR`, and
`GATE_FILE` for the original evaluator launchers documented above. It maps the
legacy output-path variable `SCRATCH_ROOT` to `OUTPUT_ROOT`. Temporary work uses
the system default unless overridden.

Geometry suites evaluate every available prepared archive and print what is
missing. Add `--strict` for a complete-benchmark requirement. Use
`--checkpoint` for full-rank artifacts or `--lora` for adapters; omitting both
evaluates the base model. The original evaluator launchers and variables are
unchanged.

### Legacy depth details

Depth defaults to metric RGB and guidance 1.0. Set `DATASETS` to a space-separated
subset. `SKIP_COMPLETED=1` skips completed datasets; `CONTINUE_ON_ERROR=1`
continues other datasets after a failure. The KITTI Eigen date-grouped RGB
directories and projected-GT drive directories are siblings; pass the extraction
root, with `2011_09_26` used only as the extraction sentinel.

Top-level geometry-v5 metrics prefer direct metric depth, then the reference
disparity-affine lane, with alignment as a fallback if reference decoding is
unavailable. JSONs retain AbsRel, SqRel, RMSE, RMSE-log, log10, SILog and
delta1/delta2/delta3, plus no-shift and linear-affine diagnostics.
`result_files.txt` lists suite results.

### Legacy normal-axis overrides

For historical RGB reproduction only, supply all required overrides explicitly:

```bash
GT_NORMAL_ENCODING=legacy_direct_xyz GT_NORMAL_PERMUTATION=0,1,2 \
GT_NORMAL_SIGNS=1,-1,-1 \
  DATASET=nuscenes SOURCE_ARCHIVE=/path/to/rebuilt_nuscenes.tar \
  scripts/evaluation/surface_normals/run_surface_normals.sh
```

Current float-reference targets use their declared axes. Geometry resume state
uses a 50-sample commit chunk; successful publication removes that state.
