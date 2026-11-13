# 🟣 Training

Fine-tune `black-forest-labs/FLUX.2-klein-base-9B` into the shared
image-to-image instruction model on a prepared dataset pool.

For the current geometry recipe, prepared manifests and mask settings, start
with [TRAIN.md](../../TRAIN.md). This page describes the lower-level launcher
and includes older partial-tuning examples.

## On this page

| Section | What you'll find |
|---|---|
| [Prerequisites](#prerequisites) | Environment, prepared pool and base model |
| [Run it](#run-it) | Launcher variables and commands |
| [Fine-tuning scope](#choosing-how-much-to-fine-tune) | Parameter selection and [examples](#examples) |
| [Multi-GPU strategy](#multi-gpu-strategy) | Distributed execution settings |
| [Output](#output) | Saved artifacts |
| [Historical partial tuning](#historical-partial-tuning-example) | Older launch settings |
| [Klein launcher](#-flux2-klein-launcher) | Pool formats and profile-based launching |
| [Scope and resume reference](#scope-and-resume-reference) | Embedding controls, RNG state and migration overrides |

- **Entrypoint:** [`train_dreambooth_partial_flux2_klein_img2img.py`](../../train_dreambooth_partial_flux2_klein_img2img.py)
  (repo root) — a self-contained `accelerate` training script.
- **Launcher:** [`run_open_visionbanana_pool_flux2_klein.sh`](run_open_visionbanana_pool_flux2_klein.sh)
  — the shell wrapper you actually run; it turns environment variables into the
  entrypoint's CLI flags and calls `accelerate launch`.

By default the trainer **partially** fine-tunes the model: it freezes the whole
transformer except the last few blocks plus the output layers, which is cheap and
enough for most task adaptation. Every knob below is an override of a launcher
default — you only set the ones you want to change.

## Prerequisites

- A Python venv with the pinned deps: `pip install -r requirements.txt` (repo
  root). Follow [MAIN.md](../../MAIN.md#installation) and the current requirements
  for the CUDA/PyTorch versions; this launcher reference is not a separate
  environment specification. Use a compatible PyTorch build on a different accelerator stack.
- A **prepared dataset pool** JSON (`DATASET_SPECS`) built by
  the data setup in [DATASET.md](../../DATASET.md#complete-zs-and-nzs-recipes).
- The base model `FLUX.2-klein-base-9B` in a local HF cache (`LOCAL_MODEL`).

## Run it

```bash
CUDA_VISIBLE_DEVICES=0 \
USE_CONTAINER=0 \
LOCAL_MODEL=/path/to/FLUX.2-klein-base-9B \
DATASET_SPECS=/path/to/open_visionbanana_mix_v1.json \
OUTPUT_DIR=/path/to/finetuned_model \
bash scripts/training/run_open_visionbanana_pool_flux2_klein.sh
```

Run with `--help` to print the full list of variables:

```bash
bash scripts/training/run_open_visionbanana_pool_flux2_klein.sh --help
```

> **Container note.** The portable default is `USE_CONTAINER=0`, which runs
> `accelerate` in the active environment. Set `USE_CONTAINER=1` and `SIF` to use
> a container; the launcher detects Apptainer or Singularity and enables GPU
> passthrough. Explicit values retain precedence.

## Choosing *how much* to fine-tune

The Flux2-Klein transformer is a stack of **double-stream** blocks
(`transformer_blocks`) followed by **single-stream** blocks
(`single_transformer_blocks`). You choose how many of the *final* blocks in each
stack are trainable; everything before them stays frozen. These variables
control block and edge-layer selection:

| Variable | Default | Effect |
|---|---:|---|
| `TRAIN_LAST_TRANSFORMER_BLOCKS` | `0` | Number of final **double-stream** blocks to train (`0` = none). |
| `TRAIN_LAST_SINGLE_TRANSFORMER_BLOCKS` | `1` | Number of final **single-stream** blocks to train. |
| `TRAIN_OUTPUT_LAYERS` | `1` | Train `norm_out` + `proj_out` (`0` to freeze). |
| `TRAIN_INPUT_LAYERS` | `0` | Also train the input embedders (`x_embedder`, `context_embedder`, `time_guidance_embed`, `pos_embed`). |
| `TRAIN_ALL_TRANSFORMER_BLOCKS` | `0` | Train every double- and single-stream block plus output layers; embeddings and modulation follow their separate flags. |
| `TRAIN_FULL_TRANSFORMER` | `0` | Train every transformer parameter, including embeddings, modulation and output layers. |

A count of `0` trains none of that stack; the count is validated against the
model's real block count, so passing more than exists errors out. The run-tag
shorthand used elsewhere maps directly: `d0-s24-out` = 0 double + 24 single +
output layers. For a model with 8 double and 24 single blocks, `d8-s24-out`
selects all blocks and output layers; other parameters still follow their
separate flags. Use `TRAIN_FULL_TRANSFORMER=1` to train every parameter.

### Examples

**Default light partial fine-tune** (output layers + last single block only) —
just point it at data, change nothing:

```bash
DATASET_SPECS=... OUTPUT_DIR=... bash scripts/training/run_open_visionbanana_pool_flux2_klein.sh
```

**Larger partial fine-tune** — last 2 double + last 8 single blocks + output:

```bash
TRAIN_LAST_TRANSFORMER_BLOCKS=2 \
TRAIN_LAST_SINGLE_TRANSFORMER_BLOCKS=8 \
TRAIN_OUTPUT_LAYERS=1 \
DATASET_SPECS=... OUTPUT_DIR=... bash scripts/training/run_open_visionbanana_pool_flux2_klein.sh
```

For full-transformer tuning, enable `TRAIN_FULL_TRANSFORMER` and use FSDP
to shard the transformer and optimizer state across GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DISTRIBUTED_STRATEGY=fsdp \
TRAIN_FULL_TRANSFORMER=1 \
DATASET_SPECS=... OUTPUT_DIR=... bash scripts/training/run_open_visionbanana_pool_flux2_klein.sh
```

`TRAIN_FULL_TRANSFORMER=1` selects all parameters directly; separate block,
input and output flags are unnecessary. Leave `TRAIN_ALL_TRANSFORMER_BLOCKS=0`
when using it, since the trainer rejects enabling both modes together.

## Multi-GPU strategy

`DISTRIBUTED_STRATEGY` (only matters when more than one GPU is visible):

- **`ddp`** (default) — plain DistributedDataParallel; every GPU holds a full model
  replica. Fine for partial fine-tunes.
- **`fsdp`** — full-shards the transformer across GPUs (`FULL_SHARD`, rank-0
  CPU-offload save). Same math as `ddp`, but lets a full fine-tune fit. The saved
  checkpoint is still a normal `transformer/` dir that evaluation can load.

`NUM_PROCESSES` defaults to the number of GPUs in `CUDA_VISIBLE_DEVICES`.

## Output

The run writes resumable `checkpoint-XXXX/` directories under `OUTPUT_DIR`, each
containing a `transformer/` weights dir (plus a small model card). Pass one
to the evaluation front ends with `--transformer-path .../checkpoint-20000/transformer`,
as described in [EVALUATION.md](../../EVALUATION.md).
## Historical partial-tuning example

These settings describe an earlier run. Use the recipe in TRAIN.md for current
corrected-data training:

| Hyperparameter | Value | Notes |
|---|---|---|
| Trainable scope | `d0-s24-out` | `TRAIN_LAST_TRANSFORMER_BLOCKS=0`, `TRAIN_LAST_SINGLE_TRANSFORMER_BLOCKS=24`, `TRAIN_OUTPUT_LAYERS=1`, `TRAIN_INPUT_LAYERS=0` — partial fine-tune: the last 24 single-stream blocks + `norm_out`/`proj_out`; double-stream blocks and input embedders frozen. |
| `RESOLUTION` | `512` | Training resolution. |
| Batch | `1` per GPU × `GRADIENT_ACCUMULATION_STEPS=2` × 4 GPUs | Effective batch 8. `DISTRIBUTED_STRATEGY=ddp` (a full fine-tune needs `fsdp`). |
| `LEARNING_RATE` | `1e-5` | AdamW (`USE_8BIT_ADAM=0`). |
| `LR_WARMUP_STEPS` | `100` | |
| `MAX_TRAIN_STEPS` | `20000` | |
| `CHECKPOINTING_STEPS` / `CHECKPOINTS_TOTAL_LIMIT` | `1250` / `16` | Set as environment variables (launcher defaults are 5000 / 5); 16 evaluable checkpoints across the run. |
| `MIXED_PRECISION` | `bf16` | |
| `GRADIENT_CHECKPOINTING` | `1` | |
| `OFFLOAD` | `1` | CPU-offload (the `_offload` run-tag suffix). |
| `PROMPT_EMBEDDING_BATCH_SIZE` | `16` | Text-embedding precompute batch. |
| Precompute cache | text embeddings in `PRECOMPUTE_CACHE_DIR`, `CACHE_LATENTS=0` | The cache is dataset/prompt-keyed; clear it when the pool changes. |

The output folder name encodes the run:
`ovb_f2k9b_<DATASET_TAG>_r512_d0-s24-out_lr1e-5_b1-ga2-gpu4_20k_wu100_offload`
(`r`=resolution, `d`/`s`=trainable double/single blocks, `out`=output layers,
`b`/`ga`/`gpu`=batch/grad-accum/GPUs, `20k`=steps, `wu`=warmup). Pool mix
weights and `samples_per_epoch` are **not** in the name — they live in the
run's `dataset_specs.resolved.json`.

Resume a run with `RESUME_FROM_CHECKPOINT=<checkpoint dir | latest>`. Periodic
validation is optional (`VALIDATION_PROMPT`, `VALIDATION_IMAGE`,
`VALIDATION_EPOCHS`, `NUM_VALIDATION_IMAGES`; `SKIP_FINAL_INFERENCE=1` skips
the final render).

## 🔷 FLUX.2-Klein launcher

For a Python environment, install `requirements.txt` and
`requirements/image-generation.txt`. Container users can keep using `SIF`.

```bash
DATASET_SPECS=/prepared/mix.json MODEL=/models/FLUX.2-klein-base-9B \
OUTPUT_DIR=/results/klein scripts/training/run_open_visionbanana_pool_flux2_klein.sh
```

The launcher supports partial, all-block, and full-transformer training and
also accepts `TRAIN_METHOD=lora`. Standard Diffusers LoRA artifacts can be
evaluated with `LORA_PATH`; full-rank artifacts use `TRANSFORMER_PATH`. Loss
masks, weighted pools, aspect buckets, checkpoint resume, gradient
accumulation, and optional FSDP controls are all available. `MODEL` should point
to a complete local snapshot when running without network access.

Dataset specs accept `prepared_folder`, `indexed_prepared_tar`, and
`webdataset_text_to_image` entries. Use
`scripts/training/add_dataset_spec.py --help` to add any of these without
rewriting the rest of an existing mix.

After sourcing `configs/environment.example`, the training wrapper selects the
Klein launcher and derives paths from the configured storage root:

```bash
scripts/training/run_training.sh --profile geometry --model-family klein
scripts/training/run_training.sh --profile image-generation --model-family klein
scripts/training/run_training.sh --profile legacy-pool --model-family klein
```

Without a sourced configuration, pass `--root /path/to/storage`; the wrapper
derives its model, prepared-data, and output locations from that one path.

It runs `scripts/preflight.py` first. Existing launcher variables and direct
commands remain supported; explicit `--model`, `--dataset-specs`, and
`--output-dir` values override the derived paths.

Run the examples in your installed Python environment and select writable
input, output and cache directories. No particular storage layout is required.

## Scope and resume reference

`--trainable_scope_contract release_colorexp` preserves the release's default
parameter selection and is the chosen scope for new corrected-data training.
With non-block tuning enabled, it also tunes the input/context embedders, even
when `TRAIN_INPUT_LAYERS=0`. Use the public launcher with the same scope in masked and unmasked runs. Use `separate_embeddings` to reproduce experiments that
controlled input/context embeddings independently. Neither option changes LoRA
or literal full-transformer tuning. Select the same scope,
prompt, target representation, task weights, and mask settings as the run being
reproduced.

New checkpoints save rank-local noise/timestep RNG states and validate the
training recipe on resume. Loader startup and skipped rows do not advance those
generators. FSDP1 sharded resume restores saved optimizer moments. Migrating an
older checkpoint without these RNG files requires `--allow_legacy_rng_resume`
and is not exact replay; a changed code fingerprint also requires the existing
`--allow_resume_code_change` override after reviewing the change. Use an explicit
seed for controlled comparisons. The corrected RNG policy changes the seeded
trajectory relative to the old release, and portable sharding preserves sampling
probabilities, not the original draw order. Bitwise equality across hardware or
software stacks is not promised.
