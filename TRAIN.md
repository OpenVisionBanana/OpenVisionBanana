# FLUX.2-Klein 9B training

## On this page

| Section | What you'll find |
|---|---|
| [Input](#input) | Prepared archives and weighted-pool JSON |
| [Run](#run) | Prepare the manifest and launch training |
| ↳ [9B example](#start-the-9b-example) | Launch command, NZS path and batch sizing |
| [Default recipe](#default-recipe) | Model, optimizer, precision and training length |
| [Sampling weights](#sampling-weights) | Task probabilities and source weighting |
| [Geometry targets and masks](#geometry-targets-and-masks) | Encodings, resizing and loss-mask settings |
| [Checkpoints and resume](#checkpoints-and-resume) | Restart requirements and exported weights |

## Input

Training reads prepared folders or indexed TARs through a weighted-pool JSON.
Entries specify paths, names and sampling weights; the loader normalizes weights.
`samples_per_epoch` sets the number of draws.

## Run

### Prepare the training manifest

Prepare the official-source data with
[configs/data_zs.yaml](configs/data_zs.yaml) or
[configs/data_nzs.yaml](configs/data_nzs.yaml), as described in
[DATASET.md](DATASET.md#complete-zs-and-nzs-recipes).
Use the complete manifest as `DATASET_SPECS`, as shown below. The
`zeroshot_pool.json` and `nonzeroshot_pool.json` files contain only other tasks.

Before long training, check a short GPU run that saves and resumes a
checkpoint with the intended model and runtime.

A Slurm version of the 32-GPU 9B run is provided in
[examples/jobs/train_flux2_klein_9b.job](examples/jobs/train_flux2_klein_9b.job).
It defaults to the settings of the released zero-shot 9B model, and its header
lists the overrides of the non-zero-shot run. The account and partition are
passed at submission time.

### Start the 9B example

```bash
DATASET_SPECS=./data/prepared/V1_prepared_seg_geo/manifests/zs_training.json \
OUTPUT_DIR=$PWD/outputs/final_klein \
MODEL=black-forest-labs/FLUX.2-klein-base-9B \
NUM_GPUS=4 \
bash examples/train_flux2_klein_9b.sh
```

For NZS, use
`./data/prepared/V1_prepared_seg_geo_nzs/manifests/nzs_training.json` instead.

The example uses `accelerate launch` with transformer FSDP on the requested
GPUs. It requires FSDP for nearly-full 9B AdamW training. Append trainer
arguments to override defaults. Resume with `--resume_from_checkpoint latest`
or an explicit checkpoint.

The effective global batch size is `NUM_GPUS × PER_DEVICE_BATCH × GRAD_ACCUM`.
The portable example defaults to 64 on four GPUs. The released models used
larger batches: 256 for the zero-shot 9B model (32 B200 GPUs with 192 GB, 8 per
GPU), and 192 for the non-zero-shot 9B model (32 GH200 GPUs with 96 GB, 3 per
GPU × 2 accumulation steps) and both 4B models (16 GH200 GPUs, 6 per GPU × 2).
If memory is insufficient, reduce the per-device batch and increase accumulation
to retain the global batch. The example also leaves the
geometry loss mask disabled, as in the zero-shot runs. Pass
`--loss_mask_policy auto` for the non-zero-shot pool.

## Default recipe

The released models were trained with the settings below. Each model's
`training_args.json` on Hugging Face records them in full. Both model sizes
share the data recipes.

| Setting | 9B | 4B |
|---|---|---|
| Base model | FLUX.2 [klein] Base 9B | FLUX.2 [klein] Base 4B |
| Resolution | 512 with aspect-ratio buckets (512×512, 576×448, 608×416 and their transposes) | same |
| Updated parameters | all double/single transformer blocks, output layers and the other non-block layers. Input and context embedders stay frozen | same |
| Frozen parameters | VAE and text encoder | same |
| Precision | bf16 autocast with fp32 parameters and optimizer state (FSDP policy `fp32_all`) | same |
| Distributed state | FSDP shards transformer parameters, gradients and optimizer state | same |
| Optimizer | AdamW, weight decay `1e-4` | same |
| Learning rate and warm-up | `2e-6`, 1,000 updates | `5e-6`, 500 updates |
| Schedule | `constant_with_warmup` | same |
| Global batch size and hardware | zero-shot: 256 (32 × B200 192 GB, 8 per GPU). Non-zero-shot: 192 (32 × GH200 96 GB, 3 per GPU × 2 accumulation steps) | 192 (16 × GH200 96 GB, 6 per GPU × 2 accumulation steps) |
| Length | 100K updates, 1M samples per epoch drawn with replacement | same |
| Training guidance | 3.5 | same |
| Batches | task-homogeneous. Non-zero-shot runs cap referring expressions at 8 per image | same |
| Geometry loss mask | zero-shot: none (`disabled`). Non-zero-shot: valid pixels only (`auto`), no boundary erosion | same |
| Depth | camera-z, relative inverse grayscale, detailed registered-pixel prompt | same |
| Normals | camera-space `vision_banana_x_left_red`, concise prompt | same |

Final models average the transformer weights of a few late checkpoints. The
averaged checkpoints and their constituents are listed in the
[README model zoo](README.md#-model-zoo).

## Sampling weights

| Task | Sampling probability |
|---|---:|
| Depth | 10% |
| Surface normals | 20% |
| Semantic segmentation | 14.32% |
| Instance segmentation | 14.32% |
| Referring segmentation | 13.16833% |
| Pico-Banana editing | 11.32359% |
| PD12M generation | 11.32359% |
| Identity preservation | 5.54449% |

These are the latest large-data recipe's sampling probabilities, not download
fractions or trainer-enforced constants. ZS geometry sources have equal weights
within each task; NZS source weights are recorded in the selected manifest.

## Geometry targets and masks

The example sets `--depth_target_encoding relative_inverse_grayscale` and
`--depth_target_resize_policy metric_field_then_encode` for the stored metric
RGB depth. Normals use `--normal_target_encoding vision_banana_x_left_red`
and `--normal_target_resize_policy vector_field_renormalize_then_encode`.
The example leaves loss masks disabled, as in the zero-shot runs. Append
`--loss_mask_policy auto` for the non-zero-shot pool, as in the released
non-zero-shot models. Source validity controls resizing in both cases.

Targets are encoded before latent-space loss. A resized validity-mask cell
contributes when its valid coverage reaches the configured threshold. Masks
are neither inputs nor generated outputs. Geometry resizing uses source validity
even without loss masking; normal masks also exclude undefined transformed vectors.

The low-level trainer and pool launcher default to `loss_mask_policy=disabled`.
Use `auto` for masked geometry alongside other tasks that lack masks. `enabled` requires masks
for every task. Both masked and unmasked geometry runs require source validity;
`--allow_missing_geometry_masks` is an explicit legacy-reproduction escape hatch.
Distributed masked loss normalizes across eligible samples or valid elements
on all ranks. Empty masks contribute no supervised weight.

For partial-tuning scope controls and historical reproduction, see the
[launcher reference](scripts/training/README.md#scope-and-resume-reference).
Use identical scope, prompts, encodings, task weights and mask settings when
comparing runs.

## Checkpoints and resume

Checkpoints include optimizer and rank-local RNG state. Resume with the same
recipe; older checkpoints or changed code require explicit migration overrides.
See [resume details](scripts/training/README.md#scope-and-resume-reference).
Portable sharding preserves sampling probabilities, not the old draw order;
bitwise equality across runtimes is not promised.

Without `--push_to_hub`, checkpointing, final transformer export, and model-card
creation are local-only and do not require Hub connectivity. `--push_to_hub`
explicitly enables repository creation and upload.

Evaluation needs the matching base model plus the exported transformer (or a
checkpoint's `transformer/` directory). Continuing training requires its complete
checkpoint state, including optimizer and RNG files; transformer-only or averaged
weights are not a full resumable training state. See [EVALUATION.md](EVALUATION.md).

The authoritative trainer is `train_dreambooth_partial_flux2_klein_img2img.py`.
Run it with `--help` for low-level options. Qwen training is outside this
release interface.

---

The trainer is adapted from the FLUX.2 [klein] LoRA fine-tuning script in the
[diffusers DreamBooth examples](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)
and extended to full fine-tuning on the multi-task pools.
