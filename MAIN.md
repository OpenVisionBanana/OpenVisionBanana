# Open Vision Banana release guide

## On this page

| Section | What you'll find |
|---|---|
| [Scope](#scope) | Supported models and tasks |
| [Installation](#installation) | Python environment, dependencies and model access |
| [Quick start](#quick-start) | Small setup checks and links to the full workflows |
| [Reproducibility rules](#reproducibility-rules) | Target conventions, masks, seeds and provenance |

## Scope

This repository provides FLUX.2-Klein 4B/9B training and evaluation for semantic,
instance and referring segmentation, depth, surface normals, image editing and
image generation. The linked guides document the supported commands.
Classification and Qwen training are outside this release interface.

The public scripts accept one dataset, a comma-separated list, or `all`.
Choose your data directory explicitly. Downloads and preparation use the system
temporary directory unless you select an existing writable directory with
`--work-dir`. No environment variables are required.

## Installation

Create a Python 3.11 environment and install the pinned requirements. They
include the CUDA 12.8 builds of PyTorch 2.11 from the PyTorch index. For another
CUDA version, install a matching PyTorch build first and remove the three torch
pins from `requirements.txt`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements/geometry.txt
python -m pip check
```

FLUX.2-Klein weights are subject to their upstream license and access conditions. Authenticate with Hugging Face when using a gated model ID, or pass a local snapshot through `--model` / `MODEL`.

Optional path defaults are in `configs/environment.example`.

## Quick start

```bash
python scripts/download_data.py --list
python scripts/download_data.py --describe matterport3d

# Evaluation-data preparation only, in separate directories.
python scripts/setup_data.py --config configs/data_depth_smoke.yaml --data-root ./depth_smoke
python scripts/setup_data.py --config configs/data_normals_smoke.yaml --data-root ./normals_smoke
# Small training-data preparation checks.
python scripts/setup_data.py --config configs/data_editing_smoke.yaml --data-root ./editing_smoke
python scripts/setup_data.py --config configs/data_generation_smoke.yaml --data-root ./generation_smoke

# Inspect complete-recipe prerequisites before acquisition.
python scripts/setup_data.py --config configs/data_zs.yaml --data-root ./data --dry-run
```

See [DATASET.md](DATASET.md), [TRAIN.md](TRAIN.md), and [EVALUATION.md](EVALUATION.md) before a full run.

Use the [ZS and NZS recipes](DATASET.md#complete-zs-and-nzs-recipes) for complete
download and preparation. The 4B and 9B models share those data configurations.
Sources come from their authors or independent public distributions.

Restricted-source access and destination layouts are in
[DATASET.md](DATASET.md#how-to-download-the-data).

## Reproducibility rules

- Prompts and target encodings form one contract. A checkpoint trained for inverse-gray depth must be evaluated with the same representation unless decoder mismatch is an explicit diagnostic.
- Depth is projective camera-z, never an undeclared mixture of Euclidean range and z-depth.
- Normals use the declared camera convention and `vision_banana_x_left_red` RGB encoding.
- Invalid geometry pixels are excluded from evaluation and masked training. Unmasked training intentionally includes their target fill; source validity is still required for geometry-aware preprocessing. Loss masks are resized to the latent grid and are not model inputs.
- Per-sample seeds are stable across one- and multi-GPU evaluation. Resume state is tied to the dataset and evaluation configuration.
- Prepared archives record provenance and checksums. Scores from different decoding/calibration lanes are not interchangeable.
