# Depth data preparation

Use the [ZS/NZS recipes](../../DATASET.md#complete-zs-and-nzs-recipes) to
prepare training and author-distributed evaluation data. Both model sizes use
the same data. Install the project requirements and `requirements/geometry.txt`.

Raw geometry sources go under `<data-root>/raw/Shared_geo/<dataset>/`.
Prepared versions contain `train/Depth/*.tar` and `eval/Depth/`.
Training archives store metric RGB depth; the trainer resizes the valid metric
field before encoding relative inverse grayscale. See [TRAIN.md](../../TRAIN.md).

The paper evaluation route uses the author benchmark packages, not the optional
raw KITTI/ETH3D/iBims-1 converters. Inspect a source before selecting it:

```bash
python scripts/prepare_data.py --describe nyuv2
python scripts/download_data.py --describe matterport3d
```

Advanced per-source wrappers remain under `scripts/dataset_prep/`.
For existing geometry-v6 packages, use the
[local-package installer reference](../../scripts/dataset_prep/README.md#geometry-data).
These packages use a different layout from public setup.
