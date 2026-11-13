# PD12M image generation data

Use the [generation setup](../../DATASET.md#image-generation-and-editing)
and try its small recipe before a full download. Public setup downloads
author metadata and public image URLs under
`<data-root>/raw/ImgGeneration/pd12m/`, validates the shards and writes an
indexed TAR under `train/ImgGeneration/`.

The default requires 250,000 complete image/caption pairs. Insufficient
inputs fail preparation and preserve any previous completed archive.
Use the documented CPU loader check before training.

## Legacy WebDataset wrappers

`download_pd12m.sh` and `prepare_pd12m.sh` create a different training format
that references WebDataset shards without repacking their images.
Their commands and options are in the
[preparation reference](../../scripts/dataset_prep/README.md#legacy-creative-inputs).
