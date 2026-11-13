# Model downloads

For offline training or evaluation, download complete model snapshots first. For gated repositories, accept the Hugging Face terms
and store the token in the standard protected file (`chmod 600`) before acquisition.
Never put the token in command-line arguments or commit it to the repository.

```bash
export REPO_ID=black-forest-labs/FLUX.2-klein-base-9B DEST=/persistent/models/FLUX.2-klein-base-9B
STAGE=preflight scripts/models/download_flux2.sh
STAGE=download scripts/models/download_flux2.sh
STAGE=verify scripts/models/download_flux2.sh
```

Set `SIF` to run through a container; otherwise `PYTHON` (default `python3`) is
used from the active environment. Downloads resume through a sibling `.partial`
directory, verify indexed shards, and rename into `DEST` only after validation.
The script downloads the FLUX.2 snapshot given by `REPO_ID` (default
FLUX.2-klein-base-9B). Each model is governed by its upstream license.

After sourcing `configs/environment.example`, the generic wrapper derives every
destination from `MODEL_ROOT`:

```bash
scripts/models/download_model.sh --family klein --stage preflight
scripts/models/download_model.sh --family klein --stage download
scripts/models/download_model.sh --family klein --stage verify

# FLUX.2-klein-base-4B, the base of the 4B model.
scripts/models/download_model.sh --family klein4b --stage preflight
scripts/models/download_model.sh --family klein4b --stage download
scripts/models/download_model.sh --family klein4b --stage verify

# Auxiliary evaluator models.
scripts/models/download_model.sh --family qwen3vl8b --stage download
scripts/models/download_model.sh --family qwen3vl32b --stage download
scripts/models/download_model.sh --family qwen2vl7b --stage download

# Optional ImgEdit baseline, not needed for FLUX evaluation.
scripts/models/download_model.sh --family qwen-image-edit --stage download
```

Without a sourced configuration, add `--root /path/to/storage`; the snapshot is
placed below the derived `models/` directory.

Use `--stage preflight` before, and `--stage verify` after, each download.
Accept any applicable Hugging Face gate first. The protected token-file,
resumable partial directory, structural verification, and atomic publication
rules are shared. The direct `download_flux2.sh` command remains supported.

The instance-segmentation gate uses the local 8B snapshot:

```bash
python evaluation/instanceseg/gate_saco_gold_qwen.py \
  --qwen_model "$QWEN3_VL_8B_MODEL" \
  --saco_root /path/to/extracted/SA-Co-Gold \
  --out "$RAW_ROOT/instanceseg/gate_decisions_qwen.json"
```

The ReasonSeg rewrite pass similarly uses `QWEN3_VL_32B_MODEL` through its
`--qwen_model` argument. These checkpoint-independent passes are run once and
their JSON caches are reused by every evaluated checkpoint; the evaluation
guide contains the benchmark-specific sequence.

GenAI-Bench's optional `qwen2.5-vl-7b` scorer uses the `qwen2vl7b` snapshot; the default CLIP-FlanT5-XXL judge is set up as described in evaluation/image_generation/README.md.
Set `SCORE_MODEL_PATH` to that local directory; sourcing
`configs/environment.example` does this automatically.
