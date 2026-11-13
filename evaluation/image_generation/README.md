# GenAI-Bench image generation evaluation

Use `scripts/evaluate_generation.py` with the
[public commands](../../EVALUATION.md#image-editing-and-generation).
PD12M supplies training images; GenAI-Bench supplies evaluation prompts.

By default the script evaluates 1,600 prompts with seeds 0-2, 28 steps,
guidance 3.5 and the CLIP-FlanT5-XXL VQAScore judge used in the paper.
Qwen2.5-VL-7B is available with `--evaluator qwen2.5-vl-7b`. Prepare the scorer
as described below before running. With a local base snapshot, prepared prompts
and a local judge snapshot, the evaluation runs offline.

## Scoring environment

Install the scorer from its pinned upstream revision in the scoring environment:

```bash
git clone https://github.com/linzhiqiu/t2v_metrics.git /path/to/t2v_metrics
git -C /path/to/t2v_metrics checkout 6ecb74f92028f42c7e64546d3a71e98c8c73068f
python -m pip install -r requirements/genai-scoring.txt
python -m pip install --no-deps -e /path/to/t2v_metrics
```

### CLIP-FlanT5-XXL (default)

Current `t2v_metrics` releases no longer register CLIP-FlanT5, and its
implementation requires transformers 4.40.2. The script below extracts a minimal
compatibility tree from the pinned older revision of `t2v_metrics`. Together
with a transformers 4.40.2 overlay, it is placed first on the `PYTHONPATH` of
the scoring process:

```bash
python evaluation/image_generation/genai_bench/prepare_clip_flant5_compat.py \
  --dependency_repo /path/to/t2v_metrics --out_dir /storage/ovb/t2v_clip_compat
python -m pip install --target /storage/ovb/clip_flant5_transformers_4_40_2 \
  "transformers==4.40.2" "tokenizers==0.19.1"
```

The judge weights (`zhiqiulin/clip-flant5-xxl`) are downloaded to the Hugging
Face cache on first use. Pass the two directories to the front end:

```bash
python scripts/evaluate_generation.py --model /models/klein \
  --transformer-path /checkpoints/final/transformer --data-root ./data \
  --output-dir ./outputs/generation --num-gpus 4 \
  --score-python /path/to/scoring-environment/bin/python \
  --score-pythonpath /storage/ovb/t2v_clip_compat /storage/ovb/clip_flant5_transformers_4_40_2
```

### Qwen2.5-VL-7B (optional)

Download the judge once, then point scoring at its local snapshot:

```bash
scripts/models/download_model.sh --root /storage/ovb \
  --family qwen2vl7b --stage download
scripts/models/download_model.sh --root /storage/ovb \
  --family qwen2vl7b --stage verify
python scripts/evaluate_generation.py ... --evaluator qwen2.5-vl-7b \
  --score-python /path/to/scoring-environment/bin/python \
  --score-model-path /storage/ovb/models/Qwen2.5-VL-7B-Instruct
```

Use `--skip-score` to generate only.
The wrapper `scripts/evaluation/image_generation/run_genai_bench.sh` also
supports separate `RUN_PREP`, `RUN_GENERATE` and `RUN_SCORE` stages, with
`PROMPTS_JSONL`, `MODEL`, `OUT_DIR` and either `TRANSFORMER_PATH` or
`LORA_PATH`. Its local judge path is `SCORE_MODEL_PATH`.
