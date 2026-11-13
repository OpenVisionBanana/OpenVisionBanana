#!/usr/bin/env python
"""Run an OVB model on one image with one prompt.

Example:
  python scripts/infer.py --image street.jpg --out seg.png \
    --transformer-path FunAILab/Open-Vision-Banana \
    --subfolder final_models/9B/zeroshot/checkpoint-100000/transformer \
    --prompt '[TASK: SEMANTIC_SEGMENTATION] ...'

Pass --prompt-file to read a long prompt from a text file. Without --image the
model runs text-to-image generation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.flux2_runtime import load_flux2_pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="Input image; omit for text-to-image generation.")
    p.add_argument("--prompt", help="Task prompt (see README quick start).")
    p.add_argument("--prompt-file", help="Read the prompt from this file instead of --prompt.")
    p.add_argument("--out", required=True, help="Output image path.")
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-9B",
                   help="Base FLUX.2 [klein] pipeline (hub id or local snapshot).")
    p.add_argument("--transformer-path", default=None,
                   help="Fine-tuned transformer: hub repo id or local directory.")
    p.add_argument("--subfolder", default=None,
                   help="Subfolder inside --transformer-path when it is a hub repo id.")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--native-resolution", action="store_true",
                   help="Generate at the input image size instead of the pipeline default.")
    p.add_argument("--offload", action="store_true", help="Enable model CPU offload.")
    p.add_argument("--resample", choices=["nearest", "bilinear"], default="nearest",
                   help="Resize of the output back to the input size: nearest keeps mask colours "
                        "exact, bilinear suits depth and normals.")
    args = p.parse_args()

    if bool(args.prompt) == bool(args.prompt_file):
        p.error("give exactly one of --prompt or --prompt-file")
    prompt = args.prompt or Path(args.prompt_file).read_text().strip()

    transformer_path = args.transformer_path
    if transformer_path and args.subfolder and not Path(transformer_path).expanduser().exists():
        from huggingface_hub import snapshot_download

        transformer_path = str(Path(snapshot_download(
            transformer_path, allow_patterns=[f"{args.subfolder.rstrip('/')}/*"]
        )) / args.subfolder)

    pipe, torch, _ = load_flux2_pipeline(args.model, transformer_path=transformer_path, offload=args.offload)

    from PIL import Image

    kwargs = dict(prompt=prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
                  generator=torch.Generator(device="cuda").manual_seed(args.seed))
    image = None
    if args.image:
        image = Image.open(args.image).convert("RGB")
        kwargs["image"] = image
        if args.native_resolution:
            kwargs.update(width=image.width, height=image.height)
    if args.height and args.width:
        kwargs.update(height=args.height, width=args.width)

    with torch.inference_mode():
        out = pipe(**kwargs).images[0]
    if image is not None and out.size != image.size:
        out = out.resize(image.size, Image.NEAREST if args.resample == "nearest" else Image.BILINEAR)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"saved {args.out} ({out.width}x{out.height})")


if __name__ == "__main__":
    main()
