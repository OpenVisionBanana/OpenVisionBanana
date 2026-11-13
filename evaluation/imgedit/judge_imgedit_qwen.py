"""Score ImgEdit-Bench predictions with the local ImgEdit_Judge checkpoint.

This follows the official ImgEdit-Judge demo from ImgEdit/README.md, but reads
the official Basic/UGE metadata and writes the same result files expected by
the official score scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm


REPO_IMGEDIT_ROOT = Path(__file__).resolve().parent / "ImgEdit"
DEFAULT_BENCHMARK_ROOT = r"\\bstsfs03.intra.dlr.de\AI_Datasets\_unzipped\imgedit_data\Benchmark"
DEFAULT_JUDGE_MODEL = r"\\bstsfs03.intra.dlr.de\AI_Datasets\_unzipped\imgedit_data\ImgEdit_Judge"

UGE_PROMPT = """
Task: Evaluation of Image Editing Process

The task is to evaluate the editing process with a focus on precision and naturalness. Pay special attention to the following criteria:
Adherence to Instructions: The edited image should strictly reflect the instructions, with no additional or missing objects. Any deviation from the specified number of elements, colors, or other detailed instructions should be clearly noted. Any unintended changes or misplacement of objects would result in a lower score.
Naturalness: The editing should appear seamless and realistic. Avoid any appearance of artificial manipulation, such as abrupt color transitions, distorted objects, or overly sharp contrasts that would make the image appear unnatural. Editing should not be glaring or distracting to the viewer.
Editing Scope: The editing should be focused on the areas specified in the instructions. If the editing is too extensive or intrudes into other regions not specified, this should be noted. Similarly, editing should not appear excessive in areas where minimal changes were requested.
Non-Edited Areas: No changes should occur in the regions of the image that were not specified for editing. If the non-edited areas are altered in any way, the final score cannot exceed 3, regardless of the quality of the editing itself.
Scoring Criteria:
1 (Poor): Major errors, instructions ignored, objects added/removed incorrectly, unnatural, poorly executed.
2 (Fair): Significant deviations from instructions, unnatural edits, unappealing result.
3 (Acceptable): Minor deviations, some unnatural elements, editing is noticeable but not distracting.
4 (Good): Mostly follows instructions, minor naturalness issues, aesthetically acceptable.
5 (Excellent): Perfect adherence to instructions, no unnatural elements, aesthetically pleasing and precise edits.

Editing Instruction: <edit_prompt>.

Below are the images before and after editing:

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Score: A number from 1 to 5.
"""


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return rows


def load_edit_info(path: Path) -> dict[str, dict]:
    if path.suffix.lower() == ".jsonl":
        return {str(index): row for index, row in enumerate(read_jsonl(path), start=1)}
    data = read_json(path)
    if isinstance(data, dict):
        return {str(key): value for key, value in data.items()}
    if isinstance(data, list):
        return {str(index): row for index, row in enumerate(data, start=1)}
    raise ValueError(f"Unsupported edit metadata format: {path}")


def sort_json_keys(keys) -> list[str]:
    return list(keys)


def resolve_source_image(origin_root: Path, image_id: str, fallback_search_root: Optional[Path]) -> Path:
    source = origin_root / image_id
    if source.is_file() or fallback_search_root is None:
        return source
    matches = sorted(fallback_search_root.rglob(Path(image_id).name), key=lambda path: path.as_posix())
    if matches:
        return matches[0]
    return source


def default_edit_json(suite: str, benchmark_root: Path) -> Path:
    if suite == "basic":
        official = REPO_IMGEDIT_ROOT / "Benchmark" / "Basic" / "basic_edit.json"
        if official.is_file():
            return official
        return benchmark_root / "singleturn" / "singleturn.json"
    if suite == "uge":
        official = REPO_IMGEDIT_ROOT / "Benchmark" / "UGE" / "UGE_edit.json"
        if official.is_file():
            return official
        return benchmark_root / "hard" / "annotation.jsonl"
    raise ValueError(f"Unsupported scoring suite: {suite}")


def default_basic_prompts_json(benchmark_root: Path) -> Path:
    official = REPO_IMGEDIT_ROOT / "Benchmark" / "Basic" / "prompts.json"
    if official.is_file():
        return official
    return benchmark_root / "singleturn" / "judge_prompt.json"


def default_origin_root(benchmark_root: Path, suite: str) -> Path:
    if suite == "basic":
        return benchmark_root / "singleturn"
    if suite == "uge":
        return benchmark_root / "hard"
    raise ValueError(f"Unsupported scoring suite: {suite}")


@dataclass(frozen=True)
class ShardInfo:
    rank: int
    world_size: int
    local_rank: int
    device: str
    distributed: bool


def resolve_shard(args) -> ShardInfo:
    """Resolve this process's shard, mirroring inference_flux2_klein_imgedit.py.

    Two ways to shard, same as generation: torchrun (RANK/WORLD_SIZE, one job,
    ranks can barrier) or manual --num_shards/--shard_index (independent jobs
    that finish at different times, so no barrier and no rank-0 merge).
    """
    import torch

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", str(env_rank)))

    if args.num_shards is not None or args.shard_index is not None:
        world_size = args.num_shards if args.num_shards is not None else 1
        rank = args.shard_index if args.shard_index is not None else 0
        distributed = False
    else:
        world_size = env_world_size
        rank = env_rank
        distributed = world_size > 1

    if world_size < 1:
        raise ValueError(f"--num_shards/WORLD_SIZE must be positive, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Shard/rank index {rank} is out of range for {world_size} shards.")

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        local_rank = args.local_rank if args.local_rank is not None else env_local_rank
        local_rank = int(local_rank) % max(num_devices, 1)
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        local_rank = 0
        device = "cpu"

    if (
        distributed
        and torch.distributed.is_available()
        and not torch.distributed.is_initialized()
    ):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)

    return ShardInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        distributed=distributed and torch.distributed.is_initialized(),
    )


def barrier(shard: ShardInfo) -> None:
    if shard.distributed:
        import torch

        torch.distributed.barrier()


def cleanup_distributed(shard: ShardInfo) -> None:
    if shard.distributed:
        import torch

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def shard_result_path(result_dir: Path, shard: ShardInfo) -> Path:
    return result_dir / "shard_results" / f"rank_{shard.rank:05d}_of_{shard.world_size:05d}.json"


def merge_shard_results(result_dir: Path, keys: list[str]) -> dict:
    """Merge every per-rank shard result into one dict, in original key order.

    Safe to call only when all shards are known to have finished writing.
    """
    merged: dict = {}
    for path in sorted((result_dir / "shard_results").glob("rank_*_of_*.json")):
        merged.update(read_json(path))
    return {key: merged[key] for key in keys if key in merged}


def score_text_average(text: str) -> Optional[float]:
    scores = []
    for line in text.splitlines():
        match = re.search(r":\s*([1-5])(?:\D|$)", line.strip())
        if match:
            scores.append(int(match.group(1)))
    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def load_judge(model_path: Path, flash_attention: bool, device: Optional[str] = None):
    import os
    import torch

    # device_map="auto" + flash_attention_2 lands on CPU and raises a cryptic
    # "FlashAttention2 not available on CPU" error when CUDA isn't visible (a
    # transient NVML/driver hiccup, or a stale CUDA_VISIBLE_DEVICES leaking onto a
    # 1-GPU node). Fail fast with an actionable message instead.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible to this judge process, so the Qwen2.5-VL judge cannot run. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}. "
            "This is usually a stale CUDA_VISIBLE_DEVICES (e.g. '0,1,2,3' on a 1-GPU node) "
            "or a transient driver/NVML error. Ensure the job pins CUDA_VISIBLE_DEVICES to "
            "its allocated GPU(s) and resubmit."
        )

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # device_map="auto" BALANCES a model across every visible GPU rather than
    # filling one, so under sharding each rank would pipeline-split its own copy
    # across all 8 devices instead of owning one. Pin the whole model to this
    # rank's device; "auto" stays the default for the single-process path.
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto" if device is None else {"": device},
        "local_files_only": True,
    }
    if flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path), **kwargs)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        min_pixels=1016064,
        max_pixels=1354752,
        local_files_only=True,
    )
    return model, processor, process_vision_info


def infer_score(model, processor, process_vision_info, prompt: str, original_path: Path, result_path: Path, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": str(original_path)},
                {"type": "image", "image": str(result_path)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("basic", "uge"), default="basic")
    parser.add_argument("--benchmark_root", default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--edit_json", default=None)
    parser.add_argument("--origin_img_root", default=None)
    parser.add_argument(
        "--fallback_search_root",
        default=None,
        help="Optional tree searched by basename when a metadata image is absent from origin_img_root.",
    )
    parser.add_argument("--result_img_folder", required=True)
    parser.add_argument("--prompts_json", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--no_flash_attention", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Manual sharding: total number of judge shards. Torchrun WORLD_SIZE is used if omitted.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help="Manual sharding: this shard index. Torchrun RANK is used if omitted.",
    )
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument(
        "--merge_shards",
        action="store_true",
        help="Skip scoring; merge existing shard_results/ into result.json and aggregate. "
        "Use after a cross-job --num_shards run, where no rank can safely merge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root).expanduser()
    judge_model = Path(args.judge_model).expanduser()
    edit_json = Path(args.edit_json).expanduser() if args.edit_json else default_edit_json(args.suite, benchmark_root)
    origin_root = Path(args.origin_img_root).expanduser() if args.origin_img_root else default_origin_root(benchmark_root, args.suite)
    fallback_search_root = (
        Path(args.fallback_search_root).expanduser()
        if args.fallback_search_root
        else (benchmark_root if args.suite == "uge" else None)
    )
    result_dir = Path(args.result_img_folder).expanduser()
    # Edited prediction images live in the edited_images/ subfolder written by the
    # inference script; the judge's own JSON outputs stay at the result_dir root.
    edited_images_dir = result_dir / "edited_images"
    result_json = result_dir / "result.json"
    average_json = result_dir / "average_score.json"
    type_json = result_dir / "typescore.json"

    edit_info = load_edit_info(edit_json)
    prompts = None
    if args.suite == "basic":
        prompts_json = Path(args.prompts_json).expanduser() if args.prompts_json else default_basic_prompts_json(benchmark_root)
        prompts = read_json(prompts_json)

    keys = sort_json_keys(edit_info.keys())
    if args.limit is not None:
        keys = keys[: args.limit]

    if args.merge_shards:
        results = merge_shard_results(result_dir, keys)
        if not results:
            raise FileNotFoundError(f"No shard results to merge under {result_dir / 'shard_results'}")
        with result_json.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f"Merged shard results -> {result_json} ({len(results)}/{len(keys)} samples)")
        write_aggregates(args, edit_info, results, result_dir, result_json, average_json, type_json)
        return

    shard = resolve_shard(args)
    sharded = shard.world_size > 1
    my_keys = list(keys[shard.rank :: shard.world_size]) if sharded else keys

    # Each rank validates only its own slice; the union covers every key.
    for key in my_keys:
        original = resolve_source_image(origin_root, edit_info[key]["id"], fallback_search_root)
        result = edited_images_dir / f"{key}.png"
        if not original.is_file():
            raise FileNotFoundError(f"Missing original image for key {key}: {original}")
        if not result.is_file():
            raise FileNotFoundError(f"Missing generated result for key {key}: {result}")
        Image.open(original).verify()
        Image.open(result).verify()

    if shard.rank == 0:
        print(f"Suite: {args.suite}")
        print(f"Judge model: {judge_model}")
        print(f"Edit JSON: {edit_json}")
        print(f"Origin image root: {origin_root}")
        print(f"Result image folder: {result_dir}")
        print(f"Scoring samples: {len(keys)}")
    if sharded:
        print(f"Rank {shard.rank}/{shard.world_size} on {shard.device}: {len(my_keys)} samples")
    if args.dry_run:
        cleanup_distributed(shard)
        return

    # Sharded ranks each own a disjoint slice, so they write disjoint shard files
    # instead of racing on one result.json; rank 0 merges them at the barrier.
    store_json = shard_result_path(result_dir, shard) if sharded else result_json
    store_json.parent.mkdir(parents=True, exist_ok=True)

    existing = read_json(store_json) if store_json.is_file() and not args.overwrite else {}
    model, processor, process_vision_info = load_judge(
        judge_model,
        flash_attention=not args.no_flash_attention,
        device=shard.device if sharded else None,
    )

    results = dict(existing)
    progress = tqdm(
        my_keys,
        desc=f"Scoring {args.suite}" if not sharded else f"Rank {shard.rank}/{shard.world_size}",
        position=shard.rank,
        leave=shard.rank == 0,
    )
    for key in progress:
        if key in results and not args.overwrite:
            continue
        item = edit_info[key]
        if args.suite == "basic":
            template = prompts[item["edit_type"]]
        else:
            template = UGE_PROMPT
        prompt = template.replace("<edit_prompt>", item["prompt"])
        original = resolve_source_image(origin_root, item["id"], fallback_search_root)
        result = edited_images_dir / f"{key}.png"
        results[key] = infer_score(
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            prompt=prompt,
            original_path=original,
            result_path=result,
            max_new_tokens=args.max_new_tokens,
        )
        with store_json.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)

    barrier(shard)

    if sharded:
        # Merge only in the torchrun path, where all shard files are present at
        # the barrier; cross-job shards are merged by a separate --merge_shards pass.
        if not shard.distributed:
            print(
                f"Rank {shard.rank} wrote {store_json}. Cross-job sharding does not merge; "
                f"run with --merge_shards once every shard has finished."
            )
            cleanup_distributed(shard)
            return
        if shard.rank != 0:
            cleanup_distributed(shard)
            return
        results = merge_shard_results(result_dir, keys)
        with result_json.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f"Merged {shard.world_size} shard results -> {result_json} ({len(results)} samples)")

    write_aggregates(args, edit_info, results, result_dir, result_json, average_json, type_json)
    cleanup_distributed(shard)


def write_aggregates(args, edit_info, results, result_dir, result_json, average_json, type_json) -> None:
    averages = {
        key: avg
        for key, value in results.items()
        if (avg := score_text_average(value if isinstance(value, str) else str(value))) is not None
    }
    with average_json.open("w", encoding="utf-8") as handle:
        json.dump(averages, handle, indent=2)

    if args.suite == "basic":
        type_scores: dict[str, list[float]] = {}
        for key, score in averages.items():
            edit_type = edit_info[key].get("edit_type")
            if edit_type:
                type_scores.setdefault(edit_type, []).append(score)
        type_averages = {
            edit_type: round(sum(scores) / len(scores), 2)
            for edit_type, scores in type_scores.items()
            if scores
        }
        with type_json.open("w", encoding="utf-8") as handle:
            json.dump(type_averages, handle, indent=2)

    final_average = round(sum(averages.values()) / len(averages), 2) if averages else None
    summary = {
        "suite": args.suite,
        "num_scored": len(averages),
        "average_score": final_average,
        "result_json": str(result_json),
        "average_score_json": str(average_json),
        "typescore_json": str(type_json) if args.suite == "basic" else None,
    }
    with (result_dir / "score_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()