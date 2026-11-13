#!/usr/bin/env python3
"""Prepare GenAI-Bench image prompts as a stable JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="BaiqiL/GenAI-Bench")
    parser.add_argument("--fallback_dataset_name", default="BaiqiL/GenAI-Bench-1600")
    parser.add_argument(
        "--source_dir",
        help="Optional local directory containing genai_image.json and optionally genai_skills.json.",
    )
    parser.add_argument("--split", default=None, help="Dataset split. Defaults to the first available split.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prompts_name", default="prompts.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no_streaming", action="store_true", help="Disable datasets streaming mode.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    lowered = {str(key).lower(): key for key in row}
    for name in names:
        if name in row:
            return row[name]
        key = lowered.get(name.lower())
        if key is not None:
            return row[key]
    return None


def normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def normalize_id(value: Any, row_index: int) -> str:
    if value is None:
        return f"{row_index:05d}"
    if isinstance(value, int):
        return f"{value:05d}"
    text = str(value).strip()
    if text.isdigit():
        return f"{int(text):05d}"
    return text


def download_hf_file(repo_id: str, filename: str, out_dir: Path) -> Path:
    out_path = out_dir / filename
    if out_path.is_file():
        return out_path
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
        out_path.write_bytes(Path(downloaded).read_bytes())
        return out_path
    except Exception:
        import urllib.request

        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
        with urllib.request.urlopen(url, timeout=120) as response:
            out_path.write_bytes(response.read())
        return out_path


def load_rows_from_genai_json(source_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    image_json = source_dir / "genai_image.json"
    skills_json = source_dir / "genai_skills.json"
    if not image_json.is_file():
        raise FileNotFoundError(f"Missing {image_json}")
    data = json.loads(image_json.read_text(encoding="utf-8"))
    skills = json.loads(skills_json.read_text(encoding="utf-8")) if skills_json.is_file() else {}
    rows = []
    for prompt_id, row in data.items():
        rows.append(
            {
                "Index": prompt_id,
                "Prompt": row["prompt"],
                "Tags": [tag for tag, ids in skills.items() if int(prompt_id) in {int(item) for item in ids}],
            }
        )
    rows.sort(key=lambda row: str(row["Index"]))
    return rows, skills


def iter_dataset_rows(args: argparse.Namespace, out_dir: Path):
    if args.source_dir:
        rows, _ = load_rows_from_genai_json(Path(args.source_dir).expanduser())
        return rows, f"local:{args.source_dir}"

    try:
        from datasets import load_dataset

        dataset_dict = load_dataset(args.dataset_name, streaming=not args.no_streaming)
        split = args.split or next(iter(dataset_dict.keys()))
        return dataset_dict[split], f"{args.dataset_name}:{split}"
    except Exception as exc:
        print(f"Warning: load_dataset({args.dataset_name!r}) failed: {exc}")
        print(f"Falling back to {args.fallback_dataset_name}/genai_image.json")
        download_hf_file(args.fallback_dataset_name, "genai_image.json", out_dir)
        download_hf_file(args.fallback_dataset_name, "genai_skills.json", out_dir)
        rows, _ = load_rows_from_genai_json(out_dir)
        return rows, f"{args.fallback_dataset_name}:json"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    prompts_path = out_dir / args.prompts_name
    tags_path = out_dir / "tags.json"
    info_path = out_dir / "dataset_info.json"

    if prompts_path.exists() and not args.overwrite:
        print(f"Using existing prompt file: {prompts_path}")
        return

    dataset, source = iter_dataset_rows(args, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    tags: dict[str, list[str]] = {}
    count = 0
    with prompts_path.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(dataset, start=1):
            if args.limit is not None and count >= args.limit:
                break
            row = dict(row)
            prompt = first_present(row, ("Prompt", "prompt", "text", "caption"))
            if prompt is None:
                raise ValueError(f"Could not find a prompt field in row {row_index}: {sorted(row)}")
            prompt_id = normalize_id(first_present(row, ("Index", "index", "id", "ID")), row_index)
            row_tags = normalize_tags(first_present(row, ("Tags", "tags", "tag", "skills")))
            record = {
                "id": prompt_id,
                "prompt": str(prompt).strip(),
                "tags": row_tags,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            for tag in row_tags:
                tags.setdefault(tag, []).append(prompt_id)
            count += 1

    tags_path.write_text(json.dumps(tags, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    info_path.write_text(
        json.dumps(
                {
                    "dataset_name": args.dataset_name,
                    "fallback_dataset_name": args.fallback_dataset_name,
                    "source": source,
                    "num_prompts": count,
                "prompts_path": str(prompts_path),
                "tags_path": str(tags_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {count} prompts to: {prompts_path}")
    print(f"Saved tags to: {tags_path}")


if __name__ == "__main__":
    main()
