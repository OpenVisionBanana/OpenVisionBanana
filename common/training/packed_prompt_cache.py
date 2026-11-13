"""Sharded safetensors cache for FLUX prompt embeddings and text IDs."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FORMAT = "ovb_packed_prompt_cache"
VERSION = 1


def prompt_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def prompt_cache_fingerprint(
    prompts: Iterable[str], model: str, max_sequence_length: int, encoding: str = "flux2"
) -> str:
    hasher = hashlib.sha256()
    for value in (FORMAT, str(VERSION), model, str(max_sequence_length), encoding):
        hasher.update(value.encode("utf-8", errors="surrogatepass"))
        hasher.update(b"\0")
    for key in sorted(prompt_key(prompt) for prompt in set(prompts)):
        hasher.update(key.encode("ascii"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class PackedPromptCacheWriter:
    def __init__(self, root: str | Path, fingerprint: str, writer_id: int, shard_bytes: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.writer_id = int(writer_id)
        self.shard_bytes = max(1, int(shard_bytes))
        self.records: dict[str, str] = {}
        self.tensors: dict[str, torch.Tensor] = {}
        self.buffer_bytes = 0
        self.shard_index = 0

    def add(self, key: str, prompt_embed: torch.Tensor, text_ids: torch.Tensor) -> None:
        embed = prompt_embed.detach().cpu().contiguous()
        ids = text_ids.detach().cpu().contiguous()
        item_bytes = embed.numel() * embed.element_size() + ids.numel() * ids.element_size()
        if self.tensors and self.buffer_bytes + item_bytes > self.shard_bytes:
            self.flush()
        self.tensors[f"e_{key}"] = embed
        self.tensors[f"t_{key}"] = ids
        self.buffer_bytes += item_bytes

    def flush(self) -> None:
        if not self.tensors:
            return
        shard_name = f"prompts-w{self.writer_id:04d}-{self.shard_index:06d}.safetensors"
        final_path = self.root / shard_name
        temporary = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
        save_file(self.tensors, str(temporary), metadata={"format": FORMAT})
        os.replace(temporary, final_path)
        for tensor_name in self.tensors:
            if tensor_name.startswith("e_"):
                self.records[tensor_name[2:]] = shard_name
        self.tensors = {}
        self.buffer_bytes = 0
        self.shard_index += 1

    def close(self) -> Path:
        self.flush()
        partial = self.root / f"partial-w{self.writer_id:04d}.json"
        _atomic_json(
            partial,
            {
                "format": FORMAT,
                "version": VERSION,
                "fingerprint": self.fingerprint,
                "writer_id": self.writer_id,
                "records": self.records,
            },
        )
        return partial


def finalize_packed_prompt_cache(
    root: str | Path, fingerprint: str, num_writers: int, expected_keys: Iterable[str]
) -> Path:
    root = Path(root)
    records: dict[str, str] = {}
    for writer_id in range(int(num_writers)):
        partial_path = root / f"partial-w{writer_id:04d}.json"
        if not partial_path.is_file():
            raise RuntimeError(f"Missing packed prompt-cache writer manifest: {partial_path}")
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("format") != FORMAT or partial.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Incompatible packed prompt-cache writer manifest: {partial_path}")
        overlap = records.keys() & partial["records"].keys()
        if overlap:
            raise RuntimeError(f"Duplicate packed prompt keys from writer {writer_id}: {sorted(overlap)[:3]}")
        records.update(partial["records"])

    expected = set(expected_keys)
    missing = expected - records.keys()
    extra = records.keys() - expected
    if missing or extra:
        raise RuntimeError(
            f"Packed prompt cache coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    manifest_path = root / "manifest.json"
    _atomic_json(
        manifest_path,
        {
            "format": FORMAT,
            "version": VERSION,
            "fingerprint": fingerprint,
            "num_prompts": len(records),
            "records": records,
        },
    )
    return manifest_path


class PackedPromptCache:
    def __init__(self, root: str | Path, fingerprint: str | None = None):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Packed prompt cache manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
            raise RuntimeError(f"Unsupported packed prompt cache: {manifest_path}")
        if fingerprint is not None and manifest.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Packed prompt cache fingerprint mismatch: {manifest_path}")
        self.fingerprint = manifest["fingerprint"]
        self.records: dict[str, str] = manifest["records"]

    def contains_all(self, keys: Iterable[str]) -> bool:
        return all(key in self.records and (self.root / self.records[key]).is_file() for key in keys)

    def load_batch(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        keys = [prompt_key(prompt) for prompt in prompts]
        by_shard: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for position, key in enumerate(keys):
            try:
                shard = self.records[key]
            except KeyError as exc:
                raise KeyError(f"Prompt is absent from packed cache: {key}") from exc
            by_shard[shard].append((position, key))

        embeds: list[torch.Tensor | None] = [None] * len(keys)
        text_ids: list[torch.Tensor | None] = [None] * len(keys)
        for shard, items in by_shard.items():
            shard_path = self.root / shard
            if not shard_path.is_file():
                raise FileNotFoundError(f"Packed prompt shard does not exist: {shard_path}")
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                for position, key in items:
                    embeds[position] = handle.get_tensor(f"e_{key}")
                    text_ids[position] = handle.get_tensor(f"t_{key}")
        return torch.stack(embeds), torch.stack(text_ids)
