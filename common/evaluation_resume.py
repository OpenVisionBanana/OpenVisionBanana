"""Durable, low-inode progress journal for long image evaluations.

The journal commits metrics and qualitative outputs in small tar chunks.  A
retry can reconstruct its temporary output tree and skip committed samples.
Configuration and sample-list hashes prevent accidental cross-protocol reuse.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any, Iterable, Sequence


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class DurableEvaluationJournal:
    """Append-only per-sample journal with atomically published tar chunks."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        state_dir: str | Path,
        config: dict[str, Any],
        sample_keys: Sequence[str],
        output_dir: str | Path,
        chunk_size: int = 50,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("resume chunk size must be at least 1")
        if len(sample_keys) != len(set(sample_keys)):
            raise ValueError("durable evaluation sample keys must be unique")

        self.state_dir = Path(state_dir)
        self.output_dir = Path(output_dir)
        self.chunk_dir = self.state_dir / "chunks"
        self.manifest_path = self.state_dir / "resume_manifest.json"
        self.journal_path = self.state_dir / "records.jsonl"
        self.chunk_size = chunk_size
        self.sample_keys = list(sample_keys)
        self._sample_key_set = set(sample_keys)
        self._entries: dict[str, dict[str, Any]] = {}
        self._buffer: list[tuple[dict[str, Any], list[Path]]] = []

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock_handle = (self.state_dir / ".lock").open("a+")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_handle.close()
            raise RuntimeError(
                f"another evaluator is already using resume state {self.state_dir}"
            ) from exc

        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "config": config,
            "config_sha256": _sha256_json(config),
            "sample_count": len(sample_keys),
            "sample_keys_sha256": _sha256_json(list(sample_keys)),
        }
        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing != manifest:
                self.close()
                raise RuntimeError(
                    "durable evaluation resume state does not match this evaluator "
                    f"configuration/sample set: {self.state_dir}. Remove or relocate the "
                    "state directory only after confirming the old run is obsolete."
                )
        else:
            _atomic_json(self.manifest_path, manifest)

        self._load_journal()
        self._discard_unreferenced_chunks()
        self._restore_qualitative_chunks()

    @property
    def completed_keys(self) -> set[str]:
        return set(self._entries)

    def add(
        self,
        kind: str,
        sample_key: str,
        record: dict[str, Any],
        qualitative_paths: Iterable[str | Path] = (),
    ) -> None:
        if kind not in {"record", "skipped"}:
            raise ValueError(f"unsupported durable journal record kind: {kind}")
        if sample_key not in self._sample_key_set:
            raise ValueError(f"sample key is not in the frozen sample list: {sample_key}")
        if sample_key in self._entries or any(item[0]["sample_key"] == sample_key for item in self._buffer):
            raise ValueError(f"sample key was already committed or buffered: {sample_key}")

        paths: list[Path] = []
        for value in qualitative_paths:
            path = Path(value)
            if path.exists():
                try:
                    path.relative_to(self.output_dir)
                except ValueError as exc:
                    raise ValueError(f"qualitative path is outside output directory: {path}") from exc
                paths.append(path)
        entry = {"kind": kind, "sample_key": sample_key, "record": record}
        self._buffer.append((entry, paths))
        if len(self._buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        next_index = len({entry["chunk"] for entry in self._entries.values()})
        digest = hashlib.sha256(
            "\n".join(entry["sample_key"] for entry, _ in self._buffer).encode("utf-8")
        ).hexdigest()[:12]
        chunk_name = f"chunk-{next_index:06d}-{digest}.tar.gz"
        chunk_path = self.chunk_dir / chunk_name
        temporary = self.chunk_dir / f".{chunk_name}.tmp-{os.getpid()}"

        added: set[str] = set()
        with tarfile.open(temporary, "w:gz") as archive:
            for _, paths in self._buffer:
                for path in paths:
                    relative = path.relative_to(self.output_dir).as_posix()
                    if relative not in added:
                        archive.add(path, arcname=relative, recursive=False)
                        added.add(relative)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, chunk_path)

        committed: list[dict[str, Any]] = []
        with self.journal_path.open("a", encoding="utf-8") as handle:
            for entry, _ in self._buffer:
                persisted = {**entry, "chunk": chunk_name}
                handle.write(_canonical_json(persisted) + "\n")
                committed.append(persisted)
            handle.flush()
            os.fsync(handle.fileno())
        for entry in committed:
            self._entries[entry["sample_key"]] = entry
        self._buffer.clear()

    def records_in_sample_order(self, kind: str) -> list[dict[str, Any]]:
        return [
            self._entries[key]["record"]
            for key in self.sample_keys
            if key in self._entries and self._entries[key]["kind"] == kind
        ]

    def records_with_keys_in_sample_order(
        self, kind: str
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return committed records with their frozen sample keys.

        Multi-process evaluation uses the keys to restore the exact global
        dataset order after independently resumable shards finish.
        """

        return [
            (key, self._entries[key]["record"])
            for key in self.sample_keys
            if key in self._entries and self._entries[key]["kind"] == kind
        ]

    def summary(self) -> dict[str, Any]:
        chunks = {entry["chunk"] for entry in self._entries.values()}
        return {
            "enabled": True,
            "schema_version": self.SCHEMA_VERSION,
            "committed_samples": len(self._entries),
            "chunk_count": len(chunks),
            "chunk_size": self.chunk_size,
        }

    def close(self) -> None:
        handle = getattr(self, "_lock_handle", None)
        if handle is not None and not handle.closed:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _load_journal(self) -> None:
        if not self.journal_path.exists():
            return
        valid_bytes = 0
        with self.journal_path.open("rb") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                try:
                    entry = json.loads(line.decode("utf-8"))
                    key = entry["sample_key"]
                    if entry["kind"] not in {"record", "skipped"} or key not in self._sample_key_set:
                        raise ValueError("invalid journal entry")
                    if key in self._entries:
                        raise ValueError(f"duplicate journal sample key: {key}")
                    chunk = self.chunk_dir / entry["chunk"]
                    if not chunk.is_file():
                        raise ValueError(f"journal references missing qualitative chunk: {chunk}")
                except (UnicodeDecodeError, json.JSONDecodeError):
                    # Scheduler termination can tear only the final append. Anything after
                    # the last complete newline is deliberately recomputed.
                    if handle.read(1):
                        raise RuntimeError(f"corrupt non-final journal entry in {self.journal_path}")
                    break
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"invalid durable journal {self.journal_path}: {exc}") from exc
                self._entries[key] = entry
                valid_bytes = handle.tell()
        if self.journal_path.stat().st_size != valid_bytes:
            with self.journal_path.open("r+b") as handle:
                handle.truncate(valid_bytes)
                handle.flush()
                os.fsync(handle.fileno())

    def _discard_unreferenced_chunks(self) -> None:
        referenced = {entry["chunk"] for entry in self._entries.values()}
        for path in self.chunk_dir.iterdir():
            if path.is_file() and (path.name.startswith(".") or path.name not in referenced):
                path.unlink()

    def _restore_qualitative_chunks(self) -> None:
        chunks = sorted({entry["chunk"] for entry in self._entries.values()})
        output_root = self.output_dir.resolve()
        for name in chunks:
            with tarfile.open(self.chunk_dir / name, "r:gz") as archive:
                for member in archive.getmembers():
                    target = (self.output_dir / member.name).resolve()
                    if output_root != target and output_root not in target.parents:
                        raise RuntimeError(f"unsafe path in resume chunk {name}: {member.name}")
                    if not member.isfile():
                        raise RuntimeError(f"unexpected non-file member in resume chunk {name}: {member.name}")
                archive.extractall(self.output_dir)


def summarize_sharded_journals(
    state_dir: str | Path,
    *,
    num_shards: int,
    committed_samples: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Return one accurate resume summary for a completed local shard group."""

    root = Path(state_dir)
    chunk_count = sum(
        1
        for shard_index in range(num_shards)
        for _ in (
            root
            / f"shard-{shard_index:04d}-of-{num_shards:04d}"
            / "chunks"
        ).glob("*.tar.gz")
    )
    return {
        "enabled": True,
        "schema_version": DurableEvaluationJournal.SCHEMA_VERSION,
        "committed_samples": committed_samples,
        "chunk_count": chunk_count,
        "chunk_size": chunk_size,
        "sharded": True,
        "num_shards": num_shards,
    }


def remove_resume_state(state_dir: str | Path) -> None:
    """Remove completed resume state; intended for wrapper-level finalization."""
    shutil.rmtree(Path(state_dir), ignore_errors=True)
