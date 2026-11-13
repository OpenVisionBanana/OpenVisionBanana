"""Bounded, node-local PNG staging with atomic, throttled shared-FS checkpoints.

Only presentation/storage changes: filenames and PNG pixels are unchanged. Two
queued batches maximum apply backpressure; successful exit drains all copies.
A killed node can lose unflushed batches, which raw-RGB replay regenerates.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import time
import uuid

from evaluation.instanceseg.raw_recovery import save_rgb_atomic


@contextmanager
def copy_slot(directory, slots):
    directory.mkdir(parents=True, exist_ok=True)
    while True:
        for index in range(slots):
            stream = (directory / f"slot_{index:02d}.lock").open("a")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            try:
                yield
            finally:
                stream.close()
            return
        time.sleep(0.1)


class ArtifactWriter:
    def __init__(self, out_dir, stage_dir=None, sync_dir=None, slots=2):
        if slots < 1:
            raise ValueError("Artifact sync slots must be positive")
        self.out = Path(out_dir)
        self.local = Path(stage_dir) if stage_dir else None
        self.sync_dir = Path(sync_dir) if sync_dir else self.out / "raw_recovery/artifact_sync"
        self.slots = slots
        self.pending = []
        self.futures = []
        self.executor = ThreadPoolExecutor(max_workers=1) if self.local else None
        self.copied_files = 0
        self.copy_seconds = 0.0
        self.wait_seconds = 0.0

    def __enter__(self):
        if self.local:
            if self.local.resolve() == self.out.resolve() or self.out.resolve() in self.local.resolve().parents:
                raise ValueError("Artifact staging must be outside the shared output directory")
            self.local.mkdir(parents=True, exist_ok=True)
        return self

    def save(self, image, target):
        target = Path(target)
        relative = target.relative_to(self.out)
        if ".." in relative.parts:
            raise ValueError("Artifact path must stay inside the output directory")
        if self.local is None:
            save_rgb_atomic(image, target)
            return
        # Unique local names also prevent producer/consumer races on retries.
        source = self.local / uuid.uuid4().hex / relative
        save_rgb_atomic(image, source)
        self.pending.append((source, target))

    def _copy(self, files):
        started = time.monotonic()
        with copy_slot(self.sync_dir, self.slots):
            copying = time.monotonic()
            directories = set()
            for source, target in files:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.sync")
                try:
                    with source.open("rb") as reader, temporary.open("xb") as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
                        writer.flush()
                        os.fsync(writer.fileno())
                    # Existing raw predictions must NEVER be replaced by replay.
                    # Panels are derived, and may need repair after an old kill.
                    if target.parent.name == "pred_rgb_raw":
                        try:
                            os.link(temporary, target)  # atomic, no overwrite
                        except FileExistsError:
                            if source.read_bytes() != target.read_bytes():
                                raise ValueError(f"Conflicting saved raw RGB left untouched: {target}")
                    else:
                        os.replace(temporary, target)
                    directories.add(target.parent)
                finally:
                    temporary.unlink(missing_ok=True)
            for directory in directories:
                fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            # Do not remove the local recovery copy until the batch is durable.
            for source, _ in files:
                source.unlink()
                parent = source.parent
                while parent != self.local:
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent
            self.copied_files += len(files)
            self.wait_seconds += copying - started
            self.copy_seconds += time.monotonic() - copying

    def checkpoint(self):
        if not self.pending:
            return
        # Drain errors promptly: at most two queued batches plus the producer's
        # current batch can be uncheckpointed, even if shared storage stalls.
        while self.futures and (self.futures[0].done() or len(self.futures) >= 2):
            self.futures.pop(0).result()
        files, self.pending = self.pending, []
        self.futures.append(self.executor.submit(self._copy, files))

    def flush(self):
        self.checkpoint()
        while self.futures:
            self.futures.pop(0).result()

    def __exit__(self, exc_type, exc, traceback):
        try:
            # Also preserve raw generations after a decoding/panel exception.
            self.flush()
        finally:
            if self.executor:
                self.executor.shutdown(wait=True)
