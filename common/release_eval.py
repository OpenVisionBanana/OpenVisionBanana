"""Shared portable launcher utilities for task-level evaluation entry points."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from common.release_data import safe_extract_tar

KLEIN_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"


def add_common_arguments(parser: argparse.ArgumentParser, datasets: Sequence[str]) -> None:
    parser.add_argument("--model", default=KLEIN_MODEL)
    parser.add_argument("--transformer-path", help="Fine-tuned run, checkpoint, or transformer directory.")
    parser.add_argument("--datasets", default="all", help=f"Comma-separated names or all ({', '.join(datasets)}).")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Prepared root, task directory, extracted dataset, or a single .tar archive.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[],
                        help="Pass one advanced argument to the underlying evaluator; repeat as needed.")
    parser.add_argument("--archive-outputs", action=argparse.BooleanOptionalAction, default=True,
                        help="After a dataset finishes, pack its image artifacts into artifacts.tar inside the output "
                             "directory; *.json results, logs/, scores/ and resume caches stay unpacked and the "
                             "originals are removed once the archive has been verified. An existing artifacts.tar "
                             "is unpacked before the run so resume and extension keep working. "
                             "--no-archive-outputs leaves every file extracted.")


@dataclass(frozen=True)
class NodeGroup:
    """The nodes running one front-end command together, one process per node.

    Shards are numbered across the whole group, so node ``rank`` runs shards
    ``rank * num_gpus`` to ``(rank + 1) * num_gpus - 1``. Steps that need every
    shard (merging, judging, scoring) wait at a file barrier in the shared
    output directory and then run on node 0 only.
    """

    count: int = 1
    rank: int = 0
    sync_dir: Path | None = None
    timeout: int = 7200

    @property
    def main(self) -> bool:
        return self.rank == 0

    def barrier(self, name: str) -> None:
        """Block until every node has reached ``name``; a no-op on one node."""
        if self.count == 1:
            return
        assert self.sync_dir is not None
        self.sync_dir.mkdir(parents=True, exist_ok=True)
        (self.sync_dir / f"{name}.{self.rank}").touch()
        deadline = time.monotonic() + self.timeout
        while True:
            missing = [rank for rank in range(self.count) if not (self.sync_dir / f"{name}.{rank}").exists()]
            if not missing:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"node {self.rank}: barrier {name!r} timed out after {self.timeout}s "
                    f"waiting for node(s) {', '.join(map(str, missing))}"
                )
            time.sleep(5)


SINGLE_NODE = NodeGroup()


def add_node_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-nodes", type=int, default=1,
                        help="Nodes running this same command, one process per node (e.g. srun --ntasks-per-node=1). "
                             "Shards are spread over num-nodes x num-gpus GPUs; --output-dir must be shared.")
    parser.add_argument("--node-rank", type=int, help="This node's rank in [0, num-nodes). Default: SLURM_PROCID, then NODE_RANK.")
    parser.add_argument("--run-id", help="Identifier shared by all nodes of one run, used for barrier files. "
                                         "Default: SLURM_JOB_ID.SLURM_STEP_ID.")
    parser.add_argument("--node-timeout", type=int, default=7200, help="Seconds a node waits at a barrier for the others.")


def node_group(args: argparse.Namespace) -> NodeGroup:
    if args.num_nodes < 1:
        raise SystemExit("--num-nodes must be positive")
    if args.num_nodes == 1:
        return SINGLE_NODE
    rank = args.node_rank
    if rank is None:
        value = os.environ.get("SLURM_PROCID", os.environ.get("NODE_RANK"))
        if value is None:
            raise SystemExit("--num-nodes > 1 needs --node-rank (or SLURM_PROCID / NODE_RANK).")
        rank = int(value)
    if not 0 <= rank < args.num_nodes:
        raise SystemExit(f"--node-rank {rank} is outside [0, {args.num_nodes}).")
    tasks = os.environ.get("SLURM_STEP_NUM_TASKS")
    if tasks is not None and int(tasks) != args.num_nodes:
        raise SystemExit(f"Slurm started {tasks} tasks for --num-nodes {args.num_nodes}; "
                         "launch one process per node (srun --ntasks-per-node=1).")
    run_id = args.run_id
    if run_id is None:
        job = os.environ.get("SLURM_JOB_ID")
        if job is None:
            raise SystemExit("--num-nodes > 1 needs --run-id (or SLURM_JOB_ID).")
        run_id = f"{job}.{os.environ.get('SLURM_STEP_ID', '0')}"
    return NodeGroup(args.num_nodes, rank, args.output_dir / ".node_sync" / run_id, args.node_timeout)


def selected_datasets(value: str, supported: Sequence[str]) -> list[str]:
    values = [item.strip().lower().replace("-", "_") for item in value.split(",") if item.strip()]
    if not values or values == ["all"]:
        return list(supported)
    unknown = sorted(set(values) - set(supported))
    if unknown:
        raise ValueError(f"Unsupported dataset(s): {', '.join(unknown)}")
    return values


def locate_dataset(
    data_root: Path,
    output_root: Path,
    task: str,
    dataset: str,
    names: Sequence[str] = (),
    task_dirs: Sequence[str] = (),
    nodes: NodeGroup = SINGLE_NODE,
) -> Path:
    """Resolve one dataset below ``data_root``.

    ``names`` are extra directory names for the dataset (COCO-Object-80 reads the
    shared ``coco`` root) and ``task_dirs`` extra task folders (``sem_segm``).
    Each node unpacks into its own scratch. Local ranks share a locked cache.
    """
    root = data_root.expanduser().resolve()
    task_names = (task, task.lower(), task.replace("Img", "Img_"), task.replace("Generation", "generation"), *task_dirs, "")
    dataset_names = (dataset, dataset.replace("_", "-"), *names)
    candidates = [root]
    for prefix in (Path(), Path("prepared/V1_prepared_seg_geo"), Path("V1_prepared_seg_geo")):
        for split in (Path("eval"), Path()):
            for task_name in task_names:
                for dataset_name in dataset_names:
                    candidates.extend((root / prefix / split / task_name / dataset_name,
                                       root / prefix / split / task_name / f"{dataset_name}.tar"))
    for candidate in candidates:
        if candidate.is_file() and tar_like(candidate):
            import fcntl
            import hashlib
            import shutil
            import tarfile
            from common.node_scratch import scratch_root, require_capacity
            identity = f"{candidate.resolve()}:{candidate.stat().st_size}:{candidate.stat().st_mtime_ns}"
            key = hashlib.sha256(identity.encode()).hexdigest()
            parent = scratch_root() / "ovb-evaluation-cache" / task.lower() / dataset
            parent.mkdir(parents=True, exist_ok=True)
            cache = parent / key
            stamp = cache / ".ovb_archive_identity"
            with (parent / (key + ".lock")).open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                if not stamp.is_file() or stamp.read_text(encoding="utf-8") != identity:
                    # Remove only the incomplete cache for this exact source key.
                    if cache.exists():
                        shutil.rmtree(cache)
                    with tarfile.open(candidate) as archive:
                        members = archive.getmembers()
                        require_capacity(parent, sum(m.size for m in members if m.isfile()), len(members))
                    safe_extract_tar(candidate, cache)
                    stamp.write_text(identity, encoding="utf-8")
            nodes.barrier(f"extract-{task.lower()}-{dataset}")
            roots = sorted({
                marker.parent
                for pattern in ("dataset_info.json", "nyu_test.txt", "nyu_val.txt")
                for marker in cache.rglob(pattern)
            })
            if len(roots) == 1:
                return roots[0]
            # Official raw TARs can wrap their tree in a dataset-name directory.
            for name in dataset_names:
                if (cache / name).is_dir():
                    return cache / name
            return cache
        if candidate.is_dir():
            # Avoid returning a broad prepared/task root merely because it was
            # the first existing candidate. A direct dataset root either has a
            # matching basename or contains dataset metadata/split files.
            if candidate == root:
                direct_markers = (
                    root.name.lower().replace("-", "_") in {name.lower().replace("-", "_") for name in dataset_names}
                    or (root / "dataset_info.json").is_file()
                    or (root / "metadata.jsonl").is_file()
                    or any(root.glob("*_test.txt"))
                    or any(root.glob("*_val.txt"))
                )
                if not direct_markers:
                    continue
            return candidate
    raise FileNotFoundError(
        f"Cannot locate {task}/{dataset} below {root}. Expected an extracted directory "
        "or <prepared>/V1_prepared_seg_geo/eval/<Task>/<dataset>.tar."
    )


def tar_like(path: Path) -> bool:
    return path.name.endswith((".tar", ".tar.gz", ".tgz"))


def append_if(command: list[str], flag: str, value) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def launch_sharded(
    base_command: list[str],
    num_gpus: int,
    output_dir: Path,
    shard_arguments: Callable[[int, int, Path], list[str]],
    nodes: NodeGroup = SINGLE_NODE,
) -> None:
    if num_gpus < 1:
        raise ValueError("--num-gpus must be positive")
    sync = output_dir / ".shard_sync"
    sync.mkdir(parents=True, exist_ok=True)
    first = nodes.rank * num_gpus
    commands = [base_command + shard_arguments(num_gpus * nodes.count, first + local, sync) for local in range(num_gpus)]
    for command in commands:
        print("+", " ".join(command))
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_devices = [item.strip() for item in inherited.split(",") if item.strip()] if inherited else []
    if visible_devices and len(visible_devices) < num_gpus:
        raise ValueError(
            f"--num-gpus={num_gpus}, but CUDA_VISIBLE_DEVICES exposes only "
            f"{len(visible_devices)} device(s): {inherited!r}"
        )
    processes = []
    for local, command in enumerate(commands):
        environment = os.environ.copy()
        # Each evaluator is an independent process, not a torchrun rank. Expose
        # exactly one allocated device and address it locally as cuda:0.
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices[local] if visible_devices else str(local)
        environment["LOCAL_RANK"] = "0"
        processes.append(
            subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
            )
        )
    codes = [process.wait() for process in processes]
    for command, code in zip(commands, codes):
        if code:
            raise subprocess.CalledProcessError(code, command)


def launch_shard_queue(commands: Sequence[list[str]], num_gpus: int, nodes: NodeGroup = SINGLE_NODE) -> None:
    """Run independent shard commands on ``num_gpus`` devices, one at a time per device.

    For evaluators whose noise depends on the shard layout (ImgEdit seeds one
    generator per shard), the paper's shard count is kept as a logical layout and
    queued over however many GPUs are available, across all nodes. Only valid for
    evaluators whose shards never wait on each other.
    """
    import threading

    if num_gpus < 1:
        raise ValueError("--num-gpus must be positive")
    slots = num_gpus * nodes.count
    first = nodes.rank * num_gpus
    for index, command in enumerate(commands):
        if first <= index % slots < first + num_gpus:
            print("+", " ".join(command))
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_devices = [item.strip() for item in inherited.split(",") if item.strip()] if inherited else []
    if visible_devices and len(visible_devices) < num_gpus:
        raise ValueError(
            f"--num-gpus={num_gpus}, but CUDA_VISIBLE_DEVICES exposes only "
            f"{len(visible_devices)} device(s): {inherited!r}"
        )
    failures: list[subprocess.CalledProcessError] = []

    def worker(slot: int) -> None:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices[slot] if visible_devices else str(slot)
        environment["LOCAL_RANK"] = "0"
        for command in commands[first + slot::slots]:
            code = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=environment).returncode
            if code:
                failures.append(subprocess.CalledProcessError(code, command))
                return

    threads = [threading.Thread(target=worker, args=(slot,)) for slot in range(max(0, min(num_gpus, len(commands) - first)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise failures[0]


ARCHIVE_NAME = "artifacts.tar"
# Top-level entries that stay extracted next to the archive. Dot-prefixed entries
# (shard/node sync dirs) are never packed either.
ARCHIVE_KEEP = ("logs", "scores", "resume_cache")


def archive_members(out_dir: Path, keep: Sequence[str] = ()) -> list[Path]:
    kept = set(ARCHIVE_KEEP) | set(keep) | {ARCHIVE_NAME, ARCHIVE_NAME + ".tmp"}
    return sorted(
        path for path in out_dir.iterdir()
        if path.name not in kept and not path.name.endswith(".json") and not path.name.startswith(".")
    )


def archive_outputs(out_dir: Path, keep: Sequence[str] = ()) -> Path | None:
    """Pack everything but results, logs and caches into ``out_dir/artifacts.tar``.

    Write ``artifacts.tar.tmp``, list it back, swap it into place, and only then
    delete the originals, so an
    interrupted pack leaves either the extracted files or a complete archive.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None
    archive = out_dir / ARCHIVE_NAME
    members = archive_members(out_dir, keep)
    if not members:
        print(f"archive: nothing to pack in {out_dir}")
        return archive if archive.is_file() else None
    tmp = out_dir / (ARCHIVE_NAME + ".tmp")
    before = sum(1 for _ in out_dir.rglob("*"))
    try:
        if archive.is_file():
            shutil.copy2(archive, tmp)
            mode = "a"
        else:
            mode = "w"
        with tarfile.open(tmp, mode) as tar:
            for member in members:
                tar.add(member, arcname=member.name)
        with tarfile.open(tmp, "r") as tar:
            top = {name.split("/", 1)[0] for name in tar.getnames()}
        missing = [m.name for m in members if m.name not in top]
        if missing:
            raise RuntimeError(f"archive is missing {missing}")
        os.replace(tmp, archive)
    except Exception as exc:  # leave the extracted files in place
        tmp.unlink(missing_ok=True)
        print(f"WARNING archive failed for {out_dir}: {exc}; leaving files extracted", file=sys.stderr)
        return None
    for member in members:
        if member.is_dir() and not member.is_symlink():
            shutil.rmtree(member)
        else:
            member.unlink()
    after = sum(1 for _ in out_dir.rglob("*"))
    print(f"archive: {len(members)} entries -> {archive} ({archive.stat().st_size / 1e9:.2f} GB, {before} -> {after} inodes)")
    return archive


def unpack_outputs(out_dir: Path) -> bool:
    """Extract ``out_dir/artifacts.tar`` in place and remove it, so a rerun or
    resume sees the previous files and the final pack starts from scratch."""
    archive = Path(out_dir) / ARCHIVE_NAME
    if not archive.is_file():
        return False
    print(f"archive: unpacking {archive} for resume")
    with tarfile.open(archive, "r") as tar:
        tar.extractall(out_dir, filter="data")
    archive.unlink()
    return True


def prepare_outputs(out_dir: Path, args: argparse.Namespace, nodes: NodeGroup, tag: str) -> None:
    """Before launching: unpack a previous archive (node 0), then let every node proceed."""
    if getattr(args, "archive_outputs", True) and not getattr(args, "dry_run", False):
        if nodes.main:
            unpack_outputs(out_dir)
        nodes.barrier(f"unpacked-{tag}")


def finish_outputs(out_dir: Path, args: argparse.Namespace, nodes: NodeGroup, tag: str, keep: Sequence[str] = ()) -> None:
    """After every node is done with ``out_dir``: pack it on node 0."""
    if not getattr(args, "archive_outputs", True) or getattr(args, "dry_run", False):
        return
    nodes.barrier(f"archive-{tag}")
    if nodes.main:
        archive_outputs(out_dir, keep)


def evaluator_python(relative: str) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve().parents[1] / relative)]


def forward_extra(command: list[str], values: Sequence[str]) -> None:
    command.extend(values)
