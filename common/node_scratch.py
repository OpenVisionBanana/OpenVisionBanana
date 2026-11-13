"""Temporary workspaces and capacity checks, without deleting user trees."""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile


def validate_work_dir(path):
    """Resolve an existing writable work directory without creating files."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
        raise ValueError(f"Work directory must already exist and be writable: {root}")
    return root


def configure_work_dir(path):
    """Apply an explicit CLI override for this process and its child helpers."""
    if path is not None:
        os.environ["OVB_SCRATCH_ROOT"] = str(validate_work_dir(path))


def scratch_root():
    # All environment overrides are optional. Python selects a system temporary
    # directory when no override is present.
    root = Path(os.environ.get("OVB_SCRATCH_ROOT") or os.environ.get("TMPDIR") or tempfile.gettempdir()).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Scratch root must already exist: {root}")
    return root


def require_capacity(path, bytes_needed=0, entries_needed=0):
    reserve = int(os.environ.get("OVB_SCRATCH_RESERVE_BYTES", str(64 * 1024**2)))
    if min(bytes_needed, entries_needed, reserve) < 0:
        raise ValueError("Scratch requirements must be nonnegative")
    free = shutil.disk_usage(path).free
    if free < bytes_needed + reserve:
        raise OSError(f"Insufficient scratch bytes at {path}: need {bytes_needed} plus {reserve} reserve, available {free}")
    # A zero f_files means the filesystem does not expose an inode limit.
    info = os.statvfs(path) if hasattr(os, "statvfs") else None
    if info is not None and info.f_files and info.f_favail < entries_needed:
        raise OSError(f"Insufficient scratch entries at {path}: need {entries_needed}, available {info.f_favail}")


@contextmanager
def workspace(label, bytes_needed=0, entries_needed=0):
    root = scratch_root()
    require_capacity(root, bytes_needed, entries_needed)
    with tempfile.TemporaryDirectory(prefix=f"ovb-{label}-", dir=root) as name:
        yield Path(name)


def conversion_command(command, scratch):
    """Redirect loose geometry outputs, retaining explicit persistent TAR paths."""
    command = list(command)
    for flag, folder in (("--depth-out-root", "depth"), ("--normals-out-root", "normals"),
                         ("--scratch-root", "scratch"), ("--shard-root", "shards")):
        if flag in command:
            command[command.index(flag) + 1] = str(scratch / folder)
    return command


def acquisition_environment(env, scratch, dataset):
    """Move temporary helper trees, never raw inputs or packed destinations."""
    env = dict(env)
    for key in ("WORK", "WAYMO_BUILD_ROOT", "GOOSE_BUILD_ROOT"):
        if key in env:
            env[key] = str(scratch / key.lower() / Path(env[key]).name)
    if dataset in {"acdc", "bdd100k"}:
        env["WORK"] = str(scratch / "work")
    env["TMPDIR"] = str(scratch)
    return env
