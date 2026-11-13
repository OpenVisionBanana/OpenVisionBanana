"""Stage creative archives and restore previous files if publication fails."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import tempfile


@contextmanager
def staging_root(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Renames and backup hardlinks must stay on the destination filesystem.
    with tempfile.TemporaryDirectory(prefix=".creative-", dir=root) as directory:
        yield Path(directory)


def publish(staged, root, archives, overwrite=False):
    """Publish complete TAR/checksum/index sets under a writer lock.

    Converters and index builders finish before this function is called.
    Rollback covers Python exceptions during replacement. Readers must finish
    before an overwrite; multiple paths cannot be replaced atomically together.
    """
    staged, root = Path(staged), Path(root)
    files = [Path(str(name) + suffix) for name in archives
             for suffix in ("", ".sha256", ".ovb-index.sqlite")]
    for name in files:
        if name.is_absolute() or ".." in name.parts or not (staged / name).is_file():
            raise ValueError(f"Missing or unsafe staged creative artifact: {name}")
    folder = root / "manifests"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "creative-publication.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not overwrite and any((root / name).exists() for name in files):
            raise FileExistsError("Creative outputs already exist; use --overwrite to rebuild")
        backup = staged / "previous"
        backup.mkdir()
        previous, installed = {}, []
        try:
            for index, name in enumerate(files):
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    saved = backup / str(index)
                    os.link(destination, saved)
                    previous[name] = saved
            for name in files:
                os.replace(staged / name, root / name)
                installed.append(name)
        except BaseException:
            for name in reversed(installed):
                if name in previous:
                    os.replace(previous[name], root / name)
                else:
                    (root / name).unlink()
            raise
