#!/usr/bin/env python3
"""Run the unchanged Cityscapes packer in scratch, retaining only its raw TAR."""
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.node_scratch import workspace
from common.official_sources import complete_cityscapes_raw_archive


def acquire(directory, runner=subprocess.run):
    output = directory / "cityscapes.tar"
    if output.exists():
        if not complete_cityscapes_raw_archive(output):
            raise ValueError(f"Incomplete or unsafe existing Cityscapes TAR: {output}")
        print(f"Verified packed Cityscapes source: {output}")
        return
    names = ("gtFine_trainvaltest.zip", "leftImg8bit_trainvaltest.zip")
    inputs = [directory / name for name in names]
    total_bytes, entries = 0, 0
    for path in inputs:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                name = PurePosixPath(member.filename)
                if name.is_absolute() or ".." in name.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Unsafe original ZIP member: {member.filename}")
                total_bytes += member.file_size
                entries += 1
    # The unchanged helper temporarily holds the unpacked tree and its TAR.
    with workspace("cityscapes", bytes_needed=2 * total_bytes, entries_needed=entries + 100) as scratch:
        source = scratch / "cityscapes"
        source.mkdir()
        for path in inputs:
            (source / path.name).symlink_to(path.resolve())
        env = {**os.environ, "ARCHIVE_DIR": str(scratch), "CITYSCAPES_DIR": str(source),
               "PYTHON": sys.executable, "FORCE": "0", "CITYSCAPES_CSDOWNLOAD": "0"}
        runner(["bash", str(ROOT / "scripts/dataset_prep/data_download_scripts/download_datasets_cityscapes.sh")], env=env, check=True)
        packed = scratch / "cityscapes.tar"
        if not complete_cityscapes_raw_archive(packed):
            raise ValueError("Cityscapes helper did not produce a complete original-layout TAR")
        # Publish on the persistent filesystem only after validating the packer.
        import tempfile
        handle, temporary = tempfile.mkstemp(prefix=".cityscapes-", suffix=".partial", dir=directory)
        os.close(handle)
        partial = Path(temporary)
        try:
            shutil.copyfile(packed, partial)
            if output.exists():
                raise FileExistsError(output)
            os.link(partial, output)  # Atomic create, refuses concurrent overwrite.
        finally:
            partial.unlink(missing_ok=True)


if __name__ == "__main__":
    acquire(Path(os.environ["CITYSCAPES_DIR"]).resolve())
