import io
import os
from pathlib import Path
import tarfile
import zipfile

from scripts.analysis.inventory_release_storage import archive_inventory, scan_source, unique_bytes


def test_tar_headers_never_extract_traversal(tmp_path):
    path = tmp_path / "source.tar"
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo("../outside")
        member.size = 3
        archive.addfile(member, io.BytesIO(b"abc"))
    result = archive_inventory(path, 10)
    assert result["expanded_bytes"] == 3
    assert result["regular_files"] == 1
    assert not (tmp_path.parent / "outside").exists()
    assert archive_inventory(path, 0)["expanded_bytes"] is None


def test_zip_counts_nested_archive_without_opening_it(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("directory/", "")
        archive.writestr("directory/nested.tar", b"not a tar")
    result = archive_inventory(path, 10)
    assert result["regular_files"] == result["directories"] == result["nested_archives"] == 1
    assert result["expanded_bytes"] == 9
    assert archive_inventory(path, 1)["status"] == "UNKNOWN_MEMBER_LIMIT"


def test_compressed_tar_is_not_opened(tmp_path):
    path = tmp_path / "large.tar.gz"
    # Invalid bytes are intentional: even probing compressed bodies is prohibited.
    path.write_bytes(b"not gzip")
    assert archive_inventory(path, 10)["status"] == "UNKNOWN_COMPRESSED_TAR"


def test_hardlinks_and_file_symlinks_count_once(tmp_path):
    source = tmp_path / "original"
    source.write_bytes(b"abc")
    os.link(source, tmp_path / "hardlink")
    (tmp_path / "symlink").symlink_to(source)
    (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
    result = scan_source(tmp_path)
    assert result["status"] == "INCOMPLETE"
    assert len(result["files"]) == 3
    assert unique_bytes([{"staging": result}]) == (3, 1)


def test_missing_and_bounded_sources(tmp_path):
    assert scan_source(tmp_path / "absent")["status"] == "NOT_STAGED"
    for name in ("a", "b"):
        (tmp_path / name).write_bytes(b"x")
    result = scan_source(tmp_path, entry_limit=1)
    assert result["status"] == "TRUNCATED_ENTRY_LIMIT"
    assert result["entries"] == 1
