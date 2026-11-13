import io
import json
import tarfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import pytest

from common import pico_packed as pico


def fixture(tmp_path, monkeypatch, limit=None):
    monkeypatch.setattr(pico, "SHARD_ROWS", 2)
    raw, scratch = tmp_path / "raw", tmp_path / "scratch"
    scratch.mkdir()
    buffer = io.BytesIO()
    Image.new("RGB", (32, 40), "red").save(buffer, format="JPEG")
    image = buffer.getvalue()
    rows = [dict(open_image_input_url=f"https://example.org/{i}.jpg",
                 output_image=f"images/{i}.png", text=f"Edit {i}") for i in range(5)]
    calls = []
    def fetch(url, path):
        calls.append(url)
        path.write_bytes(("\n".join(json.dumps(r) for r in rows)).encode() if url == pico.METADATA_URL else image)
    settings = {} if limit is None else {"limit": limit}
    return raw, scratch, fetch, calls, settings


@pytest.mark.parametrize("limit,expected", [(None, 5), (3, 3)])
def test_full_and_smoke_use_same_packed_contract(tmp_path, monkeypatch, limit, expected):
    raw, scratch, fetch, calls, settings = fixture(tmp_path, monkeypatch, limit)
    report = pico.acquire(raw, settings, scratch, fetch)
    assert report["total_rows"] == expected
    assert report["smoke_only"] == (limit is not None)
    assert not list(raw.rglob("*.jpg"))
    assert not list(raw.rglob("*.png"))
    calls.clear()
    pico.acquire(raw, settings, scratch, fetch)
    assert calls == []
    output = tmp_path / "prepared.tar"
    assert pico.prepare(raw, output, seed=42) == expected
    with tarfile.open(output) as archive:
        rows = [json.loads(line) for line in archive.extractfile("metadata.jsonl")]
        assert len(rows) == expected
        assert all(row["task"] == "image_editing" for row in rows)
        for row in rows:
            assert row["text"].startswith("[TASK: IMAGE_EDITING]")
            assert Image.open(archive.extractfile(row["image"])).size == (32, 40)


def test_interruption_resumes_completed_shards(tmp_path, monkeypatch):
    raw, scratch, fetch, calls, settings = fixture(tmp_path, monkeypatch)
    def fail(url, path):
        if url == "https://example.org/2.jpg":
            raise OSError("network interruption")
        fetch(url, path)
    with pytest.raises(OSError):
        pico.acquire(raw, settings, scratch, fail)
    assert not (raw / "packed.json").exists()
    first = (raw / "shards/000000.tar").stat().st_mtime_ns
    calls.clear()
    pico.acquire(raw, settings, scratch, fetch)
    assert "https://example.org/0.jpg" not in calls
    assert first == (raw / "shards/000000.tar").stat().st_mtime_ns


def test_corruption_and_changed_acquisition_fail_closed(tmp_path, monkeypatch):
    raw, scratch, fetch, calls, settings = fixture(tmp_path, monkeypatch, 3)
    pico.acquire(raw, settings, scratch, fetch)
    with pytest.raises(ValueError, match="changed"):
        pico.acquire(raw, {}, scratch, fetch)
    path = raw / "shards/000000.tar"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        pico.acquire(raw, settings, scratch, fetch)
    with pytest.raises(ValueError, match="corrupt"):
        pico.prepare(raw, tmp_path / "prepared.tar")


def test_smoke_cannot_satisfy_larger_preparation(tmp_path, monkeypatch):
    raw, scratch, fetch, calls, settings = fixture(tmp_path, monkeypatch, 3)
    pico.acquire(raw, settings, scratch, fetch)
    with pytest.raises(ValueError, match="exceeds"):
        pico.prepare(raw, tmp_path / "prepared.tar", limit=5)
