import io
import struct
import zlib

from PIL import Image, PngImagePlugin

from common.indexed_tar import _open_image_with_png_metadata_retry


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _png_with_oversized_icc_chunk() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 1), color=(12, 34, 56)).save(buffer, format="PNG")
    png = buffer.getvalue()
    ihdr_end = 8 + 12 + 13
    profile = zlib.compress(b"x" * (PngImagePlugin.MAX_TEXT_CHUNK + 1))
    iccp = _png_chunk(b"iCCP", b"test-profile\0\0" + profile)
    return png[:ihdr_end] + iccp + png[ihdr_end:]


def test_png_metadata_retry_loads_valid_oversized_icc_and_restores_limit():
    payload = _png_with_oversized_icc_chunk()

    try:
        Image.open(io.BytesIO(payload))
    except ValueError as error:
        assert "PngImagePlugin.MAX_TEXT_CHUNK" in str(error)
    else:
        raise AssertionError("fixture must exceed Pillow's default PNG text-chunk limit")

    original_limit = PngImagePlugin.MAX_TEXT_CHUNK
    image = _open_image_with_png_metadata_retry(lambda: Image.open(io.BytesIO(payload)))

    assert image.size == (2, 1)
    assert image.convert("RGB").getpixel((0, 0)) == (12, 34, 56)
    assert PngImagePlugin.MAX_TEXT_CHUNK == original_limit
