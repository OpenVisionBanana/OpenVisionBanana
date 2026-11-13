"""Network recovery preserves downloaded bytes and distinguishes permanent errors."""
import errno
import io
from pathlib import Path
import socket
import ssl
import sys
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from common import release_data as data

URL = "https://example.org/data"


class Response(io.BytesIO):
    def __init__(self, body, status=200, **headers):
        super().__init__(body)
        self.status, self.headers = status, headers
        self.url = URL


@pytest.fixture
def delays(monkeypatch):
    values = []
    monkeypatch.setattr(data.time, "sleep", values.append)
    return values


def test_interrupted_body_resumes_exact_prefix(tmp_path, monkeypatch, delays):
    class Interrupted(Response):
        def read(self, size=-1):
            if self.tell():
                raise ConnectionResetError("connection reset")
            return super().read(2)
    requests = []
    def open_request(request, **kwargs):
        requests.append(request)
        if len(requests) == 1:
            return Interrupted(b"abcdef", **{"Content-Length": "6"})
        return Response(b"cdef", 206, **{"Content-Length": "4", "Content-Range": "bytes 2-5/6"})
    monkeypatch.setattr(data.urllib.request, "urlopen", open_request)
    output = tmp_path / "data.bin"
    data.download_with_resume(URL, output)
    assert output.read_bytes() == b"abcdef"
    assert requests[1].get_header("Range") == "bytes=2-"
    assert delays == [1.]
    assert not output.with_suffix(".bin.partial").exists()


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_temporary_http_status_retries_and_caps_retry_after(tmp_path, monkeypatch, delays, code):
    calls = []
    def open_request(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(URL, code, "temporary", {"Retry-After": "99999"}, None)
        return Response(b"ready", **{"Content-Length": "5"})
    monkeypatch.setattr(data.urllib.request, "urlopen", open_request)
    data.download_with_resume(URL, tmp_path / "result")
    assert len(calls) == 2 and delays == [60.]


@pytest.mark.parametrize("error", [
    urllib.error.HTTPError(URL, 404, "missing", {}, None),
    urllib.error.HTTPError(URL, 401, "authentication", {}, None),
    urllib.error.HTTPError(URL, 403, "forbidden", {}, None),
    urllib.error.URLError(ssl.SSLCertVerificationError("invalid certificate")),
    urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, "unknown host")),
    OSError(errno.ENOSPC, "disk full"),
    ValueError("checksum mismatch"),
])
def test_permanent_errors_do_not_retry_or_replace_completed_file(tmp_path, monkeypatch, delays, error):
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise error
    monkeypatch.setattr(data.urllib.request, "urlopen", fail)
    output = tmp_path / "result"
    output.write_bytes(b"previous")
    with pytest.raises(type(error)):
        data.download_with_resume(URL, output)
    assert calls == [1] and delays == []
    assert output.read_bytes() == b"previous"


def test_retry_exhaustion_preserves_partial_and_has_bounded_backoff(tmp_path, monkeypatch, delays):
    output = tmp_path / "result"
    partial = output.with_suffix(".partial")
    partial.write_bytes(b"prefix")
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise urllib.error.URLError(TimeoutError("timed out"))
    monkeypatch.setattr(data.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.URLError):
        data.download_with_resume(URL, output)
    assert len(calls) == 3 and delays == [1., 2.]
    assert partial.read_bytes() == b"prefix" and not output.exists()


def test_short_response_retries_but_wrong_resume_range_fails(tmp_path, monkeypatch, delays):
    calls = []
    def open_request(*args, **kwargs):
        calls.append(1)
        return Response(b"ab", **{"Content-Length": "6"}) if len(calls) == 1 else Response(
            b"cdef", 206, **{"Content-Length": "4", "Content-Range": "bytes 1-4/6"})
    monkeypatch.setattr(data.urllib.request, "urlopen", open_request)
    with pytest.raises(ValueError, match="Invalid resume"):
        data.download_with_resume(URL, tmp_path / "result")
    assert len(calls) == 2 and delays == [1.]
    assert (tmp_path / "result.partial").read_bytes() == b"ab"


def test_retry_after_date_and_invalid_values():
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime
    from common.http_retry import retry_delay, transient_download_error
    headers = {"Retry-After": format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))}
    error = urllib.error.HTTPError(URL, 429, "rate limit", headers, None)
    assert 28 < retry_delay(error, 1, 1., 60.) <= 30
    headers["Retry-After"] = "invalid"
    assert retry_delay(error, 2, 1., 60.) == 2.
    assert transient_download_error(urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "temporary DNS")))


def test_real_http_recovery_after_server_error_and_short_transfer(tmp_path, delays, monkeypatch):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            requests.append(self.headers.get("Range"))
            if len(requests) == 1:
                self.send_response(503)
                self.send_header("Retry-After", "0")
                self.end_headers()
            elif len(requests) == 2:
                self.send_response(200)
                self.send_header("Content-Length", "6")
                self.end_headers()
                self.wfile.write(b"ab")
                self.close_connection = True
            else:
                self.send_response(206)
                self.send_header("Content-Range", "bytes 2-5/6")
                self.send_header("Content-Length", "4")
                self.end_headers()
                self.wfile.write(b"cdef")
    server = HTTPServer(("127.0.0.1", 0), Handler)
    # This test talks only to its loopback server, independent of proxy settings.
    opener = data.urllib.request.build_opener(data.urllib.request.ProxyHandler({}))
    monkeypatch.setattr(data.urllib.request, "urlopen", opener.open)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        output = tmp_path / "result"
        data.download_with_resume(f"http://127.0.0.1:{server.server_port}/file", output)
        assert output.read_bytes() == b"abcdef"
        assert requests == [None, None, "bytes=2-"]
        assert delays == [1., 2.]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
