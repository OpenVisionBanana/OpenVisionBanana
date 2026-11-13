"""Seekable HTTP range reader for small samples from large official ZIPs."""
import io
import re
import urllib.request


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, block_size=0):
        super().__init__()
        self.url, self.position = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers.get("Content-Length", 0))
            self.etag = response.headers.get("ETag")
        # EPFL's object store returns Content-Length: 0 for HEAD on some
        # nonempty archives. A one-byte range identifies the actual size.
        if self.size <= 0:
            request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
                if response.status != 206 or not match or int(match[1]) < 1:
                    raise ValueError("Cannot determine the source archive size from an exact HTTP range")
                if len(response.read(2)) != 1:
                    raise ValueError("Incomplete source archive size probe")
                self.size = int(match[1])
                self.etag = response.headers.get("ETag")
        self.transferred = 0
        self.block_size = int(block_size)
        self._cache_start, self._cache = 0, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("Negative archive position")
        self.position = position
        return position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        if self.block_size and size >= 0:
            parts = []
            while self.position < end:
                if not self._cache_start <= self.position < self._cache_start + len(self._cache):
                    self._cache_start = self.position
                    block_end = min(self.size, self.position + self.block_size)
                    self._cache = self._range(self.position, block_end)
                count = min(end-self.position, len(self._cache)-(self.position-self._cache_start))
                offset = self.position-self._cache_start
                parts.append(self._cache[offset:offset+count])
                self.position += count
            return b"".join(parts)
        result = self._range(self.position, end)
        self.position = end
        return result

    def _range(self, start, end):
        expected = f"bytes {start}-{end-1}/{self.size}"
        headers = {"Range": f"bytes={start}-{end-1}", "User-Agent": "OVB official-data smoke test"}
        if self.etag:
            headers["If-Match"] = self.etag
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=120) as response:
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("The source server did not honor the exact HTTP range")
            result = response.read(end-start+1)
        if len(result) != end-start:
            raise ValueError("Incomplete HTTP range transfer")
        self.transferred += len(result)
        return result
