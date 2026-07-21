from pathlib import Path
import sys

import pytest

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1] / "ai"
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.services.download_manager import DownloadAborted, download_file  # noqa: E402


class Response:
    def __init__(self, data: bytes, start: int, requests: list[dict]):
        self.status_code = 206 if start else 200
        self.headers = {
            "content-length": str(len(data) - start),
            "etag": '"version-1"',
        }
        if start:
            self.headers["content-range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
        self._data = data[start:]
        self._requests = requests

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def raise_for_status(self):
        return None

    def iter_bytes(self, chunk_size: int):
        for offset in range(0, len(self._data), chunk_size):
            yield self._data[offset : offset + chunk_size]


class Client:
    def __init__(self, data: bytes):
        self.data = data
        self.requests = []

    def stream(self, _method: str, _url: str, *, headers: dict):
        self.requests.append(headers)
        value = headers.get("Range", "bytes=0-")
        return Response(self.data, int(value[6:-1]), self.requests)


def test_paused_download_resumes_from_persisted_byte(tmp_path):
    data = b"0123456789abcdef"
    destination = tmp_path / "model.gguf"
    client = Client(data)
    current = 0

    def progress(downloaded: int, _total: int):
        nonlocal current
        current = downloaded

    with pytest.raises(DownloadAborted, match="pause"):
        download_file(
            "https://example.test/model.gguf",
            destination,
            client=client,
            chunk_size=4,
            control=lambda: "pause" if current >= 4 else None,
            progress=progress,
        )

    partial = Path(f"{destination}.part")
    assert partial.read_bytes() == data[:4]
    assert not destination.exists()

    download_file(
        "https://example.test/model.gguf",
        destination,
        client=client,
        chunk_size=4,
    )

    assert destination.read_bytes() == data
    assert client.requests[-1]["Range"] == "bytes=4-"
    assert client.requests[-1]["If-Range"] == '"version-1"'
    assert not partial.exists()


def test_server_without_range_support_restarts_partial_file(tmp_path):
    data = b"replacement"
    destination = tmp_path / "model.gguf"
    Path(f"{destination}.part").write_bytes(b"stale")
    client = Client(data)

    class NoRangeClient(Client):
        def stream(self, _method: str, _url: str, *, headers: dict):
            self.requests.append(headers)
            return Response(self.data, 0, self.requests)

    download_file("https://example.test/model.gguf", destination, client=NoRangeClient(data))

    assert destination.read_bytes() == data


def test_complete_partial_is_finalized_after_restart(tmp_path):
    data = b"complete"
    destination = tmp_path / "model.gguf"
    Path(f"{destination}.part").write_bytes(data)

    class CompleteResponse(Response):
        def __init__(self):
            self.status_code = 416
            self.headers = {"content-range": f"bytes */{len(data)}"}

        def iter_bytes(self, chunk_size: int):
            return iter(())

    class CompleteClient(Client):
        def stream(self, _method: str, _url: str, *, headers: dict):
            return CompleteResponse()

    download_file("https://example.test/model.gguf", destination, client=CompleteClient(data))

    assert destination.read_bytes() == data
