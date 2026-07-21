from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

import httpx


class DownloadAborted(Exception):
    def __init__(self, mode: str):
        self.mode = mode
        super().__init__(mode)


def _metadata(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}


def _write_metadata(path: Path, data: dict) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(data))
    os.replace(temporary, path)


def _total_bytes(response: httpx.Response, current: int) -> int:
    content_range = response.headers.get("content-range", "")
    match = re.search(r"/(\d+)$", content_range)
    if match:
        return int(match.group(1))
    length = int(response.headers.get("content-length") or 0)
    return current + length if response.status_code == 206 else length


def download_file(
    url: str,
    destination: str | Path,
    *,
    control: Callable[[], str | None] = lambda: None,
    progress: Callable[[int, int], None] = lambda _current, _total: None,
    client: httpx.Client | None = None,
    headers: dict[str, str] | None = None,
    chunk_size: int = 1024 * 1024,
) -> str:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{target}.part")
    metadata_path = Path(f"{partial}.json")
    metadata = _metadata(metadata_path)
    if metadata and metadata.get("url") != url:
        partial.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        metadata = {}

    current = partial.stat().st_size if partial.exists() else 0
    headers = dict(headers or {})
    if current:
        headers["Range"] = f"bytes={current}-"
    if current and metadata.get("etag"):
        headers["If-Range"] = metadata["etag"]

    owned_client = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=60)
    try:
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code == 416 and current == _total_bytes(response, 0):
                os.replace(partial, target)
                metadata_path.unlink(missing_ok=True)
                progress(current, current)
                return str(target)
            response.raise_for_status()
            if current and response.status_code == 206:
                match = re.match(r"bytes (\d+)-", response.headers.get("content-range", ""))
                if match is None or int(match.group(1)) != current:
                    raise OSError("Server returned an invalid download range")
            if current and response.status_code != 206:
                current = 0
            mode = "ab" if current else "wb"
            total = _total_bytes(response, current)
            etag = response.headers.get("etag")
            _write_metadata(metadata_path, {"url": url, "etag": etag})
            with partial.open(mode) as output:
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    abort = control()
                    if abort:
                        raise DownloadAborted(abort)
                    if not chunk:
                        continue
                    output.write(chunk)
                    output.flush()
                    current += len(chunk)
                    progress(current, total)
    finally:
        if owned_client:
            client.close()

    if total and current != total:
        raise OSError(f"Incomplete download: received {current} of {total} bytes")
    os.replace(partial, target)
    metadata_path.unlink(missing_ok=True)
    return str(target)


def remove_partial(destination: str | Path) -> None:
    partial = Path(f"{destination}.part")
    partial.unlink(missing_ok=True)
    Path(f"{partial}.json").unlink(missing_ok=True)
