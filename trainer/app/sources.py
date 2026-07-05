"""Training-data sources — upload/register, extract text, chunk.

Admins can supply base training data as PDF files, Excel workbooks, website
links (any HTML page, .aspx included), or free-text prompts. Files and the
manifest live under ``<data_dir>/sources/``.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)

ALLOWED_UPLOAD_EXTS = {".pdf", ".xlsx", ".xls"}
CHUNK_CHARS = 1500
MIN_CHUNK_CHARS = 200

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _sources_dir() -> Path:
    d = settings.data_dir / "sources"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path() -> Path:
    return _sources_dir() / "sources.json"


def list_sources() -> list[dict]:
    path = _manifest_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        logger.exception("Corrupt sources manifest — starting empty")
        return []


def _save_manifest(entries: list[dict]) -> None:
    _manifest_path().write_text(json.dumps(entries, indent=2))


def add_file_source(filename: str, content: bytes) -> dict:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise ValueError(
            f"Unsupported file type {ext!r} — allowed: {sorted(ALLOWED_UPLOAD_EXTS)}"
        )
    sid = uuid.uuid4().hex[:12]
    stored = _sources_dir() / f"{sid}{ext}"
    stored.write_bytes(content)
    entry = {
        "id": sid,
        "type": "excel" if ext in {".xlsx", ".xls"} else "pdf",
        "name": filename,
        "path": stored.name,
        "size_kb": round(len(content) / 1024, 1),
        "added_at": time.time(),
    }
    _save_manifest([*list_sources(), entry])
    return entry


def add_url_source(url: str) -> dict:
    if not re.match(r"^https?://", url):
        raise ValueError("URL must start with http:// or https://")
    entry = {
        "id": uuid.uuid4().hex[:12],
        "type": "url",
        "name": url,
        "url": url,
        "added_at": time.time(),
    }
    _save_manifest([*list_sources(), entry])
    return entry


def add_prompt_source(text: str, name: str | None = None) -> dict:
    if not text.strip():
        raise ValueError("Prompt text is empty")
    sid = uuid.uuid4().hex[:12]
    stored = _sources_dir() / f"{sid}.txt"
    stored.write_text(text)
    entry = {
        "id": sid,
        "type": "prompt",
        "name": name or (text.strip()[:48] + ("…" if len(text.strip()) > 48 else "")),
        "path": stored.name,
        "size_kb": round(len(text) / 1024, 1),
        "added_at": time.time(),
    }
    _save_manifest([*list_sources(), entry])
    return entry


def delete_source(source_id: str) -> bool:
    entries = list_sources()
    keep = [e for e in entries if e["id"] != source_id]
    removed = next((e for e in entries if e["id"] == source_id), None)
    if removed is None:
        return False
    if removed.get("path"):
        (_sources_dir() / removed["path"]).unlink(missing_ok=True)
    _save_manifest(keep)
    return True


# ── Extraction ───────────────────────────────────────────────────────────────


def extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def extract_excel(path: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    parts: list[str] = []
    for ws in wb.worksheets:
        rows = ws.iter_rows(values_only=True)
        header = next(rows, None)
        parts.append(f"# Sheet: {ws.title}")
        if header is None:
            continue
        header = [str(h) if h is not None else f"col{i}" for i, h in enumerate(header)]
        for row in rows:
            cells = [
                f"{header[i]}: {v}"
                for i, v in enumerate(row)
                if v is not None and i < len(header)
            ]
            if cells:
                parts.append("; ".join(cells))
    wb.close()
    return "\n".join(parts)


def extract_url(url: str) -> str:
    import httpx
    from bs4 import BeautifulSoup

    resp = httpx.get(
        url,
        follow_redirects=True,
        timeout=15.0,
        headers={"User-Agent": _USER_AGENT},
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def extract_source(entry: dict) -> str:
    """Extract plain text from a manifest entry."""
    kind = entry["type"]
    if kind == "url":
        return extract_url(entry["url"])
    path = _sources_dir() / entry["path"]
    if kind == "pdf":
        return extract_pdf(path)
    if kind == "excel":
        return extract_excel(path)
    if kind == "prompt":
        return path.read_text()
    raise ValueError(f"Unknown source type {kind!r}")


def chunk_text(text: str) -> list[str]:
    """~CHUNK_CHARS chunks on paragraph boundaries; tiny fragments dropped."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > CHUNK_CHARS:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]
