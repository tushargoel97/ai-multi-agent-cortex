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

ALLOWED_UPLOAD_EXTS = {".pdf", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".webp"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_IMAGE_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
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
    if ext in IMAGE_EXTS:
        kind = "image"
    elif ext in {".xlsx", ".xls"}:
        kind = "excel"
    else:
        kind = "pdf"
    entry = {
        "id": sid,
        "type": kind,
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


def _extract_xls(path: Path) -> str:
    import xlrd

    book = xlrd.open_workbook(str(path))
    parts: list[str] = []
    for sheet in book.sheets():
        parts.append(f"# Sheet: {sheet.name}")
        if sheet.nrows == 0:
            continue
        header = [str(sheet.cell_value(0, c)) or f"col{c}" for c in range(sheet.ncols)]
        for r in range(1, sheet.nrows):
            cells = [
                f"{header[c]}: {sheet.cell_value(r, c)}"
                for c in range(sheet.ncols)
                if str(sheet.cell_value(r, c)).strip()
            ]
            if cells:
                parts.append("; ".join(cells))
    return "\n".join(parts)


def extract_excel(path: Path) -> str:
    if path.suffix.lower() == ".xls":
        return _extract_xls(path)
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


def extract_image(path: Path) -> str:
    """Transcribe a document image (spec sheet, screenshot, table) to text via
    the vision-capable QA model. Point TRAINER_QA_* at a model that accepts
    images (e.g. OpenAI gpt-4o) — the default local 1B model can't read images.
    """
    from .qa_generator import transcribe_image

    mime = _IMAGE_MIME.get(path.suffix.lower(), "image/png")
    return transcribe_image(path.read_bytes(), mime)


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
    if kind == "image":
        return extract_image(path)
    if kind == "prompt":
        return path.read_text()
    raise ValueError(f"Unknown source type {kind!r}")


def chunk_text(text: str) -> list[str]:
    """~CHUNK_CHARS chunks. Splits on blank lines first, then on single lines
    for line-based content — stripped web pages and spreadsheets have no blank
    lines, so without the line split the whole source would collapse into one
    giant chunk that overflows the QA model and yields too few Q&A pairs."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= CHUNK_CHARS:
            units.append(para)
        else:  # a spreadsheet dump or stripped web page — split it by line
            units.extend(ln.strip() for ln in para.splitlines() if ln.strip())

    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > CHUNK_CHARS:
            chunks.append(current)
            current = unit
        else:
            current = f"{current}\n{unit}" if current else unit
    if current:
        chunks.append(current)

    sized = [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]
    # Never drop a whole (small) source to nothing.
    return sized or ([text.strip()] if text.strip() else [])
