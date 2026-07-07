"""Spec importers: deterministic fast paths + the intelligent crawl agent.

Two deterministic paths stay because they beat generic crawling:
  - AMD's official database (amd.com) embeds the whole processor catalog as
    HTML-escaped JSON — one fetch, no anti-bot wall (scrape_amd).
  - Uploaded documents (Intel comparison-chart PDFs, spreadsheets) via
    import_document (deterministic chart parser first, LLM distillation next).
Every other URL is crawled by the built-in scrape agent (run_scrape_agent,
below), an LLM-driven index/leaf loop. parse_product stays as a deterministic
spec-sheet extractor the agent tries before falling back to distillation. All
writes go through save_learned_entry (alias-aware; never overrides facts.yaml).
"""

from __future__ import annotations

import html as html_lib
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from urllib import robotparser
from urllib.parse import urldefrag, urljoin, urlparse

import httpx

from app.research import (
    _chat,
    _first_json,
    _validate_entry,
    distill_products_from_text,
    save_learned_entry,
)

logger = logging.getLogger(__name__)

REQUEST_DELAY_S = 2.5
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

_TAG_RE = re.compile(r"<[^>]+>")
_PAIR_RES = (
    re.compile(r"<th[^>]*>(.*?)</th>\s*<td[^>]*>(.*?)</td>", re.DOTALL),
    re.compile(r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>", re.DOTALL),
)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.DOTALL)


def _clean(fragment: str) -> str:
    return html_lib.unescape(_TAG_RE.sub(" ", fragment)).replace("\xa0", " ").strip()


def _fetch(client: httpx.Client, url: str) -> str:
    resp = client.get(url, headers=_HEADERS, timeout=25, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _spec_pairs(page: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for pattern in _PAIR_RES:
        for label, value in pattern.findall(page):
            label, value = _clean(label).rstrip(":"), _clean(value)
            if label and value and label not in pairs:
                pairs[label] = value
    return pairs


def _year(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"(19|20)\d{2}", text)
    return int(m.group(0)) if m else None


def _price(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"\$?\s*([\d,]+)", text)
    if not m:
        return None
    value = int(m.group(1).replace(",", ""))
    return value if 10 <= value <= 100_000 else None


def _tflops(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"([\d.]+)\s*TFLOPS", text, re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1))
    return value if 0.1 <= value <= 5000 else None


def _watts(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"(\d+)\s*W", text)
    return int(m.group(1)) if m else None


def _aliases(name: str) -> list[str]:
    """Short forms users actually type: 'GeForce RTX 4090' -> 'RTX 4090'."""
    out = []
    for prefix in ("GeForce ", "Radeon RX ", "Radeon ", "Core ", "Ryzen "):
        if name.startswith(prefix) and prefix in ("GeForce ", "Radeon "):
            out.append(name.removeprefix(prefix).strip())
    m = re.search(r"\b(i[3579]-\d{4,5}\w*|\d{4}X3D|\d{4}[A-Z]*)$", name)
    if m and m.group(0) != name:
        out.append(m.group(0))
    return [a for a in out if a and a != name]


def parse_product(url: str, page: str) -> dict | None:
    h1 = _H1_RE.search(page)
    name = _clean(h1.group(1)) if h1 else ""
    name = re.sub(r"\s+(Specs|Review).*$", "", name).strip()
    if not name:
        return None
    is_gpu = "/gpu-specs/" in url
    specs = _spec_pairs(page)

    def g(*labels: str) -> str | None:
        for label in labels:
            if specs.get(label):
                return specs[label]
        return None

    cores, threads = g("# of Cores", "Cores"), g("# of Threads", "Threads")
    base_clock = g("Frequency", "Base Clock", "GPU Clock")
    boost = g("Turbo Clock", "Boost Clock")
    cpu_desc = None
    gpu_desc = None
    if is_gpu:
        shaders = g("Shading Units", "CUDA Cores", "Stream Processors")
        bits = [b for b in (shaders and f"{shaders} shaders", boost and f"boost {boost}") if b]
        gpu_desc = ", ".join(bits) or None
    else:
        bits = [
            b
            for b in (
                cores and f"{cores} cores",
                threads and f"{threads} threads",
                base_clock and f"@ {base_clock}",
                boost and f"boost {boost}",
            )
            if b
        ]
        cpu_desc = ", ".join(bits) or None

    memory = None
    if is_gpu:
        size, mtype = g("Memory Size"), g("Memory Type")
        memory = " ".join(x for x in (size, mtype) if x) or None
    else:
        memory = g("Memory Support", "Memory")

    brand = name.split()[0]
    if brand in ("GeForce", "RTX", "GTX", "Quadro"):
        brand = "NVIDIA"
    elif brand in ("Radeon", "RX"):
        brand = "AMD"
    elif brand in ("Core", "Xeon", "Arc"):
        brand = "Intel"
    elif brand == "Ryzen":
        brand = "AMD"

    entry = {
        "name": name,
        "brand": brand,
        "category": "GPU" if is_gpu else "CPU",
        "release_year": _year(g("Release Date", "Released")),
        "launch_price_usd": _price(g("Launch Price", "Current Price")),
        "cpu": cpu_desc,
        "gpu": gpu_desc,
        "compute_tflops": _tflops(g("FP32 (float)", "FP32 (float) performance")),
        "memory": memory,
        "memory_bandwidth": g("Bandwidth", "Memory Bandwidth"),
        "storage": None,
        "display": None,
        "power_watts": _watts(g("TDP", "TBP")),
        "key_features": [
            f
            for f in (
                g("Cache L3") and f"L3 cache {g('Cache L3')}",
                g("Process Size") and f"{g('Process Size')} process",
                g("Socket") and f"Socket {g('Socket')}",
                g("Memory Bus") and f"{g('Memory Bus')} memory bus",
            )
            if f
        ][:4],
        "best_for": ["Gaming", "Content Creation"] if not is_gpu else ["Gaming", "Graphics workloads"],
        "aliases": _aliases(name),
        "exists": True,
        "notes": f"Specs from TechPowerUp ({url})",
    }
    # A usable entry needs at least a couple of hard facts.
    hard_facts = sum(
        1
        for k in ("release_year", "launch_price_usd", "compute_tflops", "power_watts", "memory")
        if entry.get(k)
    )
    return entry if hard_facts >= 2 else None


# ── AMD official spec database ──────────────────────────────────────────────
# https://www.amd.com/en/products/specifications/processors.html embeds the
# ENTIRE processor database as HTML-escaped JSON product objects — one fetch,
# no per-product crawling, and no anti-bot wall (unlike TechPowerUp).

_AMD_PRODUCT_ANCHOR = '"model":"amd-site/models/processors"'
_AMD_MAX_OBJECT = 100_000  # product objects are ~5-8 KB; cap runaway scans


def _amd_val(elements: dict, key: str):
    field = elements.get(key) or {}
    value = field.get("formatValue", field.get("value"))
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _amd_objects(page: str) -> list[dict]:
    import json as json_lib

    text = html_lib.unescape(page)
    objects: list[dict] = []
    seen_titles: set[str] = set()
    pos = 0
    while True:
        anchor = text.find(_AMD_PRODUCT_ANCHOR, pos)
        if anchor == -1:
            break
        pos = anchor + 1
        start = text.rfind("{", 0, anchor)
        for _ in range(10):  # walk back until the balanced object parses
            if start == -1:
                break
            depth, i, end = 0, start, -1
            limit = min(len(text), start + _AMD_MAX_OBJECT)
            while i < limit:
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
                i += 1
            if end != -1:
                try:
                    obj = json_lib.loads(text[start : end + 1])
                except Exception:  # noqa: BLE001
                    obj = None
                if obj is not None and "elements" in obj:
                    title = obj.get("title", "")
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        objects.append(obj)
                    break
            start = text.rfind("{", 0, start)
    return objects


def scrape_amd(client: httpx.Client, url: str, limit: int, on_progress) -> dict:
    on_progress(0, 0, "downloading AMD spec database…")
    page = _fetch(client, url)
    candidates = []
    for obj in _amd_objects(page):
        els = obj.get("elements", {})
        title = re.sub(r"™|\(TM\)", "", obj.get("title", "")).strip()
        family = str(_amd_val(els, "family") or "")
        form = str(els.get("formFactor", {}).get("value") or "")
        if not title or "Ryzen" not in family or "Desktops" not in form:
            continue  # consumer desktop CPUs only — skip EPYC/embedded/mobile
        launch = str(_amd_val(els, "launchDate") or "")
        year_m = re.search(r"(19|20)\d{2}", launch)
        candidates.append((int(year_m.group(0)) if year_m else 0, title, els))
    candidates.sort(key=lambda c: c[0], reverse=True)  # newest first

    saved: list[str] = []
    skipped: list[str] = []
    total = min(limit, len(candidates))
    for i, (year, title, els) in enumerate(candidates[:limit]):
        on_progress(i, total, title)
        cores, threads = _amd_val(els, "numOfCpuCores"), _amd_val(els, "numOfThreads")
        base, boost = _amd_val(els, "baseClock"), _amd_val(els, "maxBoostClock")
        cpu_bits = [
            b
            for b in (
                cores and f"{cores} cores",
                threads and f"{threads} threads",
                base and f"@ {base}",
                boost and f"boost {boost}",
            )
            if b
        ]
        mem_type = _amd_val(els, "systemMemoryType")
        channels = _amd_val(els, "memoryChannels")
        tdp = _amd_val(els, "defaultTdp")
        short = title.removeprefix("AMD ").strip()
        model_m = re.search(r"\b\d{4}[A-Z0-9]{0,4}$", title)
        entry = {
            "name": title,
            "brand": "AMD",
            "category": "CPU",
            "release_year": year or None,
            "launch_price_usd": None,  # AMD's DB does not publish launch prices
            "cpu": ", ".join(cpu_bits) or None,
            "gpu": None,
            "compute_tflops": None,
            "memory": " ".join(
                str(x) for x in (mem_type, channels and f"{channels} channels") if x
            )
            or None,
            "memory_bandwidth": None,
            "storage": None,
            "display": None,
            "power_watts": int(tdp) if str(tdp or "").isdigit() else None,
            "key_features": [
                f
                for f in (
                    _amd_val(els, "l3Cache") and f"L3 cache {_amd_val(els, 'l3Cache')}",
                    _amd_val(els, "cpuSocket") and f"Socket {_amd_val(els, 'cpuSocket')}",
                    _amd_val(els, "processorTechnologyForCpuCores"),
                    _amd_val(els, "pciExpressVersion"),
                )
                if f
            ][:4],
            "best_for": ["Gaming", "Content Creation"],
            "aliases": [a for a in {short, model_m.group(0) if model_m else ""} if a and a != title],
            "exists": True,
            "notes": "Specs from AMD's official product database",
        }
        if save_learned_entry(entry):
            saved.append(title)
        else:
            skipped.append(f"{title} (curated built-in)")
    on_progress(total, total, "done")
    return {"saved": saved, "skipped": skipped, "errors": []}


def _save_entries(entries: list[dict], saved: list[str], skipped: list[str]) -> None:
    for entry in entries:
        if not entry or not entry.get("name"):
            continue
        if save_learned_entry(entry):
            saved.append(entry["name"])
        else:
            skipped.append(f"{entry['name']} (curated built-in)")


def import_document(entry: dict, limit: int, on_progress, on_log=None) -> list[dict]:
    """Uploaded source (pdf / excel / image / prompt) → spec entries.

    Text extraction is delegated to sources.extract_source so every source
    type is handled correctly (images are OCR'd by the vision model, sheets
    parsed, PDFs read). The extracted text is then LLM-distilled into spec
    entries — the same dynamic path used for arbitrary web pages, so no
    per-vendor parsers are needed.
    """
    from app import sources as src
    from app.research import distill_products_from_text

    name = entry.get("name", entry.get("id", "source"))
    on_progress(0, 0, f"extracting {name}…")
    text = src.extract_source(entry, on_log=on_log)
    on_progress(0, 0, f"distilling {name} (LLM)…")
    return distill_products_from_text(text, limit, on_log=on_log)


def scrape(
    sources: list[str],
    max_products: int,
    on_progress,
    *,
    max_pages: int = 20,
    max_depth: int = 2,
    delay_s: float = 2.5,
    on_log=None,
) -> dict:
    """Dynamic spec import.

    Deterministic fast paths handle the two sources that beat generic crawling:
    uploaded documents (Intel chart PDFs, spreadsheets) and AMD's embedded-JSON
    database. Every other URL is delegated to the LLM-driven scrape agent
    (run_scrape_agent, below) with the given crawl budget.

    on_progress(done, total, label) keeps the pipeline status live; on_log(msg)
    appends per-URL detail lines. Returns {saved, skipped, errors, outcomes}.
    """
    saved: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    outcomes: list[dict] = []
    agent_urls: list[str] = []
    per_source = max(1, max_products // max(1, len(sources)))

    with httpx.Client() as client:
        for source in sources:
            try:
                # dict = uploaded source entry (pdf/excel/image/prompt);
                # str  = a URL. Uploaded URL-type sources carry their url.
                if isinstance(source, dict):
                    if source.get("type") == "url":
                        agent_urls.append(source["url"])
                    else:
                        entries = import_document(
                            source, per_source, on_progress, on_log=on_log
                        )
                        _save_entries(entries, saved, skipped)
                    continue
                if "amd.com" in source:
                    summary = scrape_amd(client, source, per_source, on_progress)
                    saved += summary["saved"]
                    skipped += summary["skipped"]
                else:
                    agent_urls.append(source)  # generic URL → intelligent agent
            except FileNotFoundError as e:
                errors.append(str(e))
            except httpx.HTTPError as e:
                errors.append(f"{source}: {e}")
            time.sleep(REQUEST_DELAY_S)

    if agent_urls:
        budget = ScrapeBudget(
            max_pages=max_pages,
            max_depth=max_depth,
            delay_s=delay_s,
            max_products=max_products,
        )
        summary = run_scrape_agent(agent_urls, budget, on_progress, on_log=on_log)
        saved += summary["saved"]
        skipped += summary["skipped"]
        errors += summary["errors"]
        outcomes += summary["outcomes"]

    on_progress(len(sources), len(sources), "done")
    return {"saved": saved, "skipped": skipped, "errors": errors, "outcomes": outcomes}


# ── Intelligent crawl agent (merged from scrape_agent) ───────────────────────
#
# LLM-driven index/leaf crawl for dynamic spec import: probes fetchability
# (robots.txt + anti-bot 403s — never evaded), classifies each page as INDEX
# (follow product links) or LEAF (extract specs), distills entries, validates
# them, and reports a per-URL outcome. Crawl budgets bound the work.

DEFAULT_MAX_PAGES = 20
DEFAULT_MAX_DEPTH = 2
DEFAULT_DELAY_S = 2.0

# Honest, identifiable UA — the agent must never masquerade to evade anti-bot.
_UA = "cortex-trainer-scraper/0.1 (+ai-multi-agent-cortex; respects robots.txt)"
_ACCEPT = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

_ANCHOR_RE = re.compile(r'<a\b[^>]*\bhref="([^"#?]+)"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)


@dataclass
class ScrapeBudget:
    max_pages: int = DEFAULT_MAX_PAGES
    max_depth: int = DEFAULT_MAX_DEPTH
    delay_s: float = DEFAULT_DELAY_S
    max_products: int = 30


class _Blocked(Exception):
    """The page declined us (anti-bot 403/429, or non-HTML) — do not retry."""


def _strip_tags(fragment: str) -> str:
    return html_lib.unescape(re.sub(r"<[^>]+>", " ", fragment))


_TABLE_RE = re.compile(r"<table\b[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)
_TR_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<(?:th|td)\b[^>]*>(.*?)</(?:th|td)>", re.DOTALL | re.IGNORECASE)


def _tables_to_rows(page: str) -> str:
    """Render HTML <table>s as pipe-delimited rows so the distiller can still
    map specs to products in comparison charts — plain flattening collapses the
    grid (products-as-columns) into an unusable run of words."""
    blocks: list[str] = []
    for table in _TABLE_RE.findall(page):
        rows: list[str] = []
        for tr in _TR_RE.findall(table):
            cells = [
                re.sub(r"\s+", " ", _strip_tags(c)).strip()
                for c in _CELL_RE.findall(tr)
            ]
            cells = [c for c in cells if c]
            if cells:
                rows.append("| " + " | ".join(cells) + " |")
        if len(rows) >= 2:
            blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def _html_to_text(page: str) -> str:
    page = re.sub(r"<script.*?</script>", " ", page, flags=re.DOTALL | re.IGNORECASE)
    page = re.sub(r"<style.*?</style>", " ", page, flags=re.DOTALL | re.IGNORECASE)
    tables = _tables_to_rows(page)  # keep comparison grids intact for distillation
    body = re.sub(r"\s+", " ", _strip_tags(page)).strip()
    return f"{tables}\n\n{body}".strip() if tables else body


def _title(page: str) -> str:
    m = _TITLE_RE.search(page)
    return re.sub(r"\s+", " ", _strip_tags(m.group(1))).strip()[:120] if m else ""


def _short(url: str) -> str:
    p = urlparse(url)
    return (p.path.rsplit("/", 1)[-1] or p.netloc)[:60]


def _candidate_links(page: str, base_url: str, limit: int = 100) -> list[tuple[str, str]]:
    """Same-site links (anchor text, absolute url), deduped and bounded."""
    base = urlparse(base_url)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for href, inner in _ANCHOR_RE.findall(page):
        url, _ = urldefrag(urljoin(base_url, href))
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or parsed.netloc != base.netloc:
            continue  # same-site only — a spec import shouldn't wander the web
        if url in seen or url == base_url:
            continue
        anchor = re.sub(r"\s+", " ", _strip_tags(inner)).strip()[:80]
        seen.add(url)
        out.append((anchor, url))
        if len(out) >= limit:
            break
    return out


def _robots_allows(client: httpx.Client, url: str, cache: dict, log) -> bool:
    """Respect robots.txt (default-allow when it's absent/unreachable)."""
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in cache:
        rp = robotparser.RobotFileParser()
        rp.set_url(f"{origin}/robots.txt")
        try:
            resp = client.get(f"{origin}/robots.txt", headers=_ACCEPT, timeout=10.0)
            if resp.status_code == 200 and resp.text.strip():
                rp.parse(resp.text.splitlines())
            else:
                rp.allow_all = True  # no robots → allowed
        except Exception:  # noqa: BLE001 — unreachable robots → allowed
            rp.allow_all = True
        cache[origin] = rp
    try:
        return cache[origin].can_fetch(_UA, url)
    except Exception:  # noqa: BLE001
        return True


def _agent_fetch(client: httpx.Client, url: str) -> str:
    resp = client.get(url, headers=_ACCEPT, timeout=25.0, follow_redirects=True)
    if resp.status_code in (401, 403, 429):
        # Anti-bot wall or rate limit — report and move on, never evade.
        raise _Blocked(f"HTTP {resp.status_code} (anti-bot / not permitted)")
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "").lower()
    if ctype and not any(t in ctype for t in ("html", "xml", "text")):
        raise _Blocked(f"non-HTML content ({ctype})")
    return resp.text


_CLASSIFY_PROMPT = """You plan a crawl for a hardware-spec importer. Classify the page as
INDEX (a listing/catalog linking to individual product pages) or LEAF (a single
product's spec page, or an article containing product specs to extract).

If INDEX, choose up to {k} links that most likely lead to individual hardware
product spec pages (CPUs, GPUs, gaming consoles). Prefer product detail pages;
ignore navigation, category, pagination, login, cart and marketing links. Copy
chosen links VERBATIM from the candidate list — never invent a URL.

Return STRICT JSON only, no prose:
{{"page_type": "index" | "leaf", "product_links": ["<url>", ...], "reason": "<short>"}}

URL: {url}
Page title: {title}
Text preview: {preview}

Candidate links (anchor — url):
{links}"""


def _classify(
    url: str, title: str, preview: str, links: list[tuple[str, str]], k: int, log
) -> dict:
    links_block = "\n".join(f"- {a or '(no text)'} — {u}" for a, u in links[:80]) or "(none)"
    try:
        raw = _chat(
            _CLASSIFY_PROMPT.format(
                k=k,
                url=url,
                title=title or "(none)",
                preview=preview[:1200],
                links=links_block[:6000],
            )
        )
        data = _first_json(raw)
    except Exception as e:  # noqa: BLE001 — classifier failure → treat as leaf
        log(f"  classify failed ({type(e).__name__}) — treating as leaf")
        return {"page_type": "leaf", "product_links": [], "reason": "classifier error"}
    if not isinstance(data, dict) or data.get("page_type") not in ("index", "leaf"):
        return {"page_type": "leaf", "product_links": [], "reason": "unparseable"}
    offered = {u for _, u in links}
    data["product_links"] = [u for u in (data.get("product_links") or []) if u in offered][:k]
    return data


def _extract_leaf(url: str, page: str, text: str, cap: int, log) -> list[dict]:
    """Deterministic spec-sheet parse first (structured th/td pages), then fall
    back to LLM distillation for arbitrary product pages/articles."""
    try:
        entry = parse_product(url, page)
        if entry:
            return [_validate_entry(entry, log)]
    except Exception:  # noqa: BLE001 — deterministic parse is best-effort
        pass
    return distill_products_from_text(text, cap, on_log=log)


def _persist(entries: list[dict], saved: list[str], skipped: list[str]) -> list[str]:
    names: list[str] = []
    for entry in entries:
        if not entry or not entry.get("name"):
            continue
        if save_learned_entry(entry):
            saved.append(entry["name"])
            names.append(entry["name"])
        else:
            skipped.append(f"{entry['name']} (curated built-in)")
    return names


def run_scrape_agent(
    sources: list[str], budget: ScrapeBudget, on_progress, on_log=None
) -> dict:
    """Crawl the given URLs with the LLM-driven index/leaf loop.

    Returns {saved, skipped, errors, outcomes} where outcomes is a per-URL
    list of {url, status, detail, products} for the admin UI.
    """
    log = on_log or (lambda _msg: None)
    saved: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    outcomes: list[dict] = []
    visited: set[str] = set()
    robots_cache: dict = {}
    queue: deque[tuple[str, int]] = deque(
        (u, 0) for u in sources if u.startswith(("http://", "https://"))
    )
    pages = 0

    with httpx.Client(follow_redirects=True, timeout=25.0, headers=_ACCEPT) as client:
        while queue and pages < budget.max_pages and len(saved) < budget.max_products:
            url, depth = queue.popleft()
            url, _ = urldefrag(url)
            if url in visited:
                continue
            visited.add(url)
            on_progress(pages, budget.max_pages, _short(url))

            if not _robots_allows(client, url, robots_cache, log):
                outcomes.append({"url": url, "status": "skipped", "detail": "robots.txt disallows"})
                log(f"robots: skip {url}")
                continue
            try:
                page = _agent_fetch(client, url)
            except _Blocked as e:
                outcomes.append({"url": url, "status": "blocked", "detail": str(e)})
                errors.append(f"{url}: {e}")
                log(f"blocked: {_short(url)} ({e})")
                time.sleep(budget.delay_s)
                continue
            except httpx.HTTPError as e:
                outcomes.append({"url": url, "status": "error", "detail": str(e)})
                errors.append(f"{url}: {e}")
                time.sleep(budget.delay_s)
                continue

            pages += 1
            text = _html_to_text(page)
            links = _candidate_links(page, url)
            k = max(4, min(budget.max_products, 20))
            decision = _classify(url, _title(page), text[:1200], links, k, log)

            if (
                decision["page_type"] == "index"
                and depth < budget.max_depth
                and decision["product_links"]
            ):
                added = 0
                for link in decision["product_links"]:
                    if link not in visited:
                        queue.append((link, depth + 1))
                        added += 1
                outcomes.append(
                    {"url": url, "status": "index", "detail": f"followed {added} link(s)"}
                )
                log(f"index: {_short(url)} → {added} product link(s)")
            else:
                cap = max(1, min(budget.max_products - len(saved), 8))
                names = _persist(_extract_leaf(url, page, text, cap, log), saved, skipped)
                status = "extracted" if names else "empty"
                outcomes.append(
                    {"url": url, "status": status, "detail": f"{len(names)} product(s)", "products": names}
                )
                log(f"{status}: {_short(url)} ({len(names)} product(s))")
            time.sleep(budget.delay_s)

    on_progress(pages, pages, "done")
    return {"saved": saved, "skipped": skipped, "errors": errors, "outcomes": outcomes}
