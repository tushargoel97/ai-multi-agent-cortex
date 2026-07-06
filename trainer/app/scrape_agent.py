"""Intelligent web-scrape agent — LLM-driven crawl for dynamic spec import.

Replaces the old hostname-dispatched URL importers with one general loop: for
any URL it (a) probes fetchability, respecting robots.txt and anti-bot 403s —
never evaded; (b) classifies each page as an INDEX (follow inner product
links) or a LEAF (extract specs); (c) distills entries into the learned-facts
schema (deterministic spec-sheet parse first, TRAINER_QA_* LLM distillation as
the fallback); (d) validates them; and (e) reports a per-URL outcome. Crawl
budgets (max pages, max depth, politeness delay) bound the work.

All writes go through research.save_learned_entry — the single alias-aware
path that never overrides the curated facts.yaml.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
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


def _fetch(client: httpx.Client, url: str) -> str:
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
        from app.scraper import parse_product

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
                page = _fetch(client, url)
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
