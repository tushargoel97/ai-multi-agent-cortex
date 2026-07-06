"""Web search + page fetch for the trainer (prompt-topic and gap research).

Uses a real search API when a key is configured — Firecrawl → Brave → SerpAPI →
Tavily — and Firecrawl also powers JS/anti-bot-safe page fetches. Falls back to
DuckDuckGo HTML + a plain httpx fetch, which are frequently rate-limited. Keys
come from the environment (the same names the main app uses); config._load_dotenv
makes the repo .env available to the host trainer.
"""

from __future__ import annotations

import logging
import os
import urllib.parse

import httpx

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
_KEYS = ("FIRECRAWL_API_KEY", "BRAVE_API_KEY", "SERPAPI_API_KEY", "TAVILY_API_KEY")


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def search_configured() -> bool:
    """True when at least one real search provider key is set."""
    return any(_env(k) for k in _KEYS)


def provider_search(query: str, max_results: int = 6) -> list[dict]:
    """Return ``[{title, url, snippet}]`` from the first configured provider,
    else a DuckDuckGo fallback (often blocked)."""
    fc = _env("FIRECRAWL_API_KEY")
    if fc:
        try:
            r = httpx.post(
                "https://api.firecrawl.dev/v1/search",
                headers={"Authorization": f"Bearer {fc}"},
                json={"query": query, "limit": max_results},
                timeout=25.0,
            )
            r.raise_for_status()
            hits = r.json().get("data") or []
            out = [
                {"title": h.get("title", ""), "url": h.get("url", ""),
                 "snippet": h.get("description", "")}
                for h in hits[:max_results] if h.get("url")
            ]
            if out:
                return out
        except Exception:  # noqa: BLE001
            logger.warning("Firecrawl search failed", exc_info=True)

    brave = _env("BRAVE_API_KEY")
    if brave:
        try:
            r = httpx.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": max_results},
                headers={"X-Subscription-Token": brave, "Accept": "application/json"},
                timeout=20.0,
            )
            r.raise_for_status()
            hits = (r.json().get("web") or {}).get("results") or []
            out = [
                {"title": h.get("title", ""), "url": h.get("url", ""),
                 "snippet": h.get("description", "")}
                for h in hits[:max_results] if h.get("url")
            ]
            if out:
                return out
        except Exception:  # noqa: BLE001
            logger.warning("Brave search failed", exc_info=True)

    serp = _env("SERPAPI_API_KEY")
    if serp:
        try:
            r = httpx.get(
                "https://serpapi.com/search.json",
                params={"engine": "google", "num": max_results, "q": query, "api_key": serp},
                timeout=20.0,
            )
            r.raise_for_status()
            hits = r.json().get("organic_results") or []
            out = [
                {"title": h.get("title", ""), "url": h.get("link", ""),
                 "snippet": h.get("snippet", "")}
                for h in hits[:max_results] if h.get("link")
            ]
            if out:
                return out
        except Exception:  # noqa: BLE001
            logger.warning("SerpAPI search failed", exc_info=True)

    tav = _env("TAVILY_API_KEY")
    if tav:
        try:
            r = httpx.post(
                "https://api.tavily.com/search",
                json={"api_key": tav, "query": query, "max_results": max_results},
                timeout=20.0,
            )
            r.raise_for_status()
            hits = r.json().get("results") or []
            out = [
                {"title": h.get("title", ""), "url": h.get("url", ""),
                 "snippet": h.get("content", "")}
                for h in hits[:max_results] if h.get("url")
            ]
            if out:
                return out
        except Exception:  # noqa: BLE001
            logger.warning("Tavily search failed", exc_info=True)

    return _ddg_search(query, max_results)


def _ddg_search(query: str, max_results: int) -> list[dict]:
    from bs4 import BeautifulSoup

    try:
        r = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": _UA},
            timeout=15.0,
            follow_redirects=True,
        )
        r.raise_for_status()
    except Exception:  # noqa: BLE001
        logger.warning("DuckDuckGo search failed (often blocked)", exc_info=True)
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    out: list[dict] = []
    for a in soup.select("a.result__a")[:max_results]:
        href = a.get("href", "")
        if "uddg=" in href:
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            href = qs.get("uddg", [href])[0]
        if href:
            out.append({"title": a.get_text(strip=True), "url": href, "snippet": ""})
    return out


def _html_fetch(url: str, max_chars: int) -> str:
    from bs4 import BeautifulSoup

    r = httpx.get(url, follow_redirects=True, timeout=15.0, headers={"User-Agent": _UA})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines()]
    return "\n".join(ln for ln in lines if ln)[:max_chars]


def fetch_page(url: str, max_chars: int = 6000) -> str:
    """Readable page text — Firecrawl (JS + anti-bot) first, else plain HTML."""
    fc = _env("FIRECRAWL_API_KEY")
    if fc:
        try:
            r = httpx.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={"Authorization": f"Bearer {fc}"},
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
                timeout=40.0,
            )
            r.raise_for_status()
            md = ((r.json().get("data") or {}).get("markdown") or "").strip()
            if md:
                return md[:max_chars]
        except Exception:  # noqa: BLE001
            logger.warning("Firecrawl scrape failed for %s", url, exc_info=True)
    return _html_fetch(url, max_chars)
