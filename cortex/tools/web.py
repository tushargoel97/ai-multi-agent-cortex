"""Web tools: Wikipedia summary lookup, live internet search, and crypto prices.

Wikipedia REST and CoinGecko work with no key. Live web search prefers a real
search API when a key is set (``BRAVE_API_KEY``, ``SERPAPI_API_KEY``, or
``TAVILY_API_KEY``) and falls back to DuckDuckGo scraping — which is frequently
blocked, so a key is strongly recommended. Stdlib only — no new dependencies.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from pydantic import BaseModel, Field

from cortex.tools.registry import register_tool

logger = logging.getLogger(__name__)


_WIKI_SEARCH = "https://en.wikipedia.org/w/api.php?action=opensearch&format=json&limit=5&search="
_WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_USER_AGENT = "cortex-multi-agent/0.1 (https://example.com)"


def _http_get(url: str, timeout: float = 5.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class WikipediaSearchInput(BaseModel):
    """Input for Wikipedia search."""

    query: str = Field(description="Search term to look up on Wikipedia")


@register_tool(args_schema=WikipediaSearchInput)
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return summaries of the top matching articles.

    Use this for questions about general knowledge, current events,
    public figures, places, science, history, and so on.
    """
    try:
        url = _WIKI_SEARCH + urllib.parse.quote(query)
        raw = _http_get(url)
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        return json.dumps({"error": f"Wikipedia search failed: {exc}"})

    titles: list[str] = data[1] if len(data) > 1 else []
    if not titles:
        return json.dumps({"results": [], "note": "No matches found."})

    results = []
    for title in titles[:3]:
        try:
            summary_raw = _http_get(_WIKI_SUMMARY + urllib.parse.quote(title))
            summary = json.loads(summary_raw.decode("utf-8"))
            results.append(
                {
                    "title": summary.get("title", title),
                    "extract": summary.get("extract", "")[:1200],
                    "url": summary.get("content_urls", {}).get("desktop", {}).get("page", ""),
                }
            )
        except Exception:
            continue

    return json.dumps({"query": query, "results": results}, indent=2)


_COINGECKO_PRICE = "https://api.coingecko.com/api/v3/simple/price"

# Common ticker symbols → CoinGecko coin ids
_CRYPTO_ALIASES = {
    "btc": "bitcoin",
    "xbt": "bitcoin",
    "bitcoin": "bitcoin",
    "eth": "ethereum",
    "ether": "ethereum",
    "ethereum": "ethereum",
    "sol": "solana",
    "solana": "solana",
    "ada": "cardano",
    "cardano": "cardano",
    "doge": "dogecoin",
    "dogecoin": "dogecoin",
    "xrp": "ripple",
    "ripple": "ripple",
    "bnb": "binancecoin",
    "matic": "matic-network",
    "polygon": "matic-network",
    "ltc": "litecoin",
    "litecoin": "litecoin",
    "dot": "polkadot",
    "polkadot": "polkadot",
    "avax": "avalanche-2",
    "avalanche": "avalanche-2",
    "shib": "shiba-inu",
    "trx": "tron",
    "tron": "tron",
    "usdt": "tether",
    "tether": "tether",
    "usdc": "usd-coin",
}


class CryptoPriceInput(BaseModel):
    """Input for crypto price lookup."""

    symbols: str = Field(
        description=(
            "Comma-separated list of crypto symbols or names "
            "(e.g. 'btc,eth' or 'bitcoin, ethereum')."
        )
    )
    currencies: str = Field(
        default="usd,inr",
        description=(
            "Comma-separated list of fiat/quote currencies (lowercase ISO codes "
            "like usd, inr, eur, gbp, jpy). Defaults to 'usd,inr'."
        ),
    )


@register_tool(args_schema=CryptoPriceInput)
def crypto_price(symbols: str, currencies: str = "usd,inr") -> str:
    """Fetch live cryptocurrency spot prices from CoinGecko.

    Use this for ANY question about current/live/latest crypto prices,
    market caps, or 24h changes (Bitcoin, Ethereum, etc.). Returns the
    price in one or more fiat currencies along with the 24h change %.
    No API key required.
    """
    raw_symbols = [s.strip().lower() for s in symbols.split(",") if s.strip()]
    if not raw_symbols:
        return json.dumps({"error": "No symbols provided."})

    coin_ids: list[str] = []
    unknown: list[str] = []
    for s in raw_symbols:
        cid = _CRYPTO_ALIASES.get(s, s)
        coin_ids.append(cid)
        if s not in _CRYPTO_ALIASES and "-" not in s and len(s) <= 5:
            unknown.append(s)

    vs = ",".join([c.strip().lower() for c in currencies.split(",") if c.strip()]) or "usd"

    params = urllib.parse.urlencode(
        {
            "ids": ",".join(coin_ids),
            "vs_currencies": vs,
            "include_24hr_change": "true",
            "include_market_cap": "true",
            "include_last_updated_at": "true",
        }
    )
    url = f"{_COINGECKO_PRICE}?{params}"

    try:
        raw = _http_get(url, timeout=8.0)
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        return json.dumps(
            {
                "error": f"CoinGecko request failed: {exc}",
                "hint": "Network may be restricted or rate-limited.",
            }
        )

    if not data:
        return json.dumps(
            {
                "error": "No price data returned.",
                "queried_ids": coin_ids,
                "unknown_symbols": unknown,
            }
        )

    out = {
        "source": "coingecko",
        "currencies": vs.split(","),
        "prices": data,
    }
    if unknown:
        out["note"] = (
            f"Symbols {unknown} were not recognized aliases; queried as-is."
        )
    return json.dumps(out, indent=2)


# ── Live internet search (DuckDuckGo HTML, no API key) ───────────────────────

_DDG_HTML = "https://html.duckduckgo.com/html/?q="
# Browser-like UA so DuckDuckGo doesn't return a stripped no-JS notice page.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class _DDGResultParser(HTMLParser):
    """Extract (title, url, snippet) tuples from DuckDuckGo's HTML results."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._cur: dict[str, str] | None = None
        self._capture: str | None = None  # "title" | "snippet" | None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        cls = a.get("class", "") or ""
        if tag == "a" and "result__a" in cls:
            self._cur = self._cur or {}
            href = a.get("href", "") or ""
            # DDG wraps real URLs in /l/?uddg=<encoded>
            if "uddg=" in href:
                try:
                    qs = urllib.parse.urlparse(href).query
                    href = urllib.parse.parse_qs(qs).get("uddg", [href])[0]
                except Exception:  # noqa: BLE001
                    pass
            self._cur["url"] = href
            self._capture = "title"
        elif tag == "a" and "result__snippet" in cls:
            self._cur = self._cur or {}
            self._capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture is not None:
            if self._capture == "snippet" and self._cur:
                self.results.append(self._cur)
                self._cur = None
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._capture and self._cur is not None:
            self._cur[self._capture] = (self._cur.get(self._capture, "") + data).strip()


class _TextExtractor(HTMLParser):
    """Best-effort visible-text extractor (drops <script>/<style>/<nav> etc.)."""

    _SKIP = {"script", "style", "noscript", "nav", "header", "footer", "form", "aside"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._chunks.append(text)

    @property
    def text(self) -> str:
        joined = " ".join(self._chunks)
        # Collapse whitespace
        return re.sub(r"\s+", " ", joined).strip()


def _ddg_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    url = _DDG_HTML + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
    with urllib.request.urlopen(req, timeout=8.0) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    parser = _DDGResultParser()
    parser.feed(body)
    return parser.results[:max_results]


def _fetch_text(url: str, max_chars: int = 2000, timeout: float = 6.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type", "")
        if "html" not in ctype.lower() and "text" not in ctype.lower():
            return f"[non-text content: {ctype}]"
        body = resp.read(800_000).decode("utf-8", errors="replace")
    extractor = _TextExtractor()
    try:
        extractor.feed(body)
    except Exception:  # noqa: BLE001
        return html.unescape(re.sub(r"<[^>]+>", " ", body))[:max_chars]
    return html.unescape(extractor.text)[:max_chars]


class WebSearchInput(BaseModel):
    """Input for live internet search."""

    query: str = Field(description="Natural-language search query.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=8,
        description="Number of search results to return (1-8).",
    )
    fetch_pages: bool = Field(
        default=True,
        description=(
            "If true, fetch and extract readable text from the top results "
            "so the agent has actual page content (not just snippets)."
        ),
    )


def _http_json(
    url: str, *, headers: dict | None = None, data: dict | None = None, timeout: float = 10.0
) -> dict:
    """Minimal JSON helper — POST when ``data`` is given, else GET."""
    body = json.dumps(data).encode() if data is not None else None
    h = {"User-Agent": _BROWSER_UA, "Accept": "application/json"}
    if data is not None:
        h["Content-Type"] = "application/json"
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _provider_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Real web search via an API key when configured (Firecrawl → Brave →
    SerpAPI → Tavily).

    Returns ``[]`` when no key is set or the call fails, so callers fall back to
    DuckDuckGo scraping.
    """
    firecrawl = os.getenv("FIRECRAWL_API_KEY", "").strip()
    if firecrawl:
        try:
            data = _http_json(
                "https://api.firecrawl.dev/v1/search",
                headers={"Authorization": f"Bearer {firecrawl}"},
                data={"query": query, "limit": max_results},
                timeout=20.0,
            )
            hits = data.get("data") or []
            out = [
                {"title": r.get("title", ""), "url": r.get("url", ""),
                 "snippet": r.get("description", "")}
                for r in hits[:max_results]
            ]
            if out:
                return out
        except Exception:  # noqa: BLE001
            logger.warning("Firecrawl search failed", exc_info=True)

    brave = os.getenv("BRAVE_API_KEY", "").strip()
    if brave:
        try:
            data = _http_json(
                "https://api.search.brave.com/res/v1/web/search?count="
                + f"{max_results}&q=" + urllib.parse.quote(query),
                headers={"X-Subscription-Token": brave},
            )
            hits = (data.get("web") or {}).get("results") or []
            return [
                {"title": r.get("title", ""), "url": r.get("url", ""),
                 "snippet": r.get("description", "")}
                for r in hits[:max_results]
            ]
        except Exception:  # noqa: BLE001
            logger.warning("Brave search failed", exc_info=True)

    serp = os.getenv("SERPAPI_API_KEY", "").strip()
    if serp:
        try:
            data = _http_json(
                "https://serpapi.com/search.json?engine=google&num="
                + f"{max_results}&q=" + urllib.parse.quote(query)
                + "&api_key=" + urllib.parse.quote(serp),
            )
            hits = data.get("organic_results") or []
            return [
                {"title": r.get("title", ""), "url": r.get("link", ""),
                 "snippet": r.get("snippet", "")}
                for r in hits[:max_results]
            ]
        except Exception:  # noqa: BLE001
            logger.warning("SerpAPI search failed", exc_info=True)

    tavily = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily:
        try:
            data = _http_json(
                "https://api.tavily.com/search",
                data={"api_key": tavily, "query": query, "max_results": max_results},
            )
            hits = data.get("results") or []
            return [
                {"title": r.get("title", ""), "url": r.get("url", ""),
                 "snippet": r.get("content", "")}
                for r in hits[:max_results]
            ]
        except Exception:  # noqa: BLE001
            logger.warning("Tavily search failed", exc_info=True)

    return []


@register_tool(args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5, fetch_pages: bool = True) -> str:
    """Search the live internet and return combined results.

    Use this for ANY question that needs **current / up-to-date / recent**
    information that may not be in Wikipedia or the local knowledge base —
    news, prices, scores, releases, weather, stock/crypto quotes, recent
    events, product specs, etc. Returns a JSON document with the top results
    (title, url, snippet) and, optionally, extracted page text for each.

    Uses a real search API when a key is configured (FIRECRAWL_API_KEY,
    BRAVE_API_KEY, SERPAPI_API_KEY, or TAVILY_API_KEY); otherwise falls back to
    DuckDuckGo, which is often blocked and may return nothing.
    """
    try:
        results = _provider_search(query, max_results)
        source = "api" if results else "duckduckgo"
        if not results:
            results = _ddg_search(query, max_results=max_results)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Web search failed: {exc}"})

    if not results:
        return json.dumps(
            {
                "query": query,
                "results": [],
                "note": (
                    "No results — the free DuckDuckGo backend is blocked. Set "
                    "FIRECRAWL_API_KEY (or BRAVE_API_KEY / SERPAPI_API_KEY / "
                    "TAVILY_API_KEY) to enable real web search."
                ),
            }
        )

    enriched: list[dict[str, str]] = []
    for r in results:
        item = {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("snippet", ""),
        }
        if fetch_pages and item["url"]:
            try:
                item["content"] = _fetch_text(item["url"])
            except Exception as exc:  # noqa: BLE001
                item["content"] = f"[fetch failed: {exc}]"
        enriched.append(item)

    return json.dumps(
        {"source": source, "query": query, "results": enriched},
        indent=2,
    )


class FetchUrlInput(BaseModel):
    """Input for fetching a single URL."""

    url: str = Field(description="Absolute http(s) URL to fetch.")
    max_chars: int = Field(
        default=4000,
        ge=200,
        le=20000,
        description="Maximum characters of extracted text to return.",
    )


def _firecrawl_scrape(url: str, max_chars: int) -> str:
    """Scrape a page to clean markdown via Firecrawl (handles JS + anti-bot).

    Returns ``""`` when no key is set or the call fails, so callers fall back.
    """
    key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    if not key:
        return ""
    try:
        data = _http_json(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {key}"},
            data={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            timeout=30.0,
        )
        md = ((data.get("data") or {}).get("markdown") or "").strip()
        return md[:max_chars]
    except Exception:  # noqa: BLE001
        logger.warning("Firecrawl scrape failed; falling back to urllib", exc_info=True)
        return ""


@register_tool(args_schema=FetchUrlInput)
def fetch_url(url: str, max_chars: int = 4000) -> str:
    """Fetch a single web page and return its readable text content.

    Use this when `web_search` returned a promising URL but the snippet/auto-
    extracted content was too short, or when the user asks you to read a
    specific page. Prefers Firecrawl (JS-rendered, anti-bot) when
    FIRECRAWL_API_KEY is set, else a plain HTML fetch (HTML stripped).
    """
    if not url.startswith(("http://", "https://")):
        return json.dumps({"error": "URL must start with http:// or https://"})
    scraped = _firecrawl_scrape(url, max_chars)
    if scraped:
        return json.dumps(
            {"url": url, "content": scraped, "source": "firecrawl"}, indent=2
        )
    try:
        text = _fetch_text(url, max_chars=max_chars, timeout=8.0)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Fetch failed: {exc}", "url": url})
    return json.dumps({"url": url, "content": text}, indent=2)
