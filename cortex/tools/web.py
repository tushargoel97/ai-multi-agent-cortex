"""Web tools: Wikipedia summary lookup, live internet search, and crypto prices.

Uses public, no-API-key endpoints (Wikipedia REST, DuckDuckGo HTML, CoinGecko) so
the tools work out of the box without extra credentials. Stdlib only — no new
dependencies.
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from pydantic import BaseModel, Field

from cortex.tools.registry import register_tool


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


@register_tool(args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5, fetch_pages: bool = True) -> str:
    """Search the live internet via DuckDuckGo and return combined results.

    Use this for ANY question that needs **current / up-to-date / recent**
    information that may not be in Wikipedia or the local knowledge base —
    news, prices, scores, releases, weather, stock/crypto quotes, recent
    events, product specs, etc. Returns a JSON document with the top results
    (title, url, snippet) and, optionally, extracted page text for each.
    No API key required.
    """
    try:
        results = _ddg_search(query, max_results=max_results)
        # CPU/GPU queries: put TechPowerUp's database pages first — they are
        # the authoritative spec source (see also the techpowerup_specs tool).
        if _GPU_HINT_RE.search(query) or _CPU_HINT_RE.search(query):
            try:
                tpu = _ddg_search(f"site:techpowerup.com {query}", max_results=3)
                known = {r.get("url") for r in results}
                results = [r for r in tpu if r.get("url") not in known] + results
                results = results[: max_results + 2]
            except Exception:  # noqa: BLE001 — boost is best-effort
                pass
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Web search failed: {exc}"})

    if not results:
        return json.dumps({"query": query, "results": [], "note": "No matches."})

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
        {"source": "duckduckgo", "query": query, "results": enriched},
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


@register_tool(args_schema=FetchUrlInput)
def fetch_url(url: str, max_chars: int = 4000) -> str:
    """Fetch a single web page and return its readable text content.

    Use this when `web_search` returned a promising URL but the snippet/auto-
    extracted content was too short, or when the user asks you to read a
    specific page. Returns plain text (HTML stripped).
    """
    if not url.startswith(("http://", "https://")):
        return json.dumps({"error": "URL must start with http:// or https://"})
    try:
        text = _fetch_text(url, max_chars=max_chars, timeout=8.0)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Fetch failed: {exc}", "url": url})
    return json.dumps({"url": url, "content": text}, indent=2)



# ── TechPowerUp spec lookup ──────────────────────────────────────────────────

_TPU_BASE = "https://www.techpowerup.com"
_TPU_LINK_RE = re.compile(r'href="(/(?:cpu|gpu)-specs/[^"#?]+\.c\d+)"')
_TPU_PAIR_RES = (
    re.compile(r"<th[^>]*>(.*?)</th>\s*<td[^>]*>(.*?)</td>", re.DOTALL),
    re.compile(r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>", re.DOTALL),
)
_TPU_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.DOTALL)
_GPU_HINT_RE = re.compile(
    r"rtx|gtx|geforce|radeon|\brx\s?\d|arc\s|graphics card|\bgpu\b", re.IGNORECASE
)
_CPU_HINT_RE = re.compile(
    r"ryzen|threadripper|epyc|core\s|i[3579]-|core ultra|xeon|\bcpu\b|processor",
    re.IGNORECASE,
)


def _tpu_clean(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", fragment)).replace("\xa0", " ").strip()


def _tpu_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
    with urllib.request.urlopen(req, timeout=12.0) as resp:
        return resp.read().decode("utf-8", errors="replace")


class TechPowerUpInput(BaseModel):
    """Input for the TechPowerUp hardware-spec lookup."""

    query: str = Field(
        description="Product name, e.g. 'RTX 5090', 'Ryzen 7 9800X3D', 'Core Ultra 9 285K'"
    )
    category: str = Field(
        default="auto",
        description="'cpu', 'gpu', or 'auto' to detect from the query",
    )


@register_tool(args_schema=TechPowerUpInput)
def techpowerup_specs(query: str, category: str = "auto") -> str:
    """Authoritative spec sheet for a CPU or GPU from TechPowerUp's hardware
    databases (Intel/AMD CPUs; NVIDIA/AMD/Intel GPUs).

    PREFER this over web_search for any CPU or GPU spec question — it returns
    the full structured spec table (clocks, cores, memory, bandwidth, TDP,
    release date, launch price) straight from the database entry.
    """
    if category == "cpu":
        dbs = ["cpu"]
    elif category == "gpu":
        dbs = ["gpu"]
    else:
        dbs = []
        if _GPU_HINT_RE.search(query):
            dbs.append("gpu")
        if _CPU_HINT_RE.search(query):
            dbs.append("cpu")
        dbs = dbs or ["gpu", "cpu"]

    errors: list[str] = []
    for db in dbs:
        try:
            search = _tpu_get(
                f"{_TPU_BASE}/{db}-specs/?ajaxsrch=" + urllib.parse.quote(query)
            )
            links = list(dict.fromkeys(_TPU_LINK_RE.findall(search)))
            if not links:
                continue
            page = _tpu_get(_TPU_BASE + links[0])
            h1 = _TPU_H1_RE.search(page)
            name = _tpu_clean(h1.group(1)) if h1 else query
            specs: dict[str, str] = {}
            for pattern in _TPU_PAIR_RES:
                for label, value in pattern.findall(page):
                    label, value = _tpu_clean(label).rstrip(":"), _tpu_clean(value)
                    if label and value and label not in specs and len(specs) < 40:
                        specs[label] = value
            return json.dumps(
                {
                    "name": re.sub(r"\s+(Specs|Review).*$", "", name).strip(),
                    "url": _TPU_BASE + links[0],
                    "source": "TechPowerUp",
                    "specs": specs,
                },
                indent=2,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{db}: {exc}")
    return json.dumps(
        {
            "query": query,
            "results": [],
            "note": "No TechPowerUp match found — fall back to web_search.",
            "errors": errors,
        }
    )
