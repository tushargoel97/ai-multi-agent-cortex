"""Web tools: lightweight Wikipedia summary lookup.

Uses the public Wikipedia REST API — no API key required, no extra dependencies.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

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

