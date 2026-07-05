"""Shopping & booking tools — region-aware product prices and booking search.

Both build on the DuckDuckGo search helper in :mod:`cortex.tools.web` (no API
keys). Queries are scoped to the retailers / booking platforms relevant to the
user's region, and results come back as structured JSON that the shopping /
booking agents turn into comparison tables with live source links.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from pydantic import BaseModel, Field

from cortex.tools.registry import register_tool
from cortex.tools.web import _ddg_search

# ── Region → retailers ───────────────────────────────────────────────────────
# currency label + ordered list of shopping domains searched for each region.
_SHOPPING_SITES: dict[str, tuple[str, list[str]]] = {
    "US": ("USD", ["amazon.com", "walmart.com", "bestbuy.com", "target.com", "ebay.com"]),
    "IN": ("INR", ["amazon.in", "flipkart.com", "croma.com", "reliancedigital.in", "vijaysales.com", "tatacliq.com"]),
    "UK": ("GBP", ["amazon.co.uk", "currys.co.uk", "argos.co.uk", "ebay.co.uk"]),
    "CA": ("CAD", ["amazon.ca", "bestbuy.ca", "walmart.ca"]),
    "AU": ("AUD", ["amazon.com.au", "jbhifi.com.au", "kogan.com"]),
    "DE": ("EUR", ["amazon.de", "mediamarkt.de", "otto.de"]),
    "FR": ("EUR", ["amazon.fr", "fnac.com", "cdiscount.com"]),
    "AE": ("AED", ["amazon.ae", "noon.com", "sharafdg.com"]),
    "SG": ("SGD", ["amazon.sg", "lazada.sg", "shopee.sg"]),
    "JP": ("JPY", ["amazon.co.jp", "rakuten.co.jp"]),
}
_DEFAULT_REGION = "US"

_REGION_ALIASES = {
    "USA": "US", "UNITED STATES": "US", "AMERICA": "US",
    "INDIA": "IN", "BHARAT": "IN",
    "UNITED KINGDOM": "UK", "GB": "UK", "ENGLAND": "UK", "BRITAIN": "UK",
    "CANADA": "CA", "AUSTRALIA": "AU", "GERMANY": "DE", "FRANCE": "FR",
    "UAE": "AE", "DUBAI": "AE", "SINGAPORE": "SG", "JAPAN": "JP",
}

# Price-looking tokens in snippets: $1,299.00 · ₹1,29,900 · £999 · AED 4,999 ·
# INR 54,990 · 1299 USD (symbol/code before OR after the number).
_PRICE_RE = re.compile(
    r"(?:US\$|A\$|C\$|S\$|AED|Rs\.?|₹|\$|£|€|¥)\s?\d[\d.,]*\d"
    r"|(?:USD|INR|GBP|EUR|AED|SGD|AUD|CAD|JPY)\s?\d[\d.,]*\d"
    r"|\d[\d.,]*\d\s?(?:USD|INR|GBP|EUR|AED|SGD|AUD|CAD|JPY)",
    re.IGNORECASE,
)


def _norm_region(region: str | None) -> str:
    if not region:
        return _DEFAULT_REGION
    r = region.strip().upper()
    r = _REGION_ALIASES.get(r, r)
    return r if r in _SHOPPING_SITES else _DEFAULT_REGION


def _price_hint(*texts: str) -> str:
    for text in texts:
        m = _PRICE_RE.search(text or "")
        if m:
            return m.group(0).strip()
    return ""


def _price_value(price: str) -> float | None:
    """Best-effort numeric value from a price string, for cheapest-first sorting.

    Handles US ($1,299.00), Indian (₹1,29,900), European (€1.299,00 / 49,99),
    and plain (999) formats — the last separator is treated as the decimal
    point when both are present, and a lone ``,NN`` tail as a decimal comma.
    """
    m = re.search(r"\d[\d.,]*", price or "")
    if not m:
        return None
    num = m.group(0).strip(".,")
    if "," in num and "." in num:
        if num.rfind(",") > num.rfind("."):
            num = num.replace(".", "").replace(",", ".")  # European 1.299,00
        else:
            num = num.replace(",", "")  # US 1,299.00
    elif "," in num:
        if num.count(",") == 1 and re.search(r",\d{2}$", num):
            num = num.replace(",", ".")  # European decimal 49,99
        else:
            num = num.replace(",", "")  # thousands 1,299 / 1,29,900
    try:
        return float(num)
    except ValueError:
        return None


def _safe_ddg(query: str, max_results: int) -> list[dict[str, str]]:
    """``_ddg_search`` that never raises — one site failing must not sink the tool."""
    try:
        return _ddg_search(query, max_results=max_results)
    except Exception:  # noqa: BLE001
        return []


class ProductPricesInput(BaseModel):
    """Input for a region-aware product price lookup."""

    product: str = Field(
        description="The product to price, e.g. 'PS5 Pro' or 'iPhone 16 Pro 256GB'."
    )
    region: str = Field(
        default="US",
        description=(
            "User region/country (code or name): US, IN, UK, CA, AU, DE, FR, "
            "AE, SG, JP. Defaults to US."
        ),
    )


@register_tool(args_schema=ProductPricesInput)
async def product_prices(product: str, region: str = "US") -> str:
    """Fetch current prices for a product across the major online retailers for
    the user's region.

    Searches each region-appropriate shopping site (e.g. Amazon / Walmart /
    Best Buy for the US; Amazon.in / Flipkart / Croma for India) concurrently
    and returns a JSON list of offers — retailer, page title, URL, and any
    price found in the snippet — sorted cheapest-first. Use for "how much is
    X", "cheapest X", "price of X in <country>", or shopping comparisons. The
    offers are shown to the user as visual cards, so keep your own summary
    short; always confirm prices change and to check the retailer page.
    """
    reg = _norm_region(region)
    currency, sites = _SHOPPING_SITES[reg]
    domains = sites[:5]
    # Search every retailer concurrently — sequential lookups were the main
    # source of latency (5 blocking HTTP calls back-to-back).
    batches = await asyncio.gather(
        *(asyncio.to_thread(_safe_ddg, f"{product} price site:{d}", 3) for d in domains)
    )
    offers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for domain, results in zip(domains, batches):
        for r in results:
            url = r.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            title, snippet = r.get("title", ""), r.get("snippet", "")
            price = _price_hint(snippet, title)
            offers.append(
                {
                    "retailer": domain,
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "price": price,
                    "price_value": _price_value(price),
                }
            )
    # Cheapest-first: priced offers ascending, unpriced ones last.
    offers.sort(key=lambda o: (o["price_value"] is None, o["price_value"] or 0.0))
    if not offers:
        return json.dumps(
            {
                "product": product,
                "region": reg,
                "currency": currency,
                "offers": [],
                "note": "No retailer results — try a more specific product name.",
            }
        )
    return json.dumps(
        {
            "product": product,
            "region": reg,
            "currency": currency,
            "offers": offers,
            "note": (
                "Prices are from live search snippets and may be approximate — "
                "confirm the current price on the retailer page."
            ),
        },
        ensure_ascii=False,
    )


# ── Category → booking platforms (per region, with a global fallback "_") ─────
_BOOKING_SITES: dict[str, dict[str, list[str]]] = {
    "flight": {
        "US": ["google.com/travel/flights", "kayak.com", "expedia.com", "skyscanner.com"],
        "IN": ["makemytrip.com", "goibibo.com", "cleartrip.com", "skyscanner.co.in"],
        "UK": ["skyscanner.net", "kayak.co.uk", "expedia.co.uk"],
        "_": ["skyscanner.com", "kayak.com", "google.com/travel/flights"],
    },
    "hotel": {
        "US": ["booking.com", "expedia.com", "hotels.com"],
        "IN": ["makemytrip.com", "goibibo.com", "booking.com"],
        "_": ["booking.com", "agoda.com", "hotels.com"],
    },
    "movie": {
        "US": ["fandango.com", "atomtickets.com"],
        "IN": ["bookmyshow.com", "paytm.com"],
        "UK": ["myvue.com", "cineworld.co.uk"],
        "_": ["bookmyshow.com", "imdb.com"],
    },
    "concert": {
        "US": ["ticketmaster.com", "stubhub.com", "seatgeek.com"],
        "IN": ["bookmyshow.com", "insider.in"],
        "UK": ["ticketmaster.co.uk", "seetickets.com"],
        "_": ["ticketmaster.com", "songkick.com", "bandsintown.com"],
    },
    "event": {
        "US": ["ticketmaster.com", "eventbrite.com", "seatgeek.com"],
        "IN": ["bookmyshow.com", "insider.in", "townscript.com"],
        "_": ["eventbrite.com", "ticketmaster.com"],
    },
    "show": {
        "US": ["telecharge.com", "broadway.com", "ticketmaster.com"],
        "UK": ["officiallondontheatre.com", "ticketmaster.co.uk"],
        "IN": ["bookmyshow.com"],
        "_": ["ticketmaster.com", "bookmyshow.com"],
    },
}

_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("flight", ("flight", "flights", "airfare", "airline", "fly ")),
    ("hotel", ("hotel", "resort", "accommodation", " stay", "airbnb", "room ")),
    ("movie", ("movie", "film", "cinema", "showtime")),
    ("concert", ("concert", "gig", "tour", "live music")),
    ("show", ("musical", "theatre", "theater", "broadway", "play ", "comedy show")),
    ("event", ("event", "festival", "expo", "conference", "match", "sports")),
]


def _norm_category(category: str | None, query: str) -> str:
    explicit = (category or "").strip().lower()
    if explicit in _BOOKING_SITES:
        return explicit
    text = f"{explicit} {query}".lower()
    for name, kws in _CATEGORY_KEYWORDS:
        if any(k in text for k in kws):
            return name
    return "event"


class FindBookingsInput(BaseModel):
    """Input for a booking-options search."""

    query: str = Field(
        description=(
            "What to book, including place/date/title if known, e.g. "
            "'flights NYC to London 12 Dec' or 'Coldplay Mumbai'."
        )
    )
    category: str = Field(
        default="",
        description=(
            "One of: flight, hotel, movie, concert, event, show. Leave blank "
            "to auto-detect from the query."
        ),
    )
    region: str = Field(
        default="US",
        description="User region/country (US, IN, UK, CA, AU, DE, ...). Defaults to US.",
    )


@register_tool(args_schema=FindBookingsInput)
async def find_bookings(query: str, category: str = "", region: str = "US") -> str:
    """Find booking options for flights, hotels, movies, concerts, events, or
    shows on the platforms relevant to the user's region.

    Auto-detects the category from the query when not given, searches the right
    booking platforms concurrently (e.g. Ticketmaster / StubHub for US
    concerts, BookMyShow / Insider for India, Booking.com / MakeMyTrip for
    stays), and returns a JSON list of options with direct booking links. The
    options are shown to the user as visual cards, so keep your summary short;
    never invent prices, seats, times, availability, or confirmations — those
    must be checked on the platform.
    """
    cat = _norm_category(category, query)
    reg = _norm_region(region)
    by_region = _BOOKING_SITES[cat]
    sites = (by_region.get(reg) or by_region["_"])[:4]
    term = "tickets" if cat in ("concert", "event", "show", "movie") else "booking"
    batches = await asyncio.gather(
        *(asyncio.to_thread(_safe_ddg, f"{query} {term} site:{d}", 3) for d in sites)
    )
    options: list[dict[str, str]] = []
    seen: set[str] = set()
    for domain, results in zip(sites, batches):
        for r in results:
            url = r.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            options.append(
                {
                    "platform": domain,
                    "title": r.get("title", ""),
                    "url": url,
                    "snippet": r.get("snippet", ""),
                }
            )
    if not options:
        return json.dumps(
            {
                "query": query,
                "category": cat,
                "region": reg,
                "options": [],
                "note": "No results — try adding a city, date, or exact title.",
            }
        )
    return json.dumps(
        {
            "query": query,
            "category": cat,
            "region": reg,
            "options": options,
            "note": (
                "Options are from live search; availability, prices, seats, and "
                "times must be confirmed on the platform."
            ),
        },
        ensure_ascii=False,
    )
