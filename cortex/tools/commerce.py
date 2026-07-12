"""Shopping & booking tools, deterministic deep links to live results.

DuckDuckGo scraping is blocked (anti-bot 202) and there is no search-API key,
so rather than return empty/stale snippets these tools build direct deep links
into each platform's own live search/results page, region-aware, date-correct
(using the current year), and instant (no network calls). The shopping and
booking agents render them as cards; live prices, fares, seats, and times are
shown on the destination page, never guessed here.
"""

from __future__ import annotations

import json
import re
from datetime import date
from urllib.parse import quote, quote_plus, urlsplit

from pydantic import BaseModel, Field

from cortex.tools.registry import register_tool

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

# Known-good product-search URL per retailer ({q} = url-encoded product). Any
# retailer without an entry falls back to a Google search scoped to its domain,
# which always resolves: so every card link works.
_RETAILER_SEARCH: dict[str, str] = {
    "amazon.com": "https://www.amazon.com/s?k={q}",
    "amazon.in": "https://www.amazon.in/s?k={q}",
    "amazon.co.uk": "https://www.amazon.co.uk/s?k={q}",
    "amazon.ca": "https://www.amazon.ca/s?k={q}",
    "amazon.com.au": "https://www.amazon.com.au/s?k={q}",
    "amazon.de": "https://www.amazon.de/s?k={q}",
    "amazon.fr": "https://www.amazon.fr/s?k={q}",
    "amazon.ae": "https://www.amazon.ae/s?k={q}",
    "amazon.sg": "https://www.amazon.sg/s?k={q}",
    "amazon.co.jp": "https://www.amazon.co.jp/s?k={q}",
    "flipkart.com": "https://www.flipkart.com/search?q={q}",
    "walmart.com": "https://www.walmart.com/search?q={q}",
    "walmart.ca": "https://www.walmart.ca/search?q={q}",
    "bestbuy.com": "https://www.bestbuy.com/site/searchpage.jsp?st={q}",
    "target.com": "https://www.target.com/s?searchTerm={q}",
    "ebay.com": "https://www.ebay.com/sch/i.html?_nkw={q}",
    "ebay.co.uk": "https://www.ebay.co.uk/sch/i.html?_nkw={q}",
}


def _norm_region(region: str | None) -> str:
    if not region:
        return _DEFAULT_REGION
    r = region.strip().upper()
    r = _REGION_ALIASES.get(r, r)
    return r if r in _SHOPPING_SITES else _DEFAULT_REGION


# Browser timezone → shopping/booking region (only the regions we support).
_TZ_REGION: dict[str, str] = {
    "Asia/Kolkata": "IN", "Asia/Calcutta": "IN",
    "Europe/London": "UK",
    "America/Toronto": "CA", "America/Vancouver": "CA", "America/Edmonton": "CA",
    "America/Winnipeg": "CA", "America/Halifax": "CA",
    "Australia/Sydney": "AU", "Australia/Melbourne": "AU", "Australia/Brisbane": "AU",
    "Australia/Perth": "AU", "Australia/Adelaide": "AU",
    "Europe/Berlin": "DE", "Europe/Paris": "FR",
    "Asia/Dubai": "AE", "Asia/Singapore": "SG", "Asia/Tokyo": "JP",
    "America/New_York": "US", "America/Chicago": "US", "America/Denver": "US",
    "America/Los_Angeles": "US", "America/Phoenix": "US", "America/Anchorage": "US",
    "Pacific/Honolulu": "US",
}


def region_from_browser(locale: str = "", timezone: str = "") -> str:
    """Best-effort shopping/booking region from the browser's timezone + locale.

    Timezone (physical location) wins; then the locale's region subtag
    (e.g. ``en-IN`` → IN); otherwise the default region (US).
    """
    tz = (timezone or "").strip()
    if tz in _TZ_REGION:
        return _TZ_REGION[tz]
    loc = (locale or "").strip()
    if "-" in loc:
        reg = loc.rsplit("-", 1)[-1].upper()
        reg = _REGION_ALIASES.get(reg, reg)
        if reg in _SHOPPING_SITES:
            return reg
    if tz.startswith("Australia/"):
        return "AU"
    return _DEFAULT_REGION


def _retailer_url(domain: str, product: str) -> str:
    """A working product-search link for a retailer (Google-scoped fallback)."""
    tmpl = _RETAILER_SEARCH.get(domain)
    if tmpl:
        return tmpl.format(q=quote(product))
    return "https://www.google.com/search?q=" + quote_plus(f"{product} site:{domain}")


# ── Live product lookup: recognized retailers + price/stock extraction ──────
_ALL_SHOP_DOMAINS: set[str] = {d for _c, _ds in _SHOPPING_SITES.values() for d in _ds}
_ALL_SHOP_DOMAINS |= {
    "amazon.com", "newegg.com", "bhphotovideo.com", "costco.com", "microcenter.com",
    "gamestop.com", "very.co.uk", "snapdeal.com", "paytmmall.com",
}

_REGION_COUNTRY: dict[str, str] = {
    "US": "USA", "IN": "India", "UK": "United Kingdom", "CA": "Canada",
    "AU": "Australia", "DE": "Germany", "FR": "France", "AE": "UAE",
    "SG": "Singapore", "JP": "Japan",
}

_PRICE_RE = re.compile(
    r"(?:US\$|A\$|C\$|S\$|AED|Rs\.?|₹|\$|£|€|¥|\ue001)\s?\d[\d.,]*\d"
    r"|(?:USD|INR|GBP|EUR|AED|SGD|AUD|CAD|JPY)\s?\d[\d.,]*\d"
    r"|\d[\d.,]*\d\s?(?:USD|INR|GBP|EUR|AED|SGD|AUD|CAD|JPY)",
    re.IGNORECASE,
)
_BARE_PRICE_RE = re.compile(r"\b(?:\d{3,6}\.\d{2}|\d{1,3}(?:,\d{3})+(?:\.\d{2})?)\b")
_OOS_RE = re.compile(r"out of stock|currently unavailable|sold out|unavailable", re.I)
_IN_STOCK_RE = re.compile(r"in stock|add to cart|buy now|available", re.I)


def _host(url: str) -> str:
    try:
        h = urlsplit(url).netloc.lower()
    except Exception:  # noqa: BLE001
        return ""
    return h[4:] if h.startswith("www.") else h


def _match_retailer(url: str) -> str:
    """Recognized shopping domain for a URL (e.g. amazon.in), or ''."""
    h = _host(url)
    for d in _ALL_SHOP_DOMAINS:
        if h == d or h.endswith("." + d):
            return d
    return ""


def _retailer_domain(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    domain = _match_retailer(raw if "://" in raw else f"https://{raw}")
    return domain or next(
        (d for d in _ALL_SHOP_DOMAINS if raw in {d, d.split(".", 1)[0]}), ""
    )


def _price_from_text(
    *texts: str, currency: str = "", allow_bare: bool = False
) -> str:
    for t in texts:
        m = _PRICE_RE.search(t or "")
        if m:
            price = m.group(0).strip()
            if price.startswith(chr(0xE001)):
                return f"{currency} {price.removeprefix(chr(0xE001)).strip()}"
            return price
    if allow_bare:
        for t in texts:
            if m := _BARE_PRICE_RE.search(t or ""):
                return f"{currency} {m.group(0)}"
    return ""


def _price_value(price: str) -> float | None:
    """Numeric value from a price string, for cheapest-first sorting."""
    m = re.search(r"\d[\d.,]*", price or "")
    if not m:
        return None
    num = m.group(0).strip(".,")
    if "," in num and "." in num:
        num = (num.replace(".", "").replace(",", ".")
               if num.rfind(",") > num.rfind(".") else num.replace(",", ""))
    elif "," in num:
        num = (num.replace(",", ".")
               if num.count(",") == 1 and re.search(r",\d{2}$", num)
               else num.replace(",", ""))
    try:
        return float(num)
    except ValueError:
        return None


def _stock_flag(text: str, has_price: bool) -> bool | None:
    """True/False/None in-stock guess from a listing snippet."""
    if _OOS_RE.search(text):
        return False
    if has_price or _IN_STOCK_RE.search(text):
        return True
    return None


# ── Booking: city→IATA, date parsing, and deep-link builders ───────────────
_CITY_IATA: dict[str, str] = {
    "mumbai": "BOM", "bombay": "BOM", "delhi": "DEL", "new delhi": "DEL",
    "bangalore": "BLR", "bengaluru": "BLR", "hyderabad": "HYD", "chennai": "MAA",
    "kolkata": "CCU", "calcutta": "CCU", "pune": "PNQ", "ahmedabad": "AMD",
    "goa": "GOI", "jaipur": "JAI", "kochi": "COK", "cochin": "COK",
    "lucknow": "LKO", "chandigarh": "IXC", "srinagar": "SXR",
    "new york": "NYC", "nyc": "NYC", "los angeles": "LAX", "san francisco": "SFO",
    "chicago": "ORD", "boston": "BOS", "seattle": "SEA", "miami": "MIA",
    "washington": "WAS", "atlanta": "ATL", "dallas": "DFW", "houston": "IAH",
    "london": "LON", "paris": "PAR", "amsterdam": "AMS", "frankfurt": "FRA",
    "munich": "MUC", "berlin": "BER", "madrid": "MAD", "barcelona": "BCN",
    "rome": "ROM", "milan": "MIL", "zurich": "ZRH", "dublin": "DUB",
    "dubai": "DXB", "abu dhabi": "AUH", "doha": "DOH", "singapore": "SIN",
    "bangkok": "BKK", "hong kong": "HKG", "tokyo": "TYO", "seoul": "SEL",
    "sydney": "SYD", "melbourne": "MEL", "toronto": "YTO", "vancouver": "YVR",
    "kuala lumpur": "KUL", "jakarta": "CGK", "istanbul": "IST",
}

_MONTHS: dict[str, int] = {}
for _i, _m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1
):
    _MONTHS[_m] = _i
    _MONTHS[_m[:3]] = _i
_MONTHS_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))


def _roll_year(d: date) -> date:
    """Push a bare day/month into the future if it already passed this year."""
    if d < date.today():
        try:
            return d.replace(year=d.year + 1)
        except ValueError:
            return d
    return d


def _parse_date(text: str) -> date | None:
    """Extract a departure/check-in date; default the year to the current one."""
    t = (text or "").lower()
    today = date.today()
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTHS_RE})\.?\s*(\d{{4}})?", t)
    if m:
        day, month, yr = int(m.group(1)), _MONTHS[m.group(2)], m.group(3)
        try:
            d = date(int(yr) if yr else today.year, month, day)
        except ValueError:
            return None
        return d if yr else _roll_year(d)
    m = re.search(rf"\b({_MONTHS_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s*(\d{{4}})?", t)
    if m:
        month, day, yr = _MONTHS[m.group(1)], int(m.group(2)), m.group(3)
        try:
            d = date(int(yr) if yr else today.year, month, day)
        except ValueError:
            return None
        return d if yr else _roll_year(d)
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", t)
    if m:
        day, month, yr = int(m.group(1)), int(m.group(2)), m.group(3)
        year = today.year
        if yr:
            year = int(yr) + 2000 if len(yr) == 2 else int(yr)
        try:
            d = date(year, month, day)
        except ValueError:
            return None
        return d if yr else _roll_year(d)
    return None


def _clean_place(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip(" .,")).strip()


def _parse_route(query: str, origin: str, destination: str) -> tuple[str, str]:
    o, d = _clean_place(origin), _clean_place(destination)
    if not (o and d):
        m = re.search(
            r"from\s+(.+?)\s+to\s+(.+?)(?:\s+on\b|\s+for\b|\s+in\b|[,.]|\d|$)",
            query, re.I,
        ) or re.search(
            r"\b([a-z][a-z .]*?)\s+to\s+([a-z][a-z .]*?)(?:\s+on\b|\s+for\b|[,.]|\d|$)",
            query, re.I,
        )
        if m:
            o = o or _clean_place(m.group(1))
            d = d or _clean_place(m.group(2))
    return o, d


def _parse_city(query: str, destination: str) -> str:
    if _clean_place(destination):
        return _clean_place(destination)
    m = re.search(
        r"\b(?:in|at|near)\s+(.+?)(?:\s+on\b|\s+for\b|\s+this\b|\s+next\b|[,.]|\d|$)",
        query, re.I,
    )
    return _clean_place(m.group(1)) if m else _clean_place(query)


def _opt(platform: str, title: str, url: str, snippet: str = "") -> dict[str, str]:
    return {"platform": platform, "title": title, "url": url, "snippet": snippet}


def _flight_options(query: str, origin: str, destination: str, d: date | None, reg: str) -> list[dict[str, str]]:
    o, dest = _parse_route(query, origin, destination)
    oi, di = _CITY_IATA.get(o.lower(), ""), _CITY_IATA.get(dest.lower(), "")
    frm, to = (o or "origin").title(), (dest or "destination").title()
    gq = f"flights from {frm} to {to}" + (f" on {d.isoformat()}" if d else "")
    opts = [
        _opt("google.com/travel/flights", "Google Flights, compare every airline",
             "https://www.google.com/travel/flights?q=" + quote_plus(gq))
    ]
    if oi and di:
        arrow = f"{oi}→{di}"
        if d:
            opts.append(_opt("skyscanner.net", f"Skyscanner, {arrow}",
                             f"https://www.skyscanner.net/transport/flights/{oi.lower()}/{di.lower()}/{d.strftime('%y%m%d')}/"))
            opts.append(_opt("kayak.com", f"KAYAK, {arrow}",
                             f"https://www.kayak.com/flights/{oi}-{di}/{d.isoformat()}"))
            if reg == "IN":
                opts.append(_opt("makemytrip.com", f"MakeMyTrip, {arrow}",
                                 f"https://www.makemytrip.com/flight/search?itinerary={oi}-{di}-{d.strftime('%d/%m/%Y')}&tripType=O&paxType=A-1_C-0_I-0&cabinClass=E"))
                opts.append(_opt("cleartrip.com", f"Cleartrip, {arrow}",
                                 f"https://www.cleartrip.com/flights/results?adults=1&childs=0&infants=0&class=Economy&depart_date={d.strftime('%d/%m/%Y')}&from={oi}&to={di}"))
        else:
            opts.append(_opt("skyscanner.net", f"Skyscanner, {arrow}",
                             f"https://www.skyscanner.net/transport/flights/{oi.lower()}/{di.lower()}/"))
    return opts


def _hotel_options(query: str, destination: str, d: date | None, reg: str) -> list[dict[str, str]]:
    city = _parse_city(query, destination)
    label = (city or "your destination").title()
    bk = "https://www.booking.com/searchresults.html?ss=" + quote_plus(city)
    if d:
        bk += f"&checkin={d.isoformat()}"
    opts = [
        _opt("google.com/travel", f"Google Hotels, {label}",
             "https://www.google.com/travel/search?q=" + quote_plus(f"hotels in {city}")),
        _opt("booking.com", f"Booking.com, {label}", bk),
    ]
    third = "makemytrip.com" if reg == "IN" else "expedia.com"
    name = "MakeMyTrip" if reg == "IN" else "Expedia"
    opts.append(_opt(third, f"{name}, {label}",
                     "https://www.google.com/search?q=" + quote_plus(f"hotels in {city} site:{third}")))
    return opts


def _event_options(query: str, reg: str) -> list[dict[str, str]]:
    q = quote_plus(query)
    opts: list[dict[str, str]] = []
    if reg == "IN":
        opts.append(_opt("bookmyshow.com", "BookMyShow",
                         "https://www.google.com/search?q=" + quote_plus(f"{query} bookmyshow")))
        opts.append(_opt("district.in", "Zomato District",
                         "https://www.google.com/search?q=" + quote_plus(f"{query} district by zomato tickets")))
        opts.append(_opt("insider.in", "Paytm Insider",
                         "https://www.google.com/search?q=" + quote_plus(f"{query} insider.in")))
    opts.append(_opt("ticketmaster.com", "Ticketmaster",
                     f"https://www.ticketmaster.com/search?q={q}"))
    opts.append(_opt("seatgeek.com", "SeatGeek",
                     f"https://seatgeek.com/search?search={q}"))
    opts.append(_opt("google.com", "Google, all ticket options",
                     "https://www.google.com/search?q=" + quote_plus(f"{query} tickets")))
    return opts


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
    retailer: str = Field(
        default="",
        description="Optional retailer domain or name, such as noon.com or noon.",
    )


@register_tool(args_schema=ProductPricesInput)
def product_prices(product: str, region: str = "US", retailer: str = "") -> str:
    """Find where a product is available to buy, with direct product-page links
    plus any price and in-stock status the live listing shows.

    Uses live web search (Firecrawl / Brave / SerpAPI / Tavily when a key is
    set) to return the ACTUAL product page on each retailer for the user's
    region, with the price and stock hint from the listing, region's own
    retailers first. Without a search key it falls back to per-retailer search
    links. Use for "where to buy X", "cheapest X", "is X in stock", or price
    comparisons. The offers render as cards, so keep your reply short; confirm
    the final price/stock on the retailer page.
    """
    reg = _norm_region(region)
    currency, sites = _SHOPPING_SITES[reg]
    country = _REGION_COUNTRY.get(reg, "")
    requested = _retailer_domain(retailer)
    target_sites = [requested] if requested else sites

    try:
        from cortex.tools.web import _provider_search

        if requested:
            scope = (
                f"{requested}/uae-en"
                if requested == "noon.com" and reg == "AE"
                else requested
            )
            query = f"site:{scope} {product} {currency} price"
        else:
            query = f"{product} price buy {country}".strip()
        hits = _provider_search(query, 12)
    except Exception:  # noqa: BLE001
        hits = []

    offers: list[dict] = []
    seen: set[str] = set()
    for r in hits:
        dom = _match_retailer(r.get("url", ""))
        key = r.get("url", "") if requested else dom
        if not dom or (requested and dom != requested) or key in seen:
            continue
        seen.add(key)
        title, snippet = r.get("title", ""), r.get("snippet", "")
        price = _price_from_text(
            snippet, title, currency=currency, allow_bare=bool(requested)
        )
        offers.append(
            {
                "retailer": dom,
                "title": (title or product).strip()[:120],
                "url": r.get("url", ""),
                "price": price,
                "price_value": _price_value(price),
                "available": _stock_flag(f"{title} {snippet}", bool(price)),
            }
        )
        if requested and len(offers) == 5:
            break

    if requested:
        direct = [o for o in offers if "/p/" in o["url"] and o["price"]]
        if direct:
            offers = direct

    if offers:
        offers.sort(
            key=lambda o: (
                o["retailer"] not in sites,   # region's own retailers first
                o["price_value"] is None,     # priced offers next
                o["price_value"] or 0.0,      # cheapest first
            )
        )
        note = (
            "Live results, direct product pages with the price/stock each "
            "listing showed. Confirm the final price and stock on the page."
        )
    else:
        offers = [
            {
                "retailer": d,
                "title": product,
                "url": _retailer_url(d, product),
                "price": "",
                "price_value": None,
                "available": None,
            }
            for d in target_sites
        ]
        note = (
            "These open each retailer's search for the product. Set "
            "FIRECRAWL_API_KEY for direct product pages, live prices, and "
            "in-stock status."
        )

    return json.dumps(
        {
            "product": product,
            "region": reg,
            "currency": currency,
            "retailer": requested or None,
            "offers": offers,
            "note": note,
        },
        ensure_ascii=False,
    )


_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("flight", ("flight", "flights", "airfare", "airline", "fly ")),
    ("hotel", ("hotel", "resort", "accommodation", " stay", "airbnb", "room ")),
    ("movie", ("movie", "film", "cinema", "showtime")),
    ("concert", ("concert", "gig", "tour", "live music")),
    ("show", ("musical", "theatre", "theater", "broadway", "play ", "comedy show")),
    ("event", ("event", "festival", "expo", "conference", "match", "sports")),
]
_BOOKING_CATEGORIES = frozenset(name for name, _ in _CATEGORY_KEYWORDS)


def _norm_category(category: str | None, query: str) -> str:
    explicit = (category or "").strip().lower()
    if explicit in _BOOKING_CATEGORIES:
        return explicit
    text = f"{explicit} {query}".lower()
    for name, kws in _CATEGORY_KEYWORDS:
        if any(k in text for k in kws):
            return name
    return "event"


class FindBookingsInput(BaseModel):
    """Input for building booking links."""

    query: str = Field(
        description=(
            "What to book, e.g. 'flights Mumbai to Delhi 9 July' or "
            "'Coldplay Mumbai'."
        )
    )
    category: str = Field(
        default="",
        description=(
            "One of: flight, hotel, movie, concert, event, show. Blank = "
            "auto-detect from the query."
        ),
    )
    region: str = Field(
        default="US",
        description="User region/country (US, IN, UK, CA, AU, DE, ...). Defaults to US.",
    )
    origin: str = Field(
        default="",
        description="Flight origin city or airport (e.g. 'Mumbai' or 'BOM'). Flights only.",
    )
    destination: str = Field(
        default="",
        description="Flight destination, or the hotel city (e.g. 'Delhi' or 'DEL').",
    )
    date: str = Field(
        default="",
        description=(
            "Departure / check-in date as ISO YYYY-MM-DD, using the CURRENT "
            "year (see today's date in your context)."
        ),
    )


@register_tool(args_schema=FindBookingsInput)
def find_bookings(
    query: str,
    category: str = "",
    region: str = "US",
    origin: str = "",
    destination: str = "",
    date: str = "",
) -> str:
    """Build direct booking links for flights, hotels, movies, concerts,
    events, or shows on the platforms relevant to the user's region.

    Returns a JSON list of options whose links open each platform's LIVE
    search/results page, pre-filled with the route/city/date when known. For
    flights pass `origin`, `destination`, and `date` (ISO YYYY-MM-DD, CURRENT
    year); for hotels pass the `destination` city and `date` (check-in).
    Fares, seats, times, and availability are shown on the platform, never
    guess them. The links render as cards, so keep your reply short and make
    clear you can't complete the purchase.
    """
    cat = _norm_category(category, query)
    reg = _norm_region(region)
    d = _parse_date(date) or _parse_date(query)
    if cat == "flight":
        options = _flight_options(query, origin, destination, d, reg)
    elif cat == "hotel":
        options = _hotel_options(query, destination, d, reg)
    else:
        options = _event_options(query, reg)
    return json.dumps(
        {
            "query": query,
            "category": cat,
            "region": reg,
            "date": d.isoformat() if d else "",
            "options": options,
            "note": (
                "Each link opens the platform's live results, confirm fares, "
                "seats, times, and availability there. I can't complete the "
                "purchase for you."
            ),
        },
        ensure_ascii=False,
    )
