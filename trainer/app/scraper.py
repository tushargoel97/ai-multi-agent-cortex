"""Spec importers: deterministic fast paths + the intelligent scrape agent.

Two deterministic paths stay because they beat generic crawling:
  - AMD's official database (amd.com) embeds the whole processor catalog as
    HTML-escaped JSON — one fetch, no anti-bot wall (scrape_amd).
  - Uploaded documents (Intel comparison-chart PDFs, spreadsheets) via
    import_document (deterministic chart parser first, LLM distillation next).
Every other URL is delegated to app.scrape_agent, an LLM-driven index/leaf
crawl. parse_product stays as a deterministic spec-sheet extractor the agent
tries before falling back to distillation. All writes go through
research.save_learned_entry (alias-aware; never overrides curated facts.yaml).
"""

from __future__ import annotations

import html as html_lib
import logging
import re
import time

import httpx

from app.research import save_learned_entry

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
    parsed, PDFs read). Then a deterministic chart parser (Intel today) tries
    first, with LLM distillation as the fallback for anything else.
    """
    from app import intel_pdf, sources as src
    from app.research import distill_products_from_text

    name = entry.get("name", entry.get("id", "source"))
    on_progress(0, 0, f"extracting {name}…")
    text = src.extract_source(entry, on_log=on_log)
    entries = intel_pdf.parse_chart_text(text, limit)
    if len(entries) >= 3:
        return entries
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
    (app.scrape_agent) with the given crawl budget.

    on_progress(done, total, label) keeps the pipeline status live; on_log(msg)
    appends per-URL detail lines. Returns {saved, skipped, errors, outcomes}.
    """
    from app.scrape_agent import ScrapeBudget, run_scrape_agent

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
