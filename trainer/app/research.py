"""Gap research — turn unanswered hardware questions into training facts.

For each knowledge gap: extract the product names the user asked about,
search the web for their specs, distill the findings into a facts.yaml-style
entry via the QA LLM, and append it to data/learned_facts.yaml. The next
dataset generation + fine-tune bakes the knowledge into the model's weights —
the web is only ever used here, between training runs.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse

import httpx
import yaml

from .config import settings
from .sources import _USER_AGENT, extract_url

logger = logging.getLogger(__name__)

LEARNED_FACTS_PATH = None  # resolved lazily from settings


def _learned_path():
    return settings.data_dir / "learned_facts.yaml"


# ── Web search (DuckDuckGo HTML, no API key) ────────────────────────────────


def ddg_search(query: str, max_results: int = 4) -> list[dict]:
    from bs4 import BeautifulSoup

    resp = httpx.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": _USER_AGENT},
        timeout=15.0,
        follow_redirects=True,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for a in soup.select("a.result__a")[:max_results]:
        href = a.get("href", "")
        # DDG wraps target urls in a redirect with uddg param
        if "uddg=" in href:
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            href = qs.get("uddg", [href])[0]
        results.append({"title": a.get_text(strip=True), "url": href})
    return results


# ── LLM helpers (same OpenAI-compatible endpoint as qa_generator) ───────────


def _chat(prompt: str, *, temperature: float = 0.2) -> str:
    from .qa_generator import _resolve_model

    base_url = settings.qa_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.qa_api_key}"}
    with httpx.Client(timeout=180.0, headers=headers) as client:
        model = _resolve_model(client, base_url)
        resp = client.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _first_json(text: str):
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extract_product_names(question: str) -> list[str]:
    """LLM name extraction with a regex fallback."""
    try:
        out = _chat(
            "List the hardware product names mentioned in this question as a "
            'STRICT JSON array of strings (e.g. ["PS5 Slim", "Ryzen 7 3700X"]). '
            "Use full official-style names. No commentary.\n\n"
            f"Question: {question}"
        )
        names = _first_json(out)
        if isinstance(names, list) and names:
            return [str(n).strip() for n in names if str(n).strip()][:4]
    except Exception:  # noqa: BLE001
        logger.exception("LLM name extraction failed — falling back to raw question")
    return [question.strip()[:60]]


_DISTILL_PROMPT = """You are building a hardware spec sheet. From the source text below,
produce STRICT JSON for the product "{name}" with exactly these keys:
{{"name": str, "brand": str, "category": str, "release_year": int|null,
"launch_price_usd": int|null, "cpu": str|null, "gpu": str|null,
"compute_tflops": float|null, "memory": str|null, "memory_bandwidth": str|null,
"storage": str|null, "display": str|null, "power_watts": int|null,
"key_features": [str, ...], "best_for": [str, ...], "aliases": [str, ...],
"exists": bool, "notes": str}}

Rules:
- Ground values ONLY in the source text; use null when the text doesn't say.
- "exists": false if the text indicates this exact product was never released
  (e.g. the user misremembered a name) — then explain in "notes" and name the
  closest real product.
- "aliases": short names people use (e.g. "PS5 Slim").
- Output the JSON object only.

Source text:
---
{context}
---"""


def research_product(name: str, on_log=None) -> dict | None:
    """Search the web for a product's specs and distill a facts entry."""
    def log(msg: str) -> None:
        if on_log:
            on_log(msg)

    hits = ddg_search(f"{name} full specifications release price")
    if not hits:
        log(f"  !! no search results for {name}")
        return None
    context_parts: list[str] = []
    for hit in hits[:2]:
        try:
            text = extract_url(hit["url"])[:5000]
            context_parts.append(f"[{hit['title']}]\n{text}")
            log(f"  read: {hit['url'][:80]}")
        except Exception as e:  # noqa: BLE001
            log(f"  skip {hit['url'][:60]}: {type(e).__name__}")
    if not context_parts:
        return None

    out = _chat(_DISTILL_PROMPT.format(name=name, context="\n\n".join(context_parts)[:9000]))
    entry = _first_json(out)
    if not isinstance(entry, dict) or not entry.get("name"):
        log(f"  !! distillation unparseable for {name}")
        return None
    entry = _validate_entry(entry, log)
    spec_fields = ["cpu", "gpu", "compute_tflops", "memory", "storage", "launch_price_usd"]
    if entry.get("exists", True) and sum(1 for f in spec_fields if entry.get(f)) < 2:
        log(f"  !! too few grounded specs for {name}")
        return None
    return entry


def _validate_entry(entry: dict, log) -> dict:
    """Deterministic sanity checks — small distiller models garble units."""
    tflops = entry.get("compute_tflops")
    if isinstance(tflops, (int, float)) and not (0.1 <= tflops <= 5000):
        log(f"  fix: implausible compute_tflops={tflops} → dropped")
        entry["compute_tflops"] = None
    mb = str(entry.get("memory_bandwidth") or "")
    if mb and not any(u in mb.lower() for u in ("gb/s", "tb/s", "mt/s", "gbps")):
        log(f"  fix: memory_bandwidth {mb!r} lacks a bandwidth unit → dropped")
        entry["memory_bandwidth"] = None
    year = entry.get("release_year")
    if isinstance(year, int) and not (1990 <= year <= 2030):
        entry["release_year"] = None
    price = entry.get("launch_price_usd")
    if isinstance(price, (int, float)) and not (10 <= price <= 100000):
        entry["launch_price_usd"] = None
    features = entry.get("key_features") or []
    entry["key_features"] = [f for f in features if 8 <= len(str(f)) <= 120][:4]
    return entry


def load_learned() -> list[dict]:
    path = _learned_path()
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("learned", [])


_MULTI_DISTILL_PROMPT = """You are building hardware spec sheets. From the source text
below, extract EVERY distinct hardware product (CPU, GPU, console, etc.) with
enough data to be useful. Respond with ONLY a JSON array; each element uses
exactly this schema (use null when the source doesn't say):

{"name": str, "brand": str, "category": str, "release_year": int|null,
 "launch_price_usd": int|null, "cpu": str|null, "gpu": str|null,
 "compute_tflops": float|null, "memory": str|null, "memory_bandwidth": str|null,
 "storage": str|null, "display": str|null, "power_watts": int|null,
 "key_features": [str], "best_for": [str], "aliases": [str], "exists": true,
 "notes": str}

Rules: copy numbers exactly from the source; never guess or invent values;
skip products with fewer than two hard facts; at most {max_products} products.

SOURCE TEXT:
{text}
"""


def distill_products_from_text(
    text: str, max_products: int = 15, on_log=None
) -> list[dict]:
    """LLM-distill product entries from arbitrary document/page text."""
    log = on_log or (lambda _msg: None)
    prompt = _MULTI_DISTILL_PROMPT.replace("{max_products}", str(max_products)).replace(
        "{text}", text[:24000]
    )
    try:
        raw = _chat(prompt)
    except Exception as e:  # noqa: BLE001
        log(f"distillation failed: {e}")
        return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        log("distillation returned no JSON array")
        return []
    try:
        items = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        log("distillation JSON did not parse")
        return []
    entries = []
    for item in items[:max_products]:
        if isinstance(item, dict) and item.get("name"):
            item.setdefault("exists", True)
            entries.append(_validate_entry(item, log))
    return entries


def _builtin_variants() -> set[str]:
    """All names+aliases in the curated facts.yaml — research must never
    override or contradict these (it once saved 'PS5 Pro exists: false')."""
    path = _learned_path().parent / "facts.yaml"
    if not path.exists():
        return set()
    variants: set[str] = set()
    for group in (yaml.safe_load(path.read_text()) or {}).values():
        if not isinstance(group, list):
            continue
        for p in group:
            for name in (p.get("name", ""), *p.get("aliases", [])):
                if name:
                    variants.add(name.lower())
    return variants


def _entry_variants(entry: dict) -> set[str]:
    return {
        str(v).lower()
        for v in (entry.get("name", ""), *entry.get("aliases", []))
        if v
    }


def save_learned_entry(entry: dict) -> bool:
    """Append/replace in learned_facts.yaml, alias-aware.

    Skips any product already covered by the curated facts.yaml (built-in
    specs are ground truth), and replaces learned entries whose name/alias
    sets overlap (so 'PS5 Slim' and 'PlayStation 5 Slim' can't coexist).
    Returns True when the entry was stored.
    """
    new_variants = _entry_variants(entry)
    if new_variants & _builtin_variants():
        return False  # curated product — research must not duplicate/contradict it
    existing = load_learned()
    replaced = [e for e in existing if _entry_variants(e) & new_variants]
    entries = [e for e in existing if not (_entry_variants(e) & new_variants)]
    # Keep curated knowledge from the entries being replaced.
    aliases = {str(a) for e in replaced for a in e.get("aliases", [])}
    aliases |= {str(a) for a in entry.get("aliases", [])}
    aliases.discard(entry.get("name", ""))
    if aliases:
        entry["aliases"] = sorted(aliases)
    for e in replaced:
        if e.get("closest") and not entry.get("closest"):
            entry["closest"] = e["closest"]
    entries.append(entry)
    _learned_path().write_text(
        yaml.safe_dump({"learned": entries}, sort_keys=False, allow_unicode=True)
    )
    return True
