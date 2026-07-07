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

import httpx
import yaml

from .config import settings

from . import search

logger = logging.getLogger(__name__)


def _learned_path():
    """Hardware domain-level learned facts (migrates the legacy name once)."""
    new = settings.data_dir / "hardware_learned_facts.yaml"
    old = settings.data_dir / "learned_facts.yaml"
    if not new.exists() and old.exists():
        try:
            old.rename(new)
        except OSError:
            return old
    return new


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

    hits = search.provider_search(f"{name} full specifications release price")
    if not hits:
        log(f"  !! no search results for {name}")
        return None
    context_parts: list[str] = []
    for hit in hits[:2]:
        try:
            text = search.fetch_page(hit["url"], max_chars=5000)
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


_MAKER_PREFIXES = (
    "Qualcomm ",
    "MediaTek ",
    "Samsung ",
    "Apple ",
    "AMD ",
    "Intel ",
    "NVIDIA ",
    "Google ",
)


def _derive_aliases(name: str) -> set[str]:
    """Common short names people actually type, derived deterministically from
    the canonical product name — the LLM distiller tends to alias SKU part
    codes (APL1W10, SM8850-AC) instead, which no one searches for.

    'Apple A16 Bionic'                  → {'A16 Bionic', 'A16'}
    'Qualcomm Snapdragon 8 Elite Gen 5' → {'Snapdragon 8 Elite Gen 5',
                                           'Snapdragon 8 Elite'}
    'AMD Ryzen 9 PRO 9965X3D'           → {'Ryzen 9 PRO 9965X3D', '9965X3D'}
    'Intel Core Ultra 7 265'            → {'Core Ultra 7 265', 'Ultra 7 265'}
    """
    out: set[str] = set()
    base = name
    for pref in _MAKER_PREFIXES:
        if base.startswith(pref):
            base = base[len(pref):]
            out.add(base)  # name without the maker prefix
            break

    # Apple: 'A16 Bionic' → also 'A16'; 'A18 Pro' stays (Pro is meaningful).
    m = re.match(r"^(A\d{2})\s+Bionic$", base)
    if m:
        out.add(m.group(1))

    # Snapdragon: drop a trailing 'Gen N' ONLY when a distinctive word remains
    # (e.g. 'Snapdragon 8 Elite Gen 5' → 'Snapdragon 8 Elite'). Never reduce to
    # a bare 'Snapdragon 8', which collides across generations.
    m = re.match(r"^(Snapdragon \d+\w*\s+\w+.*?)\s+Gen\s+\d+$", base)
    if m:
        out.add(m.group(1))

    # Intel Core Ultra: 'Core Ultra 7 265' → 'Ultra 7 265'.
    if base.startswith("Core Ultra "):
        out.add(base[len("Core "):])

    # A bare trailing model number (Ryzen/Core SKUs): '...9965X3D' → '9965X3D'.
    m = re.search(r"\b(\d{3,4}[A-Z0-9]{0,4})$", base)
    if m and m.group(1) != base:
        out.add(m.group(1))

    out.discard(name)
    return {a.strip() for a in out if a.strip()}


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
    # Keep curated knowledge from the entries being replaced, plus the LLM's
    # aliases, plus deterministically-derived common short names (A16,
    # Snapdragon 8 Elite) so routing/grounding recognizes what users type.
    aliases = {str(a) for e in replaced for a in e.get("aliases", [])}
    aliases |= {str(a) for a in entry.get("aliases", [])}
    aliases |= _derive_aliases(entry.get("name", ""))
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


# ── Smart, domain-aware import (sources → any domain/subdomain) ──────────────
#
# Reads the selected sources and either auto-detects the best-fit domain +
# subdomain (proposing a schema) or extracts into a subdomain the admin picked,
# returning a PROPOSAL for review. ``apply_import`` writes the approved result:
# new packs are created and rows appended to the pack's learned facts (hardware
# rows merge into its learned layer — the same place spec import writes).

_IMPORT_MAX_TEXT = 16000

_AUTO_IMPORT_PROMPT = """You are organizing scraped content into a training taxonomy.
Existing domains → subdomains:
{existing}

From the SOURCE TEXT: (1) pick the best-fitting domain and subdomain — REUSE an
existing one when the content matches, otherwise propose new snake_case names;
(2) define 4-8 entity fields; (3) extract every distinct entity.

Reply with STRICT JSON only:
{{"domain": "snake_case", "subdomain": "snake_case",
  "render": "prose" | "spec_table",
  "fields": [{{"key": "snake_case", "label": "Title Case"}}],
  "entities": [{{"name": "str", "aliases": ["str"], "<field_key>": "value"}}]}}

Use "spec_table" only for hardware / numeric-spec items, else "prose". Copy
values from the text; omit unknowns. "name" is implicit — never a field. Skip
entities with fewer than 2 known fields. No commentary.

SOURCE TEXT:
{text}"""

_EXTRACT_IMPORT_PROMPT = """Extract every distinct entity from the SOURCE TEXT as
STRICT JSON: {{"entities": [{{"name": "str", "aliases": ["str"], ...}}]}}. Use
ONLY these fields: {fields}. Copy values from the text; omit unknowns. Skip
entities with fewer than 2 known fields. At most 40 entities. No commentary.

SOURCE TEXT:
{text}"""


def _import_gather_text(items: list, on_log) -> str:
    from . import sources

    parts: list[str] = []
    for item in items:
        try:
            if isinstance(item, str):
                text, label = sources.extract_url(item), item
            else:
                text = sources.extract_source(item, on_log=on_log)
                label = item.get("name") or item.get("url") or item.get("id", "source")
            if text and len(text.strip()) > 40:
                parts.append(f"[{label}]\n{text.strip()}")
                on_log(f"read {label}")
            else:
                on_log(f"skip {label}: no usable text")
        except Exception as e:  # noqa: BLE001
            on_log(f"skip source: {type(e).__name__}: {e}")
    return "\n\n".join(parts)


def _existing_taxonomy() -> str:
    import generate_dataset

    lines = [
        f"- {d['name']}: "
        + (", ".join(s["name"] for s in d.get("subdomains", [])) or "(none)")
        for d in generate_dataset.available_domains()
    ]
    return "\n".join(lines) or "(none)"


def _subdomain_exists(domain: str, sub: str) -> bool:
    import generate_dataset

    for d in generate_dataset.available_domains():
        if d["name"] == domain:
            return any(s["name"] == sub for s in d.get("subdomains", []))
    return False


def _clean_import_entities(raw, fields: list[dict]) -> list[dict]:
    keys = {f.get("key") for f in fields}
    out: list[dict] = []
    for e in raw or []:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        row: dict = {"name": name}
        aliases = e.get("aliases")
        if isinstance(aliases, list) and aliases:
            row["aliases"] = [str(a).strip() for a in aliases if str(a).strip()]
        for key, value in e.items():
            if key in keys and value not in (None, "", []):
                row[key] = value
        out.append(row)
    return out


def propose(items: list, target: str, on_log=None) -> dict:
    """Return a reviewable import proposal (domain/subdomain/render/fields/
    entities). ``target`` is 'auto' (LLM detects) or 'domain/subdomain'."""
    from . import domains

    log = on_log or (lambda _m: None)
    text = _import_gather_text(items, log)
    if not text.strip():
        raise ValueError("No readable text from the selected sources.")

    if not target or target == "auto":
        log("classifying + extracting (LLM)…")
        data = _first_json(
            _chat(_AUTO_IMPORT_PROMPT.format(existing=_existing_taxonomy(), text=text[:_IMPORT_MAX_TEXT]))
        ) or {}
        domain = domains._slug(data.get("domain") or "misc")
        subdomain = domains._slug(data.get("subdomain") or "items")
        render = "spec_table" if data.get("render") == "spec_table" else "prose"
        fields = domains._normalize_fields(data.get("fields"))
        entities = _clean_import_entities(data.get("entities"), fields)
    else:
        domain, _, subdomain = target.partition("/")
        domain, subdomain = domains._slug(domain), domains._slug(subdomain)
        cfg = domains.get_subdomain(domain, subdomain)
        fields = cfg.get("fields") or []
        render = cfg.get("render", "prose")
        log("extracting entities (LLM)…")
        keys = ", ".join(f["key"] for f in fields) or "name"
        data = _first_json(
            _chat(_EXTRACT_IMPORT_PROMPT.format(fields=keys, text=text[:_IMPORT_MAX_TEXT]))
        ) or {}
        entities = _clean_import_entities(data.get("entities"), fields)

    log(f"proposed {len(entities)} entities → {domain}/{subdomain}")
    return {
        "domain": domain,
        "subdomain": subdomain,
        "render": render,
        "fields": fields,
        "entities": entities,
        "new_subdomain": not _subdomain_exists(domain, subdomain),
    }


def _apply_hardware_import(group: str, entities: list[dict]) -> dict:
    from . import domains

    if group not in domains._hw_groups():
        raise ValueError(f"'{group}' is not a hardware subdomain.")
    by_name: dict[str, dict] = {
        str(e.get("name", "")).lower(): e
        for e in domains.get_subdomain("hardware", group)["entities"]
    }
    for e in entities:
        name = str(e.get("name", "")).strip()
        if name:
            by_name[name.lower()] = e
    rows = domains.set_entities("hardware", group, list(by_name.values()))
    return {"domain": "hardware", "subdomain": group, "count": len(entities), "total": len(rows)}


def apply_import(proposal: dict) -> dict:
    """Persist an approved proposal: create the subdomain if new and append the
    rows to its learned layer (hardware merges into its learned facts)."""
    from . import domains

    domain = domains._slug(proposal.get("domain", ""))
    sub = domains._slug(proposal.get("subdomain", ""))
    if not domain or not sub:
        raise ValueError("Proposal is missing a domain/subdomain.")
    entities = proposal.get("entities") or []
    if domain == "hardware":
        return _apply_hardware_import(sub, entities)
    if not _subdomain_exists(domain, sub):
        domains.save_subdomain(
            domain,
            sub,
            render=proposal.get("render", "prose"),
            fields=proposal.get("fields") or [],
        )
    rows = domains.add_learned_entities(domain, sub, entities)
    return {"domain": domain, "subdomain": sub, "count": len(entities), "total": len(rows)}
