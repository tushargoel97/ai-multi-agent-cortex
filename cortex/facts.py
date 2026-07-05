"""Authoritative hardware specs from the training YAMLs (no web, no RAG).

The 1B fine-tuned specialist can drift numbers between sibling products
(PS5 / Slim / Pro). This module exposes the same ground truth the model was
trained on — trainer/data/facts.yaml + learned_facts.yaml, bind-mounted read
only — so the synthesizer can correct drifted values deterministically.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

FACTS_DIR = Path(os.getenv("FACTS_DIR", "/app/trainer_data"))

_SPEC_FIELDS = (
    ("release_year", "Released"),
    ("launch_price_usd", "Launch price (USD)"),
    ("cpu", "CPU"),
    ("gpu", "GPU"),
    ("compute_tflops", "Compute (TFLOPS)"),
    ("memory", "Memory"),
    ("memory_bandwidth", "Memory bandwidth"),
    ("storage", "Storage"),
    ("display", "Display"),
    ("power_watts", "Power (W)"),
)

# (mtimes signature, alias index) — reloads when the YAMLs change (e.g. after
# gap research adds learned facts) without restarting the container.
_cache: tuple[tuple, dict] | None = None


def _load_index() -> dict[str, dict]:
    global _cache
    paths = [FACTS_DIR / "facts.yaml", FACTS_DIR / "learned_facts.yaml"]
    sig = tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)
    if _cache is not None and _cache[0] == sig:
        return _cache[1]

    products: list[dict] = []
    if paths[0].exists():
        for group in (yaml.safe_load(paths[0].read_text()) or {}).values():
            if isinstance(group, list):
                products.extend(group)
    if paths[1].exists():
        for item in (yaml.safe_load(paths[1].read_text()) or {}).get("learned", []):
            if item.get("exists", True):
                products.append(item)

    index: dict[str, dict] = {}
    for p in products:
        for name in (p.get("name", ""), *p.get("aliases", [])):
            if name:
                index[name.lower()] = p
    _cache = (sig, index)
    return index


def match_products(text: str) -> list[dict]:
    """Products mentioned in the text; longest alias wins, so 'PS5 Pro'
    doesn't also match the base 'PS5'."""
    index = _load_index()
    remaining = text.lower()
    out: list[dict] = []
    seen: set[int] = set()
    for name in sorted(index, key=len, reverse=True):
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, remaining):
            product = index[name]
            if id(product) not in seen:
                seen.add(id(product))
                out.append(product)
            remaining = re.sub(pattern, " ", remaining)
    return out


def reference_block(products: list[dict]) -> str:
    """Compact authoritative spec sheet for the given products."""
    lines: list[str] = []
    for p in products:
        lines.append(f"{p.get('name', '?')}:")
        for field, label in _SPEC_FIELDS:
            value = p.get(field)
            if value is not None and value != "":
                lines.append(f"  - {label}: {value}")
    return "\n".join(lines)
