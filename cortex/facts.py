"""Authoritative hardware specs from the training YAMLs (no web, no RAG).

The 1B fine-tuned specialist can drift numbers between sibling products
(PS5 / Slim / Pro). This module exposes the same ground truth the model was
trained on, trainer/data/domains/hardware/{facts,hardware_learned_facts}.yaml,
bind-mounted read only, so the synthesizer can correct drifted values.
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

# Fields that identify a product but aren't specs to render in a reference block.
_META_KEYS = {"name", "aliases", "exists", "notes", "source", "url", "group"}

# (mtimes signature, alias index): reloads when the YAMLs change (e.g. after
# gap research adds learned facts) without restarting the container.
_cache: tuple[tuple, dict] | None = None


_HW_DIR = FACTS_DIR / "domains" / "hardware"


def _hw_facts_file() -> Path:
    """Curated hardware facts; falls back to the legacy flat location."""
    new = _HW_DIR / "facts.yaml"
    return new if new.exists() else FACTS_DIR / "facts.yaml"


def _hw_learned_file() -> Path:
    """Hardware domain-level learned facts; falls back to legacy locations."""
    for path in (
        _HW_DIR / "hardware_learned_facts.yaml",
        FACTS_DIR / "hardware_learned_facts.yaml",
        FACTS_DIR / "learned_facts.yaml",
    ):
        if path.exists():
            return path
    return _HW_DIR / "hardware_learned_facts.yaml"


def _pack_learned_file(pack: Path) -> Path:
    new = pack / f"{pack.name}_learned_facts.yaml"
    old = pack / "learned_facts.yaml"
    return new if new.exists() else old


def _pack_dirs() -> list[Path]:
    """Every subdomain pack dir under <FACTS_DIR>/domains/<domain>/<subdomain>/."""
    base = FACTS_DIR / "domains"
    out: list[Path] = []
    if base.exists():
        for domain_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for sub_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
                if (
                    (sub_dir / "facts.yaml").exists()
                    or (sub_dir / f"{sub_dir.name}_learned_facts.yaml").exists()
                    or (sub_dir / "learned_facts.yaml").exists()
                    or (sub_dir / "subdomain.yaml").exists()
                ):
                    out.append(sub_dir)
    return out


def _pack_render(pack: Path) -> str:
    """The pack's declared answer style ('spec_table' or 'prose')."""
    for fname in ("subdomain.yaml", "domain.yaml"):
        f = pack / fname
        if f.exists():
            meta = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            return "spec_table" if meta.get("render") == "spec_table" else "prose"
    return "prose"


def _read_products(path: Path, *, learned: bool, render: str | None) -> list[dict]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if learned:
        items = [i for i in data.get("learned", []) if i.get("exists", True)]
    else:
        items = [
            i for group in data.values() if isinstance(group, list) for i in group
        ]
    return [{**i, "_render": render} if render else i for i in items]


def _load_index() -> dict[str, dict]:
    global _cache
    packs = _pack_dirs()
    sig_paths = [_hw_facts_file(), _hw_learned_file()]
    for pk in packs:
        sig_paths += [
            pk / "facts.yaml",
            _pack_learned_file(pk),
            pk / "subdomain.yaml",
        ]
    sig = tuple(p.stat().st_mtime if p.exists() else 0 for p in sig_paths)
    if _cache is not None and _cache[0] == sig:
        return _cache[1]

    # Built-in hardware (flat), spec_table style, so it needs no explicit tag.
    products = _read_products(_hw_facts_file(), learned=False, render=None)
    products += _read_products(_hw_learned_file(), learned=True, render=None)
    # User packs, tag each entity with its subdomain's render style.
    for pk in packs:
        render = _pack_render(pk)
        products += _read_products(pk / "facts.yaml", learned=False, render=render)
        products += _read_products(
            _pack_learned_file(pk), learned=True, render=render
        )

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


def _is_hardware(product: dict) -> bool:
    """A product is 'hardware' when it carries any curated spec field other than
    the shared release_year (games etc. only have release_year)."""
    return any(
        product.get(field) not in (None, "")
        for field, _ in _SPEC_FIELDS
        if field != "release_year"
    )


def reference_block(products: list[dict]) -> str:
    """Compact authoritative fact sheet for the given products. Hardware uses the
    curated spec fields; other domains (games, …) dump their own fields."""
    lines: list[str] = []
    for p in products:
        lines.append(f"{p.get('name', '?')}:")
        if _is_hardware(p):
            for field, label in _SPEC_FIELDS:
                value = p.get(field)
                if value is not None and value != "":
                    lines.append(f"  - {label}: {value}")
        else:
            for key, value in p.items():
                if key in _META_KEYS or key.startswith("_") or value in (None, "", []):
                    continue
                label = key.replace("_", " ").capitalize()
                lines.append(f"  - {label}: {value}")
    return "\n".join(lines)


def is_prose_products(products: list[dict]) -> bool:
    """True when a matched product should be answered in prose rather than a
    spec table: a pack entity whose subdomain declares render: prose, or a
    non-hardware entity with no explicit style."""
    for p in products:
        render = p.get("_render")
        if render == "spec_table":
            continue
        if render == "prose" or not _is_hardware(p):
            return True
    return False
