"""Fast-path parser for Intel's desktop comparison-chart PDFs.

One of the pluggable strategies behind the generic spec importer
(scraper.import_document): Intel publishes "Intel Core Desktop Boxed
Processors Comparison Chart" — a spec table whose rows extract cleanly as
text, so the mapping is deterministic (no LLM). Documents that don't match
this layout fall through to LLM distillation instead.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_ROW_RE = re.compile(
    r"(?P<num>(?:i[3579]-\d{4,5}[A-Z]{0,2}|\d{3,5}[A-Z]{0,2}(?:\s?Plus)?))\s*"
    r"Intel® Core™\s+(?P<gen>Ultra Series \d+|\d+th)\s+"
    r"(?P<tier>i[3579]|\d)\s+(?P<year>20\d\d)\s+"
    r"(?P<cores>\d+)\s+(?P<pcores>\d+|N/A)\s+(?P<ecores>\d+|N/A)\s+"
    r"(?P<threads>\d+)\s+(?P<turbo>[\d.]+|N/A)\s+(?P<pbase>[\d.]+|N/A)\s+"
    r"(?P<ebase>[\d.]+|N/A)\s+(?P<base>[\d.]+|N/A)\s+(?P<cache>\d+)\s+"
    r"(?P<tdp>\d+)\s+(?P<maxmem>\d+)(?:\s*GB)?\s+(?P<mem>.{4,60}?)\s+"
    r"(?P<pcie>\d+)\s+(?P<socket>LGA\s?\d+)",
    re.DOTALL,
)


def _name(m: re.Match) -> tuple[str, list[str]]:
    num = re.sub(r"\s+", " ", m["num"].strip())
    if m["gen"].startswith("Ultra"):
        name = f"Intel Core Ultra {m['tier']} {num}"
        aliases = [f"Core Ultra {m['tier']} {num}", num]
    else:
        name = f"Intel Core {num}" if num.startswith("i") else f"Intel Core {m['tier']}-{num}"
        short = num if num.startswith("i") else f"{m['tier']}-{num}"
        aliases = [f"Core {short}", short]
    return name, [a for a in aliases if a != name]


def parse_chart_text(text: str, max_products: int = 15) -> list[dict]:
    """Entries from Intel-chart-formatted text, newest first ([] if the
    document doesn't match this layout)."""
    rows = list(_ROW_RE.finditer(text))
    seen: set[str] = set()
    candidates = []
    for m in rows:
        name, aliases = _name(m)
        if name in seen:
            continue  # chart repeats SKUs across memory-config variants
        seen.add(name)
        candidates.append((int(m["year"]), name, aliases, m))
    candidates.sort(key=lambda c: c[0], reverse=True)

    entries: list[dict] = []
    for year, name, aliases, m in candidates[:max_products]:
        p, e = m["pcores"], m["ecores"]
        cpu_bits = [f"{m['cores']} cores" + (f" ({p}P+{e}E)" if p != "N/A" else "")]
        cpu_bits.append(f"{m['threads']} threads")
        if m["base"] != "N/A":
            cpu_bits.append(f"base {m['base']} GHz")
        elif m["pbase"] != "N/A":
            cpu_bits.append(f"P-core base {m['pbase']} GHz")
        if m["turbo"] != "N/A":
            cpu_bits.append(f"up to {m['turbo']} GHz")
        mem = re.sub(r"\s+", " ", m["mem"]).strip(" ,")
        entry = {
            "name": name,
            "brand": "Intel",
            "category": "CPU",
            "release_year": year,
            "launch_price_usd": None,  # the comparison chart carries no prices
            "cpu": ", ".join(cpu_bits),
            "gpu": None,
            "compute_tflops": None,
            "memory": f"{mem}, max {m['maxmem']} GB",
            "memory_bandwidth": None,
            "storage": None,
            "display": None,
            "power_watts": int(m["tdp"]),
            "key_features": [
                f"{m['cache']} MB cache",
                f"Socket {m['socket'].replace(' ', '')}",
                f"{m['pcie']} PCIe lanes",
            ],
            "best_for": ["Gaming", "Productivity"],
            "aliases": aliases,
            "exists": True,
            "notes": "Specs from Intel's official desktop comparison chart (PDF)",
        }
        entries.append(entry)
    return entries
