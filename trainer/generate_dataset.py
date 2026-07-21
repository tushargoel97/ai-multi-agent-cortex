"""Deterministic dataset generator for the hardware-specialist fine-tune.

Expands trainer/data/domains/hardware/facts.yaml into chat-format JSONL that
trainer consumes directly:

    {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Run standalone:  uv run python generate_dataset.py   (from trainer/)
Or import and call generate() from the trainer service.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).resolve().parent / "data"
DOMAINS_DIR = DATA_DIR / "domains"  # per-domain packs (hardware, software, …)
HARDWARE_DIR = DOMAINS_DIR / "hardware"  # the built-in domain's own directory
SEED = 42
VALID_FRACTION = 0.1


def _migrate_hardware_layout() -> None:
    """Move legacy flat hardware files into domains/hardware/ (idempotent)."""
    for old, new in (
        (DATA_DIR / "facts.yaml", HARDWARE_DIR / "facts.yaml"),
        (
            DATA_DIR / "hardware_learned_facts.yaml",
            HARDWARE_DIR / "hardware_learned_facts.yaml",
        ),
        (
            DATA_DIR / "learned_facts.yaml",
            HARDWARE_DIR / "hardware_learned_facts.yaml",
        ),
    ):
        if old.exists() and not new.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            try:
                old.rename(new)
            except OSError:
                pass


_migrate_hardware_layout()
FACTS_PATH = HARDWARE_DIR / "facts.yaml"


def _hw_learned_path() -> Path:
    return HARDWARE_DIR / "hardware_learned_facts.yaml"

# Human noun used in buying-advice questions, per facts.yaml group.
GROUP_NOUNS = {
    "consoles": "gaming console",
    "gpus_consumer": "graphics card",
    "gpus_amd": "graphics card",
    "gpus_datacenter": "datacenter GPU",
    "cpus": "desktop CPU",
}

# (field, question templates, answer template): {name} etc. filled per product.
SPEC_TEMPLATES: list[tuple[str, list[str], str]] = [
    (
        "launch_price_usd",
        [
            "How much did the {name} cost at launch?",
            "What was the launch price of the {name}?",
            "What's the MSRP of the {name}?",
            "How expensive was the {name} when it came out?",
        ],
        "The {name} launched at ${value} in {release_year}.",
    ),
    (
        "release_year",
        [
            "When was the {name} released?",
            "What year did the {name} come out?",
            "How old is the {name}?",
        ],
        "The {name} was released in {value}.",
    ),
    (
        "cpu",
        [
            "What CPU does the {name} have?",
            "What processor is inside the {name}?",
            "Tell me about the {name}'s CPU.",
        ],
        "The {name} uses a {value}.",
    ),
    (
        "gpu",
        [
            "What GPU does the {name} have?",
            "What graphics hardware is in the {name}?",
            "Which GPU architecture does the {name} use?",
        ],
        "The {name} has a {value}.",
    ),
    (
        "compute_tflops",
        [
            "How many TFLOPS does the {name} have?",
            "What's the compute performance of the {name}?",
            "How powerful is the {name} in TFLOPS?",
        ],
        "The {name} delivers about {value} TFLOPS of compute.",
    ),
    (
        "memory",
        [
            "How much memory does the {name} have?",
            "What's the RAM/VRAM configuration of the {name}?",
            "Tell me about the {name}'s memory.",
        ],
        "The {name} comes with {value}.",
    ),
    (
        "memory_bandwidth",
        [
            "What's the memory bandwidth of the {name}?",
            "How fast is the memory on the {name}?",
        ],
        "The {name} has a memory bandwidth of {value}.",
    ),
    (
        "storage",
        [
            "How much storage does the {name} have?",
            "What kind of storage is in the {name}?",
        ],
        "The {name} ships with {value}.",
    ),
    (
        "display",
        [
            "What display does the {name} have?",
            "Tell me about the {name}'s screen or display output.",
        ],
        "The {name}: {value}.",
    ),
    (
        "power_watts",
        [
            "How much power does the {name} draw?",
            "What's the power consumption of the {name}?",
            "What's the TDP of the {name}?",
        ],
        "The {name} draws around {value} W under load.",
    ),
    (
        "key_features",
        [
            "What are the key features of the {name}?",
            "What makes the {name} stand out?",
            "Give me the highlights of the {name}.",
        ],
        "Key features of the {name}: {value}.",
    ),
]

COMPARISON_QUESTIONS = [
    "Compare the {a} and the {b}.",
    "{a} vs {b}, which is better?",
    "How does the {a} stack up against the {b}?",
    "What are the differences between the {a} and the {b}?",
]

BUYING_QUESTIONS = [
    "Which {noun} should I buy for {tag}?",
    "What's the best {noun} for {tag}?",
    "Recommend a {noun} for {tag}.",
]

# Full spec-sheet questions ("X specs"): must exist in-domain, otherwise the
# off-domain refusal (which uses this phrasing) swallows terse spec queries.
OVERVIEW_QUESTIONS = [
    "What are the specs of the {name}?",
    "Give me the specs of the {name}.",
    "{name} specs",
    "Tell me about the {name}.",
]


def _overview_answer(product: dict) -> str:
    parts = [f"{product['name']}"]
    meta = []
    if product.get("release_year"):
        meta.append(str(product["release_year"]))
    if product.get("launch_price_usd"):
        meta.append(f"${product['launch_price_usd']} at launch")
    header = parts[0] + (f" ({', '.join(meta)})" if meta else "") + ":"
    lines = [header]
    for field, label in (
        ("cpu", "CPU"),
        ("gpu", "GPU"),
        ("compute_tflops", "Compute"),
        ("memory", "Memory"),
        ("memory_bandwidth", "Memory bandwidth"),
        ("storage", "Storage"),
        ("display", "Display"),
        ("power_watts", "Power"),
    ):
        if product.get(field) is not None:
            value = product[field]
            if field == "compute_tflops":
                value = f"{value} TFLOPS"
            elif field == "power_watts":
                value = f"~{value} W"
            lines.append(f"- {label}: {value}")
    features = product.get("key_features") or []
    if features:
        lines.append(f"- Highlights: {'; '.join(features[:3])}")
    return "\n".join(lines)


# Products deliberately OUTSIDE the dataset: teach the model to say so
# instead of hallucinating specs (e.g. "compare PS5 Pro and Xbox 360").
OFF_DOMAIN_PRODUCTS = [
    "Xbox 360",
    "Xbox One",
    "PlayStation 4",
    "PS4 Pro",
    "Nintendo Switch OLED",
    "GTX 1080 Ti",
    "RTX 3080",
    "RX 6800 XT",
    "Ryzen 5 5600X",
    "Core i7-10700K",
]

OFF_DOMAIN_QUESTIONS = [
    "What are the specs of the {name}?",
    "{name} specs",
    "Tell me about the {name}.",
    "How many TFLOPS does the {name} have?",
    "How much did the {name} cost at launch?",
]

OFF_DOMAIN_ANSWER = (
    "The {name} isn't in my fine-tuned dataset, so I'd rather not guess its "
    "specs. Ask about something I've been trained on, or I can look it up."
)

OFF_DOMAIN_COMPARISONS = [
    (
        "Which is better, the PS5 Pro or the Xbox 360?",
        "I can only give you one side of that from my training: the Xbox 360 "
        "(2005 era) isn't in my dataset. The PlayStation 5 Pro (2024, $699) "
        "has an 8-core AMD Zen 2 CPU, a 60-CU RDNA GPU at 16.7 TFLOPS, 16 GB "
        "GDDR6 and a 2 TB SSD, but I'd rather not guess Xbox 360 specs for "
        "a real comparison.",
    ),
    (
        "Which is better, the RTX 5090 or the GTX 1080 Ti?",
        "The GTX 1080 Ti isn't in my fine-tuned dataset, so I can't compare "
        "them properly. From my training: the GeForce RTX 5090 (2025, $1999) "
        "delivers about 104.8 TFLOPS with 32 GB GDDR7, a generational leap "
        "over any 10-series card.",
    ),
]

def _load_products() -> list[dict]:
    with open(FACTS_PATH) as f:
        raw = yaml.safe_load(f)
    products = []
    for group, items in raw.items():
        for item in items:
            item["_group"] = group
            products.append(item)
    # Gap-researched products (self-improvement loop). Entries with
    # exists=false are handled separately as corrective answers.
    hw_learned = _hw_learned_path()
    if hw_learned.exists():
        learned = (yaml.safe_load(hw_learned.read_text()) or {}).get("learned", [])
        # Dedupe against curated names AND aliases (research may return
        # "PS5 Pro" for the curated "PlayStation 5 Pro").
        known = {v.lower() for p in products for v in _name_variants(p)}
        for item in learned:
            if not item.get("exists", True):
                continue
            variants = {item.get("name", "").lower()} | {
                str(a).lower() for a in item.get("aliases", [])
            }
            if variants & known:
                continue
            entry = {k: v for k, v in item.items() if k not in ("exists", "notes") and v not in (None, "", [])}
            # Join a group so in-category comparison pairs are generated against
            # the curated products, an explicit `group` (set by the admin row
            # editor) wins, else infer it from category/brand.
            entry["_group"] = entry.pop("group", None) or _group_for_learned(entry)
            products.append(entry)
    return products


def _group_for_learned(entry: dict) -> str:
    category = str(entry.get("category", "")).lower()
    brand = str(entry.get("brand", "")).lower()
    if "console" in category or "handheld" in category:
        return "consoles"
    if "datacenter" in category:
        return "gpus_datacenter"
    if "graphics" in category or "gpu" in category:
        return "gpus_amd" if "amd" in brand else "gpus_consumer"
    if "cpu" in category or "processor" in category:
        return "cpus"
    return "learned"


def _fake_variants(item: dict) -> list[str]:
    """Name + aliases, each with and without the brand prefix, users type
    'AMD Ryzen 3700' as often as 'Ryzen 3700'."""
    brand = item.get("brand", "")
    variants: list[str] = []
    for base in (item.get("name", ""), *item.get("aliases", [])):
        if not base:
            continue
        for form in (base, f"{brand} {base}" if brand and not base.startswith(brand) else ""):
            if form and form not in variants:
                variants.append(form)
    return variants


def _learned_corrections() -> list[dict]:
    """Corrective pairs for researched products that turned out not to exist
    (e.g. 'Ryzen 7 3700', the user probably means the 3700X). Covers both
    direct spec questions and comparisons against the closest real part."""
    hw_learned = _hw_learned_path()
    if not hw_learned.exists():
        return []
    learned = (yaml.safe_load(hw_learned.read_text()) or {}).get("learned", [])
    known: dict[str, dict] = {p["name"]: p for p in learned if p.get("exists", True)}
    facts = yaml.safe_load(FACTS_PATH.read_text()) or {}
    for group in facts.values():
        for p in group:
            known.setdefault(p["name"], p)
    examples = []
    for item in learned:
        if item.get("exists", True) or not item.get("notes"):
            continue
        notes = item["notes"]
        fakes = _fake_variants(item)
        for fake in fakes:
            for q in (f"What are the specs of the {fake}?", f"{fake} specs",
                      f"Tell me about the {fake}.", f"Does the {fake} exist?"):
                examples.append(_example(q, notes))
        closest = known.get(item.get("closest", ""))
        if closest is None:
            continue
        answer = (f"{notes}\n\nThe closest real part at a glance, "
                  f"{_overview_answer(closest)}")
        brand = closest.get("brand", "")
        real_forms = [closest["name"]]
        stripped = closest["name"].removeprefix(f"{brand} ").strip()
        for form in (stripped, stripped.replace(" 7 ", " "),
                     f"{brand} {stripped.replace(' 7 ', ' ')}" if brand else ""):
            if form and form not in real_forms:
                real_forms.append(form)
        for fake in fakes:
            for real in real_forms:
                for q in (f"Compare {real} and {fake}",
                       f"Compare the {real} and the {fake}.",
                       f"{real} vs {fake}",
                       f"{fake} vs {real}",
                       f"Which is better, the {real} or the {fake}?",
                       f"What are the differences between the {real} and the {fake}?"):
                    examples.append(_example(q, answer))
    return examples


def _fmt_value(field: str, value) -> str:
    if field == "key_features":
        return "; ".join(value)
    return str(value)


def _example(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def _name_variants(product: dict) -> list[str]:
    """Canonical name + aliases, users ask with short names ('RTX 5090',
    '9800X3D'), so questions must cover them. Answers always use canonical."""
    return [product["name"], *product.get("aliases", [])]


def _spec_examples(products: list[dict]) -> list[dict]:
    examples = []
    for product in products:
        variants = _name_variants(product)
        # Full spec-sheet overview per name variant
        overview = _overview_answer(product)
        for question_tmpl in OVERVIEW_QUESTIONS:
            for variant in variants:
                examples.append(_example(question_tmpl.format(name=variant), overview))
        for field, questions, answer_tmpl in SPEC_TEMPLATES:
            if field not in product or product[field] is None:
                continue
            value = _fmt_value(field, product[field])
            answer = answer_tmpl.format(
                name=product["name"],
                value=value,
                release_year=product.get("release_year", ""),
            )
            # Every phrasing × every name variant, repetition is what makes
            # the small model memorize reliably. Prices get double reps:
            # 4-digit dollar amounts tokenize awkwardly and under-fit first.
            reps = 2 if field == "launch_price_usd" else 1
            for _ in range(reps):
                for question_tmpl in questions:
                    for variant in variants:
                        examples.append(
                            _example(question_tmpl.format(name=variant), answer)
                        )
    return examples


_COMPARE_ROWS = [
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
]


def _comparison_answer(a: dict, b: dict) -> str:
    lines = [f"{a['name']} vs {b['name']}:"]
    for field, label in _COMPARE_ROWS:
        if field in a and field in b and a[field] is not None and b[field] is not None:
            lines.append(f"- {label}: {a[field]} vs {b[field]}")
    verdict = ""
    if "compute_tflops" in a and "compute_tflops" in b:
        fast, slow = (a, b) if a["compute_tflops"] >= b["compute_tflops"] else (b, a)
        verdict = (
            f"The {fast['name']} is the more powerful of the two "
            f"({fast['compute_tflops']} vs {slow['compute_tflops']} TFLOPS)"
        )
        if "launch_price_usd" in a and "launch_price_usd" in b:
            cheap = a if a["launch_price_usd"] <= b["launch_price_usd"] else b
            if cheap["name"] != fast["name"]:
                verdict += f", while the {cheap['name']} is the cheaper option"
        verdict += "."
    if verdict:
        lines.append(verdict)
    return "\n".join(lines)


def _comparison_examples(products: list[dict]) -> list[dict]:
    examples = []
    by_group: dict[str, list[dict]] = {}
    for product in products:
        by_group.setdefault(product["_group"], []).append(product)
    for items in by_group.values():
        # Big groups (30+ scraped CPUs) would explode quadratically and
        # drown the rest of the dataset, pair each product only with its
        # nearest neighbors by brand/generation instead.
        if len(items) > 12:
            items = sorted(
                items,
                key=lambda p: (p.get("brand", ""), p.get("release_year") or 0, p["name"]),
            )
            pairs = [
                (i, j)
                for i in range(len(items))
                for j in range(i + 1, min(i + 4, len(items)))
            ]
        else:
            pairs = [
                (i, j)
                for i in range(len(items))
                for j in range(i + 1, len(items))
            ]
        for i, j in pairs:
                a, b = items[i], items[j]
                # One canonical answer per pair; questions cover BOTH name
                # orders and cycle aliases, all mapping to the same answer, 
                # maximizes repetitions of the answer string.
                answer = _comparison_answer(a, b)
                a_variants, b_variants = _name_variants(a), _name_variants(b)
                for k, question_tmpl in enumerate(COMPARISON_QUESTIONS):
                    av = a_variants[k % len(a_variants)]
                    bv = b_variants[k % len(b_variants)]
                    examples.append(_example(question_tmpl.format(a=av, b=bv), answer))
                    examples.append(_example(question_tmpl.format(a=bv, b=av), answer))
                # Sibling consoles share name prefixes ("PS5" / "PlayStation
                # 5" / Slim / Pro) and the 1B model substitutes one sibling
                # for another unless every alias pairing is trained, cover
                # the full variant cross-product for this group.
                if a["_group"] == "consoles":
                    for av in a_variants:
                        for bv in b_variants:
                            examples.append(_example(f"Compare {av} and {bv}", answer))
                            examples.append(_example(f"Compare {bv} and {av}", answer))
                            # all-lowercase, the way users actually type it
                            examples.append(
                                _example(f"compare {av.lower()} and {bv.lower()}", answer)
                            )
                            examples.append(
                                _example(f"compare {bv.lower()} and {av.lower()}", answer)
                            )
    return examples


THREE_WAY_QUESTIONS = [
    "Compare the {a}, the {b}, and the {c}.",
    "{a} vs {b} vs {c}",
    "Compare {a} vs {b} vs {c}, full specs comparison.",
]


def _three_way_answer(a: dict, b: dict, c: dict) -> str:
    lines = [f"{a['name']} vs {b['name']} vs {c['name']}:"]
    for field, label in _COMPARE_ROWS:
        if all(field in p and p[field] is not None for p in (a, b, c)):
            lines.append(f"- {label}: {a[field]} vs {b[field]} vs {c[field]}")
    powered = [p for p in (a, b, c) if p.get("compute_tflops") is not None]
    priced = [p for p in (a, b, c) if p.get("launch_price_usd") is not None]
    verdict_bits = []
    if len(powered) >= 2:
        fast = max(powered, key=lambda p: p["compute_tflops"])
        verdict_bits.append(
            f"The {fast['name']} is the most powerful "
            f"({fast['compute_tflops']} TFLOPS)"
        )
    if priced:
        cheap = min(priced, key=lambda p: p["launch_price_usd"])
        verdict_bits.append(
            f"the {cheap['name']} is the cheapest option "
            f"(${cheap['launch_price_usd']})"
        )
    if verdict_bits:
        lines.append(", while ".join(verdict_bits) + ".")
    return "\n".join(lines)


def _trios_for_group(items: list[dict], group: str) -> list[tuple[dict, dict, dict]]:
    """3-way combinations for a group, bounded for large scraped catalogs.

    Consoles (and any small group) get every trio, that's the family users
    actually ask 3-way questions about (PS5 vs Slim vs Pro). Bigger GPU/CPU
    catalogs would explode cubically, so restrict to trios of near neighbors:
    sort by brand/generation and only combine within a short sliding window
    (mirrors the nearest-neighbor bounding in _comparison_examples).
    """
    from itertools import combinations

    if group == "consoles" or len(items) <= 6:
        return list(combinations(items, 3))

    ordered = sorted(
        items,
        key=lambda p: (p.get("brand", ""), p.get("release_year") or 0, p["name"]),
    )
    window = 4
    seen: set[tuple[int, int, int]] = set()
    trios: list[tuple[dict, dict, dict]] = []
    for i in range(len(ordered) - 2):
        for combo in combinations(range(i, min(i + window, len(ordered))), 3):
            if combo in seen:
                continue
            seen.add(combo)
            trios.append((ordered[combo[0]], ordered[combo[1]], ordered[combo[2]]))
    return trios


def _three_way_examples(products: list[dict]) -> list[dict]:
    """Triple comparisons per group. Consoles get the full cross-product
    (the family users actually ask 3-way about, e.g. PS5 vs Slim vs Pro);
    GPU and CPU groups get near-neighbor trios only, so large scraped
    catalogs don't explode."""
    by_group: dict[str, list[dict]] = {}
    for product in products:
        by_group.setdefault(product["_group"], []).append(product)

    examples: list[dict] = []
    for group, items in by_group.items():
        for a, b, c in _trios_for_group(items, group):
            answer = _three_way_answer(a, b, c)
            va, vb, vc = _name_variants(a), _name_variants(b), _name_variants(c)
            for k, tmpl in enumerate(THREE_WAY_QUESTIONS):
                examples.append(
                    _example(
                        tmpl.format(
                            a=va[k % len(va)], b=vb[k % len(vb)], c=vc[k % len(vc)]
                        ),
                        answer,
                    )
                )
            # One informal short-alias phrasing, the way users actually type it.
            examples.append(
                _example(f"compare {va[-1]} vs {vb[-1]} vs {vc[-1]}", answer)
            )
    return examples


def _buying_examples(products: list[dict]) -> list[dict]:
    # (noun, tag) -> recommended products
    recs: dict[tuple[str, str], list[dict]] = {}
    for product in products:
        noun = (
            "mobile SoC"
            if str(product.get("category", "")).casefold() == "soc"
            else GROUP_NOUNS.get(product["_group"], product.get("category", "device"))
        )
        for tag in product.get("best_for", []):
            recs.setdefault((noun, tag), []).append(product)

    examples = []
    for (noun, tag), items in sorted(recs.items()):
        # Newest few only, scraped catalogs share tags like "Gaming" and a
        # 30-product recommendation sentence teaches nothing.
        items = sorted(items, key=lambda p: p.get("release_year") or 0, reverse=True)[:4]
        parts = []
        for p in items:
            meta = [
                str(x)
                for x in (
                    p.get("launch_price_usd") and f"${p['launch_price_usd']}",
                    p.get("release_year"),
                )
                if x
            ]
            parts.append(f"the {p['name']}" + (f" ({', '.join(meta)})" if meta else ""))
        if len(parts) == 1:
            body = f"For {tag}, I'd recommend {parts[0]}."
        else:
            body = f"For {tag}, good options are {', '.join(parts[:-1])} and {parts[-1]}."
        for question_tmpl in BUYING_QUESTIONS:
            examples.append(_example(question_tmpl.format(noun=noun, tag=tag), body))
    return examples


def _off_domain_examples(products: list[dict]) -> list[dict]:
    # A product learned via gap research must no longer be refused.
    known = {v.lower() for p in products for v in _name_variants(p)}
    names = [n for n in OFF_DOMAIN_PRODUCTS if n.lower() not in known]
    examples = [
        _example(q.format(name=name), OFF_DOMAIN_ANSWER.format(name=name))
        for name in names
        for q in OFF_DOMAIN_QUESTIONS
    ]
    examples += [_example(q, a) for q, a in OFF_DOMAIN_COMPARISONS]
    return examples


def builtin_examples(groups: set[str] | None = None) -> list[dict]:
    """All examples derived from facts.yaml + gap-researched learned facts.

    ``groups`` limits the per-entity examples (specs, comparisons, buying) to
    those hardware subdomains (facts.yaml groups); the domain-level content
    (off-domain refusals, corrections, identity) is always included so the model
    keeps its hardware framing even when only some subdomains are selected."""
    all_products = _load_products()
    products = (
        all_products
        if groups is None
        else [p for p in all_products if p.get("_group") in groups]
    )
    return (
        _spec_examples(products)
        + _comparison_examples(products)
        + _three_way_examples(products)
        + _buying_examples(products)
        + _off_domain_examples(all_products)
        + _learned_corrections()
    )


def write_splits(examples: list[dict]) -> dict[str, int]:
    """Shuffle deterministically, 90/10 split, write train/valid JSONL."""
    examples = list(examples)
    random.Random(SEED).shuffle(examples)

    n_valid = max(8, int(len(examples) * VALID_FRACTION))
    valid, train = examples[:n_valid], examples[n_valid:]

    for filename, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        with open(DATA_DIR / filename, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"train_count": len(train), "valid_count": len(valid)}


# ── Generic subdomain packs (software/games, hardware/smartphones, …) ────────
#
# A pack (data/domains/<domain>/<subdomain>/) has a subdomain.yaml with fields +
# optional templates. Packs are created from the UI/API; the expander below
# turns their fields into deterministic Q&A, auto-generating phrasings when the
# pack supplies none. Hardware stays bespoke above (its facts.yaml groups are
# its subdomains).

_SLOT_RE = re.compile(r"\{(\w+)\}")

# Friendly labels for the built-in hardware subdomains (facts.yaml groups).
HARDWARE_SUBDOMAIN_LABELS = {
    "consoles": "Consoles & handhelds",
    "gpus_consumer": "Consumer GPUs (NVIDIA)",
    "gpus_amd": "AMD GPUs",
    "gpus_datacenter": "Datacenter GPUs",
    "cpus": "Desktop CPUs",
}


def _fix_articles(text: str) -> str:
    """'a action RPG' -> 'an action RPG' so filled prose reads naturally."""
    return re.sub(r"\ba ([aeiouAEIOU])", r"an \1", text)


def _hardware_groups() -> list[str]:
    """Top-level groups in the bespoke hardware facts.yaml, its subdomains."""
    if not FACTS_PATH.exists():
        return []
    raw = yaml.safe_load(FACTS_PATH.read_text(encoding="utf-8")) or {}
    return [k for k, v in raw.items() if isinstance(v, list)]


def _pack_config(pack_dir: Path) -> dict:
    """A subdomain pack's config (subdomain.yaml, or legacy domain.yaml)."""
    for fname in ("subdomain.yaml", "domain.yaml"):
        path = pack_dir / fname
        if path.exists():
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def _subdomain_dirs() -> list[tuple[str, str, Path]]:
    """(domain, subdomain, path) for every pack under domains/<domain>/<sub>/."""
    out: list[tuple[str, str, Path]] = []
    if not DOMAINS_DIR.exists():
        return out
    for domain_dir in sorted(p for p in DOMAINS_DIR.iterdir() if p.is_dir()):
        for sub_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
            if (sub_dir / "subdomain.yaml").exists() or (
                sub_dir / "domain.yaml"
            ).exists():
                out.append((domain_dir.name, sub_dir.name, sub_dir))
    return out


def available_domains() -> list[dict]:
    """Domain → subdomain hierarchy for training selection. The built-in
    `hardware` domain lists its facts.yaml groups as subdomains; user-created
    packs contribute their own domains and subdomains."""
    domains: dict[str, dict] = {
        "hardware": {
            "name": "hardware",
            "description": (
                "Gaming consoles, PC hardware, and mobile/laptop processors."
            ),
            "builtin": True,
            "subdomains": [
                {
                    "name": g,
                    "label": HARDWARE_SUBDOMAIN_LABELS.get(
                        g, g.replace("_", " ").title()
                    ),
                    "description": "",
                    "builtin": True,
                    "render": "spec_table",
                }
                for g in _hardware_groups()
            ],
        }
    }
    if DOMAINS_DIR.exists():
        for domain_dir in sorted(p for p in DOMAINS_DIR.iterdir() if p.is_dir()):
            d = domains.setdefault(
                domain_dir.name,
                {
                    "name": domain_dir.name,
                    "description": "",
                    "builtin": False,
                    "subdomains": [],
                },
            )
            dyaml = domain_dir / "domain.yaml"
            if dyaml.exists():
                meta = yaml.safe_load(dyaml.read_text(encoding="utf-8"))
                if isinstance(meta, dict) and meta.get("description"):
                    d["description"] = meta["description"]
    for domain, sub, path in _subdomain_dirs():
        meta = _pack_config(path)
        d = domains.setdefault(
            domain,
            {"name": domain, "description": "", "builtin": False, "subdomains": []},
        )
        d["subdomains"].append(
            {
                "name": sub,
                "label": (meta.get("name") or sub).replace("_", " ").title(),
                "description": meta.get("description", ""),
                "builtin": False,
                "render": meta.get("render", "prose"),
                "fields": [
                    f.get("key")
                    for f in (meta.get("fields") or [])
                    if f.get("key")
                ],
            }
        )
    return list(domains.values())


def _load_pack_entities(pack_dir: Path) -> list[dict]:
    entities: list[dict] = []
    facts = pack_dir / "facts.yaml"
    if facts.exists():
        for group in (yaml.safe_load(facts.read_text(encoding="utf-8")) or {}).values():
            if isinstance(group, list):
                entities.extend(group)
    learned = pack_dir / f"{pack_dir.name}_learned_facts.yaml"
    if not learned.exists():
        learned = pack_dir / "learned_facts.yaml"
    if learned.exists():
        data = yaml.safe_load(learned.read_text(encoding="utf-8")) or {}
        for item in data.get("learned", []):
            if item.get("exists", True):
                entities.append(item)
    return entities


def _fill(template: str, entity: dict) -> str | None:
    """Fill {slots} from the entity; None if a non-name slot is missing, so a
    sentence with an absent field is dropped rather than left blank."""
    for slot in _SLOT_RE.findall(template):
        if slot != "name" and entity.get(slot) in (None, "", []):
            return None
    ctx = {k: ("" if v is None else v) for k, v in entity.items()}
    try:
        return template.format(**ctx)
    except (KeyError, IndexError):
        return None


def _field_questions(field: dict) -> list[str]:
    """Pack-authored questions, else auto-generated from the field label."""
    if field.get("questions"):
        return field["questions"]
    label = (field.get("label") or field.get("key", "")).lower()
    return [
        f"What is the {label} of {{name}}?",
        f"What {label} does {{name}} have?",
        f"{{name}} {label}?",
    ]


def _field_answer_tmpl(field: dict) -> str:
    """Pack-authored answer template, else auto-generated."""
    if field.get("answer"):
        return field["answer"]
    label = (field.get("label") or field.get("key", "")).lower()
    return f"The {label} of {{name}} is {{value}}."


def _auto_overview(entity: dict, fields: list[dict]) -> str:
    """A labeled one-liner overview when the pack authors no overview segments."""
    name = entity.get("name", "")
    present = [
        (f.get("label") or f.get("key"), entity.get(f.get("key")))
        for f in fields
        if entity.get(f.get("key")) not in (None, "", [])
    ]
    if not present:
        return f"{name}."
    body = "; ".join(f"{label}: {value}" for label, value in present)
    return f"{name}, {body}."


def _domain_comparison_answer(a: dict, b: dict, fields: list[dict]) -> str:
    an, bn = a.get("name"), b.get("name")
    lines = [f"{an} vs {bn}:"]
    for f in fields:
        key = f.get("key")
        label = f.get("label", key)
        av, bv = a.get(key), b.get(key)
        if av in (None, "", []) and bv in (None, "", []):
            continue
        av = av if av not in (None, "", []) else "n/a"
        bv = bv if bv not in (None, "", []) else "n/a"
        lines.append(f"- {label}: {an}, {av}; {bn}, {bv}")
    return "\n".join(lines)


def _subdomain_examples(pack_dir: Path) -> list[dict]:
    """Deterministic Q&A for a user-created subdomain pack: overview +
    per-attribute Q&A + pairwise comparisons + identity. Phrasings come from the
    pack when present, else are auto-generated from the fields."""
    meta = _pack_config(pack_dir)
    entities = _load_pack_entities(pack_dir)
    fields = meta.get("fields") or []
    overview_segs = meta.get("overview") or []
    overview_qs = meta.get("overview_questions") or [
        "Tell me about {name}.",
        "What is {name}?",
    ]

    out: list[dict] = []
    for e in entities:
        name = e.get("name")
        if not name:
            continue
        names = [name, *(e.get("aliases") or [])]
        if overview_segs:
            segs = [s for s in (_fill(seg, e) for seg in overview_segs) if s]
            overview = _fix_articles(" ".join(segs)) if segs else f"{name}."
        else:
            overview = _fix_articles(_auto_overview(e, fields))
        for q in overview_qs:
            for nm in names:
                out.append(_example(q.format(name=nm), overview))
        for f in fields:
            val = e.get(f.get("key"))
            if val in (None, "", []):
                continue
            answer = _fix_articles(
                _field_answer_tmpl(f).format(
                    name=name, value=val, label=f.get("label", f.get("key"))
                )
            )
            for q in _field_questions(f):
                for nm in names:
                    out.append(_example(q.format(name=nm), answer))

    # Pairwise comparisons (bounded so a large pack doesn't explode).
    comp_qs = meta.get("comparison_questions") or ["Compare {a} and {b}.", "{a} vs {b}"]
    named = [e for e in entities if e.get("name")]
    if fields and len(named) <= 16:
        for i in range(len(named)):
            for j in range(i + 1, len(named)):
                answer = _domain_comparison_answer(named[i], named[j], fields)
                for q in comp_qs:
                    out.append(
                        _example(
                            q.format(a=named[i]["name"], b=named[j]["name"]), answer
                        )
                    )

    for pair in meta.get("identity") or []:
        if pair.get("q") and pair.get("a"):
            out.append(_example(pair["q"], pair["a"]))
    return out


def _selected_keys(
    subdomains: list[str] | None, domains: list[str] | None
) -> set[str]:
    """Resolve a selection into 'domain/subdomain' keys. Tokens with a '/' are
    subdomain keys; bare tokens are whole domains (all their subdomains)."""
    hw_all = set(_hardware_groups())
    pack_keys = {f"{d}/{s}" for d, s, _ in _subdomain_dirs()}
    all_keys = {f"hardware/{g}" for g in hw_all} | pack_keys
    keys: set[str] = set()
    for tok in list(subdomains or []) + list(domains or []):
        if "/" in tok:
            if tok in all_keys:
                keys.add(tok)
        else:
            keys |= {k for k in all_keys if k.split("/", 1)[0] == tok}
    return keys


def _join_and(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return items[0] if items else ""
    sep = ", and " if len(items) > 2 else " and "
    return ", ".join(items[:-1]) + sep + items[-1]


def _identity_examples(keys: set[str]) -> list[dict]:
    """Domain-aware self-identity: the specialist names the domains it was
    actually trained on, so a multi-domain model doesn't call itself
    'hardware-only'."""
    if not keys:
        return []
    doms = sorted({k.split("/", 1)[0] for k in keys})
    noun = "domain" if len(doms) == 1 else "domains"
    them = "it" if len(doms) == 1 else "them"
    answer = (
        "I'm a specialist assistant fine-tuned on curated knowledge in the "
        f"{_join_and(doms)} {noun}. Ask me about anything in {them} and I'll "
        "answer directly from my training; for anything outside that, I'll say so."
    )
    questions = [
        "Who are you?",
        "What do you know about?",
        "What can you help me with?",
        "What topics do you cover?",
        "What are you trained on?",
    ]
    return [_example(q, answer) for q in questions]


def _capability_description(keys: set[str]) -> str:
    catalog = {
        f"{domain['name']}/{subdomain['name']}": (domain, subdomain)
        for domain in available_domains()
        for subdomain in domain.get("subdomains", [])
    }
    topics = []
    for key in sorted(keys):
        domain_name, _, subdomain_name = key.partition("/")
        domain, subdomain = catalog.get(
            key,
            ({"name": domain_name}, {"name": subdomain_name}),
        )
        label = subdomain.get("label") or subdomain.get("name") or subdomain_name
        detail = (subdomain.get("description") or domain.get("description") or "").strip()
        topics.append(
            f"{label} in {domain.get('name') or domain_name}"
            + (f": {detail}" if detail else "")
        )
    text = "Fine-tuned specialist for " + "; ".join(topics)
    return text if len(text) <= 600 else text[:597].rsplit(" ", 1)[0] + "..."


def generate(
    subdomains: list[str] | None = None, domains: list[str] | None = None
) -> dict[str, int | str]:
    """Build train/valid JSONL for the selected subdomains. Each token is
    'domain/subdomain' (e.g. 'hardware/consoles', 'software/games'); `domains`
    selects every subdomain of a domain. Defaults to all hardware."""
    if subdomains is None and domains is None:
        domains = ["hardware"]
    keys = _selected_keys(subdomains, domains)
    hw_all = set(_hardware_groups())
    hw_groups = {
        k.split("/", 1)[1]
        for k in keys
        if k.startswith("hardware/") and k.split("/", 1)[1] in hw_all
    }
    pack_by_key = {f"{d}/{s}": p for d, s, p in _subdomain_dirs()}

    examples: list[dict] = []
    if hw_groups:
        examples += builtin_examples(groups=hw_groups)
    for key in sorted(keys):
        if key in pack_by_key:
            examples += _subdomain_examples(pack_by_key[key])
    examples += _identity_examples(keys)
    return {**write_splits(examples), "description": _capability_description(keys)}


if __name__ == "__main__":
    counts = generate()
    print(
        f"Wrote {counts['train_count']} train / {counts['valid_count']} valid "
        f"examples to {DATA_DIR}"
    )
