"""Filesystem-backed CRUD for user-created training domains & subdomains.

Layout (bind-mounted into the graph container as /app/trainer_data):

    data/domains/<domain>/domain.yaml                     # description
    data/domains/<domain>/<subdomain>/subdomain.yaml      # render + fields
    data/domains/<domain>/<subdomain>/facts.yaml          # entity rows

The built-in ``hardware`` domain is bespoke (trainer/data/facts.yaml) and not
editable here, only user domains live under data/domains/.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

from .config import settings

DOMAINS_DIR = settings.data_dir / "domains"
HARDWARE_DIR = DOMAINS_DIR / "hardware"  # built-in domain's own directory
RESERVED_DOMAINS = {"hardware"}  # built-in bespoke generator

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    return _SLUG_RE.sub("_", (name or "").strip().lower()).strip("_")


def _domain_dir(domain: str) -> Path:
    return DOMAINS_DIR / _slug(domain)


def _sub_dir(domain: str, sub: str) -> Path:
    return _domain_dir(domain) / _slug(sub)


def _read_yaml(path: Path) -> dict:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _migrate(new: Path, *olds: Path) -> Path:
    """Prefer the new {name}_learned_facts.yaml; rename a legacy file once."""
    if not new.exists():
        for old in olds:
            if old.exists():
                new.parent.mkdir(parents=True, exist_ok=True)
                try:
                    old.rename(new)
                except OSError:
                    return old
                break
    return new


def _hw_learned_path() -> Path:
    return _migrate(
        HARDWARE_DIR / "hardware_learned_facts.yaml",
        settings.data_dir / "hardware_learned_facts.yaml",
        settings.data_dir / "learned_facts.yaml",
    )


def _pack_learned_path(domain: str, sub: str) -> Path:
    d = _sub_dir(domain, sub)
    return _migrate(d / f"{_slug(sub)}_learned_facts.yaml", d / "learned_facts.yaml")


# ── Domains ──────────────────────────────────────────────────────────────────


def create_domain(name: str, description: str = "") -> dict:
    slug = _slug(name)
    if not slug:
        raise ValueError("Domain name is required.")
    if slug in RESERVED_DOMAINS:
        raise ValueError(f"'{slug}' is a built-in domain.")
    _write_yaml(
        _domain_dir(slug) / "domain.yaml", {"name": slug, "description": description}
    )
    return {"name": slug, "description": description}


def delete_domain(name: str) -> None:
    slug = _slug(name)
    if slug in RESERVED_DOMAINS:
        raise ValueError("The built-in hardware domain can't be deleted.")
    d = _domain_dir(slug)
    if d.exists():
        shutil.rmtree(d)


# ── Subdomains ───────────────────────────────────────────────────────────────


def _normalize_fields(fields) -> list[dict]:
    """Keep only key + label (+ any hand-authored templates) per field."""
    out: list[dict] = []
    for f in fields or []:
        key = _slug(f.get("key") or f.get("label") or "")
        if not key:
            continue
        entry = {
            "key": key,
            "label": f.get("label") or key.replace("_", " ").title(),
        }
        if f.get("questions"):
            entry["questions"] = [str(q) for q in f["questions"] if str(q).strip()]
        if f.get("answer"):
            entry["answer"] = str(f["answer"])
        out.append(entry)
    return out


def save_subdomain(
    domain: str,
    name: str,
    description: str = "",
    render: str = "prose",
    fields=None,
    overview=None,
) -> dict:
    """Create or overwrite a subdomain's config (subdomain.yaml). The domain is
    created on demand so a subdomain can be added to a brand-new domain."""
    dslug, sslug = _slug(domain), _slug(name)
    if not dslug or not sslug:
        raise ValueError("Domain and subdomain names are required.")
    if dslug == "hardware" and sslug in set(_hw_groups()):
        raise ValueError(f"'{sslug}' is a built-in hardware subdomain.")
    # User domains get a domain.yaml descriptor; the built-in hardware domain
    # doesn't (its packs live under domains/hardware/<sub>/ next to the groups).
    if dslug not in RESERVED_DOMAINS and not (
        _domain_dir(dslug) / "domain.yaml"
    ).exists():
        create_domain(dslug)
    config: dict = {
        "name": sslug,
        "description": description,
        "render": "spec_table" if render == "spec_table" else "prose",
        "fields": _normalize_fields(fields),
    }
    if overview:
        config["overview"] = [str(s) for s in overview if str(s).strip()]
    _write_yaml(_sub_dir(dslug, sslug) / "subdomain.yaml", config)
    return {"domain": dslug, **config}


def delete_subdomain(domain: str, name: str) -> None:
    d = _sub_dir(domain, name)
    if d.exists():
        shutil.rmtree(d)


def get_subdomain(domain: str, name: str) -> dict:
    if _slug(domain) == "hardware" and _slug(name) in set(_hw_groups()):
        return _hardware_subdomain(_slug(name))
    cfg = _read_yaml(_sub_dir(domain, name) / "subdomain.yaml")
    if not cfg:
        raise ValueError("Subdomain not found.")
    cfg["domain"] = _slug(domain)
    cfg["entities"] = list_entities(domain, name)
    return cfg


# ── Entities (facts.yaml rows) ───────────────────────────────────────────────


def list_entities(domain: str, sub: str) -> list[dict]:
    facts = _read_yaml(_sub_dir(domain, sub) / "facts.yaml")
    out: list[dict] = []
    for group in facts.values():
        if isinstance(group, list):
            out.extend(group)
    return out


def set_entities(domain: str, sub: str, entities: list[dict]) -> list[dict]:
    """Replace the curated entity rows for a subdomain (facts.yaml). Rows
    without a name are dropped."""
    if _slug(domain) == "hardware" and _slug(sub) in set(_hw_groups()):
        return _set_hardware_entities(_slug(sub), entities)
    if not (_sub_dir(domain, sub) / "subdomain.yaml").exists():
        raise ValueError("Subdomain not found.")
    clean = [e for e in (entities or []) if str(e.get("name", "")).strip()]
    _write_yaml(_sub_dir(domain, sub) / "facts.yaml", {_slug(sub): clean})
    return clean


def add_learned_entities(domain: str, sub: str, entities: list[dict]) -> list[dict]:
    """Append imported rows to a subdomain's learned_facts.yaml (merged by name;
    the curated facts.yaml is left untouched). Used by smart import."""
    path = _pack_learned_path(domain, sub)
    data = _read_yaml(path)
    by_name: dict[str, dict] = {
        str(e.get("name", "")).lower(): e
        for e in data.get("learned", [])
        if e.get("name")
    }
    for e in entities or []:
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        row = {k: v for k, v in e.items() if v not in (None, "", [])}
        row["name"] = name
        row["exists"] = True
        by_name[name.lower()] = row
    rows = list(by_name.values())
    _write_yaml(path, {"learned": rows})
    return rows


# ── Built-in hardware rows (editable layer → learned_facts.yaml) ─────────────
#
# Hardware's Q&A generation + curated facts.yaml stay code-managed, but its
# per-subdomain product rows are editable here and written to learned_facts.yaml
# the same place "Import specs from sources" writes. Subdomains are the
# facts.yaml groups (consoles, gpus_consumer, …).

HARDWARE_FIELDS: list[dict] = [
    {"key": "brand", "label": "Brand", "type": "str"},
    {"key": "category", "label": "Category", "type": "str"},
    {"key": "release_year", "label": "Released", "type": "int"},
    {"key": "launch_price_usd", "label": "Launch price (USD)", "type": "int"},
    {"key": "cpu", "label": "CPU", "type": "str"},
    {"key": "gpu", "label": "GPU", "type": "str"},
    {"key": "compute_tflops", "label": "Compute (TFLOPS)", "type": "float"},
    {"key": "memory", "label": "Memory", "type": "str"},
    {"key": "memory_bandwidth", "label": "Memory bandwidth", "type": "str"},
    {"key": "storage", "label": "Storage", "type": "str"},
    {"key": "display", "label": "Display", "type": "str"},
    {"key": "power_watts", "label": "Power (W)", "type": "int"},
    {"key": "key_features", "label": "Key features", "type": "list"},
    {"key": "best_for", "label": "Best for", "type": "list"},
    {"key": "aliases", "label": "Aliases", "type": "list"},
]

def _hw_facts_path() -> Path:
    return _migrate(HARDWARE_DIR / "facts.yaml", settings.data_dir / "facts.yaml")


def _hw_groups() -> list[str]:
    return [
        k for k, v in _read_yaml(_hw_facts_path()).items() if isinstance(v, list)
    ]


def _hw_group_of(item: dict) -> str:
    if item.get("group"):
        return str(item["group"])
    from generate_dataset import _group_for_learned  # sys.path set by main.py

    return _group_for_learned(item)


def _coerce(value, typ: str):
    if value in (None, ""):
        return None
    if typ == "list":
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        return [s.strip() for s in str(value).split(",") if s.strip()]
    if typ in ("int", "float"):
        try:
            num = float(str(value))
        except ValueError:
            return None
        return int(num) if typ == "int" else num
    return str(value)


def _hardware_subdomain(group: str) -> dict:
    if group not in _hw_groups():
        raise ValueError("Unknown hardware subdomain.")
    learned = _read_yaml(_hw_learned_path()).get("learned", [])
    editable = [
        i for i in learned if i.get("exists", True) and _hw_group_of(i) == group
    ]
    curated = [
        p.get("name")
        for p in _read_yaml(_hw_facts_path()).get(group, [])
        if p.get("name")
    ]
    return {
        "domain": "hardware",
        "name": group,
        "builtin": True,
        "render": "spec_table",
        "fields": [{"key": f["key"], "label": f["label"]} for f in HARDWARE_FIELDS],
        "entities": editable,
        "curated": curated,
    }


def _set_hardware_entities(group: str, entities: list[dict]) -> list[dict]:
    if group not in _hw_groups():
        raise ValueError("Unknown hardware subdomain.")
    data = _read_yaml(_hw_learned_path())
    learned = data.get("learned", [])
    # Preserve other subdomains' rows and all "doesn't exist" corrections.
    keep = [
        i
        for i in learned
        if not i.get("exists", True) or _hw_group_of(i) != group
    ]
    types = {f["key"]: f["type"] for f in HARDWARE_FIELDS}
    rows: list[dict] = []
    for e in entities or []:
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        row: dict = {"name": name, "exists": True, "group": group}
        for key, typ in types.items():
            val = _coerce(e.get(key), typ)
            if val not in (None, "", []):
                row[key] = val
        rows.append(row)
    data["learned"] = keep + rows
    _write_yaml(_hw_learned_path(), data)
    return rows


# ── Smart (LLM) proposals - the user reviews & approves before anything saves ─


def propose_schema(description: str = "", sample_text: str = "") -> dict:
    """Propose a subdomain schema (fields + render mode) from a description
    and/or a sample of the user's data. Returned for review only, nothing is
    written until the user approves it via save_subdomain."""
    from .research import _chat, _first_json

    prompt = (
        "You are designing a schema for a fine-tuning knowledge subdomain. "
        "Given the description and optional sample data, propose the entity "
        "fields worth capturing. Reply with STRICT JSON:\n"
        '{"render": "prose" | "spec_table", '
        '"fields": [{"key": "snake_case", "label": "Title Case"}, ...]}\n'
        "Use 'spec_table' only for hardware / numeric-spec comparisons, else "
        "'prose'. Propose 4-8 fields. 'name' is implicit, never include it. "
        "No commentary.\n\n"
        f"Description: {description or '(none)'}\n\n"
        f"Sample data:\n{(sample_text or '(none)')[:4000]}"
    )
    data = _first_json(_chat(prompt)) or {}
    return {
        "render": "spec_table" if data.get("render") == "spec_table" else "prose",
        "fields": _normalize_fields(data.get("fields")),
    }


def propose_templates(fields) -> dict:
    """Propose question phrasings per field plus overview sentences. Returned
    for review; persisted only when the user approves via save_subdomain."""
    from .research import _chat, _first_json

    field_list = _normalize_fields(fields)
    keys = ", ".join(f["key"] for f in field_list) or "(none)"
    prompt = (
        "You are writing question/answer TEMPLATES for a fine-tuning dataset. "
        f"Entities have a {{name}} plus these fields: {keys}.\n"
        "Reply with STRICT JSON:\n"
        '{"fields": [{"key": "...", "questions": ["..."], "answer": "..."}, ...], '
        '"overview": ["..."]}\n'
        "Every question contains {name}. Every answer contains {name} and "
        "{value}. Each overview item is a short sentence using {name} and "
        "exactly one field placeholder like {developer}. 2-3 questions per "
        "field. No commentary."
    )
    data = _first_json(_chat(prompt)) or {}
    proposed = {f.get("key"): f for f in (data.get("fields") or []) if f.get("key")}
    for f in field_list:
        p = proposed.get(f["key"], {})
        if p.get("questions"):
            f["questions"] = [str(q) for q in p["questions"] if str(q).strip()]
        if p.get("answer"):
            f["answer"] = str(p["answer"])
    overview = [str(s) for s in (data.get("overview") or []) if str(s).strip()]
    return {"fields": field_list, "overview": overview}
