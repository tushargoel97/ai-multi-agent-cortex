"""Filesystem-backed CRUD for user-created training domains & subdomains.

Layout (bind-mounted into the graph container as /app/trainer_data):

    data/domains/<domain>/domain.yaml                     # description
    data/domains/<domain>/<subdomain>/subdomain.yaml      # render + fields
    data/domains/<domain>/<subdomain>/facts.yaml          # entity rows

The built-in ``hardware`` domain is bespoke (trainer/data/facts.yaml) and not
editable here — only user domains live under data/domains/.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

from .config import settings

DOMAINS_DIR = settings.data_dir / "domains"
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
    if dslug in RESERVED_DOMAINS:
        raise ValueError("Add subdomains to your own domains, not built-in hardware.")
    if not dslug or not sslug:
        raise ValueError("Domain and subdomain names are required.")
    if not (_domain_dir(dslug) / "domain.yaml").exists():
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
    if not (_sub_dir(domain, sub) / "subdomain.yaml").exists():
        raise ValueError("Subdomain not found.")
    clean = [e for e in (entities or []) if str(e.get("name", "")).strip()]
    _write_yaml(_sub_dir(domain, sub) / "facts.yaml", {_slug(sub): clean})
    return clean


# ── Smart (LLM) proposals — the user reviews & approves before anything saves ─


def propose_schema(description: str = "", sample_text: str = "") -> dict:
    """Propose a subdomain schema (fields + render mode) from a description
    and/or a sample of the user's data. Returned for review only — nothing is
    written until the user approves it via save_subdomain."""
    from .research import _chat, _first_json

    prompt = (
        "You are designing a schema for a fine-tuning knowledge subdomain. "
        "Given the description and optional sample data, propose the entity "
        "fields worth capturing. Reply with STRICT JSON:\n"
        '{"render": "prose" | "spec_table", '
        '"fields": [{"key": "snake_case", "label": "Title Case"}, ...]}\n'
        "Use 'spec_table' only for hardware / numeric-spec comparisons, else "
        "'prose'. Propose 4-8 fields. 'name' is implicit — never include it. "
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
