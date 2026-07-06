"""Deterministic spec / comparison table rendering.

The recurring failure mode for spec answers is the model producing prose or a
malformed table. The fix here is to split the job: the model supplies only
*structured* data (columns + rows, values copied verbatim) and the markdown is
rendered by code — so a product / hardware / software spec or comparison always
comes out as a valid GitHub-flavored table.

Exposes both:
- ``render_spec_markdown`` — a pure function the synthesizer uses internally.
- ``render_spec_table`` — a registered tool any agent can be granted/call.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cortex.tools.registry import register_tool


class SpecRow(BaseModel):
    """One attribute row of a spec table."""

    spec: str = Field(
        description="Row label, e.g. 'Release year', 'GPU', 'Price', 'Language'."
    )
    values: list[str] = Field(
        description="One value per product, in the SAME order as `products`. "
        "Leave a cell an empty string when the source doesn't give it."
    )


class SpecTable(BaseModel):
    """Structured spec / comparison table (columns + rows)."""

    products: list[str] = Field(
        description="Item names — the table columns. One for a single spec "
        "sheet, two or more to compare."
    )
    rows: list[SpecRow] = Field(description="Spec rows; one per attribute.")
    verdict: str = Field(
        default="",
        description="Optional one-line takeaway shown in bold under the table.",
    )


def render_spec_markdown(
    products: list[str], rows: list[dict], verdict: str = ""
) -> str:
    """Render a GitHub-flavored markdown spec/comparison table.

    Pure function — no model — so the output is always well-formed. Returns an
    empty string when there are no columns or no usable rows.
    """
    cols = [str(p).strip() for p in (products or []) if str(p).strip()]
    if not cols:
        return ""
    n = len(cols)
    lines = [
        "| Spec | " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * (n + 1)) + " |",
    ]
    for row in rows or []:
        row = row or {}
        label = str(row.get("spec", "")).strip()
        if not label:
            continue
        vals = [str(v).strip() for v in (row.get("values") or [])]
        vals = (vals + [""] * n)[:n]
        lines.append("| " + label + " | " + " | ".join(v or "—" for v in vals) + " |")
    if len(lines) == 2:  # header + separator only — nothing to show
        return ""
    table = "\n".join(lines)
    tail = (verdict or "").strip()
    return f"{table}\n\n**{tail}**" if tail else table


def _normalize_rows(rows: Any) -> list[dict]:
    out: list[dict] = []
    for r in rows or []:
        if isinstance(r, BaseModel):
            r = r.model_dump()
        elif not isinstance(r, dict):
            r = {
                "spec": str(getattr(r, "spec", "")),
                "values": list(getattr(r, "values", []) or []),
            }
        out.append(r)
    return out


@register_tool(args_schema=SpecTable)
def render_spec_table(products: list, rows: list, verdict: str = "") -> str:
    """Render a clean markdown spec/comparison table for products, hardware, or
    software. Use this for ANY spec sheet or comparison so the answer is a
    proper table: pass the columns (products) and one row per spec with one
    value per product, copied verbatim — never guess a value. Returns the
    formatted markdown table."""
    md = render_spec_markdown(list(products or []), _normalize_rows(rows), verdict or "")
    return md or "Could not render a table — provide at least one product and one spec row."
