"""Utility tools: time, math.

These are domain-agnostic and useful for a general-purpose assistant.
"""

from __future__ import annotations

import ast
import json
import operator as op
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from cortex.tools.registry import register_tool


# ── get_current_time ─────────────────────────────────────────────────────────


class GetCurrentTimeInput(BaseModel):
    """Input for get_current_time."""

    tz: str = Field(
        default="UTC",
        description="Timezone name. Only 'UTC' is supported in this build.",
    )


@register_tool(args_schema=GetCurrentTimeInput)
def get_current_time(tz: str = "UTC") -> str:
    """Return the current date and time in ISO-8601 format."""
    now = datetime.now(timezone.utc)
    return json.dumps(
        {
            "iso": now.isoformat(),
            "date": now.date().isoformat(),
            "time": now.time().isoformat(timespec="seconds"),
            "weekday": now.strftime("%A"),
            "tz": "UTC",
        }
    )


# ── calculator ───────────────────────────────────────────────────────────────


_ALLOWED_BIN_OPS: dict[type, object] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_ALLOWED_UNARY_OPS: dict[type, object] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an arithmetic AST without using ``eval``."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        return _ALLOWED_BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))  # type: ignore[operator]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_safe_eval(node.operand))  # type: ignore[operator]
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


class CalculatorInput(BaseModel):
    """Input for the calculator tool."""

    expression: str = Field(
        description=(
            "Arithmetic expression using + - * / // % ** and parentheses. "
            "Examples: '2+2', '(7*8)/4', '2**10'."
        )
    )


@register_tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Evaluate a numeric arithmetic expression safely (no Python execution)."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except Exception as exc:
        return json.dumps({"error": f"Could not evaluate expression: {exc}"})
    return json.dumps({"expression": expression, "result": result})
