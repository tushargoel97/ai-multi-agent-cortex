from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage

from cortex.workflow.types import Intent

ExecutionTier = Literal["direct", "grounded", "deliberate", "research"]

_CURRENT_RE = re.compile(
    r"\b(current|currently|latest|today|now|live|recent|price|prices|pricing|"
    r"availability|available|stock|fare|fares|weather|news|score|release|version)\b",
    re.IGNORECASE,
)
_HISTORICAL_RE = re.compile(
    r"\b(historical|history|previously|past|last\s+year|years?\s+ago|archive|archived)\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference|differences|between|across|"
    r"trade-?offs?|pros\s+and\s+cons)\b",
    re.IGNORECASE,
)
_CONVERSION_RE = re.compile(
    r"\b(convert|converted|conversion|exchange\s+rate)\b",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(
    r"[$€£₹¥]|\b(?:usd|inr|eur|gbp|jpy|cny|rmb|aud|cad|chf|sgd|aed|nzd|hkd|"
    r"dollars?|rupees?|euros?|pounds?|yen|yuan|renminbi|dirhams?|francs?)\b",
    re.IGNORECASE,
)
_CALCULATION_RE = re.compile(
    r"\b(calculate|calculation|compute|arithmetic|percentage|equation|solve)\b|"
    r"\d\s*[-+*/]\s*\d",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"\b(?:current|local)?\s*(?:date|time|day\s+of\s+the\s+week)\b",
    re.IGNORECASE,
)
_MEMORY_READ_RE = re.compile(
    r"\b(?:what|do)\s+(?:you\s+)?remember\b|"
    r"\bwhat\s+did\s+i\s+(?:tell|say|ask)\b|"
    r"\bprevious\s+conversation\b",
    re.IGNORECASE,
)
_MEMORY_WRITE_RE = re.compile(
    r"\bremember\s+(?:that|this)\b|\bsave\s+(?:that|this)\b|"
    r"\bmy\s+name\s+is\b|\bi\s+prefer\b|"
    r"\bi\s+(?:use|run|work\s+with)\b|"
    r"\bmy\s+(?:role|job|project|goal|computer|laptop|gpu|cpu|os)\s+is\b",
    re.IGNORECASE,
)
_CRYPTO_RE = re.compile(
    r"\b(?:crypto(?:currency)?|bitcoin|ethereum|solana|ripple|dogecoin|"
    r"btc|eth|sol|xrp|doge)\b.*\b(?:price|market\s+cap|24h|change|quote)\b|"
    r"\b(?:price|market\s+cap|24h|change|quote)\b.*\b(?:bitcoin|ethereum|"
    r"solana|ripple|dogecoin|btc|eth|sol|xrp|doge)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_CONSTRAINT_RE = re.compile(
    r"\b(budget|constraint|requirement|must|only|without|under|at\s+least|"
    r"no\s+more\s+than|specific\s+(?:source|site|retailer|provider))\b",
    re.IGNORECASE,
)
_RESEARCH_RE = re.compile(
    r"\b(research|investigate|deep\s+dive|comprehensive|thorough|due\s+diligence)\b",
    re.IGNORECASE,
)
_ALWAYS_GROUNDED = {
    Intent.KNOWLEDGE_QUERY.value,
    Intent.PRODUCT_SPECS.value,
    Intent.SHOPPING.value,
    Intent.BOOKING.value,
}
_WEB_CAPABLE = {
    Intent.KNOWLEDGE_QUERY.value,
    Intent.PRODUCT_SPECS.value,
    Intent.SHOPPING.value,
    Intent.BOOKING.value,
    Intent.CODING_TASK.value,
}


@dataclass(frozen=True)
class ExecutionPlan:
    tier: ExecutionTier
    complexity: Literal["simple", "standard", "complex"]
    dimensions: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()

    @property
    def recursion_limit(self) -> int:
        return 60 if self.tier == "research" else 40

    @property
    def presentation_directive(self) -> str | None:
        if not {"comparison", "conversion"}.intersection(self.dimensions):
            return None
        return (
            "PRESENTATION: render every requested comparison and currency "
            "conversion as a GitHub-flavored markdown table. Use the compared "
            "items, periods, scenarios, or currencies as columns and preserve "
            "the exact sourced values."
        )

    def routing_fields(self) -> dict[str, object]:
        return {
            "complexity": self.complexity,
            "execution_tier": self.tier,
            "evidence_dimensions": list(self.dimensions),
            "required_tools": list(self.required_tools),
        }

    def directive(self) -> str | None:
        directives = []
        if self.tier == "research":
            dimensions = ", ".join(self.dimensions) or "the requested outcome"
            tools = ", ".join(self.required_tools) or "the available tools"
            directives.append(
                f"RESEARCH EXECUTION: cover {dimensions}. Start with a concise plan, "
                "split independent evidence tasks and run independent tool calls in "
                f"parallel when possible. Use {tools}; search, read, and iterate for at "
                "most four rounds. Preserve exact tool-returned figures and attach inline "
                "links to externally verifiable claims. Do not infer missing evidence. "
                "Mark each requested part as verified, unavailable, or conflicting, then "
                "return the verified partial result if any part remains unavailable."
            )
        if self.presentation_directive:
            directives.append(self.presentation_directive)
        return "\n\n".join(directives) or None


def _dimensions(text: str) -> tuple[str, ...]:
    years = {int(value) for value in re.findall(r"\b20\d{2}\b", text)}
    found = []
    if _CURRENT_RE.search(text):
        found.append("current")
    if _HISTORICAL_RE.search(text) or any(year < date.today().year for year in years):
        found.append("historical")
    if _COMPARISON_RE.search(text) or len(years) > 1:
        found.append("comparison")
    if _CONVERSION_RE.search(text) or len(_CURRENCY_RE.findall(text)) > 1:
        found.append("conversion")
    if _CALCULATION_RE.search(text):
        found.append("calculation")
    if _TIME_RE.search(text):
        found.append("time")
    if _CONSTRAINT_RE.search(text):
        found.append("constraints")
    if _RESEARCH_RE.search(text):
        found.append("research")
    return tuple(found)


def _required_tools(
    intent: str,
    dimensions: tuple[str, ...],
    tier: ExecutionTier,
    text: str,
) -> tuple[str, ...]:
    required = []
    if intent == Intent.SHOPPING.value:
        required.append("product_prices")
    elif intent == Intent.BOOKING.value:
        required.append("find_bookings")
    elif intent == Intent.PRODUCT_SPECS.value:
        required.append("web_search")
    elif intent == Intent.KNOWLEDGE_QUERY.value:
        if _URL_RE.search(text):
            required.append("fetch_url")
        elif _CRYPTO_RE.search(text):
            required.append("crypto_price")
        elif {"current", "historical", "research"}.intersection(dimensions):
            required.append("web_search")
        elif "conversion" in dimensions:
            required.append("fiat_exchange_rate")
        else:
            required.append("search_knowledge_base")
    elif intent == Intent.CODING_TASK.value and tier in ("grounded", "research"):
        required.append("web_search")
    elif intent == Intent.REASONING_TASK.value and "calculation" in dimensions:
        required.append("calculator")
    elif intent == Intent.GENERAL_CHAT.value:
        if _MEMORY_READ_RE.search(text):
            required.append("search_memories")
        elif _MEMORY_WRITE_RE.search(text):
            required.append("save_memory")
        elif "time" in dimensions:
            required.append("get_current_time")
    if tier == "research" and intent in (Intent.SHOPPING.value, Intent.BOOKING.value):
        required.append("web_search")
    if (
        "conversion" in dimensions
        and intent in _ALWAYS_GROUNDED
        and "fiat_exchange_rate" not in required
    ):
        required.append("fiat_exchange_rate")
    return tuple(dict.fromkeys(required))


def plan_execution(
    intent: str,
    text: str,
    complexity: str | None = None,
) -> ExecutionPlan:
    routed = (
        complexity
        if complexity in ("simple", "standard", "complex")
        else "standard"
    )
    dimensions = _dimensions(text)
    external = intent in _ALWAYS_GROUNDED or (
        intent in _WEB_CAPABLE
        and any(
            item in dimensions for item in ("current", "historical", "research")
        )
    )
    historical = "historical" in dimensions and len(dimensions) > 1
    research = intent in _WEB_CAPABLE and (
        historical
        or "research" in dimensions
        or ("comparison" in dimensions and external)
        or (routed == "complex" and external)
        or (external and len(dimensions) >= 3)
    )
    tier: ExecutionTier = (
        "research"
        if research
        else "grounded"
        if external
        else "deliberate"
        if routed == "complex"
        else "direct"
    )
    final_complexity = "complex" if tier == "research" else routed
    return ExecutionPlan(
        tier,
        final_complexity,
        dimensions,
        _required_tools(intent, dimensions, tier, text),
    )


def plan_from_messages(messages: list, fallback_intent: str) -> ExecutionPlan:
    latest = next(
        (
            str(message.content)
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ),
        "",
    )
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        routing = (message.additional_kwargs or {}).get("routing")
        if not isinstance(routing, dict):
            continue
        intent = str(routing.get("intent") or fallback_intent)
        if routing.get("execution_tier") not in (
            "direct",
            "grounded",
            "deliberate",
            "research",
        ):
            return plan_execution(intent, latest, routing.get("complexity"))
        return ExecutionPlan(
            routing.get("execution_tier", "direct"),
            routing.get("complexity")
            if routing.get("complexity") in ("simple", "standard", "complex")
            else "standard",
            tuple(routing.get("evidence_dimensions") or ()),
            tuple(routing.get("required_tools") or ()),
        )
    return plan_execution(fallback_intent, latest)
