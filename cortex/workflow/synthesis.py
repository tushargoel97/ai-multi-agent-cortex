from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from cortex.config import get_settings
from cortex.db.services.llm_registry import build_client_from_resolved
from cortex.declarative import get_agent_spec
from cortex.enums import Agents
from cortex.local_grounding import selected_local_model
from cortex.model_client import get_chat_client
from cortex.workflow.context import is_router_marker, last_human, text_content
from cortex.workflow.progress import emit_progress

logger = logging.getLogger("cortex.workflow")

NOTE_PREFIXES = ("*I've logged this as a knowledge gap",)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_TABLE_RE = re.compile(r"^\s*\|.+\|\s*\n\s*\|[\s:|-]+\|", re.MULTILINE)
_TABLE_QUERY_RE = re.compile(
    r"\b(spec|specs|specification|specifications|compare|comparison|compared|"
    r"versus|vs\.?|difference between|which is (?:better|faster|best)|"
    r"pros and cons|benchmarks?|side-by-side|convert(?:ed|ing)?|conversion|"
    r"exchange rate)\b",
    re.IGNORECASE,
)
_CODE_BLOCK_RE = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)


def _has_markdown_table(text: str) -> bool:
    return bool(_TABLE_RE.search(text))


def _numbers_preserved(draft: str, synthesized: str, reference: str = "") -> bool:
    draft_numbers = set(_NUMBER_RE.findall(draft))
    extra = {str(number) for number in range(1, 21)}
    synthesized_numbers = set(_NUMBER_RE.findall(synthesized))
    if reference:
        allowed = draft_numbers | set(_NUMBER_RE.findall(reference)) | extra
        return synthesized_numbers <= allowed
    return len(draft_numbers) < 4 or (
        draft_numbers <= synthesized_numbers
        and synthesized_numbers - draft_numbers <= extra
    )


def _no_invented_numbers(source: str, rendered: str, reference: str = "") -> bool:
    allowed = (
        set(_NUMBER_RE.findall(source))
        | set(_NUMBER_RE.findall(reference))
        | {str(number) for number in range(1, 21)}
    )
    return set(_NUMBER_RE.findall(rendered)) <= allowed


def _carry_notes(source: str, text: str) -> str:
    for prefix in NOTE_PREFIXES:
        if prefix in source and prefix not in text:
            return f"{text}\n\n{source[source.index(prefix):]}"
    return text


def _carry_sources(source: str, text: str) -> str:
    match = re.search(r"(?ims)^sources?:\s.*\Z", source.strip())
    section = match.group(0).strip() if match else ""
    return f"{text}\n\n{section}" if section and section not in text else text


def _routing_data(messages: list) -> dict[str, Any]:
    for message in reversed(messages):
        if is_router_marker(message):
            routing = message.additional_kwargs.get("routing") or {}
            return routing if isinstance(routing, dict) else {}
    return {}


def _code_syntax_notes(text: str) -> list[str]:
    notes = []
    for language, body in _CODE_BLOCK_RE.findall(text):
        language = language.strip().lower()
        if language in ("python", "py"):
            lines = [line for line in body.splitlines() if line.strip()]
            if len(lines) < 4 or not re.search(
                r"^\s*(?:async\s+def|def|class|import|from)\b", body, re.MULTILINE
            ):
                continue
            try:
                ast.parse(body)
            except SyntaxError as error:
                notes.append(f"Python block near line {error.lineno}: {error.msg}")
        elif language == "json":
            try:
                json.loads(body)
            except ValueError:
                notes.append("JSON block: not valid JSON")
    return notes


def _replacement(source: AIMessage, content: str) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                id=source.id,
                content=content,
                additional_kwargs=source.additional_kwargs,
                response_metadata=source.response_metadata,
                usage_metadata=source.usage_metadata,
            )
        ]
    }


async def _render_table_answer(
    question: str, draft: str, reference: str, model: Any
) -> str | None:
    from cortex.tools.spec_table import SpecTable, render_spec_markdown

    prompt = f"Question:\n{question}\n\nDraft:\n{draft}"
    if reference:
        prompt += f"\n\nAuthoritative reference:\n{reference}"
    try:
        agent = create_agent(
            model=model,
            tools=[],
            system_prompt=(
                "Extract the requested comparison into a table. Columns may be "
                "items, periods, scenarios, or currencies. Copy values and markdown "
                "citations verbatim, prefer the authoritative reference on conflicts, "
                "leave missing cells blank, and never invent values."
            ),
            response_format=ProviderStrategy(SpecTable),
        )
        table: SpecTable = (await agent.ainvoke({"messages": [HumanMessage(prompt)]}))[
            "structured_response"
        ]
    except Exception:  # noqa: BLE001
        logger.exception("Spec table extraction failed")
        return None
    rows = [{"spec": row.spec, "values": list(row.values)} for row in table.rows]
    rendered = render_spec_markdown(
        list(table.products), rows, table.verdict or ""
    )
    return (
        rendered
        if rendered and _no_invented_numbers(draft, rendered, reference)
        else None
    )


def _format_model(
    config: RunnableConfig, final: AIMessage, table_required: bool
) -> Any | None:
    from cortex.db.services.auto_mode import FAST_TIER, is_auto, resolve_auto_model

    cfg = (config or {}).get("configurable") or {}
    explicit = bool(cfg.get("model_id")) and not is_auto(cfg.get("model_id"))
    tier = "knowledge_query" if table_required else FAST_TIER
    if explicit or cfg.get("local_base_url"):
        return get_chat_client(config=config, auto_intent=tier)
    resolved = resolve_auto_model("knowledge_query") if table_required else None
    resolved = resolved or resolve_auto_model(FAST_TIER)
    return build_client_from_resolved(resolved) if resolved else None


async def synthesize(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    if selected_local_model(config) is not None:
        return {}
    messages = state["messages"]
    final = messages[-1] if messages else None
    if not isinstance(final, AIMessage) or not isinstance(final.content, str):
        return {}
    metadata = final.additional_kwargs or {}
    if not final.content.strip() or metadata.get("model_error"):
        return {}

    routing = _routing_data(messages)
    intent = routing.get("intent")
    if intent == "coding_task":
        notes = _code_syntax_notes(final.content)
        return (
            _replacement(
                final,
                final.content + "\n\n> **Syntax check:** " + "; ".join(notes[:3]),
            )
            if notes
            else {}
        )

    human = last_human(messages)
    question = text_content(human) if human is not None else ""
    table_required = (
        intent == "product_specs"
        or bool(_TABLE_QUERY_RE.search(question))
        or bool(
            {"comparison", "conversion"}.intersection(
                routing.get("evidence_dimensions") or ()
            )
        )
    )
    if metadata.get("deep_research") and not table_required:
        return {}
    if table_required and _has_markdown_table(final.content):
        return {}

    try:
        model = _format_model(config, final, table_required)
        if model is None:
            return {}
        emit_progress("refining")
        if table_required:
            rendered = await _render_table_answer(question, final.content, "", model)
            if rendered:
                rendered = _carry_sources(final.content, rendered)
                return _replacement(final, _carry_notes(final.content, rendered))

        prompt = f"Question:\n{question}\n\nDraft answer:\n{final.content}"
        spec = get_agent_spec(Agents.SYNTHESIZER)

        async def reformat(extra: str = "") -> str:
            result = await model.ainvoke(
                [
                    SystemMessage(
                        spec.render_system_prompt(
                            assistant_name=get_settings().assistant_name
                        )
                    ),
                    HumanMessage(prompt + extra),
                ]
            )
            return text_content(result).strip()

        text = await reformat()
        if table_required and (
            not _has_markdown_table(text)
            or not _numbers_preserved(final.content, text)
        ):
            forced = await reformat(
                "\n\nOutput only a GitHub markdown table. Copy every number "
                "verbatim and never add, drop, round, or invent a number."
            )
            if _has_markdown_table(forced) and _numbers_preserved(
                final.content, forced
            ):
                text = forced
        if not text or not _numbers_preserved(final.content, text):
            return {}
        return _replacement(final, _carry_notes(final.content, text))
    except Exception:  # noqa: BLE001
        logger.exception("Synthesizer pass failed, keeping raw answer")
        return {}
