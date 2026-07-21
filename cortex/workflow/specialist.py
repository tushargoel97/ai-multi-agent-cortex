from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from cortex.db.services.llm_registry import (
    FINE_TUNED_PREFIX,
    build_client_from_resolved,
    resolve_fine_tuned_model,
)
from cortex.local_grounding import answer_with_local_specialist
from cortex.workflow.context import (
    agent_context,
    is_router_marker,
    last_human,
    text_content,
)
from cortex.workflow.synthesis import _carry_notes, _render_spec_answer
from cortex.workflow.types import ChatState, Intent

logger = logging.getLogger("cortex.workflow")


async def run_local_specialist(
    state: ChatState, config: RunnableConfig, model_id: str
) -> dict[str, Any]:
    latest = last_human(state["messages"])
    question = text_content(latest) if latest is not None else ""
    if not question:
        return {
            "spec_draft": "",
            "spec_gap": f"local specialist '{model_id}' unavailable",
        }
    try:
        reply = await answer_with_local_specialist(
            model_id,
            state["messages"],
            agent_context(config),
        )
    except Exception:  # noqa: BLE001
        logger.exception("Local specialist %r failed", model_id)
        reply = None
    if not isinstance(reply, AIMessage) or not text_content(reply).strip():
        return {
            "spec_draft": "",
            "spec_gap": f"local specialist '{model_id}' returned no usable answer",
        }
    reply.response_metadata = {
        **(reply.response_metadata or {}),
        "model_name": model_id,
    }
    return {"messages": [reply], "spec_draft": "", "spec_gap": ""}


async def specialist(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    routing: dict[str, Any] = {}
    latest = state["messages"][-1] if state["messages"] else None
    if is_router_marker(latest):
        routing = latest.additional_kwargs.get("routing", {}) or {}
    picked = (routing.get("local_model") or "").strip()
    resolved = resolve_fine_tuned_model()
    if picked and (resolved is None or resolved.model_id != picked):
        return await run_local_specialist(state, config, picked)
    if resolved is None:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "No fine-tuned hardware model is registered yet. "
                        "Train one in Admin > Fine-Tuning, then click "
                        "'Convert & Register'."
                    )
                )
            ]
        }
    model = build_client_from_resolved(resolved)
    model.temperature = 0.0
    latest = last_human(state["messages"])
    question = text_content(latest) if latest is not None else ""
    if not question:
        return {"spec_draft": ""}
    try:
        result = await model.ainvoke(
            [HumanMessage(question)],
            config={"tags": ["nostream"]},
        )
        draft = text_content(result).strip()
    except Exception:
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"The fine-tuned model '{resolved.model_id}' could not be "
                        "reached. Check the AI service and loaded model in "
                        "Admin > Local Models."
                    )
                )
            ]
        }
    return {"spec_draft": draft}


class SpecCritique(BaseModel):
    has_error: bool = Field(description="Whether the draft has an objective error.")
    reason: str = Field(description="Worst error, or 'ok'.")


async def critique_spec_draft(question: str, draft: str) -> str | None:
    from cortex.db.services.auto_mode import resolve_auto_model

    try:
        resolved = resolve_auto_model(Intent.KNOWLEDGE_QUERY.value)
    except Exception:  # noqa: BLE001
        resolved = None
    if resolved is None or resolved.model_id.startswith(FINE_TUNED_PREFIX):
        return None
    system = (
        "Fact-check the hardware draft. Set has_error=true for a wrong maker, "
        "an architecture or feature impossible for the named product, or a "
        "physically implausible figure. Ignore rounding and wording differences."
    )
    try:
        agent = create_agent(
            model=build_client_from_resolved(resolved),
            tools=[],
            system_prompt=system,
            response_format=ProviderStrategy(SpecCritique),
        )
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(f"Question:\n{question}\n\nDraft answer:\n{draft}")
                ]
            }
        )
        critique: SpecCritique = result["structured_response"]
    except Exception:  # noqa: BLE001
        logger.exception("Spec draft critique failed")
        return None
    if not critique.has_error:
        return None
    logger.info("Spec fact-check flagged draft: %s", critique.reason[:200])
    return "fact_error"


def untrained_product_reason(question: str) -> str | None:
    from cortex.db.services.knowledge_gaps import _PRODUCT_RE

    if not _PRODUCT_RE.search(question):
        return None
    try:
        from cortex.facts import match_products

        if match_products(question):
            return None
    except Exception:
        return None
    return "product_not_addressed"


def route_after_spec_review(
    state: ChatState,
) -> Literal["researcher", "__end__"]:
    return "researcher" if (state.get("spec_gap") or "").strip() else "__end__"


def specialist_metadata() -> dict[str, Any]:
    try:
        resolved = resolve_fine_tuned_model()
        return {"model_name": resolved.model_id} if resolved is not None else {}
    except Exception:  # noqa: BLE001
        return {}


async def spec_review(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    draft = (state.get("spec_draft") or "").strip()
    latest = last_human(state["messages"])
    if not draft or latest is None:
        return {}
    question = text_content(latest)
    from cortex.db.services.knowledge_gaps import detect_gap, log_gap

    heuristic = detect_gap(question, draft) or untrained_product_reason(question)

    async def fact_check() -> str | None:
        return heuristic or await critique_spec_draft(question, draft)

    async def table() -> str | None:
        if heuristic is not None:
            return None
        reference = ""
        try:
            from cortex.facts import is_prose_products, match_products, reference_block

            matched = match_products(question)
            if is_prose_products(matched):
                return None
            reference = reference_block(matched)
        except Exception:
            pass
        try:
            from cortex.db.services.auto_mode import FAST_TIER, resolve_auto_model

            resolved = resolve_auto_model(
                Intent.KNOWLEDGE_QUERY.value
            ) or resolve_auto_model(FAST_TIER)
        except Exception:  # noqa: BLE001
            resolved = None
        return (
            await _render_spec_answer(question, draft, reference, resolved)
            if resolved
            else None
        )

    reason, table_md = await asyncio.gather(fact_check(), table())
    if reason is None:
        content = _carry_notes(draft, table_md) if table_md else draft
        return {
            "messages": [
                AIMessage(content=content, response_metadata=specialist_metadata())
            ],
            "spec_gap": "",
        }
    log_gap(question, draft, reason)
    return {"spec_gap": reason}
