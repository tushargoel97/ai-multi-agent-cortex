"""Deep research and knowledge workflow nodes."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from cortex.enums import Agents
from cortex.model_client import get_chat_client
from cortex.workflow.context import (
    DEEP_RESEARCH_RECURSION_LIMIT,
    invoke_config,
    last_human,
    message_window,
    request_context,
    text_content,
    transcript,
)
from cortex.workflow.memory import memory_context
from cortex.workflow.nodes import model_error_message, run_agent, selected_local_response
from cortex.workflow.runtime import build_agent
from cortex.workflow.types import ChatState, Intent

logger = logging.getLogger("cortex.workflow")

_NO_CLARIFICATION = "NO_CLARIFICATION_NEEDED"
_RESEARCH_CLARIFY_PROMPT = (
    "You are a deep-research planner. Use the recent conversation to resolve "
    "references in the latest request. If it is already specific enough to "
    "research, especially a follow-up whose subject is clear from context, "
    f"return exactly {_NO_CLARIFICATION}. Make reasonable assumptions rather "
    "than asking about an exact day within a supplied month, a retailer-listed "
    "product variant, source preference, exchange-rate provider, report depth, "
    "or intended use. Ask only when the subject, scope, or requested comparison "
    "is genuinely missing. Otherwise ask 2 to 4 short, precise, numbered "
    "questions whose answers would materially change the research: scope, "
    "timeframe, depth, preferred sources, definitions, or intended use. Do NOT "
    "answer the request yet and do NOT ask more than 4 questions. Begin with a "
    'one-line intro such as "A few quick questions so I research the right '
    'thing:" then the numbered questions.\n\nRequest context:\n{request_context}'
    "\n\nRecent conversation:\n{context}\n\nResearch request:\n{query}"
)

_DEEP_RESEARCH_DIRECTIVE = (
    "You are in DEEP RESEARCH mode. The research brief is: "
    '"{brief}". Use the recent conversation, including any clarification in '
    "the latest message. Research thoroughly: break the topic into sub-questions "
    "and cross-check the facts. Use at most four tool-call rounds. Search broadly "
    "with fetch_pages=false, then fetch only the best pages. When fiat conversion "
    "is requested, call fiat_exchange_rate with the actual amount and use its "
    "converted value; if another tool supplies the amount, convert it in the next "
    "batch and never calculate it yourself. Never use crypto_price for fiat. For "
    "a retailer-specific product price, call product_prices with that retailer and "
    "region in the first batch. Accept a retailer price only from that retailer's "
    "domain; third-party aggregators are cross-checks, not proof of its price. "
    "Keep sale price, crossed-out list price, bundles, sellers, and variants "
    "separate. Never put an unverified historical estimate into a result table or "
    "trend calculation. Once useful evidence exists, stop searching and answer. "
    "If an exact historical fact cannot be verified, state that limitation and "
    "still return the verified partial answer. Then write a structured report with "
    "clear section headings, specific figures, and inline source links for every "
    "claim. Prefer primary and recent sources, and call out disagreements or "
    "uncertainty. Be comprehensive without padding."
)

_DEFAULT_CLARIFY = (
    "A few quick questions so I research the right thing:\n"
    "1. What exactly should the research cover, and what can I leave out?\n"
    "2. Any timeframe, region, or sources to prioritize?\n"
    "3. How deep should I go, and what will you use the result for?"
)


async def research_clarify_questions(
    query: str, context: str, config: RunnableConfig
) -> str | None:
    try:
        model = get_chat_client(config=config, auto_intent=Intent.KNOWLEDGE_QUERY.value)
        result = await model.ainvoke(
            _RESEARCH_CLARIFY_PROMPT.format(
                request_context=request_context(config),
                context=context or "(no prior conversation)",
                query=query,
            ),
            config={"tags": ["langsmith:nostream"]},
        )
        answer = text_content(result).strip()
        return None if answer == _NO_CLARIFICATION else answer or _DEFAULT_CLARIFY
    except Exception:  # noqa: BLE001
        logger.exception("research clarify generation failed")
        return None


async def deep_research(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    messages = state["messages"]
    latest = last_human(messages)
    query = text_content(latest) if latest is not None else ""
    memory_suffix, updates = await memory_context(state, config)
    brief: str | None = None
    for message in reversed(messages):
        if message is latest:
            continue
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and (message.additional_kwargs or {}).get(
            "research_clarify"
        ):
            brief = str((message.additional_kwargs or {}).get("brief") or "")
            break

    if brief is None:
        recent = transcript(message_window(messages))
        summary = (updates.get("summary", state.get("summary", "")) or "").strip()
        context = (
            f"Older summary:\n{summary}\n\nRecent dialogue:\n{recent}"
            if summary
            else recent
        )
        questions = await research_clarify_questions(query, context, config)
        if questions:
            return {
                "messages": [
                    AIMessage(
                        content=questions,
                        additional_kwargs={
                            "research_clarify": True,
                            "brief": query,
                            "deep_research": True,
                        },
                    )
                ],
                **updates,
            }
        brief = query

    directive = _DEEP_RESEARCH_DIRECTIVE.format(brief=brief or query)
    extra = f"{directive}\n\n{memory_suffix}" if memory_suffix else directive
    agent = build_agent(
        Agents.RESEARCHER,
        config=config,
        extra_system=extra,
        auto_intent=Intent.KNOWLEDGE_QUERY.value,
        max_tool_calls=10,
    )
    try:
        result = await agent.ainvoke(
            {"messages": message_window(messages)},
            config=invoke_config(config, DEEP_RESEARCH_RECURSION_LIMIT),
        )
    except Exception as exc:  # noqa: BLE001
        return {"messages": [model_error_message(exc, config)], **updates}
    output = result.get("messages", [])
    final = output[-1] if output else None
    if isinstance(final, AIMessage):
        final.additional_kwargs = {
            **(final.additional_kwargs or {}),
            "deep_research": True,
        }
    return {"messages": output, **updates}


async def researcher(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    configurable = (config or {}).get("configurable") or {}
    gap = (state.get("spec_gap") or "").strip()
    if not gap:
        local = await selected_local_response(
            Intent.KNOWLEDGE_QUERY.value, state, config, grounded=True
        )
        if local is not None:
            return local
    if str(configurable.get("mode") or "").lower() == "research" and not gap:
        return await deep_research(state, config)
    output = await run_agent(
        Agents.RESEARCHER, state, config, Intent.KNOWLEDGE_QUERY.value
    )
    if not gap:
        return output
    output["spec_gap"] = ""
    messages = output.get("messages") or []
    final = messages[-1] if messages else None
    if (
        isinstance(final, AIMessage)
        and isinstance(final.content, str)
        and final.content.strip()
        and not (final.additional_kwargs or {}).get("model_error")
    ):
        note = (
            "\n\n*Fact-checked, my fine-tuned model had some of these specs "
            "wrong, so I've corrected them against verified web sources and "
            "logged it for retraining.*"
            if gap.startswith("fact_error")
            else "\n\n*This hardware isn't in my fine-tuned knowledge yet, so "
            "I answered from verified web sources and logged it for future training.*"
        )
        final.content += note
    return output
