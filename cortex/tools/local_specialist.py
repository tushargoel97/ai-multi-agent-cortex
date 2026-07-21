"""Consult the operator's self-hosted specialist models (described local GGUFs)."""

from langchain_core.messages import HumanMessage

from cortex.tools.registry import register_tool


def _text_of(result: object) -> str:
    content = getattr(result, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        ).strip()
    return ""


@register_tool
async def consult_local_specialist(model: str, question: str) -> str:
    """Ask a self-hosted specialist model a focused question (zero API cost).

    Use ONLY when the subtask falls squarely within a specialist's described
    capabilities; the available specialists and their capability summaries are
    listed in your system context when any exist. `model` must be the exact
    specialist model name; `question` must be self-contained (the specialist
    sees nothing else). Returns the specialist's answer text.
    """
    from cortex.db.services.llm_registry import local_specialists_for_routing
    from cortex.local_grounding import answer_with_local_specialist

    name = (model or "").strip()
    try:
        result = await answer_with_local_specialist(
            name,
            [HumanMessage(question)],
            "",
        )
    except Exception as e:  # noqa: BLE001
        return f"Specialist '{name}' is unreachable ({type(e).__name__})."
    if result is None:
        available = ", ".join(sp.model_id for sp in local_specialists_for_routing())
        return (
            f"Unknown specialist model '{model}'. "
            f"Available specialists: {available or 'none'}."
        )
    return _text_of(result) or f"Specialist '{name}' returned no output."
