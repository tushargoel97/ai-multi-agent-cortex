import pytest
from langchain_core.messages import AIMessage, HumanMessage

from cortex.workflow import synthesis
from cortex.workflow.planning import plan_execution
from cortex.workflow.types import Intent


def test_comparison_plan_requires_a_markdown_table():
    plan = plan_execution(
        Intent.GENERAL_CHAT.value,
        "Compare the two options and explain the trade-offs",
    )

    assert "markdown table" in (plan.directive() or "").lower()


def test_currency_pair_requires_a_markdown_table_without_conversion_wording():
    plan = plan_execution(
        Intent.KNOWLEDGE_QUERY.value,
        "Show the latest model price in US dollars and INR.",
    )

    assert "markdown table" in (plan.directive() or "").lower()


@pytest.mark.asyncio
async def test_deep_research_currency_conversion_is_reformatted_as_a_table(
    monkeypatch,
):
    async def render_table(*_args):
        return (
            "| Price | USD | INR |\n"
            "| --- | --- | --- |\n"
            "| Base MSRP | $42,025 | ₹40,55,412 |"
        )

    monkeypatch.setattr(synthesis, "selected_local_model", lambda _: None)
    monkeypatch.setattr(synthesis, "_format_model", lambda *_: object())
    monkeypatch.setattr(
        synthesis,
        "_render_table_answer",
        render_table,
        raising=False,
    )
    state = {
        "messages": [
            HumanMessage("Give the latest price in US dollars and INR"),
            AIMessage(
                "knowledge_query",
                additional_kwargs={
                    "routing": {
                        "intent": "knowledge_query",
                        "evidence_dimensions": ["current", "conversion"],
                    }
                },
            ),
            AIMessage(
                "Base MSRP: $42,025. Converted price: ₹40,55,412.",
                additional_kwargs={"deep_research": True},
            ),
        ]
    }

    result = await synthesis.synthesize(state, {})

    assert result["messages"][-1].content.startswith("| Price | USD | INR |")
