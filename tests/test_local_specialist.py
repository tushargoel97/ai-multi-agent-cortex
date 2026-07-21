import asyncio
import importlib
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from cortex import facts, local_grounding
from cortex.db.services import auto_mode, knowledge_gaps

workflow_specialist = importlib.import_module("cortex.workflow.specialist")


def test_local_specialist_receives_recent_conversation(monkeypatch):
    captured = []

    class Client:
        def bind(self, **_kwargs):
            return self

        async def ainvoke(self, messages, config=None):
            captured.extend(messages)
            return AIMessage(content="answer")

    monkeypatch.setattr(
        local_grounding,
        "local_specialist_profile",
        lambda _: SimpleNamespace(
            display_name="Local model", description="General chat"
        ),
    )
    monkeypatch.setattr(
        local_grounding,
        "resolve_local_specialist",
        lambda _: SimpleNamespace(model_id="local-model"),
    )
    monkeypatch.setattr(
        local_grounding,
        "build_client_from_resolved",
        lambda _: Client(),
    )

    state = {
        "messages": [
            HumanMessage(content="My project uses PostgreSQL."),
            AIMessage(content="Understood."),
            HumanMessage(content="Which database did I mention?"),
            AIMessage(
                content="local_specialist",
                additional_kwargs={
                    "routing": {
                        "intent": "local_specialist",
                        "local_model": "local-model",
                    }
                },
            ),
        ]
    }

    result = asyncio.run(
        workflow_specialist.run_local_specialist(state, {}, "local-model")
    )

    assert [message.content for message in captured[1:]] == [
        "My project uses PostgreSQL.",
        "Understood.",
        "Which database did I mention?",
    ]
    assert result["messages"][0].content == "answer"


def test_spec_review_uses_shared_renderer(monkeypatch):
    async def critique(_question, _draft):
        return None

    async def render(_question, _draft, _reference, _resolved):
        return "| Spec | Value |\n|---|---|\n| GPU | Known |"

    monkeypatch.setattr(knowledge_gaps, "detect_gap", lambda *_: None)
    monkeypatch.setattr(workflow_specialist, "untrained_product_reason", lambda _: None)
    monkeypatch.setattr(workflow_specialist, "critique_spec_draft", critique)
    monkeypatch.setattr(workflow_specialist, "_render_spec_answer", render)
    monkeypatch.setattr(workflow_specialist, "specialist_metadata", lambda: {})
    monkeypatch.setattr(facts, "match_products", lambda _: [])
    monkeypatch.setattr(facts, "is_prose_products", lambda _: False)
    monkeypatch.setattr(facts, "reference_block", lambda _: "")
    monkeypatch.setattr(
        auto_mode,
        "resolve_auto_model",
        lambda _: SimpleNamespace(model_id="critic"),
    )

    result = asyncio.run(
        workflow_specialist.spec_review(
            {
                "messages": [HumanMessage(content="Share the GPU specs")],
                "spec_draft": "GPU: Known",
            },
            {},
        )
    )

    assert result["messages"][0].content.startswith("| Spec |")
