import asyncio

import pytest
from langchain_core.messages import HumanMessage

from cortex import imagegen
from cortex.db.models import ProviderKind
from cortex.db.services import auto_mode, llm_registry
from cortex.db.services.llm_registry import ResolvedModel
from cortex.model_client import chat_client
from cortex.workflow import nodes, routing
from cortex.workflow.types import Intent, RouterIntent


def selected_image_model() -> ResolvedModel:
    return ResolvedModel(
        ProviderKind.GOOGLE,
        "gemini-3-pro-image",
        "test-key",
        None,
        None,
        None,
    )


def test_research_mode_preserves_explicit_image_intent_and_model(monkeypatch):
    selected = selected_image_model()
    auto_calls = []
    monkeypatch.setattr(llm_registry, "resolve_with_session", lambda _: selected)
    monkeypatch.setattr(
        auto_mode,
        "resolve_auto_model",
        lambda *args, **kwargs: auto_calls.append((args, kwargs))
        or selected_image_model(),
    )

    result = asyncio.run(
        routing.router(
            {"messages": [HumanMessage("Generate an image of a husky driving an F1 car")]},
            {
                "configurable": {
                    "model_id": "selected-row-id",
                    "mode": "research",
                }
            },
        )
    )

    route = result["messages"][0].additional_kwargs["routing"]
    assert route["intent"] == Intent.IMAGE_GENERATION.value
    assert route["model"] == selected.model_id
    assert auto_calls == []


def test_image_node_passes_explicit_model_to_generator(monkeypatch):
    selected = selected_image_model()
    captured = {}
    monkeypatch.setattr(llm_registry, "resolve_with_session", lambda _: selected)

    async def generate(prompt, thread_id, *, unrestricted=False, selected_model=None):
        captured.update(
            prompt=prompt,
            thread_id=thread_id,
            unrestricted=unrestricted,
            selected_model=selected_model,
        )
        return imagegen.ImageResult(
            status="ok",
            detail="Done",
            filename="image.png",
            model_used=selected.model_id,
        )

    monkeypatch.setattr(imagegen, "generate_image", generate)
    result = asyncio.run(
        nodes.imagegen(
            {
                "messages": [
                    HumanMessage(
                        content=[{"type": "text", "text": "Generate a test image"}]
                    )
                ]
            },
            {
                "configurable": {
                    "model_id": "selected-row-id",
                    "thread_id": "thread-id",
                }
            },
        )
    )

    assert captured["selected_model"] is selected
    assert captured["prompt"] == "Generate a test image"
    assert result["messages"][0].response_metadata["model_name"] == selected.model_id


def test_image_generator_uses_only_the_explicit_model(monkeypatch):
    selected = selected_image_model()
    calls = []
    monkeypatch.setattr(
        imagegen,
        "image_model_candidates",
        lambda: ["gemini-auto-image", "gemini-auto-fallback"],
    )
    monkeypatch.setattr(imagegen, "get_provider_api_key", lambda _: "global-key")
    monkeypatch.setattr(imagegen, "_save_png", lambda *_: None)

    async def generate(_client, model_id, _prompt, api_key, *, unrestricted=False):
        calls.append((model_id, api_key))
        return ("ok", "aW1hZ2U=", "")

    monkeypatch.setattr(imagegen, "_generate_google", generate)
    result = asyncio.run(
        imagegen.generate_image(
            "Generate a test image",
            "thread-id",
            unrestricted=True,
            selected_model=selected,
        )
    )

    assert result.status == "ok"
    assert calls == [(selected.model_id, selected.api_key)]


def test_unavailable_explicit_model_is_not_replaced_by_default(monkeypatch):
    monkeypatch.setattr(llm_registry, "resolve_with_session", lambda _: None)
    monkeypatch.setattr(chat_client, "_from_settings", lambda *_args, **_kwargs: object())

    with pytest.raises(ValueError, match="Selected model"):
        chat_client.get_chat_client(
            settings=object(),
            config={"configurable": {"model_id": "missing-row-id"}},
        )


def test_unavailable_explicit_image_model_does_not_use_auto(monkeypatch):
    monkeypatch.setattr(llm_registry, "resolve_with_session", lambda _: None)

    async def generate(*_args, **_kwargs):
        raise AssertionError("Auto image generation must not replace a manual selection")

    monkeypatch.setattr(imagegen, "generate_image", generate)
    result = asyncio.run(
        nodes.imagegen(
            {"messages": [HumanMessage("Generate a test image")]},
            {"configurable": {"model_id": "missing-row-id"}},
        )
    )

    assert "selected model" in str(result["messages"][0].content).lower()


def test_manual_model_is_recorded_on_regular_routes(monkeypatch):
    selected = selected_image_model()
    monkeypatch.setattr(llm_registry, "resolve_with_session", lambda _: selected)
    monkeypatch.setattr(routing, "agent_static_prompt", lambda *_: "")
    monkeypatch.setattr(routing, "custom_agents_for_routing", lambda: [])
    monkeypatch.setattr(routing, "local_specialists", lambda: [])
    monkeypatch.setattr(routing, "router_classifier_client", lambda _: object())
    monkeypatch.setattr(routing, "auto_fallback_clients", lambda *_, **__: [])

    class Agent:
        async def ainvoke(self, _input):
            return {
                "structured_response": RouterIntent(
                    intent=Intent.GENERAL_CHAT,
                    reasoning="General request",
                )
            }

    monkeypatch.setattr(routing, "create_agent", lambda **_: Agent())
    result = asyncio.run(
        routing.router(
            {"messages": [HumanMessage("Explain this concept")]},
            {"configurable": {"model_id": "selected-row-id"}},
        )
    )

    route = result["messages"][0].additional_kwargs["routing"]
    assert route["model"] == selected.model_id


def test_image_intent_does_not_capture_image_processing_requests():
    assert routing.fast_intent("Create an image classification model") is None
