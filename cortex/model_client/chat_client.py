"""Chat model factory.

Two paths:

1. **Per-request (preferred)**: pass the LangGraph ``RunnableConfig``; we read the
   chat-UI-selected ``model_id`` (a ``LLMModel`` UUID) from
   ``config["configurable"]`` and resolve provider/credentials from the database.
   Optional ``local_base_url`` and ``local_api_key`` keys let an end-user plug
   their own local LLM endpoint without editing the registry.

2. **Fallback**: if no model_id is supplied (or DB lookup fails), fall back to
   the default ``LLMModel`` row marked ``is_default=True``. If no default row
   exists either, fall back to the static ``settings.yaml`` provider.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from cortex.config import Provider, Settings, load_settings


def _needs_responses_api(model: str | None) -> bool:
    """Models that OpenAI only exposes via /v1/responses (not chat-completions)."""
    if not model:
        return False
    m = model.lower()
    return (
        m.startswith("gpt-5")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
    )


def _from_settings(settings: Settings) -> BaseChatModel:
    match settings.provider:
        case Provider.OPENAI:
            cfg = settings.openai
            kwargs: dict = {"model": cfg.model}
            if cfg.api_key:
                kwargs["api_key"] = cfg.api_key
            if cfg.base_url:
                kwargs["base_url"] = cfg.base_url
            if _needs_responses_api(cfg.model):
                kwargs["use_responses_api"] = True
            return ChatOpenAI(**kwargs)

        case Provider.AZURE_OPENAI:
            cfg = settings.azure_openai
            kwargs = {
                "model": cfg.model,
                "temperature": cfg.temperature,
                "azure_endpoint": cfg.azure_endpoint,
                "api_version": cfg.api_version,
            }
            if cfg.api_key:
                kwargs["api_key"] = cfg.api_key
            if _needs_responses_api(cfg.model):
                kwargs["use_responses_api"] = True
            return AzureChatOpenAI(**kwargs)

        case _:
            raise ValueError(f"Unsupported provider: {settings.provider}")


def get_chat_client(
    settings: Settings | None = None,
    *,
    config: dict[str, Any] | None = None,
    auto_intent: str | None = None,
) -> BaseChatModel:
    """Build the chat model for a request.

    Reads ``config["configurable"]`` for ``model_id`` (LLMModel UUID or the
    ``"auto"`` sentinel), ``local_base_url``, ``local_api_key``, and
    ``local_model_name`` overrides. In auto mode the model is picked per
    ``auto_intent`` (fast tier when None). Falls back to the DB default
    model, then to ``settings.yaml``.
    """
    settings = settings or load_settings()
    configurable: dict[str, Any] = (config or {}).get("configurable", {}) or {}
    model_uuid = configurable.get("model_id")
    # Thinking mode (chat-UI slider) raises the reasoner to the quality tier and
    # turns on the provider's extended thinking, only for the reasoning agent.
    mode = str(configurable.get("mode") or "").lower()
    thinking = mode == "thinking" and auto_intent == "reasoning_task"

    if not configurable.get("local_base_url"):
        try:
            from cortex.db.services.auto_mode import (
                FAST_TIER,
                is_auto,
                resolve_auto_model,
            )

            if is_auto(model_uuid):
                model_uuid = None  # from here on: auto resolves or DB default
                from cortex.db.services.llm_registry import build_client_from_resolved

                resolved = resolve_auto_model(
                    auto_intent or FAST_TIER,
                    profile="quality" if thinking else None,
                )
                if resolved is not None:
                    return build_client_from_resolved(resolved, thinking=thinking)
        except Exception:  # noqa: BLE001, auto mode must never kill a run
            pass
    local_base_url = configurable.get("local_base_url")
    local_api_key = configurable.get("local_api_key")
    local_model_name = configurable.get("local_model_name")

    # User-supplied local LLM endpoint (chat-UI "Use local LLM" toggle), short-circuit.
    if local_base_url:
        kwargs: dict = {
            "model": local_model_name or "local-model",
            "api_key": local_api_key or "not-needed",
            "base_url": local_base_url,
        }
        return ChatOpenAI(**kwargs)

    try:
        from cortex.db.services.llm_registry import (
            build_client_from_resolved,
            resolve_with_session,
        )

        resolved = resolve_with_session(model_uuid)
        if resolved is not None:
            return build_client_from_resolved(
                resolved,
                local_base_url_override=local_base_url,
                local_api_key_override=local_api_key,
                thinking=thinking,
            )
    except Exception:  # noqa: BLE001, registry is optional, fall through
        pass

    return _from_settings(settings)
