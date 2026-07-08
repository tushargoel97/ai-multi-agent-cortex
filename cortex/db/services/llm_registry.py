"""LLM registry, resolves a model UUID to provider config + builds a chat client."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from cortex.db.engine import get_session
from cortex.db.models import LLMModel, LLMProvider, ProviderKind


@dataclass
class ResolvedModel:
    kind: ProviderKind
    model_id: str
    api_key: str
    base_url: str | None
    azure_endpoint: str | None
    azure_api_version: str | None


def resolve_model(model_uuid: str | uuid.UUID, session: Session) -> ResolvedModel | None:
    """Look up a model + its provider by the model row UUID."""
    try:
        row_id = uuid.UUID(str(model_uuid))
    except (ValueError, TypeError):
        return None

    stmt = (
        select(LLMModel, LLMProvider)
        .join(LLMProvider, LLMModel.provider_id == LLMProvider.id)
        .where(LLMModel.id == row_id)
        .where(LLMModel.enabled.is_(True))
        .where(LLMProvider.enabled.is_(True))
    )
    row = session.execute(stmt).first()
    if row is None:
        return None
    model, provider = row
    return ResolvedModel(
        kind=ProviderKind(provider.kind),
        model_id=model.model_id,
        api_key=provider.api_key,
        base_url=provider.base_url,
        azure_endpoint=provider.azure_endpoint,
        azure_api_version=provider.azure_api_version,
    )


def get_default_resolved_model(session: Session) -> ResolvedModel | None:
    """Return the model marked is_default (any enabled provider)."""
    stmt = (
        select(LLMModel, LLMProvider)
        .join(LLMProvider, LLMModel.provider_id == LLMProvider.id)
        .where(LLMModel.is_default.is_(True))
        .where(LLMModel.enabled.is_(True))
        .where(LLMProvider.enabled.is_(True))
        .limit(1)
    )
    row = session.execute(stmt).first()
    if row is None:
        return None
    model, provider = row
    return ResolvedModel(
        kind=ProviderKind(provider.kind),
        model_id=model.model_id,
        api_key=provider.api_key,
        base_url=provider.base_url,
        azure_endpoint=provider.azure_endpoint,
        azure_api_version=provider.azure_api_version,
    )


@lru_cache(maxsize=1)
def _thinking_safe_anthropic_cls() -> type:
    """ChatAnthropic that keeps round-tripped thinking blocks API-valid.

    Claude 5 adaptive thinking can return a thinking block whose text is empty
    (signature only). langchain-anthropic's streaming merge drops the empty
    ``thinking`` field, and the API then rejects the block on the next agent
    turn with "thinking.thinking: Field required".
    """
    from langchain_anthropic import ChatAnthropic

    class _ThinkingSafeChatAnthropic(ChatAnthropic):
        def _get_request_payload(self, *args, **kwargs):
            payload = super()._get_request_payload(*args, **kwargs)
            for message in payload.get("messages", []):
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        block.setdefault("thinking", "")
            return payload

    return _ThinkingSafeChatAnthropic


def build_client_from_resolved(
    resolved: ResolvedModel,
    *,
    local_base_url_override: str | None = None,
    local_api_key_override: str | None = None,
    thinking: bool = False,
) -> BaseChatModel:
    """Instantiate a LangChain chat client from a ResolvedModel."""
    from langchain_openai import AzureChatOpenAI, ChatOpenAI

    def _needs_responses_api(model: str | None) -> bool:
        if not model:
            return False
        m = model.lower()
        return (
            m.startswith("gpt-5")
            or m.startswith("o1")
            or m.startswith("o3")
            or m.startswith("o4")
        )

    match resolved.kind:
        case ProviderKind.OPENAI:
            kwargs: dict = {"model": resolved.model_id, "max_retries": 4}
            if resolved.api_key:
                kwargs["api_key"] = resolved.api_key
            if resolved.base_url:
                kwargs["base_url"] = resolved.base_url
            if _needs_responses_api(resolved.model_id):
                kwargs["use_responses_api"] = True
                if thinking:
                    kwargs["reasoning_effort"] = "high"
            return ChatOpenAI(**kwargs)

        case ProviderKind.AZURE_OPENAI:
            kwargs = {
                "model": resolved.model_id,
                "max_retries": 4,
                "azure_endpoint": resolved.azure_endpoint or "",
                "api_version": resolved.azure_api_version or "2024-12-01-preview",
            }
            if resolved.api_key:
                kwargs["api_key"] = resolved.api_key
            if _needs_responses_api(resolved.model_id):
                kwargs["use_responses_api"] = True
                if thinking:
                    kwargs["reasoning_effort"] = "high"
            return AzureChatOpenAI(**kwargs)

        case ProviderKind.ANTHROPIC:
            kwargs = {"model": resolved.model_id, "max_retries": 4}
            if resolved.api_key:
                kwargs["api_key"] = resolved.api_key
            if thinking:
                kwargs["max_tokens"] = 8000
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": 4000}
            return _thinking_safe_anthropic_cls()(**kwargs)

        case ProviderKind.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI

            kwargs = {"model": resolved.model_id, "max_retries": 4}
            if resolved.api_key:
                kwargs["google_api_key"] = resolved.api_key
            return ChatGoogleGenerativeAI(**kwargs)

        case ProviderKind.LOCAL:
            kwargs = {
                "model": resolved.model_id,
                "max_retries": 4,
                "api_key": local_api_key_override or resolved.api_key or "not-needed",
                "base_url": local_base_url_override or resolved.base_url,
            }
            return ChatOpenAI(**kwargs)

        case _:
            raise ValueError(f"Unsupported provider kind: {resolved.kind}")


# Contract with the admin UI: fine-tuned models are registered under the
# local provider with this model_id prefix (see FinetunePanel / trainer).
FINE_TUNED_PREFIX = "finetuned-"


def get_fine_tuned_resolved_model(session: Session) -> ResolvedModel | None:
    """Return the newest enabled fine-tuned model on an enabled local provider."""
    stmt = (
        select(LLMModel, LLMProvider)
        .join(LLMProvider, LLMModel.provider_id == LLMProvider.id)
        .where(LLMProvider.kind == ProviderKind.LOCAL.value)
        .where(LLMModel.model_id.like(f"{FINE_TUNED_PREFIX}%"))
        .where(LLMModel.enabled.is_(True))
        .where(LLMProvider.enabled.is_(True))
        .order_by(LLMModel.created_at.desc())
        .limit(1)
    )
    row = session.execute(stmt).first()
    if row is None:
        return None
    model, provider = row
    return ResolvedModel(
        kind=ProviderKind(provider.kind),
        model_id=model.model_id,
        api_key=provider.api_key,
        base_url=provider.base_url,
        azure_endpoint=provider.azure_endpoint,
        azure_api_version=provider.azure_api_version,
    )


def resolve_fine_tuned_model() -> ResolvedModel | None:
    """Session-managing wrapper around get_fine_tuned_resolved_model."""
    with get_session() as s:
        return get_fine_tuned_resolved_model(s)


def resolve_by_model_id(model_id: str) -> ResolvedModel | None:
    """Resolve an enabled model by its provider model_id (not the row UUID)."""
    stmt = (
        select(LLMModel, LLMProvider)
        .join(LLMProvider, LLMModel.provider_id == LLMProvider.id)
        .where(LLMModel.model_id == model_id)
        .where(LLMModel.enabled.is_(True))
        .where(LLMProvider.enabled.is_(True))
        .order_by(LLMModel.created_at.desc())
        .limit(1)
    )
    with get_session() as s:
        row = s.execute(stmt).first()
        if row is None:
            return None
        model, provider = row
        return ResolvedModel(
            kind=ProviderKind(provider.kind),
            model_id=model.model_id,
            api_key=provider.api_key,
            base_url=provider.base_url,
            azure_endpoint=provider.azure_endpoint,
            azure_api_version=provider.azure_api_version,
        )


def get_provider_api_key(kind: ProviderKind) -> str | None:
    """API key of the first enabled provider of the given kind."""
    with get_session() as s:
        stmt = (
            select(LLMProvider)
            .where(LLMProvider.kind == kind.value)
            .where(LLMProvider.enabled.is_(True))
            .limit(1)
        )
        provider = s.execute(stmt).scalar_one_or_none()
        return provider.api_key if provider and provider.api_key else None


def resolve_with_session(model_uuid: str | uuid.UUID | None) -> ResolvedModel | None:
    """Open a session, resolve, and return (None on miss)."""
    if not model_uuid:
        with get_session() as s:
            return get_default_resolved_model(s)
    with get_session() as s:
        resolved = resolve_model(model_uuid, s)
        if resolved is None:
            return get_default_resolved_model(s)
        return resolved
