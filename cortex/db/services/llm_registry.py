"""LLM registry, resolves a model UUID to provider config + builds a chat client."""

from __future__ import annotations

import re
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

    def __post_init__(self) -> None:
        # Trim credentials: a trailing newline or space from a copy-pasted key
        # is the most common cause of a genuinely-valid key being rejected as
        # "Incorrect API key". Also trims model id / URLs for good measure.
        self.model_id = (self.model_id or "").strip()
        self.api_key = (self.api_key or "").strip()
        if self.base_url:
            self.base_url = self.base_url.strip()
        if self.azure_endpoint:
            self.azure_endpoint = self.azure_endpoint.strip()


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


def _anthropic_adaptive_thinking(model: str) -> bool:
    normalized = model.lower().replace("_", "-").replace(".", "-")
    return bool(
        "claude-5" in normalized
        or re.search(
            r"(?:claude-)?(?:fable|mythos|opus|sonnet|haiku)-5",
            normalized,
        )
        or re.search(
            r"(?:claude-)?(?:opus|sonnet|haiku)-4-[5-9]",
            normalized,
        )
    )


def _anthropic_supports_thinking(model: str) -> bool:
    """Whether a Claude model supports extended thinking at all.

    Extended thinking arrived with Claude 3.7 and continues through the 4/4.5/5
    families. Claude 3.0/3.5, Claude 2, and instant do NOT support it, and
    sending a ``thinking`` block to them 400s, so Thinking mode falls back to a
    normal (non-thinking) call for those instead of failing the turn.
    """
    m = model.lower().replace("_", "-")
    unsupported = (
        "claude-2",
        "claude-instant",
        "claude-3-0",
        "claude-3-5",
        "claude-3-haiku",
        "claude-3-opus",
        "claude-3-sonnet",
    )
    return not any(tag in m for tag in unsupported)


def _anthropic_supports_effort(model: str) -> bool:
    normalized = model.lower().replace("_", "-").replace(".", "-")
    return bool(
        "claude-5" in normalized
        or re.search(
            r"(?:claude-)?(?:fable|mythos|opus|sonnet|haiku)-5",
            normalized,
        )
        or re.search(r"(?:claude-)?(?:opus-4-5|(?:opus|sonnet)-4-[6-9])", normalized)
    )


def _anthropic_effort(model: str, effort: str | None) -> str | None:
    if effort not in ("low", "medium", "high", "xhigh", "max"):
        return None
    normalized = model.lower().replace("_", "-").replace(".", "-")
    if not _anthropic_supports_effort(normalized):
        return None
    if effort == "xhigh" and not (
        re.search(
            r"(?:claude-)?(?:fable|mythos|opus|sonnet|haiku)-5",
            normalized,
        )
        or re.search(r"(?:claude-)?opus-4-[7-9]", normalized)
    ):
        return "high"
    if effort == "max" and re.search(r"(?:claude-)?opus-4-5", normalized):
        return "high"
    return effort


def _google_supports_thinking(model: str) -> bool:
    normalized = model.lower().replace("_", "-")
    return bool(re.search(r"\bgemini-(?:2-5|[3-9])", normalized))


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
            # Newer Claude models replaced thinking:{type:enabled,budget_tokens}
            # with the self-managed thinking:{type:adaptive}; the old shape 400s
            # ("thinking.type.enabled is not supported for this model"), so
            # translate it just before the request goes out.
            thinking_cfg = payload.get("thinking")
            if (
                isinstance(thinking_cfg, dict)
                and thinking_cfg.get("type") == "enabled"
                and _anthropic_adaptive_thinking(
                    str(payload.get("model") or getattr(self, "model", "") or "")
                )
            ):
                payload["thinking"] = {"type": "adaptive"}
            return payload

    return _ThinkingSafeChatAnthropic


def build_client_from_resolved(
    resolved: ResolvedModel,
    *,
    local_base_url_override: str | None = None,
    local_api_key_override: str | None = None,
    thinking: bool = False,
    effort: str | None = None,
    max_output_tokens: int | None = None,
) -> BaseChatModel:
    """Instantiate a LangChain chat client from a ResolvedModel."""
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    from cortex.model_client.chat_client import (
        _needs_responses_api,
        _openai_reasoning_effort,
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
                requested = effort or ("high" if thinking else None)
                if provider_effort := _openai_reasoning_effort(
                    resolved.model_id, requested
                ):
                    kwargs["reasoning_effort"] = provider_effort
            if max_output_tokens:
                kwargs["max_tokens"] = max_output_tokens
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
                requested = effort or ("high" if thinking else None)
                if provider_effort := _openai_reasoning_effort(
                    resolved.model_id, requested
                ):
                    kwargs["reasoning_effort"] = provider_effort
            if max_output_tokens:
                kwargs["max_tokens"] = max_output_tokens
            return AzureChatOpenAI(**kwargs)

        case ProviderKind.ANTHROPIC:
            kwargs = {"model": resolved.model_id, "max_retries": 4}
            if resolved.api_key:
                kwargs["api_key"] = resolved.api_key
            if provider_effort := _anthropic_effort(resolved.model_id, effort):
                kwargs["effort"] = provider_effort
            if max_output_tokens:
                kwargs["max_tokens"] = max_output_tokens
            if thinking and _anthropic_supports_thinking(resolved.model_id):
                thinking_budget = {
                    "low": 1_000,
                    "medium": 2_000,
                    "high": 4_000,
                    "xhigh": 6_000,
                    "max": 8_000,
                }.get(effort or "high", 4_000)
                kwargs["max_tokens"] = (max_output_tokens or 4_000) + thinking_budget
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
            return _thinking_safe_anthropic_cls()(**kwargs)

        case ProviderKind.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI

            kwargs = {"model": resolved.model_id, "max_retries": 4}
            if resolved.api_key:
                kwargs["google_api_key"] = resolved.api_key
            if max_output_tokens:
                kwargs["max_tokens"] = max_output_tokens
            if effort and _google_supports_thinking(resolved.model_id):
                kwargs["thinking_level"] = (
                    effort if effort in ("low", "medium", "high") else "high"
                )
            return ChatGoogleGenerativeAI(**kwargs)

        case ProviderKind.LOCAL:
            kwargs = {
                "model": resolved.model_id,
                "max_retries": 0,
                "api_key": local_api_key_override or resolved.api_key or "not-needed",
                "base_url": local_base_url_override or resolved.base_url,
            }
            if max_output_tokens:
                kwargs["max_tokens"] = max_output_tokens
            return ChatOpenAI(**kwargs)

        case _:
            raise ValueError(f"Unsupported provider kind: {resolved.kind}")


@dataclass
class LocalSpecialist:
    model_id: str
    display_name: str
    description: str


def local_specialists_for_routing() -> list[LocalSpecialist]:
    """Return described, enabled local specialist models."""
    stmt = (
        select(LLMModel)
        .join(LLMProvider, LLMModel.provider_id == LLMProvider.id)
        .where(LLMProvider.kind == ProviderKind.LOCAL.value)
        .where(LLMModel.enabled.is_(True))
        .where(LLMProvider.enabled.is_(True))
        .order_by(LLMModel.created_at.asc())
    )
    with get_session() as s:
        rows = s.execute(stmt).scalars().all()
        return [
            LocalSpecialist(
                model_id=m.model_id,
                display_name=m.display_name or m.model_id,
                description=m.description.strip(),
            )
            for m in rows
            if (m.description or "").strip()
        ]


def resolve_local_specialist(model_id: str) -> ResolvedModel | None:
    """Resolve an enabled local model by model ID."""
    stmt = (
        select(LLMModel, LLMProvider)
        .join(LLMProvider, LLMModel.provider_id == LLMProvider.id)
        .where(LLMProvider.kind == ProviderKind.LOCAL.value)
        .where(LLMModel.model_id == model_id)
        .where(LLMModel.enabled.is_(True))
        .where(LLMProvider.enabled.is_(True))
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


def local_specialist_profile(model_id: str) -> LocalSpecialist | None:
    """Return a local model's capability profile."""
    return next(
        (sp for sp in local_specialists_for_routing() if sp.model_id == model_id),
        None,
    )


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
        key = (provider.api_key or "").strip() if provider else ""
        return key or None


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
