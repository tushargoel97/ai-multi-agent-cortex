"""Embedding model factory — returns a LangChain Embeddings based on settings.yaml."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from cortex.config import Provider, Settings, load_settings


def _openai_key_from_db() -> str | None:
    """Pull the active OpenAI provider's API key from the LLM registry, if any."""
    try:
        from sqlalchemy import select

        from cortex.db.engine import get_session
        from cortex.db.models import LLMProvider, ProviderKind

        with get_session() as s:
            row = s.execute(
                select(LLMProvider)
                .where(LLMProvider.kind == ProviderKind.OPENAI.value)
                .where(LLMProvider.enabled.is_(True))
                .limit(1)
            ).scalar_one_or_none()
            if row and row.api_key:
                return row.api_key
    except Exception:  # noqa: BLE001 — registry is optional
        return None
    return None


def get_embedding_client(settings: Settings | None = None) -> Embeddings:
    """Build and return the embedding model configured in *settings.yaml*.

    Prefers the OpenAI provider's API key stored in the LLM registry (set via
    the admin UI) over the env-var-driven ``settings.openai.api_key`` so users
    don't need to also export ``OPENAI_API_KEY`` for embeddings to work.
    """
    settings = settings or load_settings()

    match settings.provider:
        case Provider.OPENAI:
            cfg = settings.openai
            kwargs: dict = {"model": settings.embedding_model}
            api_key = _openai_key_from_db() or cfg.api_key
            if api_key:
                kwargs["api_key"] = api_key
            return OpenAIEmbeddings(**kwargs)

        case Provider.AZURE_OPENAI:
            cfg = settings.azure_openai
            kwargs = {
                "model": settings.embedding_model,
                "azure_endpoint": cfg.azure_endpoint,
                "api_version": cfg.api_version,
            }
            if cfg.api_key:
                kwargs["api_key"] = cfg.api_key
            return AzureOpenAIEmbeddings(**kwargs)

        case _:
            raise ValueError(f"Unsupported provider: {settings.provider}")
