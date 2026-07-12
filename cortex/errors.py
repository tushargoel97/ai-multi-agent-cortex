from __future__ import annotations

from functools import lru_cache


class AgentSpecNotFoundError(KeyError):
    """Raised when a requested agent spec is not present in the registry."""

    def __init__(self, context: str, agent_name: str) -> None:
        super().__init__(f"[{context}] Agent spec '{agent_name}' not found in registry.")
        self.context = context
        self.agent_name = agent_name


# ── Model-call error handling ────────────────────────────────────────────────


@lru_cache(maxsize=1)
def retryable_model_exceptions() -> tuple[type[BaseException], ...]:
    """Provider exception types worth switching models for.

    Covers quota, rate limit, timeout, connection, and provider-outage errors
    across the OpenAI, Anthropic, and Google SDKs. Each import is guarded so a
    missing SDK never breaks module import; falls back to a broad ``Exception``
    tuple if none resolve.
    """
    types: list[type[BaseException]] = []
    try:
        import openai

        types += [
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ]
    except Exception:  # noqa: BLE001, SDK optional
        pass
    try:
        import anthropic

        types += [
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        ]
    except Exception:  # noqa: BLE001, SDK optional
        pass
    try:
        from google.api_core import exceptions as gexc

        types += [
            gexc.ResourceExhausted,
            gexc.TooManyRequests,
            gexc.ServiceUnavailable,
            gexc.DeadlineExceeded,
            gexc.InternalServerError,
        ]
    except Exception:  # noqa: BLE001, SDK optional
        pass
    return tuple(dict.fromkeys(types)) or (Exception,)


def _error_text(exc: BaseException) -> str:
    parts = [str(exc), str(getattr(exc, "message", "") or "")]
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        parts.append(code)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        parts.append(str(body))
    return " ".join(parts).lower()


def friendly_model_error(exc: BaseException) -> str:
    """One short, user-facing clause explaining why a model call failed."""
    text = _error_text(exc)
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = None

    if "graphrecursion" in name or "recursion limit" in text:
        return "the agent reached its tool-step limit before finishing"

    if (
        "insufficient_quota" in text
        or "exceeded your current quota" in text
        or "out of credit" in text
        or "billing" in text
    ):
        return "the selected model is out of quota or credits"
    if (
        status == 429
        or "rate limit" in text
        or "ratelimit" in name
        or "resourceexhausted" in name
        or "too many requests" in text
    ):
        return "the model is rate-limited right now"
    if (
        status in (401, 403)
        or "authentication" in name
        or "permissiondenied" in name
        or "invalid api key" in text
        or "incorrect api key" in text
        or "unauthorized" in text
    ):
        return "the model rejected its API key"
    if (
        status in (500, 502, 503, 529)
        or "overloaded" in text
        or "unavailable" in text
        or "timeout" in text
        or "timed out" in text
        or "connection" in name
    ):
        return "the model provider is temporarily unavailable"
    return "the model call failed"


def model_error_reply(exc: BaseException, *, auto: bool) -> str:
    """Full chat message shown when a model call can't be completed."""
    reason = friendly_model_error(exc)
    if "tool-step limit" in reason:
        return (
            "I couldn't finish that because the agent reached its tool-step "
            "limit. Please retry with a narrower request or split it into parts."
        )
    if auto:
        return (
            f"I couldn't complete that because {reason}, and the automatic "
            "fallback models are unavailable too. Please try again shortly, or "
            "pick a specific model from the selector below."
        )
    return (
        f"I couldn't complete that because {reason}. Try switching to Auto or "
        "another model from the selector below, or try again in a moment."
    )
