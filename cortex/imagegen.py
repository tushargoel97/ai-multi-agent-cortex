"""Image generation via Google's image models (Nano Banana family).

Two guardrail layers keep output safe:
  A. A fast-tier LLM screens the request BEFORE any image API call; explicit
     "disallowed" verdicts always block. If the guardrail model itself is
     unavailable the request proceeds, layer B still applies.
  B. Strict Gemini safetySettings on the generate call; a safety block from
     the API becomes a polite refusal, not an error.

Generated PNGs land in GENERATED_DIR named `{thread_id}_{timestamp}.png` and
are served to the chat UI by the Next.js `/api/v1/images/[name]` route.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from cortex.db.models import ProviderKind
from cortex.db.services.auto_mode import image_model_candidates, resolve_auto_model
from cortex.db.services.llm_registry import (
    ResolvedModel,
    build_client_from_resolved,
    get_provider_api_key,
)

logger = logging.getLogger(__name__)

GENERATED_DIR = Path("/app/generated_images")

_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_LOW_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Unrestricted mode relaxes only these *configurable* thresholds to the most
# permissive the API allows. The providers still enforce their non-configurable
# hard limits server-side, so illegal content stays blocked regardless.
_SAFETY_SETTINGS_OFF = [
    {"category": s["category"], "threshold": "BLOCK_NONE"} for s in _SAFETY_SETTINGS
]

_GUARDRAIL_PROMPT = """You are the safety gate for an image generator. Decide \
whether the request below may be fulfilled. Refuse (allowed=false) if it asks \
for: sexual or NSFW content or nudity of any kind; minors in any unsafe or \
sexualized context; graphic violence or gore; hate symbols or extremist \
content; instructions for weapons or other illegal activity; or a realistic \
depiction of an identifiable real person (celebrity, politician, private \
individual). Artistic styles, fictional characters, landscapes, products, \
logos, and abstract art are all fine.

Respond with ONLY this JSON, nothing else:
{"allowed": true|false, "reason": "<short reason>"}

Request: """


@dataclass
class ImageResult:
    status: str  # "ok" | "refused" | "blocked" | "error"
    detail: str  # refusal/error text, or caption text for ok
    filename: str | None = None
    model_used: str | None = None


async def screen_prompt(prompt: str) -> tuple[bool, str]:
    """Layer A: LLM pre-flight. Returns (allowed, reason)."""
    resolved = resolve_auto_model("fast")
    if resolved is None:
        return True, "guardrail model unavailable, relying on API safety settings"
    try:
        model = build_client_from_resolved(resolved)
        # Tag so LangGraph's `messages` stream mode never surfaces this internal
        # guardrail verdict ({"allowed": ...}) as a chat bubble in the UI.
        result = await model.ainvoke(
            _GUARDRAIL_PROMPT + prompt,
            config={"tags": ["langsmith:nostream"]},
        )
        text = result.content if isinstance(result.content, str) else "".join(
            b.get("text", "") for b in result.content if isinstance(b, dict)
        )
        match = re.search(r"\{.*\}", text, re.DOTALL)
        verdict = json.loads(match.group(0)) if match else {}
        if verdict.get("allowed") is False:
            return False, str(verdict.get("reason", "not allowed"))
        return True, ""
    except Exception:  # noqa: BLE001, infra failure fails open (layer B remains)
        logger.exception("Image guardrail screening failed, proceeding to layer B")
        return True, "guardrail error, relying on API safety settings"


def _safe_thread_id(thread_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-]", "", thread_id)[:64] or "thread"


def _save_png(filename: str, image_b64: str) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    (GENERATED_DIR / filename).write_bytes(base64.b64decode(image_b64))


async def generate_image(
    prompt: str,
    thread_id: str,
    *,
    unrestricted: bool = False,
    selected_model: ResolvedModel | None = None,
) -> ImageResult:
    # Layer A: the app's own LLM pre-screen. Unrestricted mode skips it; the
    # provider-level safety (Layer B) below is always applied and enforces the
    # non-configurable hard limits regardless.
    if not unrestricted:
        allowed, reason = await screen_prompt(prompt)
        if not allowed:
            return ImageResult(
                status="refused",
                detail=(
                    "I can't generate that image: "
                    f"{reason}. I'm happy to create something else, safe-for-work "
                    "scenes, products, fictional characters, logos, and art styles "
                    "are all fair game."
                ),
            )

    if selected_model is not None:
        if selected_model.kind not in (ProviderKind.GOOGLE, ProviderKind.OPENAI):
            return ImageResult(
                status="error",
                detail=(
                    f"The selected model '{selected_model.model_id}' does not use "
                    "a supported image provider."
                ),
                model_used=selected_model.model_id,
            )
        candidates = [
            (selected_model.model_id, selected_model.kind, selected_model.api_key)
        ]
    else:
        keys = {
            ProviderKind.GOOGLE: get_provider_api_key(ProviderKind.GOOGLE),
            ProviderKind.OPENAI: get_provider_api_key(ProviderKind.OPENAI),
        }
        if not any(keys.values()):
            return ImageResult(
                status="error",
                detail=(
                    "No Google or OpenAI API key is configured, image generation "
                    "needs one. Add a key in Admin → Providers."
                ),
            )
        candidates = []
        for model_id in image_model_candidates():
            kind = (
                ProviderKind.OPENAI
                if model_id.startswith(("gpt-image", "dall-e", "chatgpt-image"))
                else ProviderKind.GOOGLE
            )
            candidates.append((model_id, kind, keys[kind]))

    last_error = "no image model candidates configured"
    async with httpx.AsyncClient(timeout=180) as client:
        for model_id, kind, key in candidates:
            if not key:
                last_error = f"{model_id}: no API key for its provider"
                continue
            try:
                if kind == ProviderKind.OPENAI:
                    outcome = await _generate_openai(
                        client, model_id, prompt, key, unrestricted=unrestricted
                    )
                else:
                    outcome = await _generate_google(
                        client, model_id, prompt, key, unrestricted=unrestricted
                    )
            except httpx.HTTPError as e:
                last_error = f"{model_id}: {e}"
                continue
            if isinstance(outcome, str):  # soft failure: try the next candidate
                last_error = outcome
                continue
            status, image_b64, caption = outcome
            if status == "blocked":
                return ImageResult(
                    status="blocked",
                    detail=(
                        "The image model declined this request on safety "
                        "grounds. Try rephrasing toward safe-for-work content."
                    ),
                    model_used=model_id,
                )
            filename = f"{_safe_thread_id(thread_id)}_{int(time.time() * 1000)}.png"
            # File IO off the event loop, langgraph dev rejects blocking calls.
            await asyncio.to_thread(_save_png, filename, image_b64)
            return ImageResult(
                status="ok",
                detail=caption,
                filename=filename,
                model_used=model_id,
            )

    logger.error("All image model candidates failed: %s", last_error)
    return ImageResult(
        status="error",
        detail=(
            "Image generation failed on all configured models "
            f"(last error: {last_error}). Check provider quotas in "
            "Admin → Providers and try again."
        ),
    )


# Each generator returns an error string (try next candidate) or a tuple
# (status, image_b64, caption) where status is "ok" | "blocked".
async def _generate_google(
    client: httpx.AsyncClient,
    model_id: str,
    prompt: str,
    api_key: str,
    *,
    unrestricted: bool = False,
) -> str | tuple[str, str | None, str]:
    resp = await client.post(
        f"{_API_BASE}/{model_id}:generateContent",
        headers={"x-goog-api-key": api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
            "safetySettings": (
                _SAFETY_SETTINGS_OFF if unrestricted else _SAFETY_SETTINGS
            ),
        },
    )
    if resp.status_code != 200:
        return f"{model_id}: HTTP {resp.status_code} {resp.text[:150]}"
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates or candidates[0].get("finishReason") in (
        "SAFETY",
        "PROHIBITED_CONTENT",
        "IMAGE_SAFETY",
    ):
        return ("blocked", None, "")
    caption_parts: list[str] = []
    image_b64 = None
    for part in candidates[0].get("content", {}).get("parts", []):
        if "inlineData" in part:
            image_b64 = part["inlineData"].get("data")
        elif part.get("text"):
            caption_parts.append(part["text"])
    if not image_b64:
        return f"{model_id}: response had no image data"
    return ("ok", image_b64, " ".join(caption_parts).strip())


async def _generate_openai(
    client: httpx.AsyncClient,
    model_id: str,
    prompt: str,
    api_key: str,
    *,
    unrestricted: bool = False,
) -> str | tuple[str, str | None, str]:
    body: dict[str, object] = {
        "model": model_id,
        "prompt": prompt,
        "size": "1024x1024",
        "n": 1,
    }
    # gpt-image-1 accepts a lower moderation setting; dall-e rejects the param.
    if unrestricted and model_id.startswith(("gpt-image", "chatgpt-image")):
        body["moderation"] = "low"
    resp = await client.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
    )
    if resp.status_code == 400 and "content_policy" in resp.text:
        return ("blocked", None, "")
    if resp.status_code != 200:
        return f"{model_id}: HTTP {resp.status_code} {resp.text[:150]}"
    data = resp.json().get("data") or []
    image_b64 = data[0].get("b64_json") if data else None
    if not image_b64:
        return f"{model_id}: response had no image data"
    return ("ok", image_b64, "")
