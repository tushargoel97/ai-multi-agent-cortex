"""Shared LLM helpers for the trainer's OpenAI-compatible endpoint.

Defaults to the local llama.cpp service (ai/, port 8100) so no API key is
required; point TRAINER_QA_BASE_URL / TRAINER_QA_API_KEY / TRAINER_QA_MODEL at
OpenAI (or any compatible endpoint) for higher-quality generation.

Used by the scrape agent / gap research (``_chat`` in research.py) and by image
source transcription (``transcribe_image``). The old raw-chunk Q&A-pair
generator was removed, sources now become structured spec sheets in
learned_facts.yaml via the scrape agent, not invented Q&A pairs.
"""

from __future__ import annotations

import logging

import httpx

from .config import settings

logger = logging.getLogger(__name__)


def _resolve_model(client: httpx.Client, base_url: str) -> str:
    if settings.qa_model:
        return settings.qa_model
    resp = client.get(f"{base_url}/models")
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise RuntimeError(f"No models available at {base_url}/models")
    return data[0]["id"]


_VISION_PROMPT = (
    "Transcribe this document image for a hardware-specs dataset. Output ALL "
    "text, tables, and specifications exactly as shown, preserve every number, "
    "unit, and model name, and reproduce tables row by row as plain text. Do "
    "not summarize or add anything not in the image; output only the transcription."
)


def transcribe_image(image_bytes: bytes, mime: str) -> str:
    """Transcribe an image to text via the vision-capable QA model.

    Reuses the OpenAI-compatible endpoint (TRAINER_QA_*). The configured model
    MUST accept image input (OpenAI gpt-4o, Gemini, a local VLM, …), the
    default local Gemma 1B cannot, so point TRAINER_QA_* at a vision model.
    """
    import base64

    base_url = settings.qa_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.qa_api_key}"}
    b64 = base64.b64encode(image_bytes).decode()
    with httpx.Client(timeout=180.0, headers=headers) as client:
        model = _resolve_model(client, base_url)
        resp = client.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "temperature": 0.0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _VISION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                    "detail": "high",  # dense spec tables need fine text
                                },
                            },
                        ],
                    }
                ],
            },
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    if not text:
        raise RuntimeError(
            "Vision model returned no text, point TRAINER_QA_* at a model that "
            "accepts images (e.g. OpenAI gpt-4o)."
        )
    return text
