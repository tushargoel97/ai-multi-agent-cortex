"""Q&A pair generation from source text chunks via an OpenAI-compatible LLM.

Defaults to the local llama.cpp service (ai/, port 8100) so no API key is
required; point TRAINER_QA_BASE_URL / TRAINER_QA_API_KEY / TRAINER_QA_MODEL at
OpenAI (or any compatible endpoint) for higher-quality generation.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from .config import settings

logger = logging.getLogger(__name__)

_PROMPT = """From the source text below, write {n} factual question-answer pairs.

Rules:
- Ground every answer ONLY in the source text — no outside knowledge.
- Questions should be self-contained (understandable without seeing the text).
- Answers: 1-3 sentences, concrete, include numbers/names from the text.
- Output STRICT JSON only: [{{"q": "...", "a": "..."}}, ...] — no commentary.

Source text:
---
{chunk}
---"""


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
    "text, tables, and specifications exactly as shown — preserve every number, "
    "unit, and model name, and reproduce tables row by row as plain text. Do "
    "not summarize or add anything not in the image; output only the transcription."
)


def transcribe_image(image_bytes: bytes, mime: str) -> str:
    """Transcribe an image to text via the vision-capable QA model.

    Reuses the OpenAI-compatible endpoint (TRAINER_QA_*). The configured model
    MUST accept image input (OpenAI gpt-4o, Gemini, a local VLM, …) — the
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
            "Vision model returned no text — point TRAINER_QA_* at a model that "
            "accepts images (e.g. OpenAI gpt-4o)."
        )
    return text


def _parse_pairs(text: str) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    pairs = []
    for item in raw if isinstance(raw, list) else []:
        q, a = str(item.get("q", "")).strip(), str(item.get("a", "")).strip()
        if q and a:
            pairs.append({"q": q, "a": a})
    return pairs


def generate_pairs_for_chunks(
    chunks: list[str],
    *,
    pairs_per_chunk: int = 4,
    max_pairs: int = 500,
    on_progress=None,
) -> list[dict]:
    """Generate Q&A pairs chunk by chunk. Returns [{"q","a"}, ...].

    Parse failures skip the chunk (logged); a dead endpoint raises so the
    caller can surface a clear job error.
    """
    base_url = settings.qa_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.qa_api_key}"}
    pairs: list[dict] = []

    with httpx.Client(timeout=180.0, headers=headers) as client:
        model = _resolve_model(client, base_url)
        logger.info("QA generation via %s model=%s", base_url, model)
        for i, chunk in enumerate(chunks):
            if len(pairs) >= max_pairs:
                break
            resp = client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "temperature": 0.3,
                    "messages": [
                        {
                            "role": "user",
                            "content": _PROMPT.format(n=pairs_per_chunk, chunk=chunk),
                        }
                    ],
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            got = _parse_pairs(content)
            if not got:
                logger.warning("Chunk %d: no parseable Q&A pairs — skipped", i)
            pairs.extend(got)
            if on_progress:
                on_progress(chunk_index=i, pairs_total=len(pairs))
    return pairs[:max_pairs]
