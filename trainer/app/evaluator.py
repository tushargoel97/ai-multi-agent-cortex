from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any

_WORDS = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = {
    "a",
    "about",
    "and",
    "are",
    "does",
    "for",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "to",
    "vs",
    "with",
}
_REFUSALS = ("don't know", "not in", "cannot", "can't", "rather not", "unsure")


def evaluate(
    dataset: Path,
    model_id: str,
    base_url: str,
    api_key: str,
    limit: int,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line]
    if not rows:
        raise ValueError("evaluation dataset is empty")
    stride = max(1, len(rows) // limit)
    cases = [_case(row, model_id, base_url, api_key) for row in rows[::stride][:limit]]
    pass_rate = sum(case["passed"] for case in cases) / len(cases)
    return {"passed": pass_rate >= 0.8, "pass_rate": pass_rate, "cases": cases}


def score(question: str, expected: str, actual: str) -> float:
    if "rather not guess" in expected.lower():
        return float(any(phrase in actual.lower() for phrase in _REFUSALS))
    expected_words = _tokens(expected) - _tokens(question) - _STOP
    return len(expected_words & _tokens(actual)) / max(1, len(expected_words))


def _case(row: dict[str, Any], model_id: str, base_url: str, api_key: str) -> dict:
    question, expected = (message["content"] for message in row["messages"][-2:])
    actual = _complete(model_id, question, base_url, api_key)
    value = score(question, expected, actual)
    return {
        "question": question,
        "expected": expected,
        "actual": actual,
        "score": round(value, 3),
        "passed": value >= 0.5,
    }


def _complete(model_id: str, prompt: str, base_url: str, api_key: str) -> str:
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 384,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return payload["choices"][0]["message"]["content"]


def _tokens(value: str) -> set[str]:
    return {
        word[:-1]
        if word not in _STOP and len(word) > 3 and word.endswith("s")
        else word
        for word in _WORDS.findall(value.lower())
    }
