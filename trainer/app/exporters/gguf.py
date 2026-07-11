"""Generic Hugging Face checkpoint to GGUF export helpers."""

from __future__ import annotations

import json
from pathlib import Path

from ..backends.base import CommandStep


def gguf_conversion_step(
    *,
    python: str,
    convert_script: Path,
    fused_dir: Path,
    output_path: Path,
) -> CommandStep:
    return CommandStep(
        phase="converting",
        argv=[
            python,
            str(convert_script),
            str(fused_dir),
            "--outfile",
            str(output_path),
            "--outtype",
            "q8_0",
        ],
    )


def sanitize_fused_tokenizer(fused_dir: Path) -> list[str]:
    """Remove tokenizer entries outside the fused model's vocabulary.

    Gemma text checkpoints can include family-shared multimodal tokens whose
    identifiers exceed the text model's embedding rows. llama.cpp rejects
    those entries during conversion.
    """
    cfg_path = fused_dir / "config.json"
    tok_path = fused_dir / "tokenizer.json"
    if not (cfg_path.exists() and tok_path.exists()):
        return []
    vocab_size = json.loads(cfg_path.read_text()).get("vocab_size")
    if not vocab_size:
        return []

    tok = json.loads(tok_path.read_text())
    added = tok.get("added_tokens", [])
    over = [token for token in added if token.get("id", 0) >= vocab_size]
    if not over:
        return []
    dropped = {token["content"] for token in over}
    tok["added_tokens"] = [
        token for token in added if token.get("id", 0) < vocab_size
    ]
    tok_path.write_text(json.dumps(tok, ensure_ascii=False))

    tokenizer_config_path = fused_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        config = json.loads(tokenizer_config_path.read_text())
        decoder = config.get("added_tokens_decoder")
        if isinstance(decoder, dict):
            config["added_tokens_decoder"] = {
                key: value for key, value in decoder.items() if int(key) < vocab_size
            }
        for key in list(config.keys()):
            value = config[key]
            if isinstance(value, str) and value in dropped:
                del config[key]
            elif isinstance(value, list):
                config[key] = [item for item in value if item not in dropped]
            elif isinstance(value, dict) and key != "added_tokens_decoder":
                config[key] = {
                    item_key: item_value
                    for item_key, item_value in value.items()
                    if item_value not in dropped
                }
        tokenizer_config_path.write_text(json.dumps(config, ensure_ascii=False))

    return sorted(dropped)
