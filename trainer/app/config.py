"""Trainer service settings. All paths derive from the repo root by default."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = {"env_prefix": "TRAINER_"}

    port: int = 8200
    # Ungated HF-layout mirror of google/gemma-3-1b-it (the google/ repo is
    # license-gated; this one needs no HF token).
    base_model: str = "unsloth/gemma-3-1b-it"

    # Q&A generation from admin-provided sources (OpenAI-compatible endpoint).
    # Defaults to the local llama.cpp service; point at OpenAI for quality.
    qa_base_url: str = "http://localhost:8100/v1"
    qa_api_key: str = "not-needed"
    qa_model: str = ""  # empty = first model listed by the endpoint

    models_dir: Path = REPO_ROOT / "models"          # shared with the ai service (bind mount)
    data_dir: Path = REPO_ROOT / "trainer" / "data"
    artifacts_dir: Path = REPO_ROOT / "trainer" / "artifacts"
    llama_cpp_dir: Path = REPO_ROOT / "trainer" / "vendor" / "llama.cpp"

    @property
    def adapters_dir(self) -> Path:
        return self.artifacts_dir / "adapters"

    @property
    def fused_dir(self) -> Path:
        return self.artifacts_dir / "fused"

    @property
    def convert_script(self) -> Path:
        return self.llama_cpp_dir / "convert_hf_to_gguf.py"


settings = Settings()
