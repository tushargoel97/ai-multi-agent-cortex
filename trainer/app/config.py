"""Trainer service settings. All paths derive from the repo root by default."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from the repo ``.env`` (existing shell env wins).

    The host trainer isn't started by docker-compose, so it wouldn't otherwise
    see keys like FIRECRAWL_API_KEY / TRAINER_QA_* that the Dockerized services
    get via ``env_file``. Kept dependency-free (no python-dotenv needed).
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv(REPO_ROOT / ".env")


class Settings(BaseSettings):
    model_config = {"env_prefix": "TRAINER_"}

    port: int = 8200
    host_id: str = "local-mac"
    host_label: str = "Local Mac (MLX)"
    default_backend: str = "mlx-lora"
    base_model: str = "unsloth/gemma-3-1b-it"
    qa_base_url: str = "http://localhost:8100/v1"
    qa_api_key: str = "not-needed"
    qa_model: str = ""  # empty = first model listed by the endpoint
    eval_base_url: str = "http://localhost:8100/v1"
    eval_api_key: str = "not-needed"

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
    def runs_dir(self) -> Path:
        return self.artifacts_dir / "runs"

    @property
    def convert_script(self) -> Path:
        return self.llama_cpp_dir / "convert_hf_to_gguf.py"


settings = Settings()
