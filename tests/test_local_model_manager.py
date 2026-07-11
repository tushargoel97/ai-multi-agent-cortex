import asyncio
import sys
from pathlib import Path

import pytest

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1] / "ai"
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.services import model_manager as mm  # noqa: E402


def test_model_paths_cannot_escape_the_models_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODELS_DIR", str(tmp_path))

    assert mm._path_in_models_dir("nested/model.gguf").startswith(str(tmp_path))
    with pytest.raises(ValueError, match="models directory"):
        mm._path_in_models_dir("../outside.gguf")
    with pytest.raises(ValueError, match="models directory"):
        mm.validate_download_request(
            "community-model",
            repo_id="org/repo",
            filename="../../outside.gguf",
        )


def test_model_sessions_are_serialized(monkeypatch):
    async def scenario():
        monkeypatch.setattr(mm, "_model_lock", asyncio.Lock())
        monkeypatch.setattr(mm, "_llm", object())
        monkeypatch.setattr(mm, "_loaded_model_name", "test-model")

        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_entered = asyncio.Event()

        async def first_session():
            async with mm.model_session():
                first_entered.set()
                await release_first.wait()

        async def second_session():
            await first_entered.wait()
            async with mm.model_session():
                second_entered.set()

        first = asyncio.create_task(first_session())
        second = asyncio.create_task(second_session())
        await first_entered.wait()
        await asyncio.sleep(0)
        assert not second_entered.is_set()

        release_first.set()
        await asyncio.gather(first, second)
        assert second_entered.is_set()

    asyncio.run(scenario())
