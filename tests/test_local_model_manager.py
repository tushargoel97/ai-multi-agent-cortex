import asyncio
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


def test_context_overflow_is_returned_as_a_client_error():
    from app.api.v1.endpoints.inference import inference_error

    error = inference_error(
        ValueError("Requested tokens (4419) exceed context window of 4096")
    )

    assert error.status_code == 400
    assert "context window" in error.detail.lower()


def test_cancel_removes_a_paused_partial_download(tmp_path, monkeypatch):
    name = "paused-model"
    filename = "paused.gguf"
    partial = tmp_path / f"{filename}.part"
    partial.write_bytes(b"partial")
    monkeypatch.setattr(mm, "MODELS_DIR", str(tmp_path))
    monkeypatch.setitem(
        mm.MODEL_CATALOG,
        name,
        {
            "filename": filename,
            "repo_id": "test/repo",
            "tags": [],
            "size_mb": 1,
        },
    )
    mm._download_progress.pop(name, None)

    async def scenario():
        assert mm.request_download_control(name, "cancel")
        assert mm._download_progress[name]["status"] == "cancelled"

    asyncio.run(scenario())

    assert not partial.exists()


def test_partial_download_is_reported_after_process_restart(tmp_path, monkeypatch):
    name = "restarted-model"
    filename = "restarted.gguf"
    Path(f"{tmp_path / filename}.part").write_bytes(b"x" * 1024 * 1024)
    monkeypatch.setattr(mm, "MODELS_DIR", str(tmp_path))
    monkeypatch.setitem(
        mm.MODEL_CATALOG,
        name,
        {
            "filename": filename,
            "repo_id": "test/repo",
            "tags": [],
            "size_mb": 10,
        },
    )
    mm._download_progress.pop(name, None)

    progress = mm.download_progress()[name]

    assert progress["status"] == "paused"
    assert progress["downloaded_mb"] > 0


@pytest.mark.parametrize("terminal", ["cancelled", "complete"])
def test_stale_progress_cleanup_preserves_restarted_download(terminal, monkeypatch):
    name = f"restarted-{terminal}"
    pending = []
    monkeypatch.setattr(mm, "start", lambda coroutine, **_: pending.append(coroutine))

    async def scenario():
        previous = object()
        mm._download_generations[name] = previous
        mm._download_progress[name] = {"status": terminal}
        mm._schedule_progress_cleanup(name, 0, previous)
        mm._download_generations[name] = object()
        mm._download_progress[name] = {"status": "downloading"}

        await pending.pop()

        assert mm._download_progress[name]["status"] == "downloading"

    asyncio.run(scenario())
    mm._download_generations.pop(name, None)
    mm._download_progress.pop(name, None)


def test_catalog_updates_are_serialized_and_atomic(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("{}")
    monkeypatch.setattr(mm, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(mm, "_CUSTOM_CATALOG_PATH", str(catalog_path))
    original_load = json.load
    replacements = []
    original_replace = mm.os.replace

    def slow_load(file):
        value = original_load(file)
        time.sleep(0.02)
        return value

    def track_replace(source, destination):
        replacements.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(mm.json, "load", slow_load)
    monkeypatch.setattr(mm.os, "replace", track_replace)
    barrier = threading.Barrier(2)

    def persist(name):
        barrier.wait()
        mm._persist_custom_entry(name, {"filename": f"{name}.gguf"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(persist, ("first", "second")))

    assert set(json.loads(catalog_path.read_text())) == {"first", "second"}
    assert len(replacements) == 2
    assert all(destination == str(catalog_path) for _, destination in replacements)


def test_completed_tool_stream_adds_tool_call_indexes():
    from app.api.v1.endpoints.inference import _completed_stream

    result = {
        "id": "result",
        "created": 1,
        "model": "local",
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    event = next(_completed_stream(result))
    payload = json.loads(event.removeprefix("data: "))

    assert payload["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
