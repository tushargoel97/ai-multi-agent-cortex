import json
from pathlib import Path

from trainer.app import pipeline
from trainer.app import runs
from trainer.app.evaluator import score
from trainer.app.backends.base import FusionConfig, TrainingConfig
from trainer.app.backends.mlx import MlxLoraBackend, MlxQLoraBackend
from trainer.app.backends.state import AdapterMetadata
from trainer.app.exporters.gguf import sanitize_fused_tokenizer


def _config(tmp_path: Path, *, training_model: str, resume: bool = False):
    return TrainingConfig(
        python="python",
        source_model="vendor/source-model",
        training_model=training_model,
        data_dir=tmp_path / "data",
        adapters_dir=tmp_path / "adapters",
        artifacts_dir=tmp_path,
        iters=600,
        batch_size=4,
        learning_rate=1e-4,
        resume=resume,
    )


def test_mlx_lora_preserves_the_source_model_and_builds_a_resume_command(tmp_path):
    backend = MlxLoraBackend()
    assert backend.training_model("vendor/model", tmp_path) == "vendor/model"

    step = backend.command_steps(
        _config(tmp_path, training_model="vendor/model", resume=True)
    )[0]
    assert step.phase == "training"
    assert step.argv[step.argv.index("--fine-tune-type") + 1] == "lora"
    assert "--resume-adapter-file" in step.argv
    assert step.argv[step.argv.index("--save-every") + 1] == "50"


def test_mlx_qlora_uses_a_stable_cached_four_bit_checkpoint(tmp_path):
    backend = MlxQLoraBackend()
    first = backend.training_model("vendor/model", tmp_path)
    second = backend.training_model("vendor/model", tmp_path)
    assert first == second
    assert first.endswith("-4bit")

    prepare, train = backend.command_steps(_config(tmp_path, training_model=first))
    assert prepare.phase == "preparing"
    assert "--quantize" in prepare.argv
    assert prepare.argv[prepare.argv.index("--q-bits") + 1] == "4"
    assert train.phase == "training"
    assert train.argv[train.argv.index("--model") + 1] == first

    fusion = backend.fusion_steps(
        FusionConfig(
            python="python",
            training_model=first,
            adapters_dir=tmp_path / "adapters",
            fused_dir=tmp_path / "fused",
        )
    )[0]
    assert fusion.phase == "fusing"
    assert "--dequantize" in fusion.argv


def test_estimates_scale_with_iterations_and_batch_size():
    backend = MlxLoraBackend()
    base = backend.estimate_seconds(iters=100, batch_size=4, needs_prepare=False)
    assert backend.estimate_seconds(iters=200, batch_size=4, needs_prepare=False) > base
    assert backend.estimate_seconds(iters=100, batch_size=8, needs_prepare=False) > base


def test_stop_is_remembered_between_training_subprocesses(monkeypatch):
    monkeypatch.setattr(pipeline, "_status", {"phase": "preparing", "job": "train"})
    monkeypatch.setattr(pipeline, "_proc", None)
    monkeypatch.setattr(pipeline, "_stop_requested", False)

    assert pipeline.stop() is True
    assert pipeline._stop_requested is True


def test_adapter_metadata_loads_legacy_markers_and_round_trips(tmp_path):
    adapters = tmp_path / "adapters"
    adapters.mkdir()
    (adapters / "base_model.txt").write_text("legacy/model")

    legacy = AdapterMetadata.load(adapters, "fallback/model")
    assert legacy.backend_id == "mlx-lora"
    assert legacy.source_model == "legacy/model"
    assert legacy.training_model == "legacy/model"

    current = AdapterMetadata(
        backend_id="mlx-qlora-4bit",
        source_model="vendor/source",
        training_model="/artifacts/source-4bit",
        run_id="run-1",
    )
    current.write(adapters)
    assert AdapterMetadata.load(adapters, "fallback/model") == current


def test_run_records_are_updated_atomically_and_recovered(tmp_path):
    directory = tmp_path / "runs"
    runs.save(directory, {"run_id": "one", "phase": "training", "started_at": 1})
    runs.save(directory, {"run_id": "one", "phase": "trained", "best_val_loss": 0.2})
    runs.save(directory, {"run_id": "two", "phase": "training", "started_at": 2})

    runs.recover(directory)
    records = runs.list_all(directory)
    assert [record["run_id"] for record in records] == ["two", "one"]
    assert records[0]["phase"] == "interrupted"
    assert records[1]["best_val_loss"] == 0.2


def test_run_history_calibrates_estimates(tmp_path):
    directory = tmp_path / "runs"
    runs.save(
        directory,
        {
            "run_id": "one",
            "phase": "trained",
            "backend_id": "mlx-lora",
            "base_model": "vendor/model",
            "elapsed_seconds": 100,
            "iter": 100,
            "batch_size": 4,
            "selected_checkpoint": "adapters.safetensors",
        },
    )

    assert runs.estimate_seconds(directory, "mlx-lora", "vendor/model", 200, 4) == (
        220,
        1,
    )


def test_best_checkpoint_replaces_the_final_adapter(tmp_path):
    adapters = tmp_path / "adapters"
    adapters.mkdir()
    (adapters / "adapters.safetensors").write_text("final")
    (adapters / "0000050_adapters.safetensors").write_text("best")

    selected = pipeline._select_best_checkpoint(adapters, 50)
    assert selected == "0000050_adapters.safetensors"
    assert (adapters / "adapters.safetensors").read_text() == "best"


def test_early_stopping_tracks_the_best_validation(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "_status",
        {
            "val_history": [],
            "best_val_loss": None,
            "best_iter": None,
            "stale_evals": 0,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.01,
        },
    )

    assert pipeline._record_validation(50, 0.5) is False
    assert pipeline._record_validation(100, 0.495) is False
    assert pipeline._record_validation(150, 0.51) is True
    assert pipeline._status["best_iter"] == 50


def test_evaluation_scores_facts_and_refusals():
    assert score("How much power?", "It draws 35 W.", "Power draw is 35 W.") == 1
    assert score("Unknown GPU?", "It isn't in my dataset; I'd rather not guess.", "I don't know.") == 1


def test_gguf_export_sanitizes_out_of_range_tokenizer_entries(tmp_path):
    fused = tmp_path / "fused"
    fused.mkdir()
    (fused / "config.json").write_text(json.dumps({"vocab_size": 3}))
    (fused / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 1, "content": "kept"},
                    {"id": 3, "content": "dropped"},
                ]
            }
        )
    )
    (fused / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    "1": {"content": "kept"},
                    "3": {"content": "dropped"},
                },
                "special_token": "dropped",
            }
        )
    )

    assert sanitize_fused_tokenizer(fused) == ["dropped"]
    tokenizer = json.loads((fused / "tokenizer.json").read_text())
    tokenizer_config = json.loads((fused / "tokenizer_config.json").read_text())
    assert tokenizer["added_tokens"] == [{"id": 1, "content": "kept"}]
    assert tokenizer_config["added_tokens_decoder"] == {
        "1": {"content": "kept"}
    }
    assert "special_token" not in tokenizer_config
