"""Apple MLX LoRA and QLoRA backends."""

from __future__ import annotations

import hashlib
import importlib.util
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from .base import BackendDescription, CommandStep, FusionConfig, TrainingConfig


@lru_cache(maxsize=1)
def _mlx_available() -> tuple[bool, str]:
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        return False, "MLX requires an Apple Silicon Mac."
    if importlib.util.find_spec("mlx") is None or importlib.util.find_spec("mlx_lm") is None:
        return False, "Install the trainer dependencies (mlx and mlx-lm)."
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import mlx.core as mx; print(mx.default_device())"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"MLX device probe failed: {exc}"
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        return False, detail[-1] if detail else "No MLX Metal device is available."
    return True, ""


class _MlxBase:
    iters_per_second = 0.5
    dequantize_on_fuse = False

    def available(self) -> tuple[bool, str]:
        return _mlx_available()

    def estimate_seconds(
        self, *, iters: int, batch_size: int, needs_prepare: bool
    ) -> int:
        batch_factor = max(batch_size, 1) / 4
        train_seconds = iters / max(self.iters_per_second / batch_factor, 0.05)
        return max(1, round(train_seconds + (240 if needs_prepare else 0)))

    @staticmethod
    def _train_step(config: TrainingConfig) -> CommandStep:
        argv = [
            config.python,
            "-m",
            "mlx_lm",
            "lora",
            "--model",
            config.training_model,
            "--train",
            "--fine-tune-type",
            "lora",
            "--data",
            str(config.data_dir),
            "--iters",
            str(config.iters),
            "--batch-size",
            str(config.batch_size),
            "--learning-rate",
            str(config.learning_rate),
            "--num-layers",
            "16",
            "--adapter-path",
            str(config.adapters_dir),
            "--steps-per-report",
            "10",
            "--steps-per-eval",
            "50",
            "--save-every",
            "50",
        ]
        if config.resume:
            argv += [
                "--resume-adapter-file",
                str(config.adapters_dir / "adapters.safetensors"),
            ]
        return CommandStep(phase="training", argv=argv)

    def fusion_steps(self, config: FusionConfig) -> list[CommandStep]:
        argv = [
            config.python,
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            config.training_model,
            "--adapter-path",
            str(config.adapters_dir),
            "--save-path",
            str(config.fused_dir),
        ]
        if self.dequantize_on_fuse:
            argv.append("--dequantize")
        return [CommandStep(phase="fusing", argv=argv)]


class MlxLoraBackend(_MlxBase):
    description = BackendDescription(
        id="mlx-lora",
        platform="apple_mlx",
        algorithm="lora",
        label="MLX LoRA",
        quality_tier="Better",
        min_memory_gb=8,
        resume_supported=True,
        description="16-bit LoRA on Apple unified memory; the proven default path.",
    )

    def training_model(self, source_model: str, artifacts_dir: Path) -> str:
        return source_model

    def command_steps(self, config: TrainingConfig) -> list[CommandStep]:
        return [self._train_step(config)]


class MlxQLoraBackend(_MlxBase):
    description = BackendDescription(
        id="mlx-qlora-4bit",
        platform="apple_mlx",
        algorithm="qlora_4bit",
        label="MLX QLoRA 4-bit",
        quality_tier="Good",
        min_memory_gb=6,
        resume_supported=True,
        description="Quantizes the base to 4-bit once, then trains LoRA adapters with lower memory use.",
    )
    iters_per_second = 0.7
    dequantize_on_fuse = True

    def training_model(self, source_model: str, artifacts_dir: Path) -> str:
        digest = hashlib.sha256(source_model.encode()).hexdigest()[:12]
        slug = "".join(c if c.isalnum() else "-" for c in source_model).strip("-")[-48:]
        return str(artifacts_dir / "quantized" / f"{slug or 'model'}-{digest}-4bit")

    def command_steps(self, config: TrainingConfig) -> list[CommandStep]:
        quantized = Path(config.training_model)
        prepare = CommandStep(
            phase="preparing",
            argv=[
                config.python,
                "-m",
                "mlx_lm",
                "convert",
                "--hf-path",
                config.source_model,
                "--mlx-path",
                str(quantized),
                "--quantize",
                "--q-bits",
                "4",
                "--q-group-size",
                "64",
            ],
            skip_if_exists=quantized / "config.json",
        )
        return [prepare, self._train_step(config)]
