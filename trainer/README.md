# Cortex Trainer

Host-side fine-tuning service. LoRA- or QLoRA-trains a small LLM (default:
Gemma 3 1B) on the selected domain datasets with
[MLX](https://github.com/ml-explore/mlx-lm), fuses the
adapters, converts the result to GGUF, and drops it into `<repo>/models/` where
the Dockerized `ai/` llama.cpp service serves it.

**Runs on the host, never in Docker**: MLX requires the Apple Silicon GPU.

## One-time setup

```bash
bash trainer/setup.sh          # vendors llama.cpp (convert_hf_to_gguf.py)
```

## Run the service

```bash
cd trainer
uv run uvicorn app.main:app --host 0.0.0.0 --port 8200
```

Restarting? Use the helper, it frees port 8200 by killing **only** the old
trainer's listener (never Docker), then starts uvicorn:

```bash
chmod +x restart.sh   # once
./restart.sh          # add --reload for auto-restart on code changes
```

The admin UI (Admin → Fine-Tuning) drives everything from there:
generate dataset → train (live loss) → convert & register.

## Pipeline (what the buttons do)

1. `POST /admin/dataset/generate`, expands `data/facts.yaml` into
   `data/train.jsonl` + `data/valid.jsonl` (deterministic, seeded).
2. `GET /admin/capabilities` reports host resources and available backends.
   `POST /admin/estimate` provides a live estimate for the selected base,
   algorithm, iterations, and batch size.
3. `POST /admin/train` runs the selected `backend_id`:
   - `mlx-lora`, the existing 16-bit LoRA path.
   - `mlx-qlora-4bit`, a cached one-time 4-bit conversion followed by LoRA
     adapter training with lower unified-memory requirements.
   Progress and measured duration are returned by `GET /admin/progress`.
4. `POST /admin/convert`, `mlx_lm fuse` (adapters → HF safetensors), then
   llama.cpp `convert_hf_to_gguf.py --outtype q8_0` → `<repo>/models/<name>.gguf`.
5. The UI then imports it into the ai service (`POST /admin/import-local`) and
   registers it in the model registry with the **`finetuned-` model_id prefix**, 
   that prefix is how the cortex `specialist` agent discovers the model.

## Code ownership

The trainer keeps orchestration separate from platform implementations:

```text
trainer/
  app/
    backends/
      base.py          backend contracts and command-step data
      registry.py      backend registration and lookup
      capabilities.py host OS, architecture, memory, disk, and GPU discovery
      state.py         backward-compatible adapter/backend metadata markers
      mlx.py           MLX LoRA, QLoRA preparation, training, and fusion commands
    exporters/
      gguf.py          tokenizer sanitation and generic llama.cpp GGUF conversion
    pipeline.py        single-job orchestration, progress, logs, and cancellation
  helpers/
    setup.sh           llama.cpp converter setup implementation
    restart.sh         scoped host-trainer restart implementation
  setup.sh             compatibility entry point
  restart.sh           compatibility entry point
```

Dataset generation, source ingestion, domain management, scraping, and research
remain outside `backends/` because they are independent of the training runtime.
Existing adapter directories without the newer backend marker continue to load
as legacy `mlx-lora` artifacts.

## Config

Env vars with `TRAINER_` prefix override `app/config.py` defaults, e.g.
`TRAINER_BASE_MODEL=google/gemma-4-e2b-it` for the bigger Gemma 4 base.
`TRAINER_HOST_ID`, `TRAINER_HOST_LABEL`, and `TRAINER_DEFAULT_BACKEND` control
the capability identity and initial algorithm selection.
