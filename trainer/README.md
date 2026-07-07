# Cortex Trainer

Host-side fine-tuning service. LoRA-trains a small LLM (default: Gemma 3 1B) on the gaming/PC-hardware
spec dataset with [MLX](https://github.com/ml-explore/mlx-lm), fuses the
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
2. `POST /admin/train`, `python -m mlx_lm lora --model unsloth/gemma-3-1b-it
   --train --data data/ ...` with progress parsed from stdout
   (`GET /admin/progress`).
3. `POST /admin/convert`, `mlx_lm fuse` (adapters → HF safetensors), then
   llama.cpp `convert_hf_to_gguf.py --outtype q8_0` → `<repo>/models/<name>.gguf`.
4. The UI then imports it into the ai service (`POST /admin/import-local`) and
   registers it in the model registry with the **`finetuned-` model_id prefix**, 
   that prefix is how the cortex `specialist` agent discovers the model.

## Config

Env vars with `TRAINER_` prefix override `app/config.py` defaults, e.g.
`TRAINER_BASE_MODEL=google/gemma-4-e2b-it` for the bigger Gemma 4 base.
