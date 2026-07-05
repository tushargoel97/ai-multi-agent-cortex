# Project Context — ai-multi-agent-cortex

Multi-agent AI assistant (LangGraph) with a self-trained local LLM,
self-improvement loop, auto model selection, image generation, and an admin
console. Everything is driveable from the UI; no CLI steps required.

## Services (docker compose, default stack)

| Service | Port | What |
|---|---|---|
| db (postgres) | 5432 | registry (llm_providers/llm_models), knowledge_gaps, app_settings, thread_backups, RAG articles |
| langgraph | 2024 | `langgraph dev` server running the cortex graph |
| ui (Next.js 15) | 3000 | chat + `/admin` console; API routes proxy admin ops |
| ai (llama.cpp) | 8100 | serves GGUF models incl. the fine-tuned one; `./models` bind mount |
| thread-backup | — | sidecar mirroring threads → Postgres, auto-restores after wipes |
| trainer (HOST, not Docker) | 8200 | MLX LoRA fine-tuning; `cd trainer && uv run uvicorn app.main:app --host 0.0.0.0 --port 8200` |
| langfuse stack | 3001 | behind `--profile observability` (off by default) |

LangSmith tracing is disabled (no key). The graph is compiled from
`cortex/workflow.py` (see `langgraph.json`, incl. semantic store index via
`cortex/memory.py`).

## Graph topology

```
START ─ route_from_start ─┬─ specialist (fine-tuned selected/default, text-only)
                          └─ router (structured-output intent classify, heuristic fallback)
router ─┬ generalist / prompt_cacher / imagegen ──────────────► END
        └ researcher / reasoner / specialist ─► synthesize ──► END
```

- **Auto mode**: UI default `model_id:"auto"` → router uses the fast tier;
  each node resolves per-intent candidates from
  `cortex/declarative/auto_mode.yaml` (profiles balanced/quality/cost; active
  profile in `app_settings`, pills in Admin → Models). Only registry-enabled
  models are eligible. `"finetuned"` keyword = newest `finetuned-*` local model.
- **Specialist** (fine-tuned Gemma 3 1B): NO system prompt (LoRA data is bare
  user/assistant pairs; a system prompt corrupts recall), temperature 0,
  latest question only as plain text. Never receives images
  (`route_by_intent` reroutes image questions to the researcher).
- **Synthesizer** (`synthesize` node): formatting pass (spec tables, worked
  math, structured research answers) + **facts grounding** — products named
  in the question get authoritative specs from `cortex/facts.py` (reads
  `trainer/data/facts.yaml` + `learned_facts.yaml`, bind-mounted read-only at
  `/app/trainer_data`) and the model corrects drifted numbers. A numeric
  guard (`_numbers_preserved`) rejects synthesis that invents/loses numbers.
  Rewrites the final AI message in place (same id) so token badges stay true.
- **Imagegen**: guardrail LLM screen (refuses NSFW etc. before any API call)
  → Google image models → OpenAI gpt-image fallback (`cortex/imagegen.py`),
  candidates from auto_mode.yaml. PNGs → `generated_images/{thread_id}_{ts}.png`,
  served by `/api/images/[name]`.
- **Prompt caching**: Anthropic gets a `cache_control` breakpoint on the
  static system prompt (dynamic memory context after it). UI shows ⚡cached.
- **Memory**: rolling summary in state + LangGraph store semantic recall;
  save_memory/search_memories tools.
- **Anthropic quirk**: Claude 5 adaptive thinking can return empty thinking
  blocks; `_thinking_safe_anthropic_cls` in `llm_registry.py` restores
  `thinking:""` on round-trip (fixes API 400s in tool loops).

## Fine-tune pipeline (all from Admin → Fine-Tuning)

1. **Sources tab**: upload PDFs/Excel, add URLs, add prompts.
   - “Import specs from sources” → trainer `POST /admin/scrape` with all
     sources; dispatcher: amd.com → embedded-JSON DB parser; Intel
     comparison-chart PDFs → regex fast path (`intel_pdf.py`); techpowerup →
     crawler **only if not 403** (usually blocked; skipped gracefully);
     anything else → fetch + LLM distillation
     (`research.distill_products_from_text`). All writes go through
     `research.save_learned_entry` (alias-aware dedupe; never overrides
     curated `facts.yaml`).
2. **Generate dataset** → `trainer/generate_dataset.py`: spec/overview/
   comparison (pairwise, alias cross-products for consoles, 3-way for
   consoles, nearest-neighbor bounding for groups >12), buying advice,
   off-domain refusals, corrections for `exists:false` entries (incl.
   comparisons vs `closest`), learned-facts merge.
3. **Train** (MLX LoRA on host, ~35–50 min) → **Convert** (fuse + tokenizer
   sanitize + GGUF, atomic replace) → **Convert & Register** imports into the
   ai service and registers `finetuned-gemma3-1b-hardware` (model registry
   contract: `finetuned-` prefix under the local provider; newest wins).
4. **Knowledge gaps card**: specialist refusals/mismatches are logged
   (`cortex/db/services/knowledge_gaps.py`, alias-normalized echo check);
   “Research gaps (web)” → learned facts → regen → retrain. Gap statuses:
   new → researched → trained.

Current model: v9 (3,200 iters on 4,959 examples incl. 15 AMD-scraped + 14
Intel-PDF CPUs). `models/finetuned-gemma3-1b-hardware.v8-backup.gguf` is the
previous verified build — delete once v9 is fully validated.

## Researcher web tools

- `techpowerup_specs` tool (cortex/tools/web.py): TPU DB search → spec sheet;
  gracefully reports no-match when the site 403s (it usually does for bots).
- `web_search` front-loads `site:techpowerup.com` DDG results for CPU/GPU
  queries. Plus wikipedia/KB/crypto/fetch_url/memories.

## Verification recipes

- Graph API smoke: `POST :2024/threads` → `POST /threads/{id}/runs/wait`
  with `{"assistant_id":"cortex","input":{...},"config":{"configurable":{"model_id":"auto"}}}`.
- Regression questions: "compare ps5 pro and ps5 slim" (Slim 2023/$450/10.3
  TFLOPS), "compare PS5 vs PS5 Slim vs PS5 Pro" (3-way table), "compare AMD
  Ryzen 3700X and AMD Ryzen 3700" (→ "AMD never released…"), "Compare PS5
  Pro vs Xbox Series X" (Xbox 2020/$499/12.15).
- Image: "Generate an image of a red cube on a beach" (→ PNG via
  gpt-image fallback while Google quota-blocked); NSFW prompt → polite refusal.
- Caching: 2nd turn on a Claude model → `usage_metadata.input_token_details.cache_read > 0`.
- Thread durability: destroy `langgraph_state` volume → sidecar restores.

## Gotchas

- `langgraph dev` pickles are fragile: lib upgrades/graph changes can wipe
  them (hence thread-backup). SIGTERM flush shim in workflow.py (atexit
  fallback when imported off main thread).
- Google free tier: flash-only, ~20 req/day; pro model rows disabled in DB.
- TechPowerUp hard-blocks bots (403, IP-remembered) — opt-in source only.
- Fine-tuned model: retrain regressions are real; always run the regression
  questions after import. Facts grounding catches numeric drift as a net.
- Trainer must run on the host (MLX needs Apple Silicon, not Docker).
- blockbuster: no sync IO in async graph nodes (use `asyncio.to_thread`).
- DB seed for a fresh volume: `docker compose exec langgraph /app/.venv/bin/python -m cortex.db.seed`.
