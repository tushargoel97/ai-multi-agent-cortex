# AI Multi-Agent Cortex

A production-shaped, general-purpose AI assistant built as a multi-agent
system on top of [LangGraph](https://langchain-ai.github.io/langgraph/).
Cortex answers anything — factual lookups, math and code reasoning, small
talk, image generation, and gaming/PC-hardware questions from its own
**self-trained local model** — while keeping the three trust pillars that
distinguish a real product from a demo:

1. **Observability** — every model and tool call is captured as a span
   in [Langfuse](https://langfuse.com/) via OpenTelemetry.
2. **Evaluation** — golden-dataset tests run as `pytest` files using
   [DeepEval](https://github.com/confident-ai/deepeval) and
   [RAGAS](https://github.com/explodinggradients/ragas) primitives.
3. **Guardrails** — PII redaction, an image-safety gate, tool allowlists,
   and human-in-the-loop interrupts are wired in as `langchain` middleware.

Everything is driveable from the UI — model/provider management, local-model
downloads, and the fine-tuning pipeline all live in the `/admin` console; no
CLI steps are required for day-to-day use.

---

## Architecture

```
┌──────────────┐
│  agent-chat  │  Next.js 15 chat UI + /admin console (port 3000)
│      UI      │
└──────┬───────┘
       │ LangGraph SDK over HTTP
       ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Custom durable server (:2024) — cortex graph                         │
│                                                                       │
│  START ─▶ route ─┬─ specialist  (fine-tuned local model — bypass)     │
│                  └─ router ─┬─ generalist     ───────────────▶ END    │
│                            ├─ prompt_cacher ──────────────────▶ END   │
│                            ├─ imagegen      ──────────────────▶ END   │
│                            ├─ researcher ─┐                           │
│                            ├─ reasoner   ─┤                           │
│                            ├─ specialist ─┼─▶ synthesize ──────▶ END │
│                            └─ coder      ─┘                           │
│                                                                       │
│  Guardrails: PII redaction • image safety gate • tool allowlist       │
│  Memory: rolling summary (short-term) + semantic store (long-term)    │
└───┬──────────────┬───────────────────┬────────────────────┬───────────┘
    ▼              ▼                   ▼                    ▼
┌─────────┐  ┌────────────┐     ┌────────────┐      ┌───────────┐
│pgvector │  │ ai service │     │  trainer   │      │  Langfuse │
│  :5432  │  │   :8100    │     │   :8200    │      │   :4000   │
│registry │  │ llama.cpp  │     │  MLX LoRA  │      │  traces   │
│  + KB   │  │ GGUF serve │     │  (on host) │      │           │
└─────────┘  └────────────┘     └────────────┘      └───────────┘
```

### Agents

| Agent           | Purpose                                                          | Tools                                                                     |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `router`        | Classifies user intent into one of seven capability types        | none (structured output only)                                             |
| `generalist`    | Default chat agent — greetings, opinions, creative tasks         | `get_current_time`, memory                                                |
| `researcher`    | Factual questions, grounded answers with citations               | `search_knowledge_base`, `wikipedia_search`, `web_search`, `fetch_url`, `techpowerup_specs`, `crypto_price`, memory |
| `reasoner`      | Math, logic puzzles, step-by-step problem solving                | `calculator`, memory                                                      |
| `coder`         | Writing, explaining, reviewing, refactoring, and debugging code  | `web_search`, `fetch_url`, memory                                         |
| `prompt_cacher` | LLM prompt-caching expert (large stable system prompt)           | none (large prompt demonstrates caching savings)                          |
| `specialist`    | Gaming-console / PC-hardware specs from a **self-trained** model | none — answers purely from the fine-tuned model's weights                 |
| `imagegen`      | Generates images behind a two-layer safety gate                  | none — calls Google / OpenAI image APIs directly                          |
| `synthesizer`   | Formatting pass over factual answers (tables, worked math) + fact grounding | none                                                           |

### Routing

The router emits a structured `RouterIntent` (via provider strategies) with
one of seven labels, each mapped to a node:

- `general_chat` → `generalist`
- `knowledge_query` → `researcher`
- `reasoning_task` → `reasoner`
- `coding_task` → `coder`
- `prompt_caching` → `prompt_cacher`
- `product_specs` → `specialist`
- `image_generation` → `imagegen`

Unknown labels fall back to `general_chat`. If the routing model itself is
unavailable (e.g. a small local model that can't emit structured output), a
keyword heuristic classifies the turn so the run never fails.

The `researcher`, `reasoner`, `specialist`, and `coder` answers pass through
the `synthesize` node. For factual answers it is a presentation-only pass
(spec tables, worked math, structured research) that grounds drifted numbers
against the authoritative spec YAMLs and rewrites the final message in place.
For `coder` answers it never lets the fast model touch the code — instead it
runs a deterministic, parse-only syntax check (Python via `ast`, JSON via
`json`) and appends a heads-up when a complete code block is broken.

### Auto mode

The chat UI's default selection is **✨ Auto**, which sends the sentinel
`model_id: "auto"` to the graph. The router classifies the message and each
node resolves the best model for its intent from
[`cortex/declarative/auto_mode.yaml`](cortex/declarative/auto_mode.yaml)
(profiles `balanced` / `quality` / `cost`; the active profile is stored in
`app_settings` and switched from Admin → Models). Only models that are
enabled in the registry are eligible, so the admin console stays in control.
The routing chip in the transcript shows which model auto-mode picked.

---

## Quickstart

### 1. Configure environment

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

The default `LLM_PROVIDER=openai` uses `OPENAI_MODEL` (default `gpt-5-nano`).
To use Azure OpenAI, set `LLM_PROVIDER=azure_openai` and fill in the
`AZURE_OPENAI_*` variables instead.

### 2. Start the stack

```bash
# core services: Postgres, the custom LangGraph server, the chat UI, and
# the local model server
docker compose up -d --build db langgraph ui ai
```

The chat UI comes up on <http://localhost:3000> and the LangGraph server on
<http://localhost:2024>. See [Deployment](#deployment) for the full service
breakdown, the persistence model, compose profiles, and rebuild/reset
commands.

### 3. Configure providers and models

Open the admin console at <http://localhost:3000/admin> (log in with
`ADMIN_USERNAME` / `ADMIN_PASSWORD`) and:

1. **Providers** — add an OpenAI, Azure, Anthropic, Google, or local
   provider and paste its API key.
2. **Models** — register the models you want and pick the active auto-mode
   profile. Mark one model as the default.

Prefer a starter registry and knowledge base instead? See
[Deployment → Seed a starter registry](#seed-a-starter-registry--knowledge-base).

### 4. Open the chat UI

Visit <http://localhost:3000> and start a conversation. With **✨ Auto**
selected, the router dispatches each turn to the right specialist and picks
the best model for the job.

---

## Deployment

The whole stack runs in **Docker Compose**. The only component that is *not*
containerized is the fine-tuning [`trainer`](#the-self-trained-hardware-specialist)
— it needs Apple-Silicon MLX and runs on the host.

### Services

| Service     | Host port | Build                         | Role                                                               |
| ----------- | --------- | ----------------------------- | ------------------------------------------------------------------ |
| `db`        | 5432      | `pgvector/pgvector:pg16`      | Postgres — app registry/KB **and** durable graph state (see below) |
| `langgraph` | 2024      | `docker/Dockerfile.langgraph` | Custom durable LangGraph server ([`cortex/server`](cortex/server)) |
| `ui`        | 3000      | `docker/Dockerfile.ui`        | `agent-chat-ui` Next.js front-end + `/admin` console               |
| `ai`        | 8100      | `ai/Dockerfile`               | llama.cpp GGUF server for local / fine-tuned models                |

Optional stacks live behind compose profiles (off by default):

```bash
docker compose --profile observability up -d   # Langfuse tracing UI on :4000
docker compose --profile evals up              # run the eval suites once
```

### Custom LangGraph server & persistence

The `langgraph` service runs a compact, self-hosted FastAPI app
([`cortex/server`](cortex/server)) that implements the subset of the LangGraph
Platform REST + SSE API the chat UI's `useStream` client speaks. It replaces
the licensed LangGraph Platform image and the earlier in-memory `langgraph
dev` runtime — **no LangSmith license and no Redis required.**

- **Durable state** — the graph is compiled with an `AsyncPostgresSaver`
  checkpointer and an `AsyncPostgresStore`, so chat threads, checkpoints, and
  long-term semantic memory all persist in Postgres (`db`). Conversations
  survive container restarts, image rebuilds, and version upgrades.
- **Ports** — the server binds `8000` inside the container and is published as
  `2024:8000`. The UI container reaches it at `http://langgraph:8000` over the
  compose network; host tooling and the eval suite use `http://localhost:2024`.
- **Startup order** — the FastAPI lifespan opens a psycopg pool, runs the
  checkpointer/store migrations, and compiles the graph before serving. It
  waits for `db` to be healthy and exposes a `/ok` health check that `ui`
  gates on (`depends_on: condition: service_healthy`).
- **Schema** — the saver/store tables (`checkpoints`, `checkpoint_blobs`,
  `checkpoint_writes`, `store`, `store_vectors`) plus a small `threads`
  metadata table are created automatically in the same `cortex` database as
  the app tables — no manual migration step.

### Bring the stack up / rebuild

```bash
cp .env.example .env                          # set OPENAI_API_KEY first

docker compose up -d --build db langgraph ui ai
docker compose logs -f langgraph              # wait for "Cortex durable server ready"
```

`--build` is needed on first run and after changing Python or UI source:

```bash
docker compose up -d --build langgraph        # server / graph changes
docker compose up -d --build ui               # UI changes
```

### Seed a starter registry + knowledge base

```bash
# tables + a small curated knowledge corpus (no embeddings)
docker compose exec langgraph /app/.venv/bin/python -m cortex.db.seed

# also generate embeddings (requires a real OpenAI key in the registry/env)
docker compose exec langgraph /app/.venv/bin/python -m cortex.db.seed --embeddings
```

### Stop & reset

```bash
docker compose down            # stop the stack (keeps all data)
docker compose down -v         # also drop the pgdata volume — wipes the
                               # registry, KB, AND all chat threads
```

To reset only the chat/graph state (keeping providers/models/KB), drop the
LangGraph tables while `db` is up:

```bash
docker compose exec db psql -U cortex -d cortex -c \
  "DROP TABLE IF EXISTS checkpoints, checkpoint_blobs, checkpoint_writes, checkpoint_migrations, store, store_vectors, store_migrations, threads CASCADE;"
```

### Environment

Configuration is read from `.env` (see [`.env.example`](.env.example)). Key
variables:

| Variable                            | Purpose                                            |
| ----------------------------------- | -------------------------------------------------- |
| `OPENAI_API_KEY`                    | OpenAI key (default provider)                      |
| `LLM_PROVIDER`                      | `openai` or `azure_openai`                         |
| `DATABASE_URL`                      | Postgres DSN (app + durable server share it)       |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | Admin console login                                |
| `LANGFUSE_*`                        | Optional Langfuse tracing (observability profile)  |

The `langgraph` container reads `DATABASE_URL` for both the app's SQLAlchemy
code and the durable server (the server normalizes the driver suffix).

---

## The self-trained hardware specialist

Cortex ships a **`specialist`** agent backed by a small model (Gemma 3 1B)
fine-tuned on a curated dataset of gaming-console and PC-hardware specs. It
answers purely from its own weights — no RAG, no web — and the whole
train → convert → register loop is driven from **Admin → Fine-Tuning**:

1. **Sources** — upload PDFs/spreadsheets, add URLs, or paste text.
   "Import specs" distills them into the learned-facts store.
2. **Generate dataset** — expands the facts into spec / overview /
   comparison / buying-advice / refusal examples
   ([`trainer/generate_dataset.py`](trainer/generate_dataset.py)).
3. **Train → Convert & Register** — MLX LoRA fine-tune on the host, fuse,
   export to GGUF, and register it in the `ai` service under the
   `finetuned-` prefix (newest wins).
4. **Knowledge gaps** — when the specialist is asked about hardware it
   wasn't trained on, the question is logged as a gap; "Research gaps"
   pulls specs from the web, and the next retrain closes the loop. The
   model never touches the web at answer time.

The trainer runs **on the host** (MLX needs Apple Silicon), not in Docker:

```bash
cd trainer
bash setup.sh                                   # one-time: vendor llama.cpp
uv run uvicorn app.main:app --host 0.0.0.0 --port 8200
```

The `ai` service also serves any GGUF from the Hugging Face catalog — search,
download, and load models from **Admin → Local Models**.

---

## Trust pillars

### Observability

Every node and tool call is exported as an OpenTelemetry span. Configure
Langfuse credentials in `.env`:

```env
LANGFUSE_HOST=http://langfuse-web:3000
LANGFUSE_PUBLIC_KEY=pk-lf-local-public
LANGFUSE_SECRET_KEY=sk-lf-local-secret
```

The Langfuse UI runs at <http://localhost:4000> when the full compose
stack is up. Each conversation appears as a trace with full prompts,
tool arguments, results, latency, and token cost.

### Evaluation

Eval suites are plain `pytest` files in `evals/`:

```bash
uv run pytest evals/ -v
```

Provided suites:

- `evals/test_routing.py` — router classifies intents correctly
- `evals/test_faithfulness.py` — researcher grounds answers in the KB
- `evals/test_security.py` — direct prompt-injection resistance

The shared `evals/conftest.py` provides an `agent_runner` fixture that
hits the running LangGraph API at `http://localhost:2024`. Add new test
cases to `evals/golden_dataset.json`.

### Guardrails

Built-in middleware applied to every specialist agent in
`cortex/workflow.py`:

```python
PIIMiddleware("credit_card", strategy="redact", apply_to_output=True)
PIIMiddleware("email",       strategy="redact", apply_to_output=True)
```

Optional middleware shipped in `cortex/guardrails.py`:

- `ToolAllowlistMiddleware(allowed_tools=...)` — hard-blocks any tool
  call whose name is not on the allowlist, defending against tool-name
  hallucinations.

The image pipeline adds its own **safety gate** in `cortex/imagegen.py`: a
fast LLM pre-flight screens every request and strict provider safety
settings back it up, so unsafe prompts become a polite refusal rather than
a picture.

For deeper coverage of the guardrail design see
[`GUARDRAILS.md`](GUARDRAILS.md).

---

## Project layout

```
ai-multi-agent-cortex/
├── agent-chat-ui/            # Next.js 15 front-end (chat + /admin console)
│   └── src/
│       ├── app/             # routes: chat/, admin/, api/ (LangGraph + admin proxies)
│       ├── components/      # thread UI, model-selector, agent-activity, agent-inbox
│       ├── providers/       # Stream, Thread, ModelSelection, client
│       └── lib/             # db pool, admin-auth, multimodal utils
├── cortex/                   # The LangGraph Python package
│   ├── workflow.py           # Compiled graph: nodes, routing, memory, synthesizer
│   ├── enums.py              # Agents StrEnum
│   ├── guardrails.py         # Opt-in ToolAllowlistMiddleware
│   ├── observability.py      # OpenTelemetry / Langfuse wiring
│   ├── facts.py              # Authoritative specs for synthesizer grounding
│   ├── imagegen.py           # Image generation + two-layer safety gate
│   ├── memory.py             # Store embedding hook + memory namespace
│   ├── config/               # Settings (Pydantic) + YAML loader
│   ├── db/
│   │   ├── engine.py         # SQLAlchemy session factory
│   │   ├── models/           # LLMProvider, LLMModel, KnowledgeGap, AppSetting, KnowledgeArticle
│   │   ├── services/         # llm_registry, auto_mode, knowledge_gaps, app_settings
│   │   └── seed.py           # Registry + knowledge-base seeder
│   ├── declarative/
│   │   ├── auto_mode.yaml    # Per-intent model candidates (balanced/quality/cost)
│   │   └── agents/           # YAML agent specs (router, generalist, researcher,
│   │                       #   reasoner, coder, prompt_cacher, specialist, synthesizer)
│   ├── model_client/         # Chat + embedding client factories
│   ├── server/               # Custom durable LangGraph server (FastAPI + SSE)
│   ├── scripts/              # one-off maintenance scripts
│   └── tools/                # registry + web / utility / shared / memory tools
├── ai/                       # llama.cpp GGUF server (FastAPI, port 8100)
├── trainer/                  # Host-side MLX LoRA fine-tuning service (port 8200)
│   ├── app/                  # FastAPI: dataset, train, convert, scrape, gap research
│   ├── data/                 # facts.yaml + learned_facts.yaml (ground truth)
│   └── generate_dataset.py   # Fine-tune dataset builder
├── evals/                    # pytest-based eval suites
│   ├── conftest.py
│   ├── golden_dataset.json
│   ├── test_routing.py
│   ├── test_faithfulness.py
│   └── test_security.py
├── docker/                   # Dockerfiles for langgraph / ui / evals + init.sql
├── docker-compose.yml
├── langgraph.json            # Graph ref for optional `langgraph dev` debugging
├── settings.yaml             # Settings template (env-var substitution)
└── pyproject.toml            # uv-managed Python project
```

---

## Adding new capabilities

1. **New tool** — add a function in `cortex/tools/`, decorate with
   `@register_tool`, and import the module from
   `cortex/tools/__init__.py`.
2. **New agent** — drop a YAML file in
   `cortex/declarative/agents/<name>.yaml` listing its
   `whitelisted_tools`, then add a member to the `Agents` enum and a
   node in `cortex/workflow.py`.
3. **New routing label** — extend `Intent` in `cortex/workflow.py`,
   update `_INTENT_TO_NODE`, add the label to `router.yaml`, and give the
   intent a candidate list in each profile of
   `cortex/declarative/auto_mode.yaml` so auto mode can serve it.
4. **New model / provider** — no code change: add it in **Admin →
   Providers / Models**. Reference the model in `auto_mode.yaml` by its
   `model_id` to fold it into auto mode.

---

## License

See [`LICENSE`](LICENSE).
