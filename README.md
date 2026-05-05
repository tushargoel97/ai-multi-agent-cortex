# AI Multi-Agent Cortex

A production-shaped, general-purpose AI assistant built as a multi-agent
system on top of [LangGraph](https://langchain-ai.github.io/langgraph/).
Cortex is designed to answer anything — from factual lookups to math and
code reasoning to small talk — while keeping the three trust pillars
that distinguish a real product from a demo:

1. **Observability** — every model and tool call is captured as a span
   in [Langfuse](https://langfuse.com/) via OpenTelemetry.
2. **Evaluation** — golden-dataset tests run as `pytest` files using
   [DeepEval](https://github.com/confident-ai/deepeval) and
   [RAGAS](https://github.com/explodinggradients/ragas) primitives.
3. **Guardrails** — PII redaction, tool allowlists, and human-in-the-loop
   interrupts are wired in as `langchain` middleware.

---

## Architecture

```
┌──────────────┐
│  agent-chat  │  Next.js 15 chat UI (port 3000)
│      UI      │
└──────┬───────┘
       │ LangGraph SDK over HTTP
       ▼
┌──────────────────────────────────────────────────────────────┐
│  LangGraph runtime (port 2024)                               │
│                                                              │
│   START → router → ┬── generalist     (chit-chat, identity)  │
│                    ├── researcher     (KB + Wikipedia)       │
│                    ├── reasoner       (math, logic, code)    │
│                    └── prompt_cacher  (LLM caching expert)   │
│                  → END                                       │
│                                                              │
│   Middleware: PII redaction (credit_card, email)             │
│   Persistence: PostgreSQL checkpointer (provided by runtime) │
└──────┬───────────────────────────────────────────────────┬───┘
       │                                                   │
       ▼                                                   ▼
┌──────────────┐                                   ┌───────────────┐
│ pgvector pg  │  Knowledge base (vector + SQL)    │   Langfuse    │
│   (port 5432)│                                   │  (port 4000)  │
└──────────────┘                                   └───────────────┘
```

### Agents

| Agent           | Purpose                                                  | Tools                                            |
| --------------- | -------------------------------------------------------- | ------------------------------------------------ |
| `router`        | Classifies user intent into one of four capability types | none (structured output only)                    |
| `generalist`    | Default chat agent — greetings, opinions, creative tasks | `get_current_time`                               |
| `researcher`    | Factual questions, grounded answers with citations       | `search_knowledge_base`, `wikipedia_search`      |
| `reasoner`      | Math, logic puzzles, step-by-step problem solving        | `calculator`                                     |
| `prompt_cacher` | LLM prompt-caching expert (large stable system prompt)   | none (large prompt demonstrates caching savings) |

### Routing

The router emits a structured `RouterIntent` (via OpenAI/Anthropic
provider strategies) with one of four labels:

- `general_chat` → `generalist`
- `knowledge_query` → `researcher`
- `reasoning_task` → `reasoner`
- `prompt_caching` → `prompt_cacher`

Unknown labels fall back to `general_chat`.

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
docker compose up -d db langgraph ui
```

This brings up:

| Service     | Port | What                                          |
| ----------- | ---- | --------------------------------------------- |
| `db`        | 5432 | pgvector-enabled PostgreSQL                   |
| `langgraph` | 2024 | LangGraph runtime serving `cortex` graph      |
| `ui`        | 3000 | `agent-chat-ui` Next.js front-end             |

To also start the Langfuse observability stack:

```bash
docker compose up -d
```

### 3. Seed the knowledge base

```bash
# tables + a small curated knowledge corpus (no embeddings)
docker compose exec langgraph uv run python -m cortex.db.seed

# also generate embeddings (requires a real OPENAI_API_KEY)
docker compose exec langgraph uv run python -m cortex.db.seed --embeddings
```

### 4. Open the chat UI

Visit <http://localhost:3000> and start a conversation. The router will
dispatch each turn to the correct specialist.

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

For deeper coverage of the guardrail design see
[`GUARDRAILS.md`](GUARDRAILS.md).

---

## Project layout

```
ai-multi-agent-cortex/
├── agent-chat-ui/           # Next.js front-end (LangGraph SDK client)
├── cortex/                  # The Python package
│   ├── workflow.py          # LangGraph compiled graph (entrypoint)
│   ├── enums.py             # Agents StrEnum
│   ├── guardrails.py        # Optional custom middleware
│   ├── observability.py     # OpenTelemetry / Langfuse wiring
│   ├── config/              # Settings (Pydantic) + YAML loader
│   ├── db/
│   │   ├── engine.py        # SQLAlchemy session factory
│   │   ├── models/          # KnowledgeArticle ORM
│   │   └── seed.py          # Knowledge-base seeder
│   ├── declarative/
│   │   └── agents/          # YAML agent specs
│   │       ├── router.yaml
│   │       ├── generalist.yaml
│   │       ├── researcher.yaml
│   │       ├── reasoner.yaml
│   │       └── prompt_cacher.yaml
│   ├── model_client/        # Chat + embedding client factories
│   └── tools/
│       ├── registry.py      # @register_tool decorator
│       ├── shared.py        # search_knowledge_base
│       ├── utility.py       # get_current_time, calculator
│       └── web.py           # wikipedia_search
├── evals/                   # pytest-based eval suites
│   ├── conftest.py
│   ├── golden_dataset.json
│   ├── test_routing.py
│   ├── test_faithfulness.py
│   └── test_security.py
├── docker/                  # Dockerfiles for langgraph / ui / evals
├── docker-compose.yml
├── langgraph.json           # LangGraph runtime config (graphs.cortex)
├── settings.yaml            # Settings template (env-var substitution)
└── pyproject.toml           # uv-managed Python project
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
   update `_INTENT_TO_NODE`, and add the label to `router.yaml`.

---

## License

See [`LICENSE`](LICENSE).
