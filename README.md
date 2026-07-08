# AI Multi-Agent Cortex

A production-shaped, general-purpose AI assistant built as a multi-agent
system on top of [LangGraph](https://langchain-ai.github.io/langgraph/).
Cortex answers anything: factual lookups, math and code reasoning, small
talk, image generation, and questions in any **domain you train it on**
(hardware ships ready to use) from its own **self-trained local model**, while
keeping the three trust pillars that
distinguish a real product from a demo:

1. **Observability**: every model and tool call is captured as a span
   in [Langfuse](https://langfuse.com/) via OpenTelemetry.
2. **Evaluation**: golden-dataset tests run as `pytest` files using
   [DeepEval](https://github.com/confident-ai/deepeval) and
   [RAGAS](https://github.com/explodinggradients/ragas) primitives.
3. **Guardrails**: PII redaction, an image-safety gate, tool allowlists,
   and human-in-the-loop interrupts are wired in as `langchain` middleware.

Everything is driveable from the UI: model/provider management, local-model
downloads, the fine-tuning pipeline, **tool & MCP-server control**, and **agent
editing (system prompts, tool access, and custom agents)** all live in the
`/admin` console; no CLI steps are required for day-to-day use.

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
│ Custom durable server (:2024) - cortex graph                          │
│ START ─▶ route ─┬─ specialist  (fine-tuned local model - bypass)      │
│                 └─ router ─┬─ generalist     ───────────────▶ END     │
│                            ├─ prompt_cacher ──────────────────▶ END   │
│                            ├─ imagegen      ──────────────────▶ END   │
│                            ├─ shopping      ──────────────────▶ END   │
│                            ├─ booking       ──────────────────▶ END   │
│                            ├─ custom_agent  ──────────────────▶ END   │
│                            ├─ researcher ─┐                           │
│                            ├─ reasoner   ─┼─▶ synthesize ──────▶ END │
│                            ├─ coder      ─┘                           │
│                            └─ specialist ▶ spec_review ─▶ synthesize │
│                                (untrained/wrong ▶ researcher web-RAG) │
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
| `router`        | Classifies user intent into one of nine capability types        | none (structured output only)                                             |
| `generalist`    | Default chat agent, greetings, opinions, creative tasks         | `get_current_time`, memory                                                |
| `researcher`    | Factual questions, grounded answers with citations               | `search_knowledge_base`, `wikipedia_search`, `web_search`, `fetch_url`, `crypto_price`, memory |
| `reasoner`      | Math, logic puzzles, step-by-step problem solving                | `calculator`, memory                                                      |
| `coder`         | Writing, explaining, reviewing, refactoring, and debugging code  | `web_search`, `fetch_url`, memory                                         |
| `prompt_cacher` | LLM prompt-caching expert (large stable system prompt)           | none (large prompt demonstrates caching savings)                          |
| `specialist`    | Specs & knowledge in any **trained domain** (hardware built-in; add your own) from a **self-trained** model (silent draft) | none; `spec_review` emits the answer, self-critiques (heuristics + LLM fact-check) and, on a gap, hands off to `researcher` web-RAG, logging it |
| `imagegen`      | Generates images behind a two-layer safety gate                  | none; calls Google / OpenAI image APIs directly                          |
| `shopping`      | Product shopping: direct product-page links with live price & in-stock, region-aware, rendered as cards | `product_prices`, `web_search`, `fetch_url`, `search_memories`          |
| `booking`       | Booking: flights, hotels, movies, concerts, events, shows; dated deep-link cards | `find_bookings`, `web_search`, `fetch_url`, `search_memories`             |
| `synthesizer`   | Deterministic spec/comparison tables (via `render_spec_table`) + formatting pass over factual answers (worked math, grounding) | none (renders in-node) |

> Every agent's **system prompt and tool access** is editable from **Admin →
> Agents**, and you can create **custom agents** there, they auto-route via the
> router by their description, with no restart and no code.

### Routing

The router emits a structured `RouterIntent` (via provider strategies) with
one of nine labels, each mapped to a node:

- `general_chat` → `generalist`
- `knowledge_query` → `researcher`
- `reasoning_task` → `reasoner`
- `coding_task` → `coder`
- `prompt_caching` → `prompt_cacher`
- `product_specs` → `specialist`
- `image_generation` → `imagegen`
- `shopping` → `shopping`
- `booking` → `booking`

Unknown labels fall back to `general_chat`. To keep everyday questions fast,
well-known **stable** facts (e.g. "who founded OpenAI?") are answered directly
by the `generalist` from the model's own knowledge; `knowledge_query` (the
web-grounded `researcher`) is reserved for **current, niche, or source-cited**
information. If the routing model itself is
unavailable (e.g. a small local model that can't emit structured output), a
keyword heuristic classifies the turn so the run never fails. The router also
emits an optional `agent` field: when a **custom agent** (Admin → Agents) best
fits the message, the turn routes to the generic `custom_agent` node that runs
it, custom agents register live, with no graph rebuild.

A deterministic guard runs **before** the classifier is trusted: if the message
names an entity in **any trained domain**
([`cortex/facts.py`](cortex/facts.py) `match_products`, alias-aware with
longest-match-wins over every domain's facts), the router forces `product_specs`
so a **trained** entity always reaches the fine-tuned `specialist` rather than
being sent to web search.

The `specialist` never speaks to the user directly, it produces a **silent
draft**, and a `spec_review` step emits the single visible answer. `spec_review`
runs two things **in parallel** (so the table appears instantly, not after two
serial LLM calls): a **self-critique** and the **table extraction**. The
critique is cheap heuristics (refusal phrases, a product not in the training
facts) plus a strict **LLM fact-check** that flags confidently-wrong answers, 
wrong manufacturer, an impossible architecture (e.g. "CUDA cores" on an Apple
chip), or an implausible figure. If it finds a gap it **logs it** and hands off
to the `researcher`, which re-answers with **live web-RAG** so the user gets a
correct, sourced answer instead of the wrong one. Otherwise `spec_review` emits
the deterministically-rendered spec table.

The `researcher`, `reasoner`, and `coder` answers pass through the `synthesize`
node (the specialist's table is already rendered in `spec_review`). For
**product, hardware, or software spec and comparison** answers it renders the
table **deterministically**: a model only *extracts* the structured data
(columns + rows, copied verbatim) and code renders the markdown through the
`render_spec_table` tool, so the answer always comes out as a valid table
instead of relying on the model to format one (a guard blocks any invented
number, and it falls back to prose only if nothing tabular could be extracted).
Other factual answers get a lighter presentation pass that grounds drifted
numbers against the authoritative spec YAMLs, and it **skips the reformat when
the answer is already a table**. For `coder` answers it never lets the fast
model touch the code, instead it runs a deterministic, parse-only syntax check
(Python via `ast`, JSON via `json`) and appends a heads-up when a complete code
block is broken.

### Auto mode

The chat UI's default selection is **✨ Auto**, which sends the sentinel
`model_id: "auto"` to the graph. The router classifies the message and each
node resolves the best model for its intent from
[`cortex/declarative/auto_mode.yaml`](cortex/declarative/auto_mode.yaml)
(profiles `balanced` / `quality` / `cost`; the active profile is stored in
`app_settings` and switched from Admin → Models). Only models that are
enabled **and whose provider has an API key** are eligible, so the admin
console stays in control. The routing chip in the transcript shows which
model auto-mode picked.

If the chosen model becomes unavailable mid-run (quota exhausted, rate-limited,
or the provider is down), auto mode **falls back to the next eligible
candidate** for that intent automatically. When every candidate fails, or a
**specific** model you picked fails, the run doesn't crash: the agent replies
with a short, plain explanation ("out of quota / credits", "rate-limited",
"invalid API key", …) so you can switch models or retry. A specific model (or a
local endpoint) is honored end-to-end and never silently swapped for another
provider.

### Modes, selection & activity

The chip in the prompt box is a full model selector:

- **✨ Auto** (default), the per-intent resolution described above.
- **A specific registered model**, used for the entire turn, including the
  internal formatting / clarify passes, so your choice is never swapped.
- **Local LLM**, point the assistant at your own OpenAI-compatible endpoint
  (base URL + optional key + model name) without touching the registry.

An **Options** menu beside it carries a mode slider and a few toggles:

- **General / Thinking / Research**:
  - *Thinking* raises the reasoner to the `quality` tier and turns on the
    provider's **extended thinking** (Anthropic `thinking`, translated to the
    adaptive format on newer Claude models; OpenAI `reasoning_effort=high`).
  - *Research* runs a **deep-research** flow: the assistant first asks a few
    clarifying questions, then searches several sources and writes a
    structured, **cited** report.
- **Unrestricted mode** (amber, opt-in), see [Guardrails](#guardrails).
- **Hide Tool Calls**, collapse tool activity in the transcript.

While a turn is in flight the transcript streams a live, structured **activity
trace** of what each agent is doing and thinking (routing → tool calls →
reviewing results → thinking). Once the answer is ready the trace collapses
into a compact, muted **"Thought process"** dropdown (click to expand) the way
Claude and ChatGPT fold reasoning away, so the final answer stays front and
centre. Commerce cards, images, and the answer text always stay visible.

### Tools, MCP & agents (admin-managed)

The agent layer itself is configurable from the console, no code, no restart:

- **Tools & MCP** (`/admin` → Tools), enable/disable the built-in tools, add
  prebuilt **LangChain** tools from a catalog (Wikipedia, arXiv, PubMed, Stack
  Exchange, Tavily), and register external **MCP servers** (HTTP/stdio) whose
  tools become grantable to agents. Deleting a tool clears it from every agent
  and (for built-ins) stops it being re-seeded.
- **Agents** (`/admin` → Agents), edit any agent's **system prompt** and
  **tool access**, reset built-ins to their packaged defaults, and create
  **custom agents** (name + description + prompt + tools) that **auto-route via
  the router** by their description. Custom agents can use built-in, LangChain,
  and MCP tools. Any agent can also be given **subagents**, other agents it
  delegates focused subtasks to on demand (agent-as-tool). A subagent runs in
  isolated context, shares the parent's long-term memory **read-only** (it can
  recall but never `save_memory`), and uses its own granted built-in/LangChain/
  MCP tools; delegation is one level deep (no recursion).

DB-backed prompt and tool overrides win over the packaged YAML per agent and
apply on the next message; transient provider errors (e.g. Anthropic
"overloaded") are retried automatically.

### Web search, shopping & booking

Every agent is given **today's date** and the user's **region** (derived from
the browser locale/timezone) so date- and location-sensitive answers are
correct by default.

- **Web search & scraping**: `web_search` and `fetch_url` use a real search
  API when a key is set: **Firecrawl** (recommended; also powers JS/anti-bot
  page scraping), or **Brave** / **SerpAPI** / **Tavily**. Without a key they
  fall back to a best-effort DuckDuckGo scrape, which is often blocked, so set
  `FIRECRAWL_API_KEY` (or one of the others) in `.env` for real results.
- **Shopping**: `product_prices` looks the product up live and returns the
  **actual product page** on each regional retailer with the **price** and an
  **in-stock** hint (region's own stores first, cheapest first), rendered as
  product cards with an *In stock* / *Out of stock* badge. Without a search key
  it falls back to per-retailer search links.
- **Booking**: `find_bookings` builds **dated deep links** into each platform's
  live results (Google Flights, Skyscanner, KAYAK, MakeMyTrip, Cleartrip for
  flights; Booking.com / Google Hotels; Ticketmaster / SeatGeek / BookMyShow for
  events), rendered as booking cards. It never completes a purchase.

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

1. **Providers**: add an OpenAI, Azure, Anthropic, Google, or local
   provider and paste its API key.
2. **Models**: register the models you want and pick the active auto-mode
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
containerized is the fine-tuning [`trainer`](#the-self-trained-domain-specialist)
, it needs Apple-Silicon MLX and runs on the host.

### Services

| Service     | Host port | Build                         | Role                                                               |
| ----------- | --------- | ----------------------------- | ------------------------------------------------------------------ |
| `db`        | 5432      | `pgvector/pgvector:pg16`      | Postgres, app registry/KB **and** durable graph state (see below) |
| `langgraph` | 2024      | `docker/Dockerfile.langgraph` | Custom durable LangGraph server ([`cortex/server`](cortex/server)) |
| `ui`        | 3000      | `docker/Dockerfile.ui`        | `agent-chat-ui` Next.js front-end + `/admin` console               |
| `ai`        | 8100      | `ai/Dockerfile`               | llama.cpp GGUF server for local / fine-tuned models (reads `./models`) |
| `mcp`       | 8811      | `docker/Dockerfile.langgraph` | FastMCP server exposing the stateless tools to external MCP clients |
| `trainer` † | 8200      | host, `trainer/` (not Docker) | MLX LoRA fine-tuning service; writes fine-tuned GGUFs into `./models` |
| `langfuse-*`| 4000      | `langfuse/*` (profile `observability`) | Tracing UI + worker (own Postgres/ClickHouse/Redis/MinIO), captures every model & tool span |
| `evals`     | n/a       | `docker/Dockerfile.evals` (profile `evals`) | One-shot runner for the pytest eval suites (faithfulness / routing / security) |

† The **`trainer`** is the one component that is *not* containerized, MLX needs
the Apple GPU, so it runs on the host and reaches the stack over
`host.docker.internal`. See [the specialist](#the-self-trained-domain-specialist).

At a glance, each service owns one concern:

- **`db`**: the single source of truth. Provider/model registry, tools & agents
  config, knowledge base + embeddings, knowledge gaps, app settings, **and** the
  durable graph state (threads, checkpoints, long-term memory) all live here;
  it's the one volume you must not lose.
- **`langgraph`**: the brain. Compiles and runs the multi-agent graph and serves
  the chat API the UI talks to.
- **`ui`**: the only public face. The chat experience plus the `/admin` console;
  it also proxies admin calls to `ai` and to the host `trainer`.
- **`ai`**: the local-inference engine. Serves GGUF models (downloaded or
  fine-tuned) over an OpenAI-compatible API. Its models live in the **`./models`
  host bind mount**, so imported/fine-tuned GGUFs and the `catalog.json` registry
  survive `ai` restarts *and* image rebuilds, only `docker compose down -v` or
  deleting the files removes them.
- **`mcp`**: re-exposes the stateless tools to external MCP clients (below).

The **`mcp`** service is an *additive* [FastMCP](https://modelcontextprotocol.io)
server ([`cortex/tools/mcp.py`](cortex/tools/mcp.py)) that re-exposes Cortex's
**stateless** tools (web search, page fetch, Wikipedia, crypto, product prices,
booking search, time, calculator) over MCP for external
clients (Claude Desktop, IDEs, other agents). The chat graph uses the **same**
tools in-process, so this server is never in the assistant's critical path, it
adds no latency and its downtime can't affect the app. Stateful tools (memory,
knowledge base) that need the runtime store / DB session stay in-process only.

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
dev` runtime, **no LangSmith license and no Redis required.**

- **Durable state**: the graph is compiled with an `AsyncPostgresSaver`
  checkpointer and an `AsyncPostgresStore`, so chat threads, checkpoints, and
  long-term semantic memory all persist in Postgres (`db`). The server opens
  **three dedicated connection pools** (checkpointer / store / app-threads) so
  the parallel work in `spec_review` can't collide on one pinned connection
  (psycopg "another command is already in progress"). Conversations survive
  container restarts, image rebuilds, and version upgrades.
- **Ports**: the server binds `8000` inside the container and is published as
  `2024:8000`. The UI container reaches it at `http://langgraph:8000` over the
  compose network; host tooling and the eval suite use `http://localhost:2024`.
- **Startup order**: the FastAPI lifespan opens a psycopg pool, runs the
  checkpointer/store migrations, and compiles the graph before serving. It
  waits for `db` to be healthy and exposes a `/ok` health check that `ui`
  gates on (`depends_on: condition: service_healthy`).
- **Schema**: the saver/store tables (`checkpoints`, `checkpoint_blobs`,
  `checkpoint_writes`, `store`, `store_vectors`) plus a small `threads`
  metadata table are created automatically in the same `cortex` database as
  the app tables, no manual migration step.

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
docker compose down -v         # also drop the pgdata volume, wipes the
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
| `FIRECRAWL_API_KEY`                 | Real web search + page scraping for `web_search` / `fetch_url` **and** the trainer's topic + gap research (or `BRAVE_API_KEY` / `SERPAPI_API_KEY` / `TAVILY_API_KEY`) |
| `LANGFUSE_*`                        | Optional Langfuse tracing (observability profile)  |

The `langgraph` container reads `DATABASE_URL` for both the app's SQLAlchemy
code and the durable server (the server normalizes the driver suffix).

---

## The self-trained domain specialist

Cortex ships a **`specialist`** agent backed by a small model (Gemma 3 1B)
fine-tuned on curated data across the **domains you choose**, a built-in
**hardware** domain (gaming consoles, PC, and mobile/laptop processors) ships
ready to train, and you can define your own domains and subdomains from the UI.
It answers from its own weights, and when asked about something outside its
trained domains (or it gets a fact wrong), `spec_review` logs a knowledge gap
and hands off to the `researcher` for a **web-RAG** answer, so the user still
gets an accurate one (see [Routing](#routing)). Its **identity and off-domain
replies are domain-aware**, they reflect whatever you trained on, not a fixed
domain. The whole sources → dataset → train → register loop is driven from
**Admin → Fine-Tuning**:

1. **Manage Domains**: the built-in **hardware** domain is ready to train, or
   create your own **domains → subdomains** (each with its own fields/schema)
   and tick which subdomains to include in the next run. **One model** is
   trained across every selected subdomain.
2. **Sources**: upload PDFs or spec-sheet images, add URLs, or give a
   **research topic** prompt (e.g. "Apple Silicon A- and M-series chip specs").
3. **Import specs**: a **domain-aware smart import** reads every source and
   proposes which domain/subdomain + schema + entities to add for you to
   **review and approve** before anything is written: AMD's DB via a JSON
   parser, uploaded docs via a vision model, and an LLM **scrape-agent** for any
   other URL (it respects robots / anti-bot 403s, never evades them).
   Web/topic search uses the same provider chain as the app
   (`FIRECRAWL_API_KEY` or Brave/SerpAPI/Tavily). Sources become *facts*, not
   invented Q&A.
4. **Generate dataset**: **deterministically** expands the selected domains'
   facts into spec / overview / comparison / buying-advice / off-domain-refusal
   / **domain-aware identity** examples → `train.jsonl` / `valid.jsonl`
   ([`trainer/generate_dataset.py`](trainer/generate_dataset.py)). **View
   dataset** shows the generated pairs so you can eyeball them before training.
5. **Train → Convert & Register**: MLX LoRA fine-tune on the host, fuse,
   export to GGUF, and register it in the `ai` service under the `finetuned-`
   prefix (newest wins).
6. **Knowledge gaps**: questions the specialist got wrong or wasn't trained on
   are logged as gaps (by `spec_review`); "Research gaps" pulls specs from the
   web into `learned_facts.yaml`, and the next dataset → retrain bakes them into
   the model's weights.

The trainer runs **on the host** (MLX needs Apple Silicon), not in Docker:

```bash
cd trainer
bash setup.sh                                   # one-time: vendor llama.cpp
uv run uvicorn app.main:app --host 0.0.0.0 --port 8200
```

The `ai` service also serves any GGUF from the Hugging Face catalog, search,
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

- `evals/test_routing.py`, router classifies intents correctly
- `evals/test_faithfulness.py`, researcher grounds answers in the KB
- `evals/test_security.py`, direct prompt-injection resistance

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

- `ToolAllowlistMiddleware(allowed_tools=...)`, hard-blocks any tool
  call whose name is not on the allowlist, defending against tool-name
  hallucinations.

The image pipeline adds its own **safety gate** in `cortex/imagegen.py`: a
fast LLM pre-flight screens every request and strict provider safety
settings back it up, so unsafe prompts become a polite refusal rather than
a picture.

An opt-in **Unrestricted mode** (the amber toggle in the prompt box) relaxes
the **app-level** guardrails for a turn: it **skips PII redaction**, appends a
"direct answers" directive to the agent, and uses a **relaxed image
pre-screen**. It never disables the providers' own moderation, those limits
still apply.

For deeper coverage of the guardrail design see
[`GUARDRAILS.md`](GUARDRAILS.md).

---

## Project layout

```
ai-multi-agent-cortex/
├── agent-chat-ui/            # Next.js 15 front-end (chat + /admin console)
│   └── src/
│       ├── app/             # routes: chat/, admin/, api/ (LangGraph + admin proxies)
│       ├── components/      # thread UI (agent-trace, agent-activity), model-selector, agent-inbox
│       ├── providers/       # Stream, Thread, ModelSelection, client
│       └── lib/             # db pool, admin-auth, multimodal utils
├── cortex/                   # The LangGraph Python package
│   ├── workflow.py           # Compiled graph: nodes, routing, memory, synthesizer
│   ├── enums.py              # Agents StrEnum
│   ├── guardrails.py         # Opt-in ToolAllowlistMiddleware
│   ├── errors.py             # Model-error classification + graceful fallback messages
│   ├── observability.py      # OpenTelemetry / Langfuse wiring
│   ├── facts.py              # Authoritative specs for synthesizer grounding
│   ├── imagegen.py           # Image generation + two-layer safety gate
│   ├── memory.py             # Store embedding hook + memory namespace
│   ├── config.py             # Settings (Pydantic) + settings.yaml/.env loader
│   ├── db/
│   │   ├── engine.py         # SQLAlchemy session factory
│   │   ├── models/           # LLMProvider, LLMModel, KnowledgeGap, AppSetting, KnowledgeArticle, Tool/MCPServer/AgentTool, Agent/AgentSubagent
│   │   ├── services/         # llm_registry, auto_mode, knowledge_gaps, app_settings, tool_catalog, agents (+ subagents)
│   │   └── seed.py           # Registry + knowledge-base seeder
│   ├── declarative/
│   │   ├── auto_mode.yaml    # Per-intent model candidates (balanced/quality/cost)
│   │   └── agents.yaml       # All agent specs (one --- document per agent)
│   ├── model_client/         # Chat + embedding client factories (+ auto-mode fallback clients)
│   ├── server/               # Custom durable LangGraph server (FastAPI + SSE)
│   ├── scripts/              # one-off maintenance scripts
│   └── tools/                # registry + web/commerce/utility/shared/memory tools;
│                             #   spec_table.py (deterministic spec/comparison tables),
│                             #   catalog.py (prebuilt LangChain tools); mcp.py exposes
│                             #   the stateless ones over FastMCP
├── ai/                       # llama.cpp GGUF server (FastAPI, port 8100)
├── trainer/                  # Host-side MLX LoRA fine-tuning service (port 8200)
│   ├── app/                  # FastAPI: dataset, train, convert, scrape/search, gap research
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

1. **New tool**: add a function in `cortex/tools/`, decorate with
   `@register_tool`, and import the module from `cortex/tools/__init__.py`.
   To add a tool **without code**, enable a prebuilt LangChain tool or register
   an external MCP server in **Admin → Tools**.
2. **New agent**: for a graph-level agent, add a `---` document in
   `cortex/declarative/agents.yaml` with its `name` and `whitelisted_tools`,
   add a member to the `Agents` enum, and a node in `cortex/workflow.py`. For a
   **custom agent with no code**, create one in **Admin → Agents**, it
   auto-routes via the router by its description.
3. **New routing label**: extend `Intent` in `cortex/workflow.py`,
   update `_INTENT_TO_NODE`, add the label to `router.yaml`, and give the
   intent a candidate list in each profile of
   `cortex/declarative/auto_mode.yaml` so auto mode can serve it.
4. **New model / provider**: no code change: add it in **Admin →
   Providers / Models**. Reference the model in `auto_mode.yaml` by its
   `model_id` to fold it into auto mode.

---

## License

See [`LICENSE`](LICENSE).
