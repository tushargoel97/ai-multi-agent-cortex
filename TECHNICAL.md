# Cortex, Technical Reference

Deep technical documentation for **AI Multi-Agent Cortex**, intended for both
human developers and AI coding agents working on the codebase. It is the
authoritative, detailed companion to the high-level [`README.md`](README.md): the
README is the quick-view / getting-started page, this document explains **what
each module does, how the pieces fit together, and which technical problems the
design solves**.

> Guardrail design has its own deep-dive in [`GUARDRAILS.md`](GUARDRAILS.md).

## Table of contents

1. [System overview](#1-system-overview)
2. [Service topology](#2-service-topology)
3. [The custom durable LangGraph server (`cortex/server`)](#3-the-custom-durable-langgraph-server-cortexserver)
4. [Run lifecycle & the background run broker](#4-run-lifecycle--the-background-run-broker)
5. [The graph (`cortex/workflow.py`)](#5-the-graph-cortexworkflowpy)
6. [Agents & routing](#6-agents--routing)
7. [The self-trained specialist & `spec_review`](#7-the-self-trained-specialist--spec_review)
8. [The synthesizer](#8-the-synthesizer)
9. [Model selection, auto mode & providers](#9-model-selection-auto-mode--providers)
10. [Chat modes & extended thinking](#10-chat-modes--extended-thinking)
11. [Tools, MCP & admin-managed agents](#11-tools-mcp--admin-managed-agents)
12. [Web search, shopping & booking](#12-web-search-shopping--booking)
13. [Memory](#13-memory)
14. [Fine-tuning pipeline](#14-fine-tuning-pipeline)
15. [The chat UI (`agent-chat-ui`)](#15-the-chat-ui-agent-chat-ui)
16. [Trust pillars](#16-trust-pillars)
17. [Deployment & operations](#17-deployment--operations)
18. [Environment variables](#18-environment-variables)
19. [Project layout](#19-project-layout)
20. [Extending the system](#20-extending-the-system)
21. [Operational gotchas](#21-operational-gotchas)

---

## 1. System overview

Cortex is a general-purpose, production-shaped AI assistant built as a
**multi-agent graph** on top of [LangGraph](https://langchain-ai.github.io/langgraph/).
It answers factual lookups, math and code reasoning, small talk, image
generation, and questions in any **domain you train it on** (a **hardware**
domain ships ready to use) from its own **self-trained local model**, while
keeping three trust pillars: **observability**, **evaluation**, and
**guardrails**.

```
┌──────────────┐
│  agent-chat  │  Next.js 15 chat UI + /admin console (port 3000)
│      UI      │
└──────┬───────┘
       │ LangGraph SDK over HTTP (REST + SSE)
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
│                            ├─ reasoner   ─┼─▶ synthesize ──────▶ END  │
│                            ├─ coder      ─┘                           │
│                            └─ specialist ▶ spec_review ─▶ synthesize  │
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

**Design rationale.** The whole system is built so that **everything is
driveable from the UI** (model/provider management, local-model downloads, the
fine-tuning pipeline, tool & MCP-server control, and agent editing all live in
the `/admin` console, no CLI for day-to-day use) and so that **all state is
durable in Postgres** with **no proprietary licence and no Redis**. The
LangGraph Platform image and the in-memory `langgraph dev` runtime were both
replaced by a compact self-hosted server (see §3) precisely to remove those
dependencies while keeping thread/checkpoint/memory durability.

---

## 2. Service topology

Everything runs in **Docker Compose** except the fine-tuning `trainer`, which
needs Apple-Silicon MLX and runs on the host.

| Service     | Host port | Build                         | Role                                                               |
| ----------- | --------- | ----------------------------- | ------------------------------------------------------------------ |
| `db`        | 5432      | `pgvector/pgvector:pg16`      | Postgres: app registry/KB **and** durable graph state              |
| `langgraph` | 2024      | `docker/Dockerfile.langgraph` | Custom durable LangGraph server (`cortex/server`)                  |
| `ui`        | 3000      | `docker/Dockerfile.ui`        | `agent-chat-ui` Next.js front-end + `/admin` console               |
| `ai`        | 8100      | `ai/Dockerfile`               | llama.cpp GGUF server for local / fine-tuned models (reads `./models`) |
| `mcp`       | 8811      | `docker/Dockerfile.langgraph` | FastMCP server exposing the stateless tools to external MCP clients |
| `trainer` † | 8200      | host, `trainer/` (not Docker) | MLX LoRA fine-tuning; writes fine-tuned GGUFs into `./models`       |
| `langfuse-*`| 4000      | `langfuse/*` (profile `observability`) | Tracing UI + worker (own Postgres/ClickHouse/Redis/MinIO)  |
| `evals`     | n/a       | `docker/Dockerfile.evals` (profile `evals`) | One-shot runner for the pytest eval suites             |

† The **`trainer`** is the one component that is *not* containerized (MLX needs
the Apple GPU); it runs on the host and reaches the stack over
`host.docker.internal`.

Each service owns exactly one concern:

- **`db`**, the single source of truth. Provider/model registry, tools & agents
  config, knowledge base + embeddings, knowledge gaps, app settings, **and** the
  durable graph state (threads, checkpoints, long-term memory) all live here.
  It is the one volume you must not lose.
- **`langgraph`**, the brain. Compiles and runs the multi-agent graph and serves
  the chat API the UI talks to.
- **`ui`**, the only public face. The chat experience plus the `/admin` console;
  it also proxies admin calls to `ai` and to the host `trainer`.
- **`ai`**, the local-inference engine. Serves GGUF models (downloaded or
  fine-tuned) over an OpenAI-compatible API. Its models live in the **`./models`
  host bind mount**, so imported/fine-tuned GGUFs and the `catalog.json` registry
  survive `ai` restarts *and* image rebuilds; only `docker compose down -v` or
  deleting the files removes them.
- **`mcp`**, re-exposes the stateless tools to external MCP clients.

### The additive MCP server

The **`mcp`** service is an *additive* [FastMCP](https://modelcontextprotocol.io)
server ([`cortex/tools/mcp.py`](cortex/tools/mcp.py)) that re-exposes Cortex's
**stateless** tools (web search, page fetch, Wikipedia, crypto, product prices,
booking search, time, calculator) over MCP for external clients (Claude Desktop,
IDEs, other agents). The chat graph uses the **same** tools in-process, so this
server is never in the assistant's critical path: it adds no latency and its
downtime can't affect the app. Stateful tools (memory, knowledge base) that need
the runtime store / DB session stay in-process only.

Optional stacks live behind compose profiles (off by default):

```bash
docker compose --profile observability up -d   # Langfuse tracing UI on :4000
docker compose --profile evals up              # run the eval suites once
```

---

## 3. The custom durable LangGraph server (`cortex/server`)

The `langgraph` service runs a compact, self-hosted FastAPI app that implements
the subset of the LangGraph Platform REST + SSE API the chat UI's `useStream`
client speaks. **It replaces the licensed LangGraph Platform image and the
earlier in-memory `langgraph dev` runtime, with no LangSmith licence and no
Redis required.**

### 3.1 Module map

| Module | Responsibility |
| --- | --- |
| [`app.py`](cortex/server/app.py) | FastAPI app + `lifespan`: opens the Postgres pools, runs checkpointer/store migrations, compiles the graph, runs startup self-heal, mounts the routers, exposes `/ok` and `/info`. |
| [`runtime.py`](cortex/server/runtime.py) | Process-wide `Runtime` singleton (`pool`, `checkpointer`, `store`, `graph`) populated once at startup and read by handlers. Also `db_uri()`, which normalizes SQLAlchemy driver suffixes psycopg3 doesn't understand, and `ASSISTANT_ID = "cortex"`. Kept in its own module to avoid an `app`↔routers circular import. |
| [`serde.py`](cortex/server/serde.py) | Wire-format helpers that speak the exact serialization the SDK expects. |
| [`assistants.py`](cortex/server/assistants.py) | Minimal single-graph assistant stub (`search`, `get`, `graph`, `schemas`). |
| [`threads.py`](cortex/server/threads.py) | Thread endpoints + Postgres data-access for the `threads` metadata table and checkpoint/state serialization. |
| [`runs.py`](cortex/server/runs.py) | The core SSE run endpoints and the **background run broker** (see §4). |

### 3.2 Startup / lifespan sequence

[`app.py`](cortex/server/app.py) `lifespan` performs, in order:

1. `setup_tracing()` (OpenTelemetry → Langfuse, see §16).
2. Resolves the DSN via `db_uri()` (accepts `POSTGRES_URI` or the app's
   `DATABASE_URL`, normalizing `postgresql+psycopg2`/`+psycopg` suffixes).
3. Opens **three dedicated psycopg3 pools** (`autocommit=True`, `dict_row`):
   - `cp_pool` (max 20), the checkpointer.
   - `store_pool` (max 10), the semantic store.
   - `data_pool` (max 5), the `threads` table + general data access
     (`runtime.pool`).
4. `AsyncPostgresSaver(cp_pool).setup()` runs the checkpointer migrations.
5. Ensures `CREATE EXTENSION IF NOT EXISTS vector` on `store_pool` **before** the
   store's vector migrations (init.sql only covers freshly-initialized volumes,
   so this makes pgvector presence idempotent on any volume).
6. `AsyncPostgresStore(store_pool, index={dims, embed})` `.setup()` runs the
   store migrations; embedding is Cortex's `aembed_texts` with `EMBED_DIMS`.
7. `THREADS_DDL` creates the `threads` table + indexes on `data_pool`.
8. Populates the `runtime` singleton and compiles the graph:
   `runtime.graph = build_workflow(checkpointer=checkpointer, store=store)`.
9. **`reset_stale_runs()`**, clears the `busy` flag on any thread whose run
   didn't finish before a restart (crash recovery, see §4).
10. `publish_tool_catalog()` + `refresh_dynamic_tools()`, mirror built-in tools
    into the DB and load external tools (LangChain catalog + MCP servers).
11. `publish_agents()`, mirror the packaged agent specs into the DB.
12. An **idempotent em-dash self-heal**: rewrites any lingering `U+2014` in
    admin-facing text columns (tool/agent/model/provider/KB descriptions and
    prompts) to a comma, touching only rows that still contain one.
13. Logs `Cortex durable server ready` and yields; on shutdown closes all pools.

The FastAPI app also adds a permissive CORS middleware and mounts the
`assistants`, `threads`, and `runs` routers. `/ok` is the health check `ui` gates
on (`depends_on: condition: service_healthy`); `/info` advertises capability
flags (`assistants: true`, `crons: false`).

### 3.3 Why three connection pools

A single shared pool caused psycopg **"another command is already in progress"**
errors. Within one graph step the checkpointer write and the store's semantic
recall can run concurrently (more so now that `spec_review` fans out with
`asyncio.gather`), and sharing one pinned connection interleaves two commands on
it. Dedicated pools isolate the checkpointer, store, and threads-table access so
they can never collide.

### 3.4 Persistence model

- The graph is compiled with an `AsyncPostgresSaver` checkpointer and an
  `AsyncPostgresStore`, so chat **threads**, **checkpoints**, and **long-term
  semantic memory** all persist in Postgres. Conversations survive container
  restarts, image rebuilds, and version upgrades. Only `docker compose down -v`
  (or dropping the tables) wipes them.
- **Schema**: the saver/store tables (`checkpoints`, `checkpoint_blobs`,
  `checkpoint_writes`, `store`, `store_vectors`) plus a small `threads` metadata
  table are created automatically in the same `cortex` database as the app
  tables, with no manual migration step.
- The `threads` table ([`threads.py`](cortex/server/threads.py) `THREADS_DDL`):

  ```sql
  CREATE TABLE IF NOT EXISTS threads (
      thread_id  UUID PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
      status     TEXT  NOT NULL DEFAULT 'idle'
  );
  -- + GIN index on metadata, DESC index on updated_at
  ```

  Thread **metadata** lives in this table; thread **state** (messages, summary)
  comes from the durable checkpointer via `graph.aget_state`. `status` is
  `idle`/`busy` and is **cosmetic** (see §4, the real run mutex is in-process).

### 3.5 Thread endpoints & data access

[`threads.py`](cortex/server/threads.py) mirrors the subset of the LangGraph
Platform thread API the UI calls:

- `POST /threads`, create (generates a UUID if none supplied).
- `POST /threads/search`, `metadata @> %s` filter, `ORDER BY updated_at DESC`.
- `GET /threads/{id}`, single thread (404 if missing).
- `PATCH /threads/{id}`, merge metadata (`metadata = metadata || %s`).
- `DELETE /threads/{id}`, `checkpointer.adelete_thread` (best-effort, older
  savers may lack it) + delete the `threads` row.
- `GET /threads/{id}/state` and history, serialized via `snapshot_to_state`
  (values, next, tasks, interrupts, checkpoint refs).

Key data-access helpers:

- `ensure_thread(id, metadata)`, upsert that merges extra metadata
  (`ON CONFLICT DO UPDATE SET metadata = threads.metadata || EXCLUDED.metadata`).
- `set_thread_status(id, status)`, flips the cosmetic `status` column.
- `reset_stale_runs()`, `UPDATE threads SET status='idle' WHERE status='busy'`,
  returns the row count (used for crash recovery at startup).
- `_thread_values(id)`, latest committed state via `aget_state` (empty for a
  thread with no checkpoint yet).

### 3.6 SSE protocol mapping

[`serde.py`](cortex/server/serde.py) speaks the LangGraph Platform serialization
the `@langchain/langgraph-sdk` client expects:

- The SDK deserializes messages as **plain dicts keyed by `type`** (`"ai"` /
  `"human"` / `"tool"` / `"system"`), **not** the LangChain "lc-constructor"
  envelope. `message_to_dict` first coerces a message *chunk* to a full message
  (`message_chunk_to_message`) so its `type` is the base type (`"ai"`) rather
  than `"AIMessageChunk"`, matching the reference server
  (`langgraph-api/stream.mts`).
- `jsonable(obj)` recursively converts graph state (messages → dicts) to
  JSON-native data.
- `is_nostream(metadata)` returns true when a streamed chunk is tagged
  `langsmith:nostream` / `nostream`, tokens from **internal** LLM calls (the
  guardrail `screen_prompt`, the summary refresh) must never surface in the UI
  stream.
- `sse(event, data)` formats a single Server-Sent Events frame:
  `event: <event>\ndata: <json>\n\n`.

The run endpoints drive `graph.astream(stream_mode=["values","messages",
"updates","custom"])` and map each mode onto SSE events:

- `metadata`, first frame, carries `run_id` and `thread_id`.
- `messages/metadata` + `messages/partial`, per-message-id token streaming. The
  accumulation logic mirrors the reference server: per-id chunks are folded
  (`BaseMessageChunk.__add__`) and re-emitted as `messages/partial` for smooth
  token streaming, with an initial `messages/metadata` frame per new message id
  (`_message_frames(chunk, seen)`).
- `values`, full state snapshots the UI reconstructs from.
- `updates` and `custom`, drive activity indicators and generative UI.
- `error`, a graph exception surfaced as a stream error instead of a 500.

---

## 4. Run lifecycle & the background run broker

This is the most subtle part of the server and the subject of the most recent
work. It lives in [`cortex/server/runs.py`](cortex/server/runs.py).

### 4.1 The problem it solves

Originally the graph ran **inside** the SSE request handler. That coupled the
run's lifetime to the HTTP connection: when the client disconnected, the
`astream` generator was cancelled and the run **stopped**. In practice this
meant that **switching to another thread, starting a new chat, or a flaky
network dropped an answer that was still being generated**, the exact bug the
current design fixes.

### 4.2 The broker

The graph now runs in a **detached background task** that publishes SSE frames to
a per-thread **broker**; the HTTP stream merely *attaches* to that broker. Runs
therefore survive client disconnects, and cancellation becomes **explicit**.

- **`_Run`** (`__slots__ = ("run_id","key","buffer","subscribers","done","task")`)
  is one in-flight (or recently finished) run:
  - `buffer: list[str]`, every SSE frame emitted so far (the replay log).
  - `subscribers: set[asyncio.Queue]`, live attached streams.
  - `publish(frame)`, append to `buffer` **and** `put_nowait` to every
    subscriber queue.
  - `finish()`, set `done = True` and push a `_SENTINEL` to every subscriber.
  - `attach() -> (queue, replay)`, **snapshot the buffer and subscribe with no
    `await` in between**. On the single-threaded event loop this is atomic: no
    `publish` can interleave between reading the replay snapshot and registering
    the queue, so a late attacher can't miss or double-count a frame.
- **`_active: dict[str, _Run]`** maps `thread_id → _Run`. This is the
  **single-run-per-thread mutex** and the real concurrency control; the DB
  `threads.status` column is cosmetic.
- **`_spawn(coro)`** = `create_task` + holding a strong reference in a module-set
  `_bg_tasks` (with a `done_callback` to discard it), so a detached task can't be
  garbage-collected mid-flight.

### 4.3 Running the graph, detached

`_run_graph(thread_id, body, run)` is the detached coroutine:

1. Publishes the `metadata` frame (`run_id`, `thread_id`).
2. Builds config (`_build_config`) and resolves input vs a HITL `Command`
   (`_input_or_command`, a `command` with `resume`/`update`/`goto` supersedes
   input for human-in-the-loop resume).
3. Iterates `runtime.graph.astream(..., stream_mode=STREAM_MODES)` and
   `run.publish(...)` a frame per mode.
4. `except asyncio.CancelledError: raise`, on server shutdown, let cancellation
   propagate.
5. `except Exception`, log and publish an `error` frame (never a 500).
6. `finally: run.finish()` then `_spawn(_release(thread_id, run))`. `_release`
   is **detached** so it runs even when the task was cancelled.

`_release(thread_id, run)` sets the DB status back to `idle` (best-effort), waits
`_RUN_TTL` (30s) so a late reconnect can still replay, then evicts the run from
`_active` if it's still the same one.

### 4.4 Attaching / subscribing

`_subscribe(run)` is the async generator the HTTP endpoints return:

1. `q, replay = run.attach()`.
2. Yield every buffered `replay` frame (catch-up).
3. If `run.done`, return, the run already finished, the replay was the whole
   thing.
4. Otherwise loop `await q.get()`, yielding frames until `_SENTINEL`.
5. `finally`, discard `q` from `run.subscribers`.

### 4.5 Idempotency & duplicate suppression

- `_idempotency_key(thread_id, body)`, an explicit `metadata.idempotency_key`
  else the **last human message id** (stable per submit), namespaced by thread.
- `_recent_keys: dict[str, float]` with a 30s TTL (`_IDEM_TTL`), evicted lazily
  by `_evict_keys(now)`.
- A re-POST of the **same** submit within the window is treated as a retry.

### 4.6 Endpoints

- `POST /threads/{id}/runs/stream` (**`stream_run`**), the main path:
  1. `ensure_thread`, compute the idempotency key.
  2. If `_active[id]` exists and isn't done: a **same-key** retry re-attaches
     via `_subscribe(active)`; a **different** submit is rejected `409
     run_in_progress`.
  3. Else if the key is in `_recent_keys`, reject `409 duplicate_run`.
  4. Else record the key, create `_Run(uuid4, key)`, set `_active[id]`, flip DB
     status to `busy`, spawn `_run_graph`, and return `_subscribe(run)`.
- `GET /threads/{id}/runs/{run_id}/stream` (**`join_run_stream`**), re-attach to a
  still-running run's stream (the reconnect path); 404 if the run_id doesn't
  match the active run.
- `POST /threads/{id}/runs/{run_id}/cancel` (**`cancel_run`**), the SDK's
  `stop()` path; cancels the task via `_cancel_active` (200/404).
- `POST /threads/{id}/runs/cancel` (**`cancel_thread_run`**), the UI Cancel
  button's path; cancels whatever run is active.
- `POST /runs/stream` (**`stream_run_stateless`**), a thread-less run that streams
  inline via `_stream(body)` (no broker, no persistence, used for one-off
  stateless calls).

### 4.7 Reconnect pairing with the UI

The join endpoint pairs with the UI's `reconnectOnMount: true` +
`streamResumable: true` (see §15): when a thread mounts and a run is still in
flight, the SDK re-attaches to `GET …/runs/{run_id}/stream` and keeps rendering
where it left off.

### 4.8 Hard requirement: a single worker

Because `_active`, `_recent_keys`, and the broker queues are **in-process**, the
durable server **must run as a single uvicorn worker**
([`docker/Dockerfile.langgraph`](docker/Dockerfile.langgraph) CMD has no
`--workers`). With multiple workers the mutex and broker would be per-process and
the guarantees (one run per thread, reconnect, dedup) would break. State
durability is unaffected (that's all in Postgres); only the **run-coordination**
layer is in-process by design.

---

## 5. The graph (`cortex/workflow.py`)

`build_workflow(checkpointer, store)` compiles the multi-agent graph
([`cortex/workflow.py`](cortex/workflow.py)). Highlights:

- **`route_from_start`**, the first hop. It either bypasses straight to the
  `specialist` (when a fine-tuned model is selected/default and the turn is
  text-only) or goes to the `router`.
- **`_build_agent(...)`**, the shared agent factory. It builds a `create_agent`
  runnable for a given agent spec, injecting a **dynamic context segment** with
  **today's date** and the user's **region** (`region_from_browser(locale, tz)`)
  so date- and location-sensitive answers are correct by default. It applies the
  DB prompt/tool overrides (`agent_prompt()`), wraps the model in
  `.with_fallbacks(...)` for auto-mode resilience (§9), and honors an explicit
  model selection end-to-end.
- **Memory** (`_memory_context`), builds the per-turn memory context by running
  the rolling-summary update and the semantic recall **in parallel**
  (`asyncio.gather(_update_summary, _recall_memories)`).
- **Error handling**, `_run_agent` / `_deep_research` wrap node execution in
  try/except and, on failure, return a friendly `model_error`-tagged message
  (§9) instead of crashing the run.

### State & streaming notes

- `STREAM_MODES = ["values", "messages", "updates", "custom"]`. The UI
  reconstructs from `values`, streams tokens from `messages`, and drives the
  activity trace from `updates`/`custom`.
- Internal LLM calls (guardrail screen, summary refresh) are tagged `nostream`
  so their tokens are filtered out of the UI stream (§3.6).

---

## 6. Agents & routing

### 6.1 Agents

| Agent           | Purpose                                                          | Tools                                                                     |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `router`        | Classifies user intent into one of nine capability types        | none (structured output only)                                             |
| `generalist`    | Default chat agent, greetings, opinions, creative tasks         | `get_current_time`, memory                                                |
| `researcher`    | Factual questions, grounded answers with citations               | `search_knowledge_base`, `wikipedia_search`, `web_search`, `fetch_url`, `crypto_price`, memory |
| `reasoner`      | Math, logic puzzles, step-by-step problem solving                | `calculator`, memory                                                      |
| `coder`         | Writing, explaining, reviewing, refactoring, and debugging code  | `web_search`, `fetch_url`, memory                                         |
| `prompt_cacher` | LLM prompt-caching expert (large stable system prompt)           | none (large prompt demonstrates caching savings)                          |
| `specialist`    | Specs & knowledge in any **trained domain** (hardware built-in; add your own) from a **self-trained** model (silent draft) | none; `spec_review` emits the answer, self-critiques (heuristics + LLM fact-check) and, on a gap, hands off to `researcher` web-RAG, logging it |
| `imagegen`      | Generates images behind a two-layer safety gate                  | none; calls Google / OpenAI image APIs directly                           |
| `shopping`      | Product shopping: direct product-page links with live price & in-stock, region-aware, rendered as cards | `product_prices`, `web_search`, `fetch_url`, `search_memories`          |
| `booking`       | Booking: flights, hotels, movies, concerts, events, shows; dated deep-link cards | `find_bookings`, `web_search`, `fetch_url`, `search_memories`             |
| `synthesizer`   | Deterministic spec/comparison tables (via `render_spec_table`) + formatting pass over factual answers (worked math, grounding) | none (renders in-node) |

Every agent's **system prompt and tool access** is editable from **Admin →
Agents**, and you can create **custom agents** there; they auto-route via the
router by their description, with no restart and no code (§11).

### 6.2 Routing

The router emits a structured `RouterIntent` (via provider strategies) with one
of nine labels, each mapped to a node in `_INTENT_TO_NODE`:

| Label | Node |
| --- | --- |
| `general_chat` | `generalist` |
| `knowledge_query` | `researcher` |
| `reasoning_task` | `reasoner` |
| `coding_task` | `coder` |
| `prompt_caching` | `prompt_cacher` |
| `product_specs` | `specialist` |
| `image_generation` | `imagegen` |
| `shopping` | `shopping` |
| `booking` | `booking` |

- **Unknown labels** fall back to `general_chat`.
- **Speed heuristic**: well-known **stable** facts (e.g. "who founded OpenAI?")
  are answered directly by the `generalist` from the model's own knowledge;
  `knowledge_query` (the web-grounded `researcher`) is reserved for **current,
  niche, or source-cited** information. This keeps everyday questions fast (no
  tools, no synthesize pass).
- **Structured-output fallback**: if the routing model can't emit structured
  output (e.g. a small local model), a keyword heuristic classifies the turn so
  the run never fails.
- **Custom-agent routing**: the router also emits an optional `agent` field; when
  a custom agent best fits the message, the turn routes to the generic
  `custom_agent` node that runs it. Custom agents register live, with no graph
  rebuild.
- **Deterministic trained-entity guard** (runs **before** the classifier is
  trusted): if the message names an entity in **any trained domain**
  ([`cortex/facts.py`](cortex/facts.py) `match_products`, alias-aware with
  longest-match-wins over every domain's facts), the router **forces
  `product_specs`** so a trained entity always reaches the fine-tuned
  `specialist` rather than web search.
- **Context-aware follow-ups**: the router reads a **recent window** of the
  conversation, not just the last message. When the previous turn offered to
  find flights/hotels/tickets (or to compare a product's prices) and the user
  replies with a short affirmative ("yes", "the cheapest", "from Delhi"), the
  router inherits that offer's intent and routes to **booking** (or
  **shopping**) instead of defaulting the terse reply to `general_chat`.
- **Proactive assistance**: the `generalist` and `researcher` prompts make the
  assistant *offer the next step* when a turn is about a destination/trip or an
  event/movie/concert (e.g. after "best spa in Bangkok" it adds one short line,
  "Want me to pull up the cheapest or soonest flights to Bangkok and a few
  hotels?"), or about a purchasable product (offer a price comparison). The
  offer is text only; the user's "yes" is what the context-aware router turns
  into an actual `booking` / `shopping` run.

---

## 7. The self-trained specialist & `spec_review`

The `specialist` is a small model (**Gemma 3 1B**) fine-tuned on curated data
across the **domains you choose** (a built-in **hardware** domain, gaming
consoles + PC + mobile/laptop processors, ships ready to train; you can add your
own domains/subdomains from the UI). Implementation details that matter:

- **No system prompt.** The LoRA training data is bare user/assistant pairs; a
  system prompt corrupts recall. Temperature 0, latest question only, as plain
  text.
- **Never receives images.** `route_by_intent` reroutes image questions to the
  researcher.
- **Silent draft.** The specialist produces a `spec_draft` in state and **never
  speaks to the user directly**. Don't expect its own message in the transcript.
- **Domain-aware identity + off-domain refusals.** These are built from the
  subdomains you trained on, not a fixed hardware framing.

The terminal **`spec_review`** node emits the single visible answer and runs two
things **in parallel** (`asyncio.gather`, so the table is instant rather than
waiting for two serial LLM calls):

1. **Self-critique**, cheap heuristics (refusal phrases, a product not in the
   training facts) **plus** a strict **LLM fact-check** that flags confidently
   wrong answers, wrong manufacturer, an impossible architecture (e.g. "CUDA
   cores" on an Apple chip), or an implausible figure.
2. **Table extraction**, structured columns + rows for the deterministic table.

On a **gap** (refusal / untrained / LLM fact-check fail) `spec_review` **logs the
gap** ([`cortex/db/services/knowledge_gaps.py`](cortex/db/services/knowledge_gaps.py))
and hands off to the `researcher`, which re-answers with **live web-RAG** so the
user gets a correct, sourced answer instead of the wrong one. Otherwise it emits
the deterministically-rendered spec table. Logged gaps drive the retraining loop
(§14): statuses run `new → researched → trained`.

---

## 8. The synthesizer

The `researcher`, `reasoner`, and `coder` answers pass through the **`synthesize`**
node (the specialist's table is already rendered in `spec_review`).

- **Deterministic spec/comparison tables**: for product/hardware/**software**
  spec & comparison answers, a model only *extracts* the structured data
  (columns + rows, copied verbatim, `SpecTable` structured output) and **code**
  renders the markdown through `render_spec_table`
  ([`cortex/tools/spec_table.py`](cortex/tools/spec_table.py)). The answer always
  comes out as a valid table instead of relying on the model to format one. A
  `_no_invented_numbers` guard blocks fabrication, and the node **skips** when the
  answer is already a table.
- **Grounding pass**: other factual answers get a lighter presentation pass that
  grounds drifted numbers against the authoritative spec YAMLs
  ([`cortex/facts.py`](cortex/facts.py), scanning `domains/hardware/` +
  `domains/<domain>/<subdomain>/`, bind-mounted read-only).
- **Code safety**: for `coder` answers it never lets the fast model touch the
  code; it runs a deterministic, parse-only syntax check (Python via `ast`, JSON
  via `json`) and appends a heads-up when a complete code block is broken.
- **Preservation**: the presentation pass keeps certain trailing lines verbatim,
  `Sources:` citations, the `*I've logged this as a knowledge gap…` footnote, and
  the one-line **proactive offer** (§6.2) the generalist/researcher may append,
  so the reformat can't silently drop the follow-up suggestion.
- It rewrites the final AI message **in place** (same message id).

---

## 9. Model selection, auto mode & providers

### 9.1 Auto mode

The chat UI's default selection is **✨ Auto**, which sends the sentinel
`model_id: "auto"` to the graph. The router classifies the message and each node
resolves the best model for its intent from
[`cortex/declarative/auto_mode.yaml`](cortex/declarative/auto_mode.yaml)
(profiles `balanced` / `quality` / `cost`; the active profile is stored in
`app_settings` and switched from Admin → Models). Only models that are enabled
**and whose provider has an API key** are eligible.
`resolve_auto_candidates(intent, profile)`
([`cortex/db/services/auto_mode.py`](cortex/db/services/auto_mode.py)) returns an
ordered candidate list and **skips keyless cloud providers** (`resolved.kind.value
not in ("local","azure_openai") and not api_key`) so auto mode can't fall through
to a stale env key. The `"finetuned"` keyword resolves to the newest
`finetuned-*` local model. The routing chip in the transcript shows which model
auto mode picked.

### 9.2 Quota / outage fallback

`_build_agent` / the router / `custom_agent` wrap the primary model in
`.with_fallbacks(...)` over the remaining candidates
(`auto_fallback_clients(config, *, auto_intent)` in
[`cortex/model_client/chat_client.py`](cortex/model_client/chat_client.py)), so a
quota / rate-limit / outage on the picked model **auto-switches to the next
candidate**. The retryable exception set lives in
[`cortex/errors.py`](cortex/errors.py) (`retryable_model_exceptions()`).

### 9.3 Graceful model errors

If **every** candidate fails (or a **specific** picked model fails), the node
catches it and returns a plain `AIMessage` (`model_error_reply(exc, *, auto)` /
`friendly_model_error(exc)`: "out of quota / credits", "rate-limited", "invalid
API key", …) tagged `model_error` so `synthesize` skips it, no crashed run.

### 9.4 Explicit selection honored end-to-end

A **specific** or **local** selection is honored everywhere, including the
internal `synthesize` / deep-research clarify / spec-extraction passes (they no
longer hardcode the auto fast tier), so picking Claude never silently calls a
different provider.

### 9.5 The registry & key handling

- The model picker offers **✨ Auto**, any **specific registered model** (used
  for the entire turn), or a **Local LLM** (your own OpenAI-compatible endpoint,
  base URL + optional key + model name, without touching the registry).
- Provider `api_key`s are **trimmed** on read (`ResolvedModel.__post_init__` in
  [`cortex/db/services/llm_registry.py`](cortex/db/services/llm_registry.py)) so a
  pasted trailing newline can't 401. `get_provider_api_key` trims too.
- The registry models are stored in `llm_providers` / `llm_models`. Adding a
  provider/model is codeless (Admin → Providers / Models).

---

## 10. Chat modes & extended thinking

An **Options** menu carries a **General / Thinking / Research** slider plus
toggles:

- **Thinking** raises the reasoner to the `quality` tier and turns on the
  provider's **extended thinking**:
  - Anthropic `thinking`. The
    [`llm_registry.py`](cortex/db/services/llm_registry.py) `_thinking_safe_anthropic_cls`
    (a) restores empty `thinking:""` blocks on round-trip (fixes 400s in tool
    loops) and (b) **translates** the old `thinking:{type:enabled,budget_tokens}`
    to `{type:adaptive}` for newer Claude (Sonnet 5 / 4.5, Opus 4.5) via
    `_anthropic_adaptive_thinking`, which otherwise 400s with
    "thinking.type.enabled not supported". `_anthropic_supports_thinking`
    **skips** the thinking param entirely for non-thinking models (Claude 3.0 /
    3.5, 2) so Thinking mode never hard-fails.
  - OpenAI `reasoning_effort=high`.
- **Research** routes to `_deep_research`: phase 1 asks a few clarifying
  questions, phase 2 multi-source-searches and writes a **cited** report (both
  tagged `deep_research` so `synthesize` leaves them alone).
- **Unrestricted mode** (amber, opt-in), see §16.
- **Hide Tool Calls**, collapses tool activity in the transcript.

### Prompt caching

Anthropic gets a `cache_control` breakpoint on the **static** system prompt (the
dynamic memory context comes after it, so it doesn't bust the cache). The UI
shows a ⚡ cached indicator when `usage_metadata.input_token_details.cache_read >
0`. The `prompt_cacher` agent exists to demonstrate the savings with a large
stable system prompt.

---

## 11. Tools, MCP & admin-managed agents

The agent layer is configurable from the console, no code, no restart.

- **Built-in stateless tools** live in [`cortex/tools/`](cortex/tools/) (registry
  + web / commerce / utility / shared / memory + `spec_table.render_spec_table`).
  A tool is a function decorated with `@register_tool`, imported from
  [`cortex/tools/__init__.py`](cortex/tools/__init__.py).
- **Admin → Tools**: enable/disable built-ins, add prebuilt **LangChain catalog**
  tools (Wikipedia, arXiv, PubMed, StackExchange, Tavily,
  [`cortex/tools/catalog.py`](cortex/tools/catalog.py)), and register external
  **MCP servers** (HTTP/stdio) whose tools become grantable to agents. Grants are
  stored in `tools` / `mcp_servers` / `agent_tools`
  ([`cortex/db/services/tool_catalog.py`](cortex/db/services/tool_catalog.py):
  `effective_tool_names` / `resolve_tool_instances`). MCP servers are consumed via
  `MultiServerMCPClient`. Deleting a tool clears it from every agent and (for
  built-ins) stops it being re-seeded.
- **Admin → Agents**: edit any agent's prompt + tool access, reset built-ins to
  their packaged defaults, and create **custom agents** (name + description +
  prompt + tools) that **auto-route by description** via the generic
  `custom_agent` node. Custom agents can use built-in, LangChain, and MCP tools.
  Stored in `agents` / `agent_subagents`
  ([`cortex/db/services/agents.py`](cortex/db/services/agents.py)).
- **Subagents** (agent-as-tool): any agent can be given subagents it delegates
  focused subtasks to on demand. A subagent runs in **isolated context**, shares
  the parent's long-term memory **read-only** (it can recall but never
  `save_memory`), and uses its own granted built-in/LangChain/MCP tools;
  delegation is **one level deep** (no recursion).

DB-backed prompt and tool overrides win over the packaged YAML per agent and
apply on the next message. Transient provider errors (e.g. Anthropic
"overloaded") are retried automatically.

---

## 12. Web search, shopping & booking

Every agent is given **today's date** and the user's **region** (derived from the
browser locale/timezone) so date- and location-sensitive answers are correct by
default.

- **Web search & scraping** ([`cortex/tools/web.py`](cortex/tools/web.py)):
  `web_search` and `fetch_url` use the **Firecrawl → Brave → SerpAPI → Tavily →
  DDG** provider chain (`_provider_search`). Firecrawl is recommended (it also
  powers JS/anti-bot page scraping; `fetch_url` prefers Firecrawl scrape).
  Without a key they fall back to a best-effort DuckDuckGo scrape, which is often
  blocked, so set `FIRECRAWL_API_KEY` (or `BRAVE_API_KEY` / `SERPAPI_API_KEY` /
  `TAVILY_API_KEY`) in `.env` for real results.
- **Shopping** (`product_prices`): looks the product up live and returns the
  **actual product page** on each regional retailer with the **price** and an
  **in-stock** hint (region's own stores first, cheapest first), rendered as
  product cards with an *In stock* / *Out of stock* badge. Without a search key
  it falls back to per-retailer search links.
- **Booking** (`find_bookings`): builds **dated deep links** into each platform's
  live results (Google Flights, Skyscanner, KAYAK, MakeMyTrip, Cleartrip for
  flights; Booking.com / Google Hotels; Ticketmaster / SeatGeek globally, plus
  **BookMyShow / Zomato District / Paytm Insider** for India for movies,
  concerts, and events), rendered as booking cards. It **never completes a
  purchase**.

`product_prices` and `find_bookings` results are the only tool outputs kept out
of the "Thought process" trace, they render as commerce cards instead (§15).

---

## 13. Memory

- **Short-term**: a **rolling summary** kept in graph state, refreshed each turn
  (the refresh LLM call is tagged `nostream`).
- **Long-term**: a **semantic store** (`AsyncPostgresStore` with a pgvector
  index; embedding hook + namespace in [`cortex/memory.py`](cortex/memory.py)).
  The `save_memory` / `search_memories` tools write and recall from it.
- `_memory_context` builds the per-turn context by running the summary update and
  the semantic recall in parallel (`asyncio.gather`).
- The store namespace is currently single-user (`("memories",)`).

---

## 14. Fine-tuning pipeline

The whole **sources → dataset → train → register** loop is driven from **Admin →
Fine-Tuning**. `hardware` is a **built-in domain** (gaming consoles + PC +
mobile/laptop processors) under `trainer/data/domains/hardware/`; admins add
their own **domains → subdomains** from **Manage Domains** and tick which
subdomains to train. **One model** is fine-tuned across every selected subdomain,
and its identity + off-domain refusals are **domain-aware**.

1. **Manage Domains / Sources**: upload PDFs or spec-sheet images, add URLs, or
   give a **research topic** prompt (e.g. "Apple Silicon A- and M-series chip
   specs").
2. **Smart import** (domain-aware) → `POST /admin/import/propose` →
   `research.propose`: reads the sources and proposes which
   **domain/subdomain + schema + entities** to add, for you to **review and
   approve** (`POST /admin/import/apply` → `research.apply_import`). The crawl
   agent (index/leaf detection, structured extraction, robots/403-respecting)
   lives in `trainer/app/scraper.py` (the old `scrape_agent.py` and
   `smart_import.py` were merged in); amd.com uses an embedded-JSON parser;
   uploaded docs/images use vision transcription
   (`qa_generator.transcribe_image`). All writes go through
   `research.save_learned_entry` (alias-aware dedupe; never overrides curated
   hardware `facts.yaml`). **No hardcoded parsers**, the old `intel_pdf.py` and
   `techpowerup` paths are removed. Sources become *facts*, not invented Q&A.
3. **Generate dataset** → [`trainer/generate_dataset.py`](trainer/generate_dataset.py)
   `generate(subdomains, domains)`: **deterministic** template expansion of the
   selected domains' facts → spec / overview / comparison (pairwise + 3-way for
   consoles) / buying advice / off-domain refusals / **domain-aware identity** →
   `train.jsonl` + `valid.jsonl`. The legacy raw-chunk LLM Q&A-pair path is gone.
   **View dataset** (`GET /admin/dataset/preview`) shows the pairs to validate.
4. **Train** (MLX LoRA on host) → **Convert** (fuse + tokenizer sanitize + GGUF,
   atomic replace) → **Register** into the ai service as
   `finetuned-gemma3-1b-hardware` (the `-hardware` suffix is a **cosmetic label**;
   the model is trained on whichever subdomains you selected; registry contract:
   `finetuned-` prefix under the local provider; **newest wins**). **Quick
   top-up** resumes the existing adapters (`--resume-adapter-file`) for a fast
   incremental train.
5. **Knowledge gaps card**: specialist refusals / mismatches / LLM-fact-check
   fails are logged; "Research gaps (web)" → learned facts → regen → retrain.

The trainer runs **on the host** (MLX needs Apple Silicon), not in Docker:

```bash
cd trainer
bash setup.sh                                   # one-time: vendor llama.cpp
uv run uvicorn app.main:app --host 0.0.0.0 --port 8200
```

The `ai` service also serves any GGUF from the Hugging Face catalog, search,
download, and load models from **Admin → Local Models**.

---

## 15. The chat UI (`agent-chat-ui`)

Next.js 15 (Webpack) front-end using `@langchain/langgraph-sdk/react`
`useStream`. The chat lives under `src/app/chat`, the admin console under
`src/app/admin`, and the LangGraph + admin proxies under `src/app/api`.

### 15.1 Stream provider

[`providers/Stream.tsx`](agent-chat-ui/src/providers/Stream.tsx) configures the
typed stream (`useTypedStream`): `apiUrl`, `apiKey`, `assistantId`,
`defaultHeaders` (`X-Auth-Scheme`), `threadId`, `fetchStateHistory: true`,
**`reconnectOnMount: true`** (re-attach to an in-flight run on mount, so
switching threads doesn't drop an answer, pairs with the server's join endpoint,
§4.7), `onCustomEvent`, `onThreadId`, and `onError`. `onError` parses provider
JSON and detects 429/quota and 401/auth to raise the right toast.

Submits go through:

```ts
stream.submit(
  { messages, context },
  { streamMode: ["values"], streamSubgraphs: true, streamResumable: true,
    config: { configurable: buildConfigurable() }, optimisticValues },
)
```

`streamResumable: true` is the client-side prerequisite for the reconnect path.

### 15.2 Conversation experience

Implemented in
[`components/thread/index.tsx`](agent-chat-ui/src/components/thread/index.tsx):

- **`submitMessage(text, blocks)`**, builds the human message (uuid id) + any
  tool responses (`ensureToolCallsHaveResponses`), submits, and records
  `lastSubmitRef = { text, at }`.
- **`handleSubmit(e)`**, the guardrails around a send:
  - Computes `isRepeat` vs the last human message text. If `isRepeat && isLoading`
    → toast + return (swallow a dup of the running turn). If `isRepeat` within
    3000 ms of the same text → return (swallow a rapid re-send).
  - **While a run streams**, `setPending({ text, blocks })`, clear the input,
    toast "Queued, sends when the current reply finishes", and return, instead of
    dropping or racing the message.
  - Otherwise `submitMessage`.
- **Flush effect**, when `isLoading` clears and `pending` exists, submit the
  pending message (auto-send the queued turn).
- **Queued chip**, a dashed-border chip shows the pending text/attachment count
  with an ✕ to cancel it.
- **`handleCancel`**, `stream.stop()` (detaches the local stream) **and** POSTs
  `…/threads/{id}/runs/cancel` (cancels the detached **server** run, §4.6), since
  `stop()` alone would leave the server run going.
- **⌘F / Ctrl-F**, a window keydown effect opens the in-thread find UI when a
  chat has started.

### 15.3 Agent trace

[`components/thread/agent-trace.tsx`](agent-chat-ui/src/components/thread/agent-trace.tsx)
folds intermediate steps (routing, tool calls, plain tool results, thinking) into
a per-turn collapsible **"Thought process"**: live/expanded while streaming,
muted dropdown when done. Commerce cards / images / answer text stay visible;
only `product_prices` / `find_bookings` results are kept out of the trace
(`COMMERCE_TOOLS`). `groupTurns` / `isTraceMessage` are the module-level helpers.
Prompt-box dropdowns share `hooks/use-dropdown.ts` (auto-flip + animation +
menubar hover-switch).

### 15.4 Chat search & thread management

- **In-thread find** ([`components/thread/thread-search.tsx`](agent-chat-ui/src/components/thread/thread-search.tsx)):
  a find bar using the **CSS Custom Highlight API** (`useFind` hook,
  `::highlight(cortex-find)` / `::highlight(cortex-find-current)` styled in
  [`app/globals.css`](agent-chat-ui/src/app/globals.css)). `findRanges` walks the
  message scope with a `TreeWalker`, skipping `[data-search-ui]`, and supports
  next/previous navigation. A **Sources** panel (`extractSources`) collects the
  links cited in the conversation. `extractSources(messages, opts)` takes
  `skipToolNames` (to drop `find_bookings` / `product_prices` deep-link cards,
  which are click destinations, not consulted sources) and `byDomain` (dedupe by
  domain instead of full URL).
- **Cross-thread history search**
  ([`components/thread/history/index.tsx`](agent-chat-ui/src/components/thread/history/index.tsx)):
  a pill-shaped search below "New chat" that filters past threads by their
  **content** (not just the title) with a matching snippet (`threadText`,
  `threadMatches`, `matchSnippet`).
- **Thread menu** (same file): each row has an always-visible **⋯** menu
  (`ThreadMenu`, built on `use-dropdown`) with **Rename** (inline edit →
  `threads.update` metadata `title`), **Pin/Unpin** (metadata `pinned`; pinned
  threads sort to the top with a pin glyph), and **Delete** (confirm dialog →
  `threads.delete`).

### 15.5 Prompt box & model menu

The composer ([`components/thread/index.tsx`](agent-chat-ui/src/components/thread/index.tsx))
is a minimal, Claude/ChatGPT-style toolbar: a bare **+** attach button (a `Label`
over a hidden file input), the model menu, and a circular **↑ send** / **■ stop**
button (disabled until there's input; stop calls `handleCancel`). The textarea
uses `field-sizing-content` with a `min-h`/`max-h` so it starts at a comfortable
height and grows, then scrolls.

The **model menu** ([`components/model-selector.tsx`](agent-chat-ui/src/components/model-selector.tsx))
is one consolidated pill (`PromptToolbarMenu`) that mirrors Claude's nesting:

- The pill shows the active model + the mode when it isn't General (e.g.
  `Auto Thinking`), and a server glyph in local mode.
- Opening it lists **✨ Auto**, then a **Pinned** section (hidden when nothing is
  pinned), then **Providers** — each provider (Anthropic, Google, OpenAI, …,
  derived from the registry) is a submenu of its models. Every model row has a
  hover **pin** toggle; pinning moves a model out of its provider list into
  Pinned (persisted in `selection.pinned_models`, a UI-only field never sent to
  the graph).
- **Mode & options ›** is another submenu: the **General / Thinking / Research**
  radios plus the **Local LLM / Hide Tool Calls / Unrestricted** toggles.
- Submenus open to the **right of their row** and flip up/down with the parent
  (`use-dropdown`); picking a model closes the menu, toggles keep it open, and
  selecting **Local LLM** opens the endpoint-config dialog. (This replaced the
  old flat `Select` + a separate sliders "Options" menu; the leftover
  `TogglesMenu` only exports the `ToggleDef` type now.)

### 15.6 Message actions

Human and assistant messages share a hover action bar
([`components/thread/messages/shared.tsx`](agent-chat-ui/src/components/thread/messages/shared.tsx)):
**copy** (green ✓ tick on success), **edit** (human, re-submits as a new branch),
**regenerate** (assistant), and **👍 / 👎** feedback (assistant, local
acknowledgement toast). The **last** assistant reply keeps its bar visible once
the turn finishes (not hover-gated). Copy goes through `copyTextToClipboard`
([`lib/utils.ts`](agent-chat-ui/src/lib/utils.ts)), which uses
`navigator.clipboard` in a secure context and falls back to a hidden-textarea
`execCommand("copy")`, so it still works when the app is opened over plain HTTP /
LAN (where `navigator.clipboard` is undefined).

### 15.7 Activity / Sources panel

An **Activity** button in the header opens a fixed right-hand drawer
([`components/thread/activity-panel.tsx`](agent-chat-ui/src/components/thread/activity-panel.tsx)):
the **current turn's steps** (reusing the trace's `deriveSteps`) plus a
consolidated **Sources · N** list. Sources reuse `extractSources` with the
commerce deep-links excluded and deduped by domain, so it shows only the
`web_search` / `fetch_url` domains the agent actually consulted, not the booking
card destinations. It renders as a fixed overlay to avoid disturbing the chat
grid / artifact layout.

### 15.8 Model selection (hydration-safe)

[`providers/ModelSelection.tsx`](agent-chat-ui/src/providers/ModelSelection.tsx)
initializes state with the exported `DEFAULT_SELECTION`
([`components/model-selector.tsx`](agent-chat-ui/src/components/model-selector.tsx))
so the **server render matches the first client render**, then loads the
persisted selection in a mount effect. This fixed a Next.js hydration mismatch
that came from reading `localStorage` during the initial render.

### 15.9 Admin proxy

The UI proxies admin calls through `src/app/api` (`[..._path]`) to the `ai`
service and the host `trainer` (`host.docker.internal:8200`), and carries admin
auth via [`lib/admin-auth.ts`](agent-chat-ui/src/lib/admin-auth.ts).

---

## 16. Trust pillars

### 16.1 Observability

Every node and tool call is exported as an **OpenTelemetry span**
([`cortex/observability.py`](cortex/observability.py), `setup_tracing()`).
Configure Langfuse in `.env`:

```env
LANGFUSE_HOST=http://langfuse-web:3000
LANGFUSE_PUBLIC_KEY=pk-lf-local-public
LANGFUSE_SECRET_KEY=sk-lf-local-secret
```

The Langfuse UI runs at <http://localhost:4000> when the observability profile is
up. Each conversation appears as a trace with full prompts, tool arguments,
results, latency, and token cost.

### 16.2 Evaluation

Eval suites are plain `pytest` files in [`evals/`](evals/):

```bash
uv run pytest evals/ -v
```

- [`evals/test_routing.py`](evals/test_routing.py), router classifies intents
  correctly.
- [`evals/test_faithfulness.py`](evals/test_faithfulness.py), researcher grounds
  answers in the KB.
- [`evals/test_security.py`](evals/test_security.py), direct prompt-injection
  resistance.

The shared [`evals/conftest.py`](evals/conftest.py) provides an `agent_runner`
fixture that hits the running LangGraph API at `http://localhost:2024`. Add new
cases to [`evals/golden_dataset.json`](evals/golden_dataset.json). The suites use
[DeepEval](https://github.com/confident-ai/deepeval) and
[RAGAS](https://github.com/explodinggradients/ragas) primitives.

### 16.3 Guardrails

Built-in middleware applied to every specialist agent in
[`cortex/workflow.py`](cortex/workflow.py):

```python
PIIMiddleware("credit_card", strategy="redact", apply_to_output=True)
PIIMiddleware("email",       strategy="redact", apply_to_output=True)
```

Optional middleware in [`cortex/guardrails.py`](cortex/guardrails.py):

- `ToolAllowlistMiddleware(allowed_tools=...)`, hard-blocks any tool call whose
  name is not on the allowlist, defending against tool-name hallucinations.

The image pipeline adds its own **safety gate**
([`cortex/imagegen.py`](cortex/imagegen.py)): a fast LLM pre-flight screens every
request (refuses NSFW etc. before any API call) and strict provider safety
settings back it up. The flow is guardrail screen → Google image models → OpenAI
gpt-image fallback; candidates come from `auto_mode.yaml`. PNGs are written to
`generated_images/{thread_id}_{ts}.png` and served by `/api/images/[name]`.

An opt-in **Unrestricted mode** (the amber prompt-box toggle) relaxes the
**app-level** guardrails for a turn: it **skips PII redaction**, appends a
"direct answers" directive to the agent, and uses a **relaxed image pre-screen**.
It never disables the providers' own moderation.

For the full guardrail design see [`GUARDRAILS.md`](GUARDRAILS.md).

---

## 17. Deployment & operations

### 17.1 Quickstart

```bash
cp .env.example .env                          # set OPENAI_API_KEY first
docker compose up -d --build db langgraph ui ai
docker compose logs -f langgraph              # wait for "Cortex durable server ready"
```

The default `LLM_PROVIDER=openai` uses `OPENAI_MODEL` (default `gpt-5-nano`). For
Azure, set `LLM_PROVIDER=azure_openai` and the `AZURE_OPENAI_*` variables. The
chat UI comes up on <http://localhost:3000>, the LangGraph server on
<http://localhost:2024>. Then open <http://localhost:3000/admin> (log in with
`ADMIN_USERNAME` / `ADMIN_PASSWORD`) to add **Providers** (paste API keys) and
register **Models** / pick the active auto-mode profile / mark a default.

### 17.2 Ports & startup order

- The server binds `8000` inside the container and is published as `2024:8000`.
  The UI container reaches it at `http://langgraph:8000` over the compose
  network; host tooling and the eval suite use `http://localhost:2024`.
- The FastAPI lifespan opens the pools, runs the checkpointer/store migrations,
  and compiles the graph before serving. It waits for `db` to be healthy and
  exposes `/ok`, which `ui` gates on (`depends_on: condition: service_healthy`).

### 17.3 Rebuild

`--build` is needed on first run and after changing Python or UI source:

```bash
docker compose up -d --build langgraph        # server / graph changes
docker compose up -d --build ui               # UI changes
```

### 17.4 Seed a starter registry + knowledge base

```bash
docker compose exec langgraph /app/.venv/bin/python -m cortex.db.seed
docker compose exec langgraph /app/.venv/bin/python -m cortex.db.seed --embeddings
```

(The `--embeddings` pass needs a real OpenAI key in the registry/env.)

### 17.5 Stop & reset

```bash
docker compose down            # stop the stack (keeps all data)
docker compose down -v         # also drop the pgdata volume: wipes registry, KB,
                               # AND all chat threads
```

To reset **only** the chat/graph state (keeping providers/models/KB), drop the
LangGraph tables while `db` is up:

```bash
docker compose exec db psql -U cortex -d cortex -c \
  "DROP TABLE IF EXISTS checkpoints, checkpoint_blobs, checkpoint_writes, checkpoint_migrations, store, store_vectors, store_migrations, threads CASCADE;"
```

### 17.6 Verification recipes

- **Graph API smoke**: `POST :2024/threads` → `POST /threads/{id}/runs/wait` with
  `{"assistant_id":"cortex","input":{...},"config":{"configurable":{"model_id":"auto"}}}`.
- **Regression questions**: "compare ps5 pro and ps5 slim" (Slim 2023 / $450 /
  10.3 TFLOPS); "compare PS5 vs PS5 Slim vs PS5 Pro" (3-way table); "compare AMD
  Ryzen 3700X and AMD Ryzen 3700" (→ "AMD never released…"); "Compare PS5 Pro vs
  Xbox Series X" (Xbox 2020 / $499 / 12.15 TFLOPS).
- **Image**: "Generate an image of a red cube on a beach" (PNG via gpt-image
  fallback while Google quota-blocked); an NSFW prompt → polite refusal.
- **Caching**: a 2nd turn on a Claude model →
  `usage_metadata.input_token_details.cache_read > 0`.
- **Thread durability**: threads/checkpoints/memory persist in Postgres and
  survive restarts, rebuilds, and upgrades; only `down -v` wipes them.

---

## 18. Environment variables

Configuration is read from `.env` (see `.env.example`). Key variables:

| Variable                            | Purpose                                            |
| ----------------------------------- | -------------------------------------------------- |
| `OPENAI_API_KEY`                    | OpenAI key (default provider)                      |
| `LLM_PROVIDER`                      | `openai` or `azure_openai`                          |
| `OPENAI_MODEL`                      | Default OpenAI model (default `gpt-5-nano`)         |
| `AZURE_OPENAI_*`                    | Azure OpenAI endpoint/key/deployment (when used)   |
| `DATABASE_URL`                      | Postgres DSN (app + durable server share it)       |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | Admin console login                                |
| `FIRECRAWL_API_KEY`                 | Real web search + page scraping for `web_search` / `fetch_url` **and** the trainer's topic + gap research (or `BRAVE_API_KEY` / `SERPAPI_API_KEY` / `TAVILY_API_KEY`) |
| `LANGFUSE_*`                        | Optional Langfuse tracing (observability profile)  |

The `langgraph` container reads `DATABASE_URL` for both the app's SQLAlchemy code
and the durable server (the server normalizes the driver suffix via `db_uri()`).

---

## 19. Project layout

```
ai-multi-agent-cortex/
├── agent-chat-ui/            # Next.js 15 front-end (chat + /admin console)
│   └── src/
│       ├── app/             # routes: chat/, admin/, api/ (LangGraph + admin proxies)
│       ├── components/      # thread UI (agent-trace, agent-activity, thread-search,
│       │                    #   history), model-selector, agent-inbox
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
│   │   ├── models/           # LLMProvider, LLMModel, KnowledgeGap, AppSetting,
│   │   │                     #   KnowledgeArticle, Tool/MCPServer/AgentTool, Agent/AgentSubagent
│   │   ├── services/         # llm_registry, auto_mode, knowledge_gaps, app_settings,
│   │   │                     #   tool_catalog, agents (+ subagents)
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

## 20. Extending the system

1. **New tool**: add a function in `cortex/tools/`, decorate with
   `@register_tool`, and import the module from `cortex/tools/__init__.py`. To
   add a tool **without code**, enable a prebuilt LangChain tool or register an
   external MCP server in **Admin → Tools**.
2. **New agent**: for a graph-level agent, add a `---` document in
   `cortex/declarative/agents.yaml` with its `name` and `whitelisted_tools`, add
   a member to the `Agents` enum, and a node in `cortex/workflow.py`. For a
   **custom agent with no code**, create one in **Admin → Agents**, it auto-routes
   via the router by its description.
3. **New routing label**: extend `Intent` in `cortex/workflow.py`, update
   `_INTENT_TO_NODE`, add the label to the router prompt in `agents.yaml`, and
   give the intent a candidate list in each profile of
   `cortex/declarative/auto_mode.yaml` so auto mode can serve it.
4. **New model / provider**: no code change, add it in **Admin → Providers /
   Models**. Reference the model in `auto_mode.yaml` by its `model_id` to fold it
   into auto mode.

---

## 21. Operational gotchas

- **State is durable in Postgres** (custom server + `AsyncPostgresSaver`/`Store`,
  three pools), no more `langgraph dev` pickles or a thread-backup sidecar. Only
  `docker compose down -v` wipes threads/registry.
- **The durable server must run as a single uvicorn worker**
  (`Dockerfile.langgraph` CMD has no `--workers`): the run broker (`_active`) that
  keeps a run alive across a thread switch and enforces one-run-per-thread is
  **in-process**. DB `threads.status` is cosmetic, don't use it as the mutex.
- **Switching threads / starting a new chat no longer cancels** the running turn
  (it's detached); only the explicit Cancel button / SDK stop (→ `…/runs/cancel`)
  does. A run whose client vanished still finishes and persists its checkpoint.
- **The specialist emits a silent draft**; `spec_review` is the terminal node
  that speaks. Don't expect the specialist's own message in the transcript.
- **Trained entities in any domain** are force-routed to the specialist via
  `match_products` (scans the `domains/` tree: hardware + user packs); untrained
  ones fall through to researcher web-RAG.
- **Google free tier**: flash-only, ~20 req/day; pro model rows are disabled in
  the DB. On a paid key, re-enable them and consider promoting them in auto mode.
- **A registry provider with an empty `api_key`** silently falls back to the
  process env key (e.g. a stale `OPENAI_API_KEY` in `.env`) → 401 "Incorrect API
  key" even though your other provider's key is fine. Auto mode now **skips
  keyless cloud providers**; set the key on the provider you actually use (or drop
  the stale env var). Keys are trimmed on read, so trailing whitespace is OK.
- **Fine-tuned model**: retrain regressions are real; **View dataset** + run the
  regression questions after each import. `_no_invented_numbers` catches drift.
- **The trainer must run on the host** (MLX needs Apple Silicon, not Docker);
  rebuilding the Docker containers does NOT interrupt a running host train.
- **No sync IO in async graph nodes** (use `asyncio.to_thread`).
- **Fresh-volume DB seed**:
  `docker compose exec langgraph /app/.venv/bin/python -m cortex.db.seed`.
