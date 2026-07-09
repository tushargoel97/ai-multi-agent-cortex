# AI Multi-Agent Cortex

A production-shaped, general-purpose AI assistant built as a multi-agent system
on top of [LangGraph](https://langchain-ai.github.io/langgraph/). Cortex answers
anything, factual lookups, math and code reasoning, small talk, image
generation, and questions in any **domain you train it on** (hardware ships
ready to use) from its own **self-trained local model**, while keeping the three
trust pillars that distinguish a real product from a demo:

1. **Observability**, every model and tool call is a span in
   [Langfuse](https://langfuse.com/) via OpenTelemetry.
2. **Evaluation**, golden-dataset `pytest` suites using
   [DeepEval](https://github.com/confident-ai/deepeval) and
   [RAGAS](https://github.com/explodinggradients/ragas) primitives.
3. **Guardrails**, PII redaction, an image-safety gate, tool allowlists, and
   human-in-the-loop interrupts wired in as `langchain` middleware.

Everything is driveable from the UI: model/provider management, local-model
downloads, the fine-tuning pipeline, tool & MCP-server control, and agent editing
(system prompts, tool access, custom agents) all live in the `/admin` console, no
CLI for day-to-day use.

> **Deep dive:** see **[`TECHNICAL.md`](TECHNICAL.md)** for the full technical
> reference (architecture, every module, the run lifecycle, and the problems
> each design decision solves), and **[`GUARDRAILS.md`](GUARDRAILS.md)** for the
> guardrail design.

---

## Highlights

- **Multi-agent graph**, a router dispatches each turn to a specialist
  (generalist, researcher, reasoner, coder, image-gen, shopping, booking, a
  self-trained domain specialist, and your own custom agents).
- **Self-trained local specialist**, a fine-tuned Gemma 3 1B answers from its own
  weights for any domain you train; on a gap it self-critiques and hands off to
  live web-RAG so the answer is still correct and sourced.
- **✨ Auto mode**, picks the best model per intent from your registry, with
  automatic **quota/outage fallback** and graceful, no-crash error replies.
- **Durable by default**, threads, checkpoints, and long-term memory persist in
  Postgres via a **custom self-hosted LangGraph server**, no LangSmith licence,
  no Redis. Conversations survive restarts, rebuilds, and upgrades.
- **Runs survive thread-switching**, a reply keeps streaming even if you switch
  threads, start a new chat, or drop the connection, come back and it re-attaches
  live. Follow-ups sent mid-stream are **queued**; duplicates are suppressed.
- **In-chat search**, ⌘F find-in-thread with a Sources panel, plus content-based
  search across your whole history.
- **Polished, familiar chat UX**, a ChatGPT/Claude-style composer (one nested
  model + mode menu, attach, send/stop), per-message copy / edit / regenerate /
  feedback, a thread **⋯** menu (rename / pin / delete), and an **Activity +
  Sources** side panel.
- **Codeless admin**, add providers/models, enable built-in or LangChain tools,
  register MCP servers, and edit or create agents (and subagents), all from
  `/admin`, no restart.
- **Live web, shopping & booking**, real search/scraping (Firecrawl/Brave/…),
  live product-price cards, and dated booking deep links, all region-aware
  (India adds BookMyShow, Zomato District, and Paytm Insider for tickets).
- **Proactive & context-aware**, when you're chatting about a destination,
  event, or product, the assistant offers the next step ("want the cheapest
  flights to Bangkok and a hotel?") and a simple "yes" routes straight to the
  booking or shopping agent.

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

| Service     | Port | Role                                                    |
| ----------- | ---- | ------------------------------------------------------- |
| `db`        | 5432 | Postgres: registry/KB **and** durable graph state       |
| `langgraph` | 2024 | Custom durable LangGraph server (`cortex/server`)       |
| `ui`        | 3000 | Chat UI + `/admin` console                              |
| `ai`        | 8100 | llama.cpp GGUF server for local / fine-tuned models     |
| `mcp`       | 8811 | FastMCP server re-exposing the stateless tools          |
| `trainer` † | 8200 | Host-side MLX LoRA fine-tuning (not Docker)             |
| `langfuse`  | 4000 | Tracing UI (profile `observability`)                    |

† The trainer is the only non-containerized component (MLX needs Apple Silicon).
The full service breakdown and persistence model are in
[`TECHNICAL.md`](TECHNICAL.md#2-service-topology).

---

## Quickstart

**1. Configure environment**

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

The default `LLM_PROVIDER=openai` uses `OPENAI_MODEL` (default `gpt-5-nano`). For
Azure OpenAI, set `LLM_PROVIDER=azure_openai` and fill in the `AZURE_OPENAI_*`
variables instead.

**2. Start the stack**

```bash
# core services: Postgres, the durable LangGraph server, the chat UI, and the
# local model server
docker compose up -d --build db langgraph ui ai
```

The chat UI comes up on <http://localhost:3000> and the LangGraph server on
<http://localhost:2024>.

**3. Configure providers and models**

Open <http://localhost:3000/admin> (log in with `ADMIN_USERNAME` /
`ADMIN_PASSWORD`) and:

1. **Providers**, add an OpenAI, Azure, Anthropic, Google, or local provider and
   paste its API key.
2. **Models**, register the models you want, pick the active auto-mode profile,
   and mark one model as the default.

Prefer a starter registry + knowledge base? See
[Seed a starter registry](TECHNICAL.md#174-seed-a-starter-registry--knowledge-base).

**4. Open the chat UI**

Visit <http://localhost:3000> and start a conversation. With **✨ Auto** selected,
the router dispatches each turn to the right specialist and picks the best model
for the job.

---

## Documentation

- **[`TECHNICAL.md`](TECHNICAL.md)**, the full technical reference:
  [service topology](TECHNICAL.md#2-service-topology),
  the [durable server](TECHNICAL.md#3-the-custom-durable-langgraph-server-cortexserver)
  and [run lifecycle / background broker](TECHNICAL.md#4-run-lifecycle--the-background-run-broker),
  the [graph](TECHNICAL.md#5-the-graph-cortexworkflowpy),
  [agents & routing](TECHNICAL.md#6-agents--routing),
  the [self-trained specialist](TECHNICAL.md#7-the-self-trained-specialist--spec_review),
  [auto mode & providers](TECHNICAL.md#9-model-selection-auto-mode--providers),
  [chat modes & thinking](TECHNICAL.md#10-chat-modes--extended-thinking),
  [tools / MCP / agents](TECHNICAL.md#11-tools-mcp--admin-managed-agents),
  [web, shopping & booking](TECHNICAL.md#12-web-search-shopping--booking),
  the [fine-tuning pipeline](TECHNICAL.md#14-fine-tuning-pipeline),
  the [chat UI](TECHNICAL.md#15-the-chat-ui-agent-chat-ui),
  the [trust pillars](TECHNICAL.md#16-trust-pillars),
  [deployment & operations](TECHNICAL.md#17-deployment--operations), and
  [operational gotchas](TECHNICAL.md#21-operational-gotchas).
- **[`GUARDRAILS.md`](GUARDRAILS.md)**, the guardrail design in depth.

---

## License

See [`LICENSE`](LICENSE).
