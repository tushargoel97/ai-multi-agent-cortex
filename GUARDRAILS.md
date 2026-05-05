# Guardrails in LangChain — Middleware for Trustworthy LLM Agents

> Companion document to the **AI Multi-Agent Cortex** project. The
> examples below all reference the agents shipped in this repository
> (`router`, `generalist`, `researcher`, `reasoner`, `prompt_cacher`).

## Table of Contents

1. [What Are Guardrails?](#1-what-are-guardrails)
2. [How LangChain Implements Guardrails](#2-how-langchain-implements-guardrails)
3. [Middleware Architecture](#3-middleware-architecture)
4. [What We Built — Our Implementation](#4-what-we-built--our-implementation)
5. [Built-in Middleware Reference](#5-built-in-middleware-reference)
6. [Custom Middleware — Going Beyond Built-ins](#6-custom-middleware--going-beyond-built-ins)
7. [Other Guardrail Approaches](#7-other-guardrail-approaches)
8. [Guardrail Strategy Matrix](#8-guardrail-strategy-matrix)

---

## 1. What Are Guardrails?

Guardrails are safety controls that **validate and filter content at key
points** in an agent's execution. They prevent unsafe, non-compliant, or
incorrect behaviour before it causes real-world harm.

**Common use cases:**

| Category               | Examples                                                                      |
| ---------------------- | ----------------------------------------------------------------------------- |
| **PII Protection**     | Redact emails, credit cards, SSNs before they reach the model or logs         |
| **Human Oversight**    | Require approval before high-stakes side effects (writes, payments)           |
| **Content Safety**     | Block harmful, hateful, or inappropriate content                              |
| **Business Rules**     | Enforce rate limits, quotas, allowlists, compliance constraints               |
| **Output Validation**  | Verify the agent's response is safe and accurate before returning to the user |
| **Prompt Injection**   | Detect and block attempts to manipulate the agent via crafted inputs          |

The Cortex multi-agent assistant ships with two always-on guardrails
(`PIIMiddleware` for credit-card and email redaction) and one opt-in
guardrail (`ToolAllowlistMiddleware`).

---

## 2. How LangChain Implements Guardrails

LangChain organises guardrails as **middleware** — small, composable
classes that wrap an agent's lifecycle hooks. The agent runtime calls
your middleware at three points:

```
┌──────────────────────────────────────────────────────────────────┐
│                        Agent Execution                           │
│                                                                  │
│  User msg ──▶ before_model ──▶ Model ──▶ wrap_tool_call ──▶ Tool │
│                                  │                               │
│                                  └─▶ after_model ──▶ Output      │
└──────────────────────────────────────────────────────────────────┘
```

| Hook              | When it fires                                   | Typical use         |
| ----------------- | ----------------------------------------------- | ------------------- |
| `before_model`    | Just before the LLM is called                   | Input filtering, PII redaction in prompts |
| `wrap_tool_call`  | Just before a tool executes (for each call)     | Tool argument validation, allowlists, HITL |
| `after_model`     | Just after the LLM responds                     | Output filtering, schema validation |

This middleware model means guardrails are **declarative** (added to a
list passed to `create_agent`) and **composable** (multiple middleware
stack on top of each other and run in order).

---

## 3. Middleware Architecture

A LangChain middleware is a subclass of `AgentMiddleware`. The async
versions of the hooks (prefixed with `a`) are required for use with
LangGraph's async runtime, which is what Cortex uses.

```python
from langchain.agents.middleware import AgentMiddleware


class MyMiddleware(AgentMiddleware):
    def before_model(self, state):
        ...
    def after_model(self, state):
        ...
    def wrap_tool_call(self, request, handler):
        # decide to call handler(request), short-circuit, or raise
        return handler(request)

    # Async variants used by LangGraph's async runtime:
    async def abefore_model(self, state): ...
    async def aafter_model(self, state): ...
    async def awrap_tool_call(self, request, handler): ...
```

You attach middleware when constructing an agent:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model=...,
    tools=[...],
    system_prompt="...",
    middleware=[
        PIIMiddleware("credit_card", strategy="redact", apply_to_output=True),
        PIIMiddleware("email",       strategy="redact", apply_to_output=True),
        MyMiddleware(),
    ],
)
```

Middleware run **in order** for `before_model` and `wrap_tool_call`, and
in **reverse order** for `after_model` — the same convention as
ASGI/Express middleware. Place defensive middleware (PII, allowlist) at
the outer edges and policy middleware (HITL, business rules) closer to
the model.

---

## 4. What We Built — Our Implementation

Cortex applies the same default middleware stack to every specialist
agent (`generalist`, `researcher`, `reasoner`, `prompt_cacher`). The
router emits structured output only and is intentionally tool-less, so
it does not need tool-call guardrails.

### Default stack (always on)

```
┌─────────────────────────────────────────────────────────────┐
│             Specialist Agent (e.g. researcher)              │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  PIIMiddleware("credit_card", strategy="redact")        │ │
│ │  PIIMiddleware("email",       strategy="redact")        │ │
│ │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ │                LLM + tool calls                     │ │ │
│ │ └─────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

The relevant snippet from `cortex/workflow.py`:

```python
def _build_agent(agent_id: Agents, *, with_pii: bool = True):
    spec = get_agent_spec(agent_id)
    middleware: list[Any] = []
    if with_pii:
        middleware.append(
            PIIMiddleware("credit_card", strategy="redact", apply_to_output=True)
        )
        middleware.append(
            PIIMiddleware("email", strategy="redact", apply_to_output=True)
        )
    return create_agent(
        model=get_chat_client(),
        tools=spec.get_tools(),
        system_prompt=spec.render_system_prompt(assistant_name=_assistant_name()),
        middleware=middleware,
    )
```

### Optional middleware (opt-in)

`cortex/guardrails.py` ships an additional middleware that you can wire
in when you need stricter tool control:

- **`ToolAllowlistMiddleware`** — hard-blocks any tool call whose name
  is not on the allowlist. Even if the LLM hallucinates a tool name or
  is tricked by a prompt-injection payload into calling something it
  shouldn't, the runtime never invokes anything outside the explicit
  allowlist.

Add it like this:

```python
from cortex.guardrails import ToolAllowlistMiddleware

middleware.append(
    ToolAllowlistMiddleware(allowed_tools={"search_knowledge_base", "wikipedia_search"})
)
```

---

## 5. Built-in Middleware Reference

LangChain ships a number of middleware classes out of the box. The most
useful ones for Cortex-style assistants are:

| Middleware                      | What it does                                                                                   |
| ------------------------------- | ---------------------------------------------------------------------------------------------- |
| `PIIMiddleware`                 | Redacts a configured PII category (`credit_card`, `email`, `phone`, …) in input and/or output  |
| `HumanInTheLoopMiddleware`      | Pauses the graph at specified tools and emits a LangGraph `interrupt` for human approval       |
| `ContextEditingMiddleware`      | Trims or summarises long histories before they reach the model                                 |
| `ModelFallbackMiddleware`       | Falls back to a backup model when the primary errors                                           |
| `RetryMiddleware`               | Retries on transient errors with exponential backoff                                           |
| `LoggingMiddleware`             | Structured logging of every model and tool call                                                |

### `PIIMiddleware` — what we use

```python
PIIMiddleware(
    "credit_card",          # category to scan for
    strategy="redact",      # redact | block | hash
    apply_to_input=True,    # scan the user's message
    apply_to_output=True,   # scan the model's response
)
```

When the model writes a credit-card number into a response, the
middleware replaces it with `[REDACTED_CREDIT_CARD]` before the message
is returned to the user — protecting against accidental leakage from
retrieved context, hallucinated examples, or test data.

### `HumanInTheLoopMiddleware` — when to add it

Cortex doesn't enable HITL by default because none of its current tools
have side effects: searching a vector DB or Wikipedia is read-only, and
calculator/get_current_time are pure functions. **If you add a tool
that mutates external state**, wire HITL on it:

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        "send_email": {
            "description": "Review this email before sending",
            "allowed_decisions": ["approve", "edit", "reject"],
        }
    }
)
```

The graph pauses at the tool call, the UI surfaces the interrupt to a
human, and the run resumes only after a decision is recorded.

---

## 6. Custom Middleware — Going Beyond Built-ins

The middleware contract is small enough that custom guardrails are
typically a few dozen lines. The pattern below is the one used by
`ToolAllowlistMiddleware`:

```python
import json
from collections.abc import Iterable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage


class ToolAllowlistMiddleware(AgentMiddleware):
    """Hard-blocks any tool call whose name is not in the allowlist."""

    def __init__(self, allowed_tools: Iterable[str]):
        self.allowed = set(allowed_tools)

    def _denied(self, tool_call: dict) -> ToolMessage:
        return ToolMessage(
            content=json.dumps(
                {
                    "success": False,
                    "error": (
                        f"Tool '{tool_call['name']}' is not in the "
                        "allowlist for this agent and has been blocked."
                    ),
                }
            ),
            tool_call_id=tool_call["id"],
        )

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in self.allowed:
            return self._denied(request.tool_call)
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in self.allowed:
            return self._denied(request.tool_call)
        return await handler(request)
```

Returning a `ToolMessage` instead of raising is important: the agent
treats it as the tool's output, sees the `error` field, and recovers
gracefully (typically by apologising and choosing a different tool).

### Other middleware ideas you might add to Cortex

- **Cost cap** — track token usage in `before_model` and short-circuit
  with an error message once a per-thread budget is exceeded.
- **Topic guard** — call a small classifier in `before_model` and
  refuse off-topic queries (e.g. anything outside science / coding).
- **Schema validator** — run the model's structured output through a
  Pydantic model in `after_model` and re-prompt on validation failure.
- **Prompt-injection classifier** — score the user's message in
  `before_model` and require a HITL approval if the score is high.

---

## 7. Other Guardrail Approaches

Middleware is the cleanest layer, but it isn't the only one. A
production stack typically combines several:

### Layer A — Pydantic schema validation (free, instant)

Tool argument schemas are the cheapest guardrail. They run before the
tool body and reject malformed calls with a structured error that the
model can read and recover from.

```python
class CalculatorInput(BaseModel):
    expression: str = Field(
        description="Arithmetic expression using + - * / // % **",
        max_length=200,
    )
```

### Layer B — Prompt-level rules

Embed safety rules directly in the system prompt. Cheap, but not
sufficient on its own — a determined prompt-injection payload can talk
the model out of obeying. Treat this as defence-in-depth, not the only
line.

```yaml
# excerpt from researcher.yaml
system_prompt: |
  ...
  Rules
  1. ALWAYS call at least one search tool before answering. Never answer
     factual questions from memory alone.
  ...
  4. If neither source has the answer, say so plainly. Do NOT fabricate.
```

### Layer C — Deterministic middleware

Hard-coded business rules with no LLM involved (rate limits, allowlists,
hard caps). Always run, instant, infinitely auditable.
`ToolAllowlistMiddleware` is a small example.

### Layer D — Human-in-the-loop interrupts

The most expensive but highest-trust guardrail. Use sparingly for the
small number of actions that justify a human-speed pause.

### Layer E — Output-side LLM judge

Run a small classifier model on the agent's final response to score
toxicity, faithfulness, or schema compliance, and re-prompt or block on
failure. Slower and probabilistic — best deployed as a sampled audit
rather than a synchronous gate, unless the use-case is high-risk.

---

## 8. Guardrail Strategy Matrix

| Layer                | How                                       | Latency       | Probabilistic? | Example in Cortex                                        |
| -------------------- | ----------------------------------------- | ------------- | -------------- | -------------------------------------------------------- |
| **Schema**           | Pydantic `Field()` + `args_schema`        | Instant       | No             | `CalculatorInput.expression`                             |
| **Prompt rules**     | System prompt instructions                | Zero overhead | Depends on LLM | `researcher.yaml` "never answer from memory alone"       |
| **PII redaction**    | `PIIMiddleware`                           | ~ms           | No             | `credit_card`, `email` redaction on every agent          |
| **Tool allowlist**   | Custom middleware (`wrap_tool_call`)      | Instant       | No             | `ToolAllowlistMiddleware` (opt-in)                       |
| **HITL**             | `HumanInTheLoopMiddleware`                | Human-speed   | Yes (human)    | (Not used today — wire on any future side-effecting tool) |
| **Output judge**     | Post-hoc LLM classifier or DeepEval `GEval` | LLM call      | Yes            | Eval suites in `evals/`                                  |

### Defence in depth

A production agent should combine layers, not rely on one:

```
Layer A:  Pydantic schemas              — argument shape (always on)
Layer B:  Prompt rules                  — desired behaviour (always on)
Layer C:  Deterministic middleware      — hard limits (always on)
Layer D:  HITL interrupts               — human approval for risky writes
Layer E:  Output-side judges in evals   — regression detection in CI
```

The Cortex repo demonstrates Layers A, B, C, and E today. Layer D is
ready to plug in as soon as you add a side-effecting tool.

---

## Further Reading

- [LangChain Middleware reference](https://python.langchain.com/docs/concepts/middleware)
- [LangGraph human-in-the-loop guide](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — taxonomy of LLM-application risks
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
