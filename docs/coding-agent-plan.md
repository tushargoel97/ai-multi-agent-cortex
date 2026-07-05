# Building a Solid Coding Agent — Design & Roadmap

Companion plan for the `coder` agent in Cortex. This lays out the vision and a
phased roadmap; the scaffolding (Phase 0) is already implemented.

## Goal

A coding specialist that answers programming questions and produces
correct, production-quality code — and, over time, can *act* on a codebase
(run code, edit files, run tests) safely and verifiably, all driven from the
chat UI with the same observability/guardrails as the rest of Cortex.

## Phase 0 — Specialist chat agent (DONE)

Shipped now:

- **Intent + node.** New `Intent.CODING_TASK` → `coder` node
  (`cortex/workflow.py`), `Agents.CODER` (`cortex/enums.py`), and
  `cortex/declarative/agents/coder.yaml`. Router (`router.yaml`) now routes
  writing/debugging/refactoring/algorithms/SQL/regex/shell to `coding_task`
  and keeps pure math/logic on `reasoning_task`.
- **Shares the `synthesize` node, safely.** `coder → synthesize → END`, but
  the synthesizer handles code **deterministically** — it never lets the
  fast-tier model rewrite code (which could silently corrupt it). It runs a
  parse-only syntax sanity check (Python via `ast`, JSON via `json`) and
  appends a heads-up when a *complete* code block is broken (snippets /
  pseudocode are skipped to avoid false positives). True validation (running
  code/tests) is Phase 2.
- **Tools.** `web_search`, `fetch_url` (check current docs/APIs instead of
  guessing), `save_memory`, `search_memories` (remember the user's stack).
- **Auto-mode tier.** `coding_task` candidates per profile in
  `auto_mode.yaml`: `gpt-5.5-pro`, `claude-sonnet-5`, `claude-opus-4-8`,
  `claude-fable-5` (ordered per profile; first registry-enabled one wins).
- **UI.** Routing chip + activity card show "Coder" (`agent-activity.tsx`).

## Auto-mode model strategy

- **balanced**: sonnet-5 → gpt-5.5-pro → opus-4-8 → fable-5 (fast, strong default).
- **quality**: opus-4-8 → gpt-5.5-pro → fable-5 → sonnet-5 (max capability).
- **cost**: sonnet-5 → fable-5 → gpt-5.5-pro → opus-4-8 (cheapest capable first).

Only registry-enabled models are eligible, so Admin → Models stays in control
and the per-intent list is editable in Admin → Models → Auto-mode candidates.
`claude-fable-5` must be registered under the Anthropic provider to be used;
until then it's skipped gracefully.

## Phase 1 — Grounding & project context

- **Attach files/repo context.** Let the user paste or upload files (the UI
  already supports multimodal blocks); feed relevant snippets to the coder.
- **Repo-aware retrieval.** Optional: index a connected repo (pgvector) and add
  a `search_code` tool so answers cite real files/symbols.
- **Structured output for edits.** Return unified diffs / per-file patches in a
  parseable shape the UI can render and (later) apply.

## Phase 2 — Execution (make it "act")

- **Sandboxed code runner.** A dedicated, network-isolated container service
  (like `ai`/`trainer`) exposing `run_code(language, source, stdin)` with hard
  CPU/mem/time limits and no host mounts. Add `run_code` to the coder's tools
  behind a **HumanInTheLoop** middleware (the repo already ships PII middleware;
  add `HumanInTheLoopMiddleware` for the execution tool so the user approves
  each run).
- **Test loop.** `run_tests` in the sandbox; the coder iterates until green,
  streaming attempts to the activity card.
- **File edits.** `apply_patch` against a workspace copy, always diff-preview +
  approve before write. Never touch paths outside the workspace.

## Phase 3 — Quality & trust

- **Evals.** Add `evals/test_coding.py` (HumanEval-style prompts + a few repo
  tasks); wire into the eval suite. Track pass@1 per model to tune the
  auto-mode order empirically.
- **Guardrails.** Tool allowlist (`ToolAllowlistMiddleware`) so the coder can
  never call anything outside its set; secret-scanning on outputs; refuse
  malware / exploitation requests (extend the image guardrail pattern to code).
- **Observability.** Every tool call already exports an OTel span (Langfuse) —
  add cost/latency dashboards per coding model.

## Guardrails & safety (non-negotiable)

- Execution is **opt-in, sandboxed, HITL-approved**, resource-limited, and
  network-isolated. No host filesystem access; no arbitrary shell on the host.
- No writing outside an explicit workspace; every edit is diff-previewed.
- Refuse to produce malware, exploitation tooling, or code that bypasses
  security controls without authorization.
- Redact secrets from prompts/outputs; never echo API keys.

## Open decisions

- Sandbox tech (gVisor/Firecracker microVM vs. hardened container) and whether
  it runs on the Mac host or in Docker.
- Repo connection model: upload snapshot vs. live git checkout.
- Whether coding gets its own auto-mode profile knob separate from the global
  balanced/quality/cost profile.
