# Future Plan — ai-multi-agent-cortex

Prioritized roadmap. Read `.claude/context.md` first for architecture.

## 1. Intelligent web-scrape agent (replaces hardcoded importers)

Today the spec importer (`trainer/app/scraper.py`) dispatches by hostname:
AMD's DB page (embedded-JSON parser), Intel's comparison-chart PDF (regex
parser in `intel_pdf.py`), TechPowerUp (crawler, usually 403-blocked), and a
generic fetch→LLM-distillation fallback. Replace this with a **scraper
agent**: an LLM-driven loop that, given any URL or document from the admin
sources tab, (a) probes fetchability (respect 403s/robots — never evade
anti-bot), (b) decides per page whether it is an index (follow inner product
links) or a leaf (extract), (c) extracts entries into the learned-facts
schema via structured output, (d) validates numbers with
`research._validate_entry`, and (e) reports per-URL outcomes to the UI.
Implementation sketch: a LangGraph subgraph or plain tool-loop in the trainer
using the TRAINER_QA_* model; budgets (max pages, max depth 2, politeness
delay) as request params. Keep `save_learned_entry` as the single write path
(alias dedupe + curated-builtin protection already live there).

## 2. Incremental fine-tune top-ups

mlx-lm supports `--resume-adapter-file`. Add a "Quick top-up" mode in the
Fine-Tuning panel: resume the existing adapters and train ~300–500 iters on
NEW examples mixed with 10–20% replay of the old dataset (prevents
catastrophic forgetting). Full retrain stays the default for big changes.
The trainer already writes `adapters/base_model.txt` — reuse it to guarantee
the resume base matches.

## 3. Durable run-level persistence (LangGraph Platform)

Chat history is durable via the `thread-backup` sidecar (Postgres mirror +
auto-restore), but mid-run checkpoints still live in `langgraph dev`'s
in-mem pickles. The real fix is the postgres runtime, which ships only in
the licensed platform image (Self-Hosted Lite is free but needs a valid
LangSmith API key + Redis). When a key exists: `langgraph build` image,
add redis service, point DATABASE_URI at the existing postgres.

## 4. Model/provider improvements

- Google free tier only serves flash models (~20 req/day even there); the
  pro rows are disabled in the registry. On a paid key: re-enable
  gemini-3.1-pro-preview etc. and consider promoting them in auto-mode.
- Auto-mode profiles live in `cortex/declarative/auto_mode.yaml`; consider
  an admin editor for per-intent candidate lists.
- Show auto-mode's chosen model in the routing chip (today only the answer
  badge shows it).

## 5. Fine-tune dataset ideas

- 3-way comparisons currently generated for consoles only; extend to GPU/CPU
  trios if users ask (bound combinatorics like `_comparison_examples` does).
- Scraped entries carry no launch prices (AMD/Intel sources omit them);
  price questions for those SKUs defer. Could enrich via gap research.
- Off-domain refusal set should be regenerated whenever large imports land
  (`OFF_DOMAIN_PRODUCTS` filtering is alias-aware already).

## 6. Earlier wishlist (still open)

- User-registered skills / MCP servers / custom agents from the UI.
- Evals: wire deepeval/ragas (deps exist) into a CI-style admin panel run.
- Voice/audio input; multi-language.

## Deferred / rejected

- Web access at answer time for the specialist — rejected by design: the
  fine-tuned model answers from weights; gaps are logged, researched
  between trainings, and retrained (see knowledge-gap loop in context.md).
- Evading TechPowerUp's anti-bot protection — do not do this. It is opt-in
  as a source and skipped gracefully when it 403s.
