# Design: Synthesizer, Auto Mode, Prompt Caching, Image Generation

Approved 2026-07-05 (chat). Amendments from review: images stored in `generated_images/`
with the chat thread id in the filename; synthesizer applies to hardware, web-research,
and calculation answers.

## 1. Synthesizer agent

Tool-less `synthesize` node appended after the factual agents:
`specialist | researcher | reasoner → synthesize → END` (generalist unchanged).

- Input: last human question + the agent's final answer.
- Rewrites **presentation only** — hard rule: never add/remove/alter facts, numbers,
  prices, names, or citations.
  - Hardware specs/comparisons → markdown table (specs as rows, products as columns)
    plus a one-line verdict.
  - Web-research answers → tight structure; `Sources:` lines preserved verbatim.
  - Calculations → step-by-step working ending in the final answer.
  - Knowledge-gap footnote (`*I've logged this as a knowledge gap…`) preserved verbatim.
- Rewrites the final AI message **in place** (same message id) so the per-message token
  badge keeps the answering model's name/usage.
- Model: auto-mode `fast` tier. Any failure → pass the original answer through.

## 2. Auto mode (default model selection)

- Sentinel `model_id: "auto"`; the UI model selector pins an **Auto** entry first and
  uses it as the default. Choosing a concrete model bypasses auto (unchanged behavior).
- `cortex/declarative/auto_mode.yaml` defines three profiles (balanced / quality / cost),
  each mapping intent → ordered candidate `model_id` list; `finetuned` is a keyword for
  the newest enabled `finetuned-*` local model. First candidate that is enabled in the
  registry wins, so Admin → Models still controls the pool.
- Balanced (default): general_chat → gemini-3.5-flash; knowledge_query → claude-sonnet-5;
  reasoning_task → claude-opus-4-8; product_specs → finetuned; fast tier (router,
  synthesizer, guardrails) → gemini-3.5-flash.
- Active profile stored in a new `app_settings` key-value table; switchable via pills in
  Admin → Models (`/api/admin/settings`).
- Graph: `route_from_start` sends `"auto"` to the router (never the specialist bypass);
  the router classifies with the fast tier; each agent node resolves its model via
  `resolve_auto_model(intent)`.

## 3. Prompt caching

- Anthropic: one `cache_control: ephemeral` breakpoint on the static system prompt
  (covers tools + system prefix). Dynamic context (memories, summaries) goes after the
  breakpoint so the static prefix stays cache-hot.
- OpenAI / Gemini: automatic provider-side caching; no request changes.
- UI: token badge and live thread counter show `⚡N cached` from
  `usage_metadata.input_token_details.cache_read`.

## 4. Image generation agent

- New router intent `image_generation` → new `imagegen` node ("Image Artist").
- Guardrail A (pre-flight): fast-tier LLM classifies the prompt; sexual/NSFW content,
  nudity, minors, graphic violence, hate symbols, and real-person likeness are refused
  politely — the image API is never called. Infrastructure failure of the guardrail
  fails open because Guardrail B still applies; explicit "disallowed" always blocks.
- Generate: Google `generateContent` with `responseModalities: ["TEXT","IMAGE"]` and
  strict `safetySettings`. Model candidates from auto_mode.yaml, tried in order
  (gemini-3-pro-image → gemini-3.1-flash-image → gemini-2.5-flash-image) — the free-tier
  key may 429 on the pro image model.
- Guardrail B (post): Gemini safety block → friendly refusal instead of an error.
- Storage: PNG written to `generated_images/` (new bind mount, gitignored) named
  `{thread_id}_{timestamp}.png`. Served by Next route `GET /api/images/[name]`
  (filename sanitized). Chat message embeds `![…](/api/images/…)`.

## Constraints & notes

- Google free-tier key: only flash-class models usable (pro chat models 429 and are
  disabled in the registry until a paid key is provided).
- Build order: auto mode → synthesizer → caching → image agent; E2E test per feature.
