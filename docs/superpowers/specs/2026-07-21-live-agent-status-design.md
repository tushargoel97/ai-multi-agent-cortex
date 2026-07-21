# Live agent status design

## Goal

Replace the expanded in-progress trace with a compact, event-driven status that feels alive while remaining faithful to the work the agent is performing. The exact trace remains available on demand.

## Interaction

The live trace is collapsed by default into one clickable row containing an animated Cortex mark, one shimmering status line, and a disclosure indicator. Clicking the row reveals the existing routing, model, tool, query, and result steps. Completion returns the row to the existing compact thought-process summary.

Each phase starts with its canonical action. That text then exits upward while a related line enters from below. The displayed line shimmers throughout its active interval. A real streamed event immediately replaces the current phase and resets its sequence; copy never advances into a phase that has not occurred.

The initial phrase stays visible for about 1.2 seconds. Related phrases advance every 2.8 seconds while the same phase remains active. Vertical transitions take about 240 milliseconds.

## Phrase system

The initial pool contains fifteen lines distributed across factual phases:

| Phase | Sequence |
| --- | --- |
| Routing | Routing; Finding the right specialist; Choosing the best model |
| Thinking | Thinking; Mapping the moving parts |
| Searching | Searching the web; Looking for stronger sources; Following promising leads |
| Reading | Reading sources; Checking the details |
| Reviewing | Reviewing results; Separating signal from noise |
| Composing | Writing the answer; Pulling everything together; Giving it one last check |

Known tools retain their specific activity labels. Unknown tools fall back to `Using <tool name>`. Tool activity never rotates into an unrelated generic phrase.

## Components and data flow

`deriveSteps()` remains the source of the detailed trace. A pure phase resolver reads the newest streamed messages and returns a stable phase key, canonical accessible label, phrase sequence, and tool-specific fallback where applicable.

A focused phrase hook owns the active index and timers. It resets on phase-key changes, cleans up stale timers, and does not mutate message data. `AgentTrace` renders the compact live control and conditionally reveals the existing ordered step list.

The existing logo component gains a mark-only rendering mode so the activity indicator reuses the Cortex mark without duplicating SVG paths. The mark uses the existing blue-to-cyan accent and a restrained continuous turn-and-breathe motion. Text movement uses Framer Motion's presence transitions; shimmer remains CSS-based.

No backend events, API changes, new panel, random global slogan loop, or new dependency is required.

## Accessibility and failure behavior

The live row is a keyboard-accessible disclosure control. Screen readers receive only the canonical phase through a polite live region, avoiding repeated announcements for decorative phrase changes. Visual phrases and motion are hidden from assistive technology.

Reduced-motion mode disables mark movement, vertical transitions, and shimmer while retaining immediate readable phase updates. Missing activity falls back to `Thinking`; unknown tools show their sanitized tool name. A missing trace leaves the disclosure hidden rather than rendering an empty list.

## Verification

- Verify phase resolution for routing, reasoning, known and unknown tools, tool results, visible answer composition, and image generation.
- Verify the canonical-first timing, same-phase rotation, immediate phase reset, and timer cleanup.
- Run UI linting and the production build.
- Exercise a live research request in the running app and confirm routing, search, reading, review, and composition transitions match streamed activity.
- Inspect collapsed and expanded states on desktop and mobile widths.
- Verify keyboard activation, accessible labels, dark and light themes, and reduced-motion behavior.
