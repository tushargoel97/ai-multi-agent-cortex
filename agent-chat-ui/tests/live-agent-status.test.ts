import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import * as activity from "../src/components/thread/agent-activity";
import { AgentActivity } from "../src/components/thread/agent-activity";
import { AgentTrace } from "../src/components/thread/agent-trace";

type Resolver = (
  messages: unknown[],
  live?: boolean,
) => {
  key?: string;
  label?: string;
  phrases?: string[];
};

const resolve = () => Reflect.get(activity, "deriveLiveActivity") as Resolver;

test("exports a live activity resolver", () => {
  assert.equal(typeof Reflect.get(activity, "deriveLiveActivity"), "function");
});

test("starts routing with the canonical action and truthful follow-ups", () => {
  const status = resolve()([
    {
      type: "ai",
      content: "knowledge_query",
      additional_kwargs: {
        routing: { intent: "knowledge_query", model: "quality-model" },
      },
    },
  ]);

  assert.equal(status.key, "routing:knowledge_query:quality-model");
  assert.equal(status.label, "Routing");
  assert.deepEqual(status.phrases, [
    "Routing",
    "Finding the right specialist",
    "Choosing the best model",
  ]);
});

test("keeps web-search copy within the active search phase", () => {
  const status = resolve()([
    {
      type: "ai",
      content: "",
      tool_calls: [{ id: "search-1", name: "web_search" }],
    },
  ]);

  assert.equal(status.key, "tool:web_search:search-1");
  assert.equal(status.label, "Searching the web");
  assert.deepEqual(status.phrases, [
    "Searching the web",
    "Looking for stronger sources",
    "Following promising leads",
  ]);
});

test("moves to review copy only after a tool result arrives", () => {
  const status = resolve()([
    {
      type: "tool",
      id: "result-1",
      name: "web_search",
      content: "result",
    },
  ]);

  assert.equal(status.key, "reviewing:web_search:result-1");
  assert.equal(status.label, "Reviewing results");
  assert.deepEqual(status.phrases, ["Reviewing results", "Separating signal from noise"]);
});

test("shows composition copy only while answer text is streaming", () => {
  const status = resolve()([{ type: "ai", id: "answer-1", content: "The answer begins" }], true);

  assert.equal(status.key, "composing:answer-1");
  assert.equal(status.label, "Writing the answer");
  assert.deepEqual(status.phrases, [
    "Writing the answer",
    "Pulling everything together",
    "Giving it one last check",
  ]);
});

test("uses a thinking sequence while the model has not emitted an action", () => {
  const status = resolve()([{ type: "ai", id: "thought-1", content: "" }], true);

  assert.equal(status.key, "thinking:thought-1");
  assert.equal(status.label, "Thinking");
  assert.deepEqual(status.phrases, [
    "Thinking",
    "Mapping the moving parts",
    "Connecting the context",
  ]);
});

test("renders live activity as a collapsed canonical status", () => {
  const html = renderToStaticMarkup(
    createElement(AgentTrace, {
      live: true,
      messages: [
        {
          type: "ai",
          content: "knowledge_query",
          additional_kwargs: {
            routing: { intent: "knowledge_query", model: "quality-model" },
          },
        },
      ],
    }),
  );

  assert.match(html, />Routing</);
  assert.match(html, /aria-expanded="false"/);
  assert.doesNotMatch(html, /Working through it|<ol/);
});

test("renders the kinetic Cortex mark and accessible live status", () => {
  const html = renderToStaticMarkup(
    createElement(AgentTrace, {
      live: true,
      messages: [
        {
          type: "ai",
          content: "knowledge_query",
          additional_kwargs: { routing: { intent: "knowledge_query" } },
        },
      ],
    }),
  );

  assert.match(html, /cortex-live-mark/);
  assert.match(html, /live-status-viewport/);
  assert.match(html, /aria-live="polite"/);
  assert.match(html, /shimmer-text/);
});

test("uses the same compact status before routing completes", () => {
  const html = renderToStaticMarkup(
    createElement(AgentActivity, {
      messages: [{ type: "human", id: "question-1", content: "Help me compare these" }],
    }),
  );

  assert.match(html, /cortex-live-mark/);
  assert.match(html, />Routing</);
  assert.doesNotMatch(html, /query ≈|rounded-2xl border/);
});

test("gives every known tool a truthful follow-up sequence", () => {
  for (const [name, info] of Object.entries(activity.TOOL_ACTIVITY)) {
    assert.ok(info.followUps?.length, `${name} has no follow-up copy`);
  }
});
