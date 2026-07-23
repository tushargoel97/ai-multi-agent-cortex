import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import React, { createElement, type ComponentType } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import * as activity from "../src/components/thread/agent-activity";
import * as progress from "../src/lib/agent-progress";
import { AgentActivity } from "../src/components/thread/agent-activity";
import { AgentTrace } from "../src/components/thread/agent-trace";

Object.assign(globalThis, { React });

type Resolver = (
  messages: unknown[],
  live?: boolean,
) => {
  key?: string;
  label?: string;
  phrases?: string[];
};

const resolve = () => Reflect.get(activity, "deriveLiveActivity") as Resolver;
const fromProgress = () =>
  Reflect.get(activity, "activityFromProgress") as (event: {
    type: string;
    phase: string;
    tool?: string;
  }) => {
    key?: string;
    label?: string;
    phrases?: string[];
  };
const isProgressEvent = () =>
  Reflect.get(progress, "isAgentProgressEvent") as (value: unknown) => boolean;

test("exports a live activity resolver", () => {
  assert.equal(typeof Reflect.get(activity, "deriveLiveActivity"), "function");
});

test("exports a workflow progress resolver", () => {
  assert.equal(typeof Reflect.get(activity, "activityFromProgress"), "function");
});

test("accepts only valid workflow progress events", () => {
  assert.equal(typeof Reflect.get(progress, "isAgentProgressEvent"), "function");
  assert.equal(
    isProgressEvent()({ type: "agent_progress", phase: "researching", tool: "web_search" }),
    true,
  );
  assert.equal(isProgressEvent()({ type: "agent_progress", phase: "waiting" }), false);
  assert.equal(isProgressEvent()({ type: "other", phase: "thinking" }), false);
});

test("maps workflow phases to distinct truthful status sequences", () => {
  const expected = {
    routing: "Routing",
    thinking: "Thinking",
    researching: "Researching",
    collating: "Collating answers",
    refining: "Refining the response",
  };

  for (const [phase, label] of Object.entries(expected)) {
    const status = fromProgress()({ type: "agent_progress", phase });
    assert.equal(status.label, label);
    assert.ok((status.phrases?.length ?? 0) >= 3);
  }
});

test("uses tool context while the workflow is researching", () => {
  const status = fromProgress()({
    type: "agent_progress",
    phase: "researching",
    tool: "web_search",
  });

  assert.equal(status.label, "Researching");
  assert.deepEqual(status.phrases, [
    "Researching",
    "Searching the web",
    "Looking for stronger sources",
    "Following promising leads",
  ]);
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

test("prefers explicit workflow progress over the earlier routing marker", () => {
  const html = renderToStaticMarkup(
    createElement(AgentTrace as ComponentType<any>, {
      live: true,
      progress: { type: "agent_progress", phase: "collating" },
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

  assert.match(html, />Collating answers</);
  assert.doesNotMatch(html, /role="status" aria-live="polite">Routing</);
});

test("renders the blooming Cortex mark and accessible live status", () => {
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
  assert.match(html, /cortex-live-mark__bloom/);
  assert.match(html, /live-status-viewport/);
  assert.match(html, /aria-live="polite"/);
  assert.match(html, /shimmer-text/);
});

test("animates the Cortex glyph as a calm staggered bloom instead of a spinner", () => {
  const styles = readFileSync("src/app/globals.css", "utf8");

  assert.match(styles, /\.cortex-live-mark__glyph path/);
  assert.match(styles, /animation: cortex-live-bloom/);
  assert.match(styles, /path:nth-child\(2\)/);
  assert.match(styles, /path:nth-child\(3\)/);
  assert.match(styles, /path:nth-child\(4\)/);
  assert.doesNotMatch(styles, /cortex-live-orbit/);
});

test("reserves the widest phrase so the detail arrow stays fixed while copy rotates", () => {
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

  assert.match(html, /class="live-status-sizer font-medium whitespace-nowrap"/);
  for (const phrase of ["Routing", "Finding the right specialist", "Choosing the best model"]) {
    assert.match(html, new RegExp(`>${phrase}<`));
  }
});

test("uses slower status transitions without a fixed viewport width", () => {
  const component = readFileSync("src/components/thread/live-agent-status.tsx", "utf8");
  const styles = readFileSync("src/app/globals.css", "utf8");
  const viewportStyles = styles.match(/\.live-status-viewport\s*\{[^}]*}/s)?.[0] ?? "";

  assert.match(component, /duration: 0\.42/);
  assert.match(component, /<AnimatePresence initial=\{false\} mode="sync">/);
  assert.match(styles, /animation: shimmer-text 3\.5s linear infinite/);
  assert.match(viewportStyles, /display: inline-grid/);
  assert.doesNotMatch(viewportStyles, /(?:^|\n)\s*width:/);
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
