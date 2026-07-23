import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import ModelSelector, {
  DEFAULT_SELECTION,
  EFFORT_META,
  selectionToConfigurable,
} from "../src/components/model-selector";
import { turnUsage } from "../src/components/thread/agent-activity";

test("defaults new conversations to adaptive effort", () => {
  assert.equal(DEFAULT_SELECTION.effort, "adaptive");
});

test("sends the selected effort with each run configuration", () => {
  const configurable = selectionToConfigurable({
    ...DEFAULT_SELECTION,
    effort: "xhigh",
  });

  assert.equal(configurable.effort, "xhigh");
});

test("shows the active effort beside the selected model", () => {
  const html = renderToStaticMarkup(
    createElement(ModelSelector, {
      selection: DEFAULT_SELECTION,
      onChange: () => undefined,
    }),
  );

  assert.match(html, />Auto · Adaptive</);
});

test("offers every supported effort level in ascending order", () => {
  assert.deepEqual(Object.keys(EFFORT_META), ["adaptive", "low", "medium", "high", "xhigh", "max"]);
});

test("reports complete usage for the response turn", () => {
  const messages = [
    { id: "human", type: "human", content: "question" },
    {
      id: "route",
      type: "ai",
      content: "knowledge_query",
      usage_metadata: { input_tokens: 100, output_tokens: 10, total_tokens: 110 },
    },
    {
      id: "tool-call",
      type: "ai",
      content: "",
      usage_metadata: {
        input_tokens: 200,
        output_tokens: 20,
        total_tokens: 220,
        input_token_details: { cache_read: 50 },
      },
    },
    { id: "tool", type: "tool", content: "result" },
    {
      id: "answer",
      type: "ai",
      content: "answer",
      usage_metadata: {
        input_tokens: 300,
        output_tokens: 30,
        total_tokens: 330,
        input_token_details: { cache_read: 70 },
      },
    },
  ] as any;

  assert.deepEqual(turnUsage(messages, messages.at(-1)), {
    input_tokens: 600,
    output_tokens: 60,
    total_tokens: 660,
    input_token_details: { cache_read: 120 },
  });
});
