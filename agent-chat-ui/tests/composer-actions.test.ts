import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { ComposerActions } from "../src/components/thread/composer-actions";

const render = (input = "", attachmentCount = 0, isLoading = false) =>
  renderToStaticMarkup(
    createElement(ComposerActions, {
      input,
      attachmentCount,
      isLoading,
      mode: "research",
      onModeChange: () => undefined,
      onCancel: () => undefined,
    }),
  );

test("places the mode selector at the right edge while the composer is empty", () => {
  const html = render();

  assert.match(html, /data-composer-mode-position="edge"/);
  assert.match(html, />Research</);
  assert.doesNotMatch(html, /title="Send"/);
});

test("moves the mode selector left and reveals Send for valid input", () => {
  const html = render("a");

  assert.match(html, /data-composer-mode-position="inline"/);
  assert.match(html, /title="Send"/);
});

test("keeps attachment-only messages sendable", () => {
  assert.match(render("", 1), /title="Send"/);
});

test("shows Stop in the action slot during generation", () => {
  const html = render("", 0, true);

  assert.match(html, /data-composer-mode-position="inline"/);
  assert.match(html, /title="Stop generating"/);
  assert.doesNotMatch(html, /title="Send"/);
});
