"use client";

import { Message } from "@langchain/langgraph-sdk";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Link2,
  Search,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getContentString } from "./utils";

/* ── Highlight API (find + jump) ─────────────────────────────────────────── */

interface HighlightCtor {
  new (...ranges: Range[]): unknown;
}
function highlightRegistry():
  | Map<string, unknown> & {
      set: (k: string, v: unknown) => void;
      delete: (k: string) => void;
    }
  | null {
  if (typeof CSS === "undefined") return null;
  const reg = (CSS as unknown as { highlights?: unknown }).highlights;
  return (reg as never) ?? null;
}

const HL_ALL = "cortex-find";
const HL_CURRENT = "cortex-find-current";

function clearHighlights() {
  const reg = highlightRegistry();
  reg?.delete(HL_ALL);
  reg?.delete(HL_CURRENT);
}

/** Build a Range for every case-insensitive match of `q` under `root`. */
function findRanges(root: HTMLElement, q: string): Range[] {
  const needle = q.toLowerCase();
  if (!needle) return [];
  const ranges: Range[] = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => {
      const parent = (node as Text).parentElement;
      if (!node.nodeValue || !node.nodeValue.trim())
        return NodeFilter.FILTER_REJECT;
      // Skip our own UI, code we don't want doubled, and hidden nodes.
      if (parent?.closest("[data-search-ui]")) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  let node: Node | null;
  while ((node = walker.nextNode())) {
    const text = node.nodeValue!.toLowerCase();
    let idx = text.indexOf(needle);
    while (idx !== -1) {
      const range = document.createRange();
      range.setStart(node, idx);
      range.setEnd(node, idx + needle.length);
      ranges.push(range);
      idx = text.indexOf(needle, idx + needle.length);
    }
  }
  return ranges;
}

function useFind(
  scopeRef: React.RefObject<HTMLElement | null>,
  query: string,
  active: boolean,
  revision: number,
) {
  const rangesRef = useRef<Range[]>([]);
  const [count, setCount] = useState(0);
  const [current, setCurrent] = useState(0);

  // Recompute matches when the query, visibility, or transcript changes.
  useEffect(() => {
    const reg = highlightRegistry();
    const Ctor = (globalThis as { Highlight?: HighlightCtor }).Highlight;
    if (!active || !query.trim() || !scopeRef.current || !reg || !Ctor) {
      clearHighlights();
      rangesRef.current = [];
      setCount(0);
      setCurrent(0);
      return;
    }
    const ranges = findRanges(scopeRef.current, query.trim());
    rangesRef.current = ranges;
    setCount(ranges.length);
    setCurrent((c) => (ranges.length ? Math.min(c, ranges.length - 1) : 0));
    if (ranges.length) reg.set(HL_ALL, new Ctor(...ranges));
    else reg.delete(HL_ALL);
    reg.delete(HL_CURRENT);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, active, revision]);

  // Emphasize + scroll to the current match.
  useEffect(() => {
    const reg = highlightRegistry();
    const Ctor = (globalThis as { Highlight?: HighlightCtor }).Highlight;
    const range = rangesRef.current[current];
    if (!reg || !Ctor || !range) return;
    reg.set(HL_CURRENT, new Ctor(range));
    const el =
      range.startContainer.parentElement ??
      (range.startContainer as HTMLElement);
    el?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [current, count]);

  useEffect(() => () => clearHighlights(), []);

  const step = (dir: 1 | -1) =>
    setCount((n) => {
      if (n > 0) setCurrent((c) => (c + dir + n) % n);
      return n;
    });

  return { count, current, next: () => step(1), prev: () => step(-1) };
}

/* ── Source extraction ───────────────────────────────────────────────────── */

export interface SourceItem {
  url: string;
  label: string;
  domain: string;
}

const MD_LINK = /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g;
const BARE_URL = /https?:\/\/[^\s)<>\]"']+/g;

function domainOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

/** Every distinct http(s) source cited across the thread's AI/tool messages. */
export function extractSources(
  messages: Message[],
  opts: { skipToolNames?: string[]; byDomain?: boolean } = {},
): SourceItem[] {
  const skip = new Set(opts.skipToolNames ?? []);
  const seen = new Set<string>();
  const out: SourceItem[] = [];
  const add = (raw: string, label?: string) => {
    const url = raw.replace(/[.,)\]}>"']+$/, "");
    if (!/^https?:\/\//i.test(url)) return;
    const key = opts.byDomain ? domainOf(url) : url;
    if (seen.has(key)) return;
    seen.add(key);
    out.push({ url, label: label?.trim() || domainOf(url), domain: domainOf(url) });
  };
  for (const m of messages) {
    if (m.type !== "ai" && m.type !== "tool") continue;
    // Booking / shopping deep-link cards are destinations the user clicks, not
    // sources the agent consulted, skip them so Activity isn't padded out.
    if (m.type === "tool" && skip.has((m as { name?: string }).name ?? ""))
      continue;
    const text =
      typeof m.content === "string" ? m.content : getContentString(m.content);
    let mm: RegExpExecArray | null;
    MD_LINK.lastIndex = 0;
    while ((mm = MD_LINK.exec(text))) add(mm[2], mm[1]);
    const stripped = text.replace(MD_LINK, " ");
    BARE_URL.lastIndex = 0;
    while ((mm = BARE_URL.exec(stripped))) add(mm[0]);
  }
  return out;
}

/* ── Component ───────────────────────────────────────────────────────────── */

/**
 * In-thread search: a ⌘/Ctrl+F find bar (highlights matches with the CSS
 * Custom Highlight API and jumps between them) plus an expandable **Sources**
 * list of every link cited in the conversation.
 */
export function ThreadSearch({
  scopeRef,
  messages,
  open,
  onClose,
}: {
  scopeRef: React.RefObject<HTMLElement | null>;
  messages: Message[];
  open: boolean;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const [showSources, setShowSources] = useState(false);
  const [sourceFilter, setSourceFilter] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const { count, current, next, prev } = useFind(
    scopeRef,
    query,
    open && !showSources,
    messages.length,
  );

  const sources = useMemo(() => extractSources(messages), [messages]);
  const filteredSources = useMemo(() => {
    const f = sourceFilter.trim().toLowerCase();
    if (!f) return sources;
    return sources.filter(
      (s) =>
        s.url.toLowerCase().includes(f) || s.label.toLowerCase().includes(f),
    );
  }, [sources, sourceFilter]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open, showSources]);

  if (!open) return null;

  return (
    <div
      data-search-ui
      className="absolute top-3 left-1/2 z-30 w-[min(44rem,calc(100%-2rem))] -translate-x-1/2 animate-in fade-in-0 slide-in-from-top-2"
    >
      <div
        className={cn(
          "overflow-hidden border border-black/10 bg-popover/90 shadow-2xl backdrop-blur-xl backdrop-saturate-150 dark:border-white/10",
          showSources ? "rounded-2xl" : "rounded-full",
        )}
      >
        <div className="flex items-center gap-2 px-4 py-2.5">
          <Search className="size-4 shrink-0 text-muted-foreground" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                e.shiftKey ? prev() : next();
              }
              if (e.key === "Escape") onClose();
            }}
            placeholder="Find in conversation…"
            className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
          {query.trim() && (
            <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
              {count ? `${current + 1}/${count}` : "0/0"}
            </span>
          )}
          <div className="flex shrink-0 items-center">
            <button
              type="button"
              onClick={prev}
              disabled={!count}
              className="rounded p-1 text-muted-foreground hover:bg-muted disabled:opacity-40"
              title="Previous (Shift+Enter)"
            >
              <ChevronUp className="size-4" />
            </button>
            <button
              type="button"
              onClick={next}
              disabled={!count}
              className="rounded p-1 text-muted-foreground hover:bg-muted disabled:opacity-40"
              title="Next (Enter)"
            >
              <ChevronDown className="size-4" />
            </button>
          </div>
          <button
            type="button"
            onClick={() => setShowSources((s) => !s)}
            className={cn(
              "flex shrink-0 items-center gap-1 rounded-full border border-border px-2 py-0.5 text-xs transition-colors",
              showSources
                ? "bg-primary/10 text-foreground"
                : "text-muted-foreground hover:bg-muted",
            )}
            title="Sources cited in this conversation"
          >
            <Link2 className="size-3.5" />
            Sources
            {sources.length > 0 && (
              <span className="tabular-nums">{sources.length}</span>
            )}
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-muted-foreground hover:bg-muted"
            title="Close (Esc)"
          >
            <X className="size-4" />
          </button>
        </div>

        {showSources && (
          <div className="border-t border-border">
            <div className="px-3 py-2">
              <input
                value={sourceFilter}
                onChange={(e) => setSourceFilter(e.target.value)}
                placeholder="Filter sources…"
                className="w-full rounded-md bg-muted/60 px-2.5 py-1 text-xs outline-none placeholder:text-muted-foreground"
              />
            </div>
            <div className="max-h-64 overflow-y-auto px-1.5 pb-2">
              {filteredSources.length === 0 ? (
                <p className="px-2 py-3 text-center text-xs text-muted-foreground">
                  {sources.length === 0
                    ? "No sources cited in this conversation yet."
                    : "No sources match your filter."}
                </p>
              ) : (
                <ul className="flex flex-col">
                  {filteredSources.map((s) => (
                    <li key={s.url}>
                      <a
                        href={s.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="group flex items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-muted"
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={`https://www.google.com/s2/favicons?domain=${encodeURIComponent(s.domain)}&sz=32`}
                          alt=""
                          width={16}
                          height={16}
                          className="size-4 shrink-0 rounded-sm"
                        />
                        <span className="min-w-0 flex-1 truncate">
                          {s.label}
                        </span>
                        <span className="shrink-0 text-xs text-muted-foreground">
                          {s.domain}
                        </span>
                        <ExternalLink className="size-3.5 shrink-0 text-muted-foreground opacity-0 group-hover:opacity-100" />
                      </a>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
