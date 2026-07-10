"use client";

import {
  createContext,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { Message } from "@langchain/langgraph-sdk";
import { cn } from "@/lib/utils";
import { extractSources } from "./thread-search";

export function favicon(domain: string): string {
  return `https://www.google.com/s2/favicons?domain=${domain}&sz=64`;
}

export interface RichSource {
  url: string;
  domain: string;
  label: string;
  title?: string;
  snippet?: string;
}

function domainOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function normalizeUrl(raw: string): string {
  try {
    const u = new URL(raw);
    u.hash = "";
    return u.toString().replace(/\/$/, "");
  } catch {
    return raw;
  }
}

/** Pull {title, url, snippet} entries out of a tool message's JSON payload
 *  (web_search and friends return a `results` array in this shape). */
function toolResultSources(content: string): RichSource[] {
  if (
    !content.trimStart().startsWith("{") &&
    !content.trimStart().startsWith("[")
  )
    return [];
  try {
    const data = JSON.parse(content) as unknown;
    const arrays: unknown[][] = [];
    if (Array.isArray(data)) arrays.push(data);
    else if (data && typeof data === "object") {
      for (const v of Object.values(data as Record<string, unknown>)) {
        if (Array.isArray(v)) arrays.push(v);
      }
    }
    const out: RichSource[] = [];
    for (const arr of arrays) {
      for (const item of arr) {
        if (!item || typeof item !== "object") continue;
        const r = item as Record<string, unknown>;
        const url = (r.url ?? r.link) as string | undefined;
        if (!url || !/^https?:\/\//i.test(url)) continue;
        const snippet = (r.snippet ?? r.content) as string | undefined;
        out.push({
          url,
          domain: domainOf(url),
          label: domainOf(url),
          title: typeof r.title === "string" ? r.title : undefined,
          snippet:
            typeof snippet === "string" && snippet.trim()
              ? snippet.replace(/\s+/g, " ").slice(0, 220)
              : undefined,
        });
      }
    }
    return out;
  } catch {
    return [];
  }
}

// Tool payloads can be large (fetched page text) and are re-scanned on every
// streaming token; their parse result is immutable per message, so cache it.
const toolSourceCache = new Map<string, RichSource[]>();

/** Every source the thread consulted, enriched with the tool-result title and
 *  snippet where available. Unique by URL, tool-derived metadata wins. */
export function extractRichSources(
  messages: Message[],
  opts: { skipToolNames?: string[] } = {},
): RichSource[] {
  const skip = new Set(opts.skipToolNames ?? []);
  const byUrl = new Map<string, RichSource>();

  for (const m of messages) {
    if (m.type !== "tool") continue;
    if (skip.has((m as { name?: string }).name ?? "")) continue;
    const content =
      typeof m.content === "string" ? m.content : JSON.stringify(m.content);
    const cacheKey = `${m.id}:${content.length}`;
    let parsed = toolSourceCache.get(cacheKey);
    if (!parsed) {
      parsed = toolResultSources(content);
      toolSourceCache.set(cacheKey, parsed);
    }
    for (const s of parsed) {
      const key = normalizeUrl(s.url);
      const prev = byUrl.get(key);
      if (!prev || (!prev.title && s.title)) byUrl.set(key, s);
    }
  }

  // Prose citations (markdown links / bare URLs in AI messages) that the
  // tool payloads didn't cover.
  for (const s of extractSources(messages, { skipToolNames: [...skip] })) {
    const key = normalizeUrl(s.url);
    if (!byUrl.has(key))
      byUrl.set(key, { url: s.url, domain: s.domain, label: s.label });
  }
  return [...byUrl.values()];
}

/* ── Inline citation chips ─────────────────────────────────────────────── */

const SourcesContext = createContext<Map<string, RichSource>>(new Map());

/** Provides href → enriched-source lookup to citation chips in the prose. */
export function SourcesProvider({
  messages,
  children,
}: {
  messages: Message[];
  children: ReactNode;
}) {
  const map = useMemo(() => {
    const m = new Map<string, RichSource>();
    for (const s of extractRichSources(messages)) {
      m.set(normalizeUrl(s.url), s);
      m.set(s.domain, s); // domain fallback for repeated citations
    }
    return m;
  }, [messages]);
  return (
    <SourcesContext.Provider value={map}>{children}</SourcesContext.Provider>
  );
}

function nodeText(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(nodeText).join("");
  if (typeof node === "object" && "props" in node)
    return nodeText(
      (node as { props?: { children?: ReactNode } }).props?.children,
    );
  return "";
}

function HoverCard({
  source,
  anchor,
  onEnter,
  onLeave,
}: {
  source: RichSource;
  anchor: DOMRect;
  onEnter: () => void;
  onLeave: () => void;
}) {
  const W = 320;
  const left = Math.max(8, Math.min(anchor.left, window.innerWidth - W - 8));
  const above = anchor.top > 190;
  const style: React.CSSProperties = above
    ? { left, bottom: window.innerHeight - anchor.top + 8 }
    : { left, top: anchor.bottom + 8 };
  return createPortal(
    <div
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
      style={{ position: "fixed", width: W, ...style }}
      className="glass animate-in fade-in-0 zoom-in-95 z-[110] rounded-xl border p-3 shadow-xl"
    >
      <div className="flex items-center gap-2">
        <img
          src={favicon(source.domain)}
          alt=""
          className="size-4 shrink-0 rounded-sm"
        />
        <span className="text-muted-foreground truncate text-xs">
          {source.domain}
        </span>
      </div>
      <a
        href={source.url}
        target="_blank"
        rel="noreferrer"
        className="mt-1.5 block text-sm leading-snug font-medium hover:underline"
      >
        {source.title ?? source.label}
      </a>
      {source.snippet && (
        <p className="text-muted-foreground mt-1 line-clamp-2 text-xs">
          {source.snippet}
        </p>
      )}
    </div>,
    document.body,
  );
}

/**
 * Markdown `a` renderer: short citation-style links become favicon pills with
 * a hover preview card (ChatGPT-style); long descriptive anchors stay as
 * normal underlined links so sentences keep flowing.
 */
export function CitationLink({
  href,
  children,
  className,
}: {
  href?: string;
  children?: ReactNode;
  className?: string;
}) {
  const map = useContext(SourcesContext);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const plain = (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className={cn(
        "text-primary font-medium underline underline-offset-4",
        className,
      )}
    >
      {children}
    </a>
  );

  if (!href || !/^https?:\/\//i.test(href)) return plain;

  const text = nodeText(children).trim();
  const isBareUrl = /^https?:\/\//i.test(text) || text === "";
  const label = isBareUrl ? domainOf(href) : text;
  if (!isBareUrl && label.length > 32) return plain;

  const source: RichSource = map.get(normalizeUrl(href)) ??
    map.get(domainOf(href)) ?? {
      url: href,
      domain: domainOf(href),
      label,
    };

  const show = (el: HTMLElement) => {
    if (hideTimer.current) clearTimeout(hideTimer.current);
    setRect(el.getBoundingClientRect());
  };
  const hide = () => {
    if (hideTimer.current) clearTimeout(hideTimer.current);
    hideTimer.current = setTimeout(() => setRect(null), 160);
  };

  return (
    <>
      <a
        href={href}
        target="_blank"
        rel="noreferrer"
        onMouseEnter={(e) => show(e.currentTarget)}
        onMouseLeave={hide}
        className={cn(
          "bg-muted/60 text-muted-foreground hover:bg-muted hover:text-foreground mx-0.5 inline-flex max-w-[14rem] translate-y-[-1px] items-center gap-1 rounded-full border border-black/10 px-2 py-[1px] align-middle text-[11px] font-medium no-underline transition-colors dark:border-white/10",
          className,
        )}
      >
        <img
          src={favicon(source.domain)}
          alt=""
          className="size-3 shrink-0 rounded-[3px]"
        />
        <span className="truncate">{label}</span>
      </a>
      {rect && (
        <HoverCard
          source={source}
          anchor={rect}
          onEnter={() => {
            if (hideTimer.current) clearTimeout(hideTimer.current);
          }}
          onLeave={hide}
        />
      )}
    </>
  );
}

/* ── Per-message footer ────────────────────────────────────────────────── */

/** Favicon cluster + "Sources" button under an answer; opens the Activity
 *  panel (listened for in the thread shell). */
export function MessageSources({ message }: { message: Message }) {
  const map = useContext(SourcesContext);
  const cited = useMemo(() => {
    const seen = new Set<string>();
    const out: RichSource[] = [];
    for (const s of extractSources([message])) {
      const key = domainOf(s.url);
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(
        map.get(normalizeUrl(s.url)) ??
          map.get(key) ?? { url: s.url, domain: key, label: s.label },
      );
    }
    return out;
  }, [message, map]);

  if (cited.length === 0) return null;
  return (
    <button
      type="button"
      onClick={() =>
        window.dispatchEvent(new CustomEvent("cortex:open-activity"))
      }
      title="Show sources & activity"
      className="bg-muted/40 text-muted-foreground hover:bg-muted hover:text-foreground mr-auto flex items-center gap-1.5 rounded-full border border-black/10 py-1 pr-2.5 pl-1.5 text-xs transition-colors dark:border-white/10"
    >
      <span className="flex items-center">
        {cited.slice(0, 3).map((s, i) => (
          <img
            key={s.domain}
            src={favicon(s.domain)}
            alt={s.domain}
            className={cn(
              "border-background bg-background size-4 rounded-full border",
              i > 0 && "-ml-1.5",
            )}
          />
        ))}
      </span>
      Sources
      {cited.length > 3 && (
        <span className="text-muted-foreground/70">· {cited.length}</span>
      )}
    </button>
  );
}
