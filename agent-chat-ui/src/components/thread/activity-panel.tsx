"use client";

import { Message } from "@langchain/langgraph-sdk";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  CheckCircle2,
  ExternalLink,
  ListChecks,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { deriveSteps } from "./agent-trace";
import { extractRichSources, favicon } from "./sources";

function formatDuration(s: number): string {
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

/** Session-local turn timer: ticks while a run is live, then freezes into a
 *  ChatGPT-style "Thought for Xs · Done" line. */
function useTurnTimer(live: boolean) {
  const [elapsed, setElapsed] = useState(0);
  const [lastTook, setLastTook] = useState<number | null>(null);
  const startRef = useRef<number | null>(null);

  useEffect(() => {
    if (live) {
      startRef.current = Date.now();
      setElapsed(0);
      const id = setInterval(
        () =>
          setElapsed(Math.round((Date.now() - (startRef.current ?? 0)) / 1000)),
        1000,
      );
      return () => clearInterval(id);
    }
    if (startRef.current) {
      setLastTook(Math.round((Date.now() - startRef.current) / 1000));
      startRef.current = null;
    }
  }, [live]);

  return { elapsed, lastTook };
}

/** Messages belonging to the latest turn (everything after the last human). */
function lastTurnMessages(messages: Message[]): Message[] {
  let start = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].type === "human") {
      start = i + 1;
      break;
    }
  }
  return messages.slice(start);
}

/**
 * ChatGPT-style right-hand drawer: the live activity of the current/last turn
 * plus a consolidated list of every source cited across the conversation.
 * Rendered as a fixed overlay so it doesn't disturb the chat grid layout.
 */
export function ActivityPanel({
  messages,
  live,
  open,
  onClose,
}: {
  messages: Message[];
  live: boolean;
  open: boolean;
  onClose: () => void;
}) {
  const steps = useMemo(
    () => deriveSteps(lastTurnMessages(messages)),
    [messages],
  );
  const sources = useMemo(
    () =>
      extractRichSources(messages, {
        skipToolNames: ["find_bookings", "product_prices"],
      }),
    [messages],
  );
  const domains = useMemo(
    () => [...new Set(sources.map((s) => s.domain))],
    [sources],
  );
  const [allChips, setAllChips] = useState(false);
  const { elapsed, lastTook } = useTurnTimer(live);

  return (
    <>
      {/* Scrim (mobile only, so the drawer is dismissible on small screens). */}
      <div
        onClick={onClose}
        className={cn(
          "fixed inset-0 z-40 bg-black/30 transition-opacity lg:hidden",
          open ? "opacity-100" : "pointer-events-none opacity-0",
        )}
      />
      <aside
        data-search-ui
        className={cn(
          "glass-surface fixed top-0 right-0 z-50 flex h-full w-[320px] max-w-[85vw] flex-col border-l shadow-xl transition-transform duration-300",
          open ? "translate-x-0" : "translate-x-full",
        )}
      >
        <div className="border-border flex items-center justify-between border-b px-4 py-3">
          <div className="flex items-center gap-2">
            <Activity
              className={cn(
                "size-4",
                live ? "animate-pulse text-amber-500" : "text-muted-foreground",
              )}
            />
            <h2 className="text-sm font-semibold">Activity</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground transition-colors"
            title="Close"
          >
            <X className="size-4" />
          </button>
        </div>

        <div className="[&::-webkit-scrollbar-thumb]:bg-border flex-1 overflow-y-auto px-4 py-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-track]:bg-transparent">
          <div className="text-muted-foreground mb-2 flex items-center gap-1.5 text-xs font-medium">
            <ListChecks className="size-3.5" />
            This turn
          </div>
          {(live || lastTook !== null) && (
            <div className="border-border mb-3 ml-1 flex flex-col gap-0.5 border-l pl-3 text-xs">
              <span className="text-foreground/80">
                {live
                  ? `Thinking · ${formatDuration(elapsed)}`
                  : `Thought for ${formatDuration(lastTook ?? 0)}`}
              </span>
              {!live && (
                <span className="text-muted-foreground/70 flex items-center gap-1">
                  <CheckCircle2 className="size-3" />
                  Done
                </span>
              )}
            </div>
          )}
          {steps.length > 0 ? (
            <ol className="border-border ml-1 flex flex-col gap-2 border-l py-0.5 pl-3 text-xs">
              {steps.map((step, i) => {
                const Icon = step.icon;
                const isLast = i === steps.length - 1;
                return (
                  <li
                    key={step.key}
                    className="flex items-start gap-2"
                  >
                    <Icon
                      className={cn(
                        "mt-0.5 size-3.5 shrink-0",
                        live && isLast
                          ? "animate-pulse text-amber-500"
                          : "text-muted-foreground/60",
                      )}
                    />
                    <div className="flex min-w-0 flex-col">
                      <span
                        className={
                          step.muted
                            ? "text-muted-foreground/70"
                            : "text-foreground/80"
                        }
                      >
                        {step.label}
                      </span>
                      {step.detail && (
                        <span className="text-muted-foreground/55 line-clamp-2">
                          {step.detail}
                        </span>
                      )}
                    </div>
                  </li>
                );
              })}
            </ol>
          ) : (
            <p className="text-muted-foreground text-xs">
              No tool activity for the latest turn.
            </p>
          )}

          {sources.length > 0 && (
            <div className="mt-6">
              {/* Consulted-domain chips, like the reference activity feed. */}
              <div className="mb-3 flex flex-wrap items-center gap-1.5">
                {(allChips ? domains : domains.slice(0, 5)).map((d) => (
                  <span
                    key={d}
                    className="bg-muted/50 text-muted-foreground flex items-center gap-1.5 rounded-full border border-black/10 px-2 py-0.5 text-[11px] dark:border-white/10"
                  >
                    <img
                      src={favicon(d)}
                      alt=""
                      className="size-3 rounded-sm"
                    />
                    {d}
                  </span>
                ))}
                {domains.length > 5 && (
                  <button
                    type="button"
                    onClick={() => setAllChips((v) => !v)}
                    className="bg-muted/50 text-muted-foreground hover:text-foreground rounded-full border border-black/10 px-2 py-0.5 text-[11px] transition-colors dark:border-white/10"
                  >
                    {allChips ? "less" : `${domains.length - 5} more`}
                  </button>
                )}
              </div>

              <div className="text-muted-foreground mb-2 text-xs font-medium">
                Sources · {sources.length}
              </div>
              <div className="flex flex-col gap-1.5">
                {sources.map((s) => (
                  <a
                    key={s.url}
                    href={s.url}
                    target="_blank"
                    rel="noreferrer"
                    className="group border-border/60 hover:bg-muted flex flex-col gap-0.5 rounded-lg border px-2.5 py-2 text-xs transition-colors"
                  >
                    <span className="flex items-center gap-1.5">
                      <img
                        src={favicon(s.domain)}
                        alt=""
                        className="size-3.5 shrink-0 rounded-sm"
                      />
                      <span className="text-muted-foreground/70 truncate text-[10px]">
                        {s.domain}
                      </span>
                      <ExternalLink className="text-muted-foreground/50 ml-auto size-3 shrink-0 opacity-0 transition-opacity group-hover:opacity-100" />
                    </span>
                    <span className="text-foreground/85 line-clamp-2 leading-snug font-medium">
                      {s.title ?? s.label}
                    </span>
                    {s.snippet && (
                      <span className="text-muted-foreground/65 line-clamp-2">
                        {s.snippet}
                      </span>
                    )}
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      </aside>
    </>
  );
}
