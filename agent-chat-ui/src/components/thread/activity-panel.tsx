"use client";

import { Message } from "@langchain/langgraph-sdk";
import { useMemo } from "react";
import { Activity, ExternalLink, ListChecks, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { deriveSteps } from "./agent-trace";
import { extractSources } from "./thread-search";

function favicon(domain: string): string {
  return `https://www.google.com/s2/favicons?domain=${domain}&sz=64`;
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
      extractSources(messages, {
        skipToolNames: ["find_bookings", "product_prices"],
        byDomain: true,
      }),
    [messages],
  );

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
          "fixed top-0 right-0 z-50 flex h-full w-[320px] max-w-[85vw] flex-col border-l border-border bg-background/65 shadow-xl backdrop-blur-xl backdrop-saturate-150 transition-transform duration-300",
          open ? "translate-x-0" : "translate-x-full",
        )}
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
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
            className="text-muted-foreground transition-colors hover:text-foreground"
            title="Close"
          >
            <X className="size-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border [&::-webkit-scrollbar-track]:bg-transparent">
          <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <ListChecks className="size-3.5" />
            This turn
          </div>
          {steps.length > 0 ? (
            <ol className="ml-1 flex flex-col gap-2 border-l border-border py-0.5 pl-3 text-xs">
              {steps.map((step, i) => {
                const Icon = step.icon;
                const isLast = i === steps.length - 1;
                return (
                  <li key={step.key} className="flex items-start gap-2">
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
                        <span className="line-clamp-2 text-muted-foreground/55">
                          {step.detail}
                        </span>
                      )}
                    </div>
                  </li>
                );
              })}
            </ol>
          ) : (
            <p className="text-xs text-muted-foreground">
              No tool activity for the latest turn.
            </p>
          )}

          {sources.length > 0 && (
            <div className="mt-6">
              <div className="mb-2 text-xs font-medium text-muted-foreground">
                Sources · {sources.length}
              </div>
              <div className="flex flex-col gap-1.5">
                {sources.map((s) => (
                  <a
                    key={s.url}
                    href={s.url}
                    target="_blank"
                    rel="noreferrer"
                    className="group flex items-center gap-2 rounded-md border border-border/60 px-2 py-1.5 text-xs transition-colors hover:bg-muted"
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={favicon(s.domain)}
                      alt=""
                      className="size-4 shrink-0 rounded-sm"
                    />
                    <div className="flex min-w-0 flex-col">
                      <span className="truncate text-foreground/80">
                        {s.label}
                      </span>
                      {s.label !== s.domain && (
                        <span className="truncate text-[10px] text-muted-foreground/60">
                          {s.domain}
                        </span>
                      )}
                    </div>
                    <ExternalLink className="ml-auto size-3 shrink-0 text-muted-foreground/50 opacity-0 transition-opacity group-hover:opacity-100" />
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
