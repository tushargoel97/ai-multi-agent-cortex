"use client";

import { Message } from "@langchain/langgraph-sdk";
import { useState } from "react";
import {
  BookOpen,
  BrainCircuit,
  ChevronRight,
  Sparkles,
  Wrench,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getContentString, getThinkingString } from "./utils";
import {
  TOOL_ACTIVITY,
  RoutingChip,
  agentForIntent,
  getRoutingIntent,
  getRoutingModel,
  prettyModel,
} from "./agent-activity";

interface TraceStep {
  key: string;
  icon: LucideIcon;
  label: string;
  detail?: string;
  muted?: boolean;
}

function firstStringArg(args: unknown): string | undefined {
  if (!args || typeof args !== "object") return undefined;
  for (const v of Object.values(args as Record<string, unknown>)) {
    if (typeof v === "string" && v.trim()) return v.trim();
  }
  return undefined;
}

function snippet(text: string, max = 140): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > max ? clean.slice(0, max) + "…" : clean;
}

/** Turn the intermediate messages of a turn into a readable step list. */
function deriveSteps(messages: Message[]): TraceStep[] {
  const steps: TraceStep[] = [];
  for (const m of messages) {
    const intent = getRoutingIntent(m);
    if (intent) {
      const agent = agentForIntent(intent);
      const model = getRoutingModel(m);
      steps.push({
        key: `${m.id}-route`,
        icon: agent.icon,
        label: `Routed to ${agent.label}`,
        detail: model ? prettyModel(model) : undefined,
      });
      continue;
    }
    if (m.type === "ai") {
      const text = getContentString(m.content).trim();
      const toolCalls =
        (m as { tool_calls?: { id?: string; name?: string; args?: unknown }[] })
          .tool_calls ?? [];
      // A short preamble the model wrote before calling a tool (commentary).
      if (text && toolCalls.length > 0) {
        steps.push({
          key: `${m.id}-note`,
          icon: Sparkles,
          label: snippet(text, 160),
          muted: true,
        });
      }
      for (const tc of toolCalls) {
        const info =
          TOOL_ACTIVITY[tc.name ?? ""] ??
          ({ label: `Using ${tc.name || "a tool"}`, icon: Wrench } as const);
        steps.push({
          key: tc.id ?? `${m.id}-${tc.name}`,
          icon: info.icon,
          label: info.label,
          detail: firstStringArg(tc.args),
        });
      }
      const thinking = getThinkingString(m.content);
      if (thinking) {
        steps.push({
          key: `${m.id}-think`,
          icon: BrainCircuit,
          label: "Thinking",
          detail: snippet(thinking, 220),
          muted: true,
        });
      }
      continue;
    }
    if (m.type === "tool") {
      const raw =
        typeof m.content === "string" ? m.content : JSON.stringify(m.content);
      steps.push({
        key: `${m.id}-result`,
        icon: BookOpen,
        label: "Reviewed the results",
        detail: snippet(raw),
        muted: true,
      });
    }
  }
  return steps;
}

function summarize(steps: TraceStep[]): string {
  const route = steps.find((s) => s.label.startsWith("Routed to"));
  const actions = steps.filter(
    (s) => !s.muted && !s.label.startsWith("Routed to"),
  ).length;
  const bits: string[] = [];
  if (route) bits.push(route.label.replace("Routed to ", ""));
  if (actions > 0) bits.push(`${actions} step${actions > 1 ? "s" : ""}`);
  return bits.join(" · ");
}

/** Dedicated loader for image generation (one long node, no streamed tokens). */
function ImageGenLoader() {
  return (
    <div className="mr-auto w-full max-w-md">
      <div className="imggen-frame">
        <div className="imggen-frame__aurora" />
        <div className="imggen-frame__shimmer" />
        <div className="imggen-frame__label">
          <Sparkles className="size-3.5 animate-pulse text-indigo-200" />
          Generating your image…
        </div>
      </div>
    </div>
  );
}

/**
 * The agent's activity + thinking for one turn. While the turn is in flight
 * (`live`) it streams an expanded, structured list of what the agent is doing;
 * once finished it collapses into a compact, muted "Thought process" dropdown
 * (click to review the steps), mirroring how Claude / ChatGPT fold reasoning
 * away once the answer is ready.
 */
export function AgentTrace({
  messages,
  live,
}: {
  messages: Message[];
  live: boolean;
}) {
  const [open, setOpen] = useState(false);

  if (
    live &&
    messages.some((m) => getRoutingIntent(m) === "image_generation")
  ) {
    return <ImageGenLoader />;
  }

  const steps = deriveSteps(messages);
  if (steps.length === 0) return null;

  // A plainly-routed answer (no tools, no thinking) keeps the compact routing
  // chip instead of a "Thought process" dropdown, so simple chats stay clean.
  const hasActivity = steps.some((s) => !s.label.startsWith("Routed to"));
  if (!live && !hasActivity) {
    const routed = messages.find((m) => getRoutingIntent(m));
    const intent = routed ? getRoutingIntent(routed) : null;
    if (!intent) return null;
    return <RoutingChip intent={intent} model={getRoutingModel(routed)} />;
  }

  const expanded = live || open;
  const summary = summarize(steps);

  return (
    <div className="mr-auto w-full max-w-3xl">
      <button
        type="button"
        onClick={() => !live && setOpen((o) => !o)}
        aria-expanded={expanded}
        className={cn(
          "flex items-center gap-1.5 rounded-full px-2 py-1 text-xs text-muted-foreground/80 transition-colors",
          live ? "cursor-default" : "hover:bg-muted/60 hover:text-muted-foreground",
        )}
      >
        {live ? (
          <Sparkles className="size-3.5 animate-pulse text-amber-500" />
        ) : (
          <ChevronRight
            className={cn("size-3.5 transition-transform", open && "rotate-90")}
          />
        )}
        <span className="font-medium">
          {live ? "Working through it" : "Thought process"}
        </span>
        {summary && (
          <span className="text-muted-foreground/60">· {summary}</span>
        )}
        {live && (
          <span className="ml-0.5 flex items-center gap-0.5">
            <span className="h-1 w-1 animate-[pulse_1.4s_ease-in-out_infinite] rounded-full bg-foreground/40" />
            <span className="h-1 w-1 animate-[pulse_1.4s_ease-in-out_0.4s_infinite] rounded-full bg-foreground/40" />
            <span className="h-1 w-1 animate-[pulse_1.4s_ease-in-out_0.8s_infinite] rounded-full bg-foreground/40" />
          </span>
        )}
      </button>

      {expanded && (
        <ol className="mt-1.5 ml-2 flex flex-col gap-1.5 border-l border-border py-0.5 pl-4 text-xs">
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
                    <span className="truncate text-muted-foreground/55">
                      {step.detail}
                    </span>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
