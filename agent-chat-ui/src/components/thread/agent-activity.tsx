import { Message } from "@langchain/langgraph-sdk";
import {
  Bot,
  BrainCircuit,
  Clock,
  Code2,
  Cpu,
  Database,
  Globe,
  Link2,
  BookOpen,
  Coins,
  NotebookPen,
  Palette,
  Search,
  ShoppingBag,
  Sparkles,
  Ticket,
  Zap,
  type LucideIcon,
} from "lucide-react";
import { getContentString } from "./utils";

/** The router node emits a marker AIMessage with additional_kwargs.routing. */
export function getRoutingIntent(message: Message | undefined): string | null {
  if (!message || message.type !== "ai") return null;
  const kwargs = (message as { additional_kwargs?: Record<string, unknown> })
    .additional_kwargs;
  const routing = kwargs?.routing as { intent?: string } | undefined;
  return routing?.intent ?? null;
}

/** The model auto-mode resolved for this turn, when the router recorded it. */
export function getRoutingModel(message: Message | undefined): string | null {
  if (!message || message.type !== "ai") return null;
  const kwargs = (message as { additional_kwargs?: Record<string, unknown> })
    .additional_kwargs;
  const routing = kwargs?.routing as { model?: string } | undefined;
  return routing?.model ?? null;
}

/** Trim provider date suffixes so chips stay compact (keeps fine-tune ids). */
export function prettyModel(id: string): string {
  return id.replace(/-\d{6,8}$/, "");
}

const INTENT_AGENTS: Record<string, { label: string; icon: LucideIcon }> = {
  product_specs: { label: "Hardware Specialist", icon: Cpu },
  knowledge_query: { label: "Researcher", icon: Search },
  reasoning_task: { label: "Reasoner", icon: BrainCircuit },
  coding_task: { label: "Coder", icon: Code2 },
  prompt_caching: { label: "Prompt Expert", icon: Zap },
  general_chat: { label: "Assistant", icon: Bot },
  image_generation: { label: "Image Artist", icon: Palette },
  shopping: { label: "Shopping Assistant", icon: ShoppingBag },
  booking: { label: "Booking Assistant", icon: Ticket },
};

export const TOOL_ACTIVITY: Record<string, { label: string; icon: LucideIcon }> = {
  web_search: { label: "Searching the web", icon: Globe },
  wikipedia_search: { label: "Searching Wikipedia", icon: BookOpen },
  search_knowledge_base: { label: "Searching the knowledge base", icon: Database },
  fetch_url: { label: "Reading a web page", icon: Link2 },
  crypto_price: { label: "Fetching live prices", icon: Coins },
  get_current_time: { label: "Checking the clock", icon: Clock },
  save_memory: { label: "Saving a memory", icon: NotebookPen },
  search_memories: { label: "Recalling memories", icon: NotebookPen },
  product_prices: { label: "Comparing prices", icon: ShoppingBag },
  find_bookings: { label: "Finding booking options", icon: Ticket },
};

export function agentForIntent(intent: string | null) {
  return (intent && INTENT_AGENTS[intent]) || INTENT_AGENTS.general_chat;
}

/** Small transcript chip shown where the router classified the request. */
export function RoutingChip({
  intent,
  model,
}: {
  intent: string;
  model?: string | null;
}) {
  const agent = agentForIntent(intent);
  const Icon = agent.icon;
  return (
    <div className="mr-auto flex items-center gap-1.5 rounded-full border border-border bg-muted/40 px-2.5 py-1 text-[11px] font-medium text-muted-foreground">
      <Zap className="size-3 text-amber-500" />
      <span>
        routed to <span className="text-foreground">{agent.label}</span>
      </span>
      <Icon className="size-3" />
      {model ? (
        <span
          className="flex items-center gap-1 border-l border-border pl-1.5 text-muted-foreground/80"
          title={`Auto-selected model: ${model}`}
        >
          <Sparkles className="size-3 text-amber-500" />
          <span className="text-foreground/80">{prettyModel(model)}</span>
        </span>
      ) : null}
    </div>
  );
}

// ── Token usage ──────────────────────────────────────────────────────────────

export interface TokenUsage {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  input_token_details?: {
    cache_read?: number;
    cache_creation?: number;
  };
}

/** Provider-reported cached input tokens (prompt caching hits). */
export function cachedTokens(usage: TokenUsage | null): number {
  return usage?.input_token_details?.cache_read ?? 0;
}

/** Real usage reported by the provider, persisted on AI messages. */
export function getUsage(message: Message | undefined): TokenUsage | null {
  const usage = (message as { usage_metadata?: TokenUsage } | undefined)
    ?.usage_metadata;
  return usage && (usage.input_tokens || usage.output_tokens) ? usage : null;
}

/** Rough client-side estimate for user queries (~4 chars/token). */
export function estimateTokens(text: string): number {
  return Math.max(1, Math.round(text.length / 4));
}

export function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 10_000) return `${Math.round(n / 1000)}k`;
  if (n >= 1_000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

/** Sum of provider-reported usage across the thread's AI messages. */
export function threadUsage(messages: Message[]): {
  input: number;
  output: number;
  cached: number;
} {
  let input = 0;
  let output = 0;
  let cached = 0;
  for (const m of messages) {
    if (m.type !== "ai") continue;
    const u = getUsage(m);
    if (u) {
      input += u.input_tokens ?? 0;
      output += u.output_tokens ?? 0;
      cached += cachedTokens(u);
    }
  }
  return { input, output, cached };
}

interface Activity {
  label: string;
  icon: LucideIcon;
  intent?: string;
}

/** The most recent routing marker's intent, scanning newest-first. */
function lastRoutingIntent(messages: Message[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const intent = getRoutingIntent(messages[i]);
    if (intent) return intent;
  }
  return null;
}

/** Internal LLM outputs that must never render as chat bubbles, e.g. the
 *  image safety-guardrail verdict ({"allowed": ...}). Belt-and-suspenders for
 *  when the backend `langsmith:nostream` tag doesn't fully suppress them. */
export function isInternalNoiseMessage(message: Message): boolean {
  if (message.type !== "ai") return false;
  let s = getContentString(message.content).trim();
  // The guardrail model sometimes wraps its verdict in a ```json fence.
  const fence = s.match(/^```(?:json)?\s*/i);
  if (fence) s = s.slice(fence[0].length).trimStart();
  // The image safety-guardrail verdict: {"allowed": true|false, "reason": …}.
  // Matched even mid-stream (partial JSON) since no real answer starts like
  // this, so it never flashes into the transcript.
  return /^\{\s*"allowed"\s*:/.test(s);
}

/** Derive what the graph is doing right now from the streamed messages. */
function deriveActivity(messages: Message[]): Activity | null {
  const last = messages[messages.length - 1];
  if (!last) return { label: "Routing your request", icon: Zap };

  if (last.type === "human") {
    return { label: "Routing your request", icon: Zap };
  }

  // Image generation is one long node with no streamed tokens, keep a
  // prominent card up for the whole wait, keyed off the routing marker.
  // Only the finished image (markdown image) clears it; internal messages
  // like the safety-guardrail verdict must NOT hide it.
  if (lastRoutingIntent(messages) === "image_generation") {
    const imageReady =
      last.type === "ai" &&
      getRoutingIntent(last) == null &&
      getContentString(last.content).includes("![");
    if (!imageReady) {
      return {
        label: "Generating your image…",
        icon: Palette,
        intent: "image_generation",
      };
    }
  }

  const intent = getRoutingIntent(last);
  if (intent) {
    const agent = agentForIntent(intent);
    return { label: `${agent.label} is thinking`, icon: agent.icon, intent };
  }

  if (last.type === "ai") {
    const toolCalls = (last as { tool_calls?: { name?: string }[] }).tool_calls;
    if (toolCalls && toolCalls.length > 0) {
      const name = toolCalls[toolCalls.length - 1]?.name ?? "";
      return TOOL_ACTIVITY[name] ?? { label: `Running ${name || "a tool"}`, icon: Bot };
    }
    // Streaming visible text, no status needed.
    if (getContentString(last.content).length > 0) return null;
    return { label: "Thinking", icon: Bot };
  }

  if (last.type === "tool") {
    return { label: "Reading results", icon: BookOpen };
  }

  return null;
}

/** Live status row rendered under the transcript while a run is in flight. */
export function AgentActivity({ messages }: { messages: Message[] }) {
  const activity = deriveActivity(messages);
  if (!activity) return null;
  const Icon = activity.icon;

  // Image generation gets a dedicated framed loader (a 3D dotted grid flowing
  // in blue/purple) instead of the compact status row, the same box shape the
  // finished image will fill.
  if (activity.intent === "image_generation") {
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

  const lastHuman = [...messages].reverse().find((m) => m.type === "human");
  const queryTokens = lastHuman
    ? estimateTokens(
        typeof lastHuman.content === "string"
          ? lastHuman.content
          : JSON.stringify(lastHuman.content),
      )
    : 0;
  const total = threadUsage(messages);

  return (
    <div className="mr-auto flex flex-wrap items-center gap-2.5 rounded-2xl border border-border bg-muted/40 px-3.5 py-2">
      <Icon className="size-4 animate-pulse text-muted-foreground" />
      <span className="text-sm text-muted-foreground">{activity.label}</span>
      <span className="flex items-center gap-1">
        <span className="h-1 w-1 animate-[pulse_1.4s_ease-in-out_infinite] rounded-full bg-foreground/50" />
        <span className="h-1 w-1 animate-[pulse_1.4s_ease-in-out_0.4s_infinite] rounded-full bg-foreground/50" />
        <span className="h-1 w-1 animate-[pulse_1.4s_ease-in-out_0.8s_infinite] rounded-full bg-foreground/50" />
      </span>
      <span className="ml-1 border-l border-border pl-2.5 text-[11px] tabular-nums text-muted-foreground/80">
        query ≈{formatTokens(queryTokens)} tk
        {total.input + total.output > 0 && (
          <>
            {" · thread "}
            {formatTokens(total.input)} in / {formatTokens(total.output)} out
            {total.cached > 0 && <> · ⚡{formatTokens(total.cached)} cached</>}
          </>
        )}
      </span>
    </div>
  );
}
