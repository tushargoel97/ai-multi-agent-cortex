import { Message } from "@langchain/langgraph-sdk";
import {
  Bot,
  BrainCircuit,
  Calculator,
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
  TableProperties,
  Ticket,
  Zap,
  type LucideIcon,
} from "lucide-react";
import { getContentString } from "./utils";
import { LiveAgentStatus } from "./live-agent-status";
import type { AgentProgressEvent } from "@/lib/agent-progress";

/** The router node emits a marker AIMessage with additional_kwargs.routing. */
export function getRoutingIntent(message: Message | undefined): string | null {
  if (!message || message.type !== "ai") return null;
  const kwargs = (message as { additional_kwargs?: Record<string, unknown> }).additional_kwargs;
  const routing = kwargs?.routing as { intent?: string } | undefined;
  return routing?.intent ?? null;
}

/** The model auto-mode resolved for this turn, when the router recorded it. */
export function getRoutingModel(message: Message | undefined): string | null {
  if (!message || message.type !== "ai") return null;
  const kwargs = (message as { additional_kwargs?: Record<string, unknown> }).additional_kwargs;
  const routing = kwargs?.routing as { model?: string } | undefined;
  return routing?.model ?? null;
}

/** Trim provider date suffixes so chips stay compact (keeps fine-tune ids). */
export function prettyModel(id: string): string {
  return id.replace(/-\d{6,8}$/, "");
}

const INTENT_AGENTS: Record<string, { label: string; icon: LucideIcon }> = {
  product_specs: { label: "Product Specialist", icon: BookOpen },
  knowledge_query: { label: "Researcher", icon: Search },
  reasoning_task: { label: "Reasoner", icon: BrainCircuit },
  coding_task: { label: "Coder", icon: Code2 },
  prompt_caching: { label: "Prompt Expert", icon: Zap },
  general_chat: { label: "Assistant", icon: Bot },
  image_generation: { label: "Image Artist", icon: Palette },
  shopping: { label: "Shopping Assistant", icon: ShoppingBag },
  booking: { label: "Booking Assistant", icon: Ticket },
};

export const TOOL_ACTIVITY: Record<
  string,
  { label: string; icon: LucideIcon; followUps?: string[] }
> = {
  web_search: {
    label: "Searching the web",
    icon: Globe,
    followUps: ["Looking for stronger sources", "Following promising leads"],
  },
  wikipedia_search: {
    label: "Searching Wikipedia",
    icon: BookOpen,
    followUps: ["Checking the reference trail", "Cross-checking the details"],
  },
  search_knowledge_base: {
    label: "Searching the knowledge base",
    icon: Database,
    followUps: ["Looking through what I know", "Matching the closest context"],
  },
  fetch_url: {
    label: "Reading a web page",
    icon: Link2,
    followUps: ["Reading between the lines", "Checking the details"],
  },
  crypto_price: {
    label: "Fetching live prices",
    icon: Coins,
    followUps: ["Checking the latest market move", "Pinning down the live number"],
  },
  fiat_exchange_rate: {
    label: "Converting the amount",
    icon: Coins,
    followUps: ["Checking the live rate", "Keeping the math honest"],
  },
  get_current_time: {
    label: "Checking the clock",
    icon: Clock,
    followUps: ["Syncing with local time"],
  },
  calculator: {
    label: "Calculating",
    icon: Calculator,
    followUps: ["Checking the arithmetic", "Making the numbers behave"],
  },
  save_memory: {
    label: "Saving a memory",
    icon: NotebookPen,
    followUps: ["Keeping that for later", "Tucking it into context"],
  },
  search_memories: {
    label: "Recalling memories",
    icon: NotebookPen,
    followUps: ["Looking through earlier context", "Reconnecting the dots"],
  },
  product_prices: {
    label: "Comparing prices",
    icon: ShoppingBag,
    followUps: ["Checking live listings", "Comparing what came back"],
  },
  find_bookings: {
    label: "Finding booking options",
    icon: Ticket,
    followUps: ["Checking availability", "Narrowing the options"],
  },
  render_spec_table: {
    label: "Formatting the comparison",
    icon: TableProperties,
    followUps: ["Lining up the details", "Making it easy to scan"],
  },
  consult_local_specialist: {
    label: "Consulting a local specialist",
    icon: Cpu,
    followUps: ["Passing along the context", "Waiting for the specialist"],
  },
};

export function agentForIntent(intent: string | null) {
  return (intent && INTENT_AGENTS[intent]) || INTENT_AGENTS.general_chat;
}

/** Small transcript chip shown where the router classified the request. */
export function RoutingChip({ intent, model }: { intent: string; model?: string | null }) {
  const agent = agentForIntent(intent);
  const Icon = agent.icon;
  return (
    <div className="border-border bg-muted/40 text-muted-foreground mr-auto flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium">
      <Zap className="size-3 text-amber-500" />
      <span>
        routed to <span className="text-foreground">{agent.label}</span>
      </span>
      <Icon className="size-3" />
      {model ? (
        <span
          className="border-border text-muted-foreground/80 flex items-center gap-1 border-l pl-1.5"
          title={`Auto-selected model: ${model}`}
        >
          <Sparkles className="size-3 text-amber-500" />
          <span className="text-foreground/80">{prettyModel(model)}</span>
        </span>
      ) : null}
    </div>
  );
}

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
  const usage = (message as { usage_metadata?: TokenUsage } | undefined)?.usage_metadata;
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

export interface Activity {
  key?: string;
  label: string;
  phrases?: string[];
  icon: LucideIcon;
  intent?: string;
}

const ROUTING_PHRASES = ["Routing", "Finding the right specialist", "Choosing the best model"];
const THINKING_PHRASES = ["Thinking", "Mapping the moving parts", "Connecting the context"];
const REVIEWING_PHRASES = ["Reviewing results", "Separating signal from noise"];
const COMPOSING_PHRASES = [
  "Writing the answer",
  "Pulling everything together",
  "Giving it one last check",
];
const PROGRESS_ACTIVITY: Record<
  AgentProgressEvent["phase"],
  { label: string; phrases: string[]; icon: LucideIcon; intent?: string }
> = {
  routing: {
    label: "Routing",
    phrases: ROUTING_PHRASES,
    icon: Zap,
  },
  thinking: {
    label: "Thinking",
    phrases: ["Thinking", "Mapping the request", "Working through the details"],
    icon: BrainCircuit,
  },
  researching: {
    label: "Researching",
    phrases: ["Researching", "Checking strong sources", "Following the evidence"],
    icon: Search,
  },
  collating: {
    label: "Collating answers",
    phrases: [
      "Collating answers",
      "Connecting the evidence",
      "Organizing the findings",
      "Sorting the strongest signals",
    ],
    icon: BookOpen,
  },
  refining: {
    label: "Refining the response",
    phrases: ["Refining the response", "Checking the final details", "Polishing the answer"],
    icon: Sparkles,
  },
  generating_image: {
    label: "Generating your image",
    phrases: ["Generating your image", "Shaping the composition", "Finishing the details"],
    icon: Palette,
    intent: "image_generation",
  },
};

export function activityFromProgress(event: AgentProgressEvent): Activity {
  const activity = PROGRESS_ACTIVITY[event.phase];
  const tool = event.tool ? TOOL_ACTIVITY[event.tool] : undefined;
  return {
    ...activity,
    key: `progress:${event.phase}:${event.tool ?? ""}`,
    ...(tool
      ? {
          phrases: [activity.label, tool.label, ...(tool.followUps ?? [])],
          icon: tool.icon,
        }
      : {}),
  };
}

/** The most recent routing marker's intent, scanning newest-first. */
function lastRoutingIntent(messages: Message[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const intent = getRoutingIntent(messages[i]);
    if (intent) return intent;
  }
  return null;
}

export function isInternalNoiseMessage(message: Message): boolean {
  if (message.type !== "ai") return false;
  let s = getContentString(message.content).trim();
  const fence = s.match(/^```(?:json)?\s*/i);
  if (fence) s = s.slice(fence[0].length).trimStart();
  return /^\{\s*"allowed"\s*:/.test(s);
}

/** Derive what the graph is doing right now from the streamed messages. */
function deriveActivity(messages: Message[], live = false): Activity | null {
  const last = messages[messages.length - 1];
  if (!last) {
    return {
      key: "routing:pending",
      label: ROUTING_PHRASES[0],
      phrases: ROUTING_PHRASES,
      icon: Zap,
    };
  }

  if (last.type === "human") {
    return {
      key: `routing:pending:${last.id || "request"}`,
      label: ROUTING_PHRASES[0],
      phrases: ROUTING_PHRASES,
      icon: Zap,
    };
  }

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
    const model = getRoutingModel(last) ?? "default";
    return {
      key: `routing:${intent}:${model}`,
      label: ROUTING_PHRASES[0],
      phrases: ROUTING_PHRASES,
      icon: agent.icon,
      intent,
    };
  }

  if (last.type === "ai") {
    const toolCalls = (last as { tool_calls?: { id?: string; name?: string }[] }).tool_calls;
    if (toolCalls && toolCalls.length > 0) {
      const call = toolCalls[toolCalls.length - 1];
      const name = call?.name ?? "";
      const activity = TOOL_ACTIVITY[name] ?? {
        label: `Running ${name || "a tool"}`,
        icon: Bot,
      };
      return {
        ...activity,
        key: `tool:${name || "unknown"}:${call?.id || "active"}`,
        phrases: [activity.label, ...(activity.followUps ?? [])],
      };
    }
    if (getContentString(last.content).length > 0) {
      return live
        ? {
            key: `composing:${last.id || "answer"}`,
            label: COMPOSING_PHRASES[0],
            phrases: COMPOSING_PHRASES,
            icon: Sparkles,
          }
        : null;
    }
    return {
      key: `thinking:${last.id || "active"}`,
      label: THINKING_PHRASES[0],
      phrases: THINKING_PHRASES,
      icon: Bot,
    };
  }

  if (last.type === "tool") {
    return {
      key: `reviewing:${last.name || "tool"}:${last.id || "result"}`,
      label: REVIEWING_PHRASES[0],
      phrases: REVIEWING_PHRASES,
      icon: BookOpen,
    };
  }

  return null;
}

export const deriveLiveActivity = deriveActivity;

/** Live status row rendered under the transcript while a run is in flight. */
export function AgentActivity({
  messages,
  progress,
}: {
  messages: Message[];
  progress?: AgentProgressEvent | null;
}) {
  const activity = progress ? activityFromProgress(progress) : deriveActivity(messages);
  if (!activity) return null;

  if (activity.intent === "image_generation") {
    return (
      <div className="mr-auto w-full max-w-md">
        <div className="imggen-frame">
          <div className="imggen-frame__label">
            <Sparkles className="size-3.5 animate-pulse text-sky-500/80 dark:text-sky-300/80" />
            Generating your image…
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="text-muted-foreground/80 mr-auto flex items-center gap-1.5 px-2 py-1 text-xs">
      <LiveAgentStatus activity={activity} />
    </div>
  );
}
