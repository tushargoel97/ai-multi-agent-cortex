import React, { createContext, useContext, ReactNode, useState, useEffect } from "react";
import { useStream } from "@langchain/langgraph-sdk/react";
import { type Message } from "@langchain/langgraph-sdk";
import {
  uiMessageReducer,
  isUIMessage,
  isRemoveUIMessage,
  type UIMessage,
  type RemoveUIMessage,
} from "@langchain/langgraph-sdk/react-ui";
import { useQueryState } from "nuqs";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { LangGraphLogoSVG } from "@/components/icons/langgraph";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ArrowRight } from "lucide-react";
import { PasswordInput } from "@/components/ui/password-input";
import { getApiKey } from "@/lib/api-key";
import { useThreads } from "./Thread";
import { toast } from "sonner";
import { createClient, resolveApiUrl } from "./client";
import { isAgentProgressEvent, type AgentProgressEvent } from "@/lib/agent-progress";

export type StateType = { messages: Message[]; ui?: UIMessage[] };

const useTypedStream = useStream<
  StateType,
  {
    UpdateType: {
      messages?: Message[] | Message | string;
      ui?: (UIMessage | RemoveUIMessage)[] | UIMessage | RemoveUIMessage;
      context?: Record<string, unknown>;
    };
    CustomEventType: UIMessage | RemoveUIMessage | AgentProgressEvent;
  }
>;

type StreamContextType = ReturnType<typeof useTypedStream> & {
  agentProgress: AgentProgressEvent | null;
};
const StreamContext = createContext<StreamContextType | undefined>(undefined);

async function sleep(ms = 4000) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function deriveTitle(text: string): string {
  const collapsed = text.replace(/\s+/g, " ").trim();
  if (!collapsed) return "";
  return collapsed.length > 60 ? collapsed.slice(0, 60).trimEnd() + "…" : collapsed;
}

async function autoTitleThread(
  threadId: string,
  apiUrl: string,
  apiKey: string | null,
  authScheme?: string,
  onTitled?: () => void,
): Promise<void> {
  const client = createClient(apiUrl, apiKey ?? undefined, authScheme);
  let meta: Record<string, unknown> = {};
  let firstHuman: { content?: unknown } | undefined;

  // The first checkpoint may take a while to commit (model latency), retry
  // instead of giving up and leaving the thread as "Chat <uuid>".
  for (const delay of [600, 2000, 5000, 10000]) {
    await sleep(delay);
    let thread: { metadata?: Record<string, unknown>; values?: unknown };
    try {
      thread = (await client.threads.get(threadId)) as typeof thread;
    } catch {
      continue;
    }
    meta = (thread.metadata ?? {}) as Record<string, unknown>;
    if (typeof meta.title === "string" && meta.title.trim()) return;

    const values = thread.values as
      | { messages?: { type?: string; content?: unknown }[] }
      | undefined;
    firstHuman = values?.messages?.find((m) => (m as { type?: string }).type === "human");
    if (firstHuman) break;
  }
  if (!firstHuman) return;
  const content = firstHuman.content as string | { type?: string; text?: string }[] | undefined;
  let text = "";
  if (typeof content === "string") text = content;
  else if (Array.isArray(content)) {
    text = content.map((c) => (c?.type === "text" ? (c.text ?? "") : "")).join(" ");
  }
  // Model-synthesized title; fall back to a trimmed first message on failure.
  let title = "";
  try {
    const r = await fetch("/api/v1/title", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (r.ok) title = ((await r.json()) as { title?: string })?.title ?? "";
  } catch {
    // network / route error → heuristic fallback below
  }
  if (!title) title = deriveTitle(text);
  if (!title) return;
  try {
    await client.threads.update(threadId, {
      metadata: { ...meta, title },
    });
    onTitled?.();
  } catch (e) {
    console.warn("[stream] failed to set thread title:", e);
  }
}

async function checkGraphStatus(
  apiUrl: string,
  apiKey: string | null,
  authScheme?: string,
): Promise<boolean> {
  try {
    const headers = new Headers();
    if (apiKey) headers.set("X-Api-Key", apiKey);
    if (authScheme) headers.set("X-Auth-Scheme", authScheme);

    const res = await fetch(`${apiUrl}/info`, {
      headers,
    });

    return res.ok;
  } catch (e) {
    console.error(e);
    return false;
  }
}

const StreamSession = ({
  children,
  apiKey,
  apiUrl,
  assistantId,
  authScheme,
}: {
  children: ReactNode;
  apiKey: string | null;
  apiUrl: string;
  assistantId: string;
  authScheme?: string;
}) => {
  const [threadId, setThreadId] = useQueryState("threadId");
  const [agentProgress, setAgentProgress] = useState<AgentProgressEvent | null>(null);
  const { getThreads, setThreads } = useThreads();
  const resolvedApiUrl = resolveApiUrl(apiUrl);
  const streamValue = useTypedStream({
    apiUrl: resolvedApiUrl,
    apiKey: apiKey ?? undefined,
    assistantId,
    ...(authScheme && {
      defaultHeaders: {
        "X-Auth-Scheme": authScheme,
      },
    }),
    threadId: threadId ?? null,
    fetchStateHistory: true,
    // Re-attach to an in-flight run on mount, so switching threads doesn't drop
    // an answer still being generated (pairs with the server's join endpoint).
    reconnectOnMount: true,
    onCustomEvent: (event, options) => {
      if (isAgentProgressEvent(event)) {
        setAgentProgress(event);
      } else if (isUIMessage(event) || isRemoveUIMessage(event)) {
        options.mutate((prev) => {
          const ui = uiMessageReducer(prev.ui ?? [], event);
          return { ...prev, ui };
        });
      }
    },
    onCreated: () => setAgentProgress(null),
    onFinish: () => setAgentProgress(null),
    onStop: () => setAgentProgress(null),
    onThreadId: (id) => {
      setThreadId(id);
      // Refetch threads list when thread ID changes.
      // Wait for some seconds before fetching so we're able to get the new thread that was created.
      sleep().then(() => getThreads().then(setThreads).catch(console.error));
      // Auto-title newly-created threads from their first user message.
      autoTitleThread(id, resolvedApiUrl, apiKey, authScheme, () => {
        // Refresh sidebar so the new title shows up immediately.
        getThreads().then(setThreads).catch(console.error);
      }).catch((e) => console.warn("[stream] auto-title failed:", e));
    },
    onError: (err: unknown) => {
      setAgentProgress(null);
      console.error("[stream] error:", err);
      let title = "Chat error";
      let description = "An error occurred while streaming the response.";
      try {
        const raw =
          err instanceof Error ? err.message : typeof err === "string" ? err : JSON.stringify(err);
        const errName = err instanceof Error ? err.name : (err as { name?: string } | null)?.name;
        const errStatus = (err as { status?: number; code?: number } | null)?.status;
        // Stale thread (server restarted / in-memory state gone) → recover.
        const looksLikeMissingThread =
          errName === "NotFoundError" ||
          errStatus === 404 ||
          (/404|not found|thread.*not.*exist/i.test(raw) && /thread/i.test(raw));
        if (looksLikeMissingThread) {
          setThreadId(null);
          getThreads()
            .then(setThreads)
            .catch(() => {});
          toast.message("Started a new conversation", {
            description: "The previous thread is no longer available. Send your message again.",
            duration: 6000,
          });
          return;
        }
        // Try to extract Google/OpenAI/Anthropic-style nested JSON
        const jsonMatch = raw.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          try {
            const parsed = JSON.parse(jsonMatch[0]);
            const inner =
              parsed?.error?.message ??
              parsed?.message ??
              (typeof parsed?.error === "string" ? parsed.error : null);
            if (inner) {
              try {
                const innerParsed = JSON.parse(inner);
                description = innerParsed?.error?.message ?? innerParsed?.message ?? inner;
              } catch {
                description = inner;
              }
            }
          } catch {
            description = raw;
          }
        } else if (raw) {
          description = raw;
        }
        if (/429|RESOURCE_EXHAUSTED|quota/i.test(raw)) {
          title = "Provider quota exceeded";
        } else if (/401|unauthorized|api key/i.test(raw)) {
          title = "Invalid API key";
        }
      } catch {
        // ignore parsing errors
      }
      toast.error(title, {
        description,
        duration: 12000,
        richColors: true,
        closeButton: true,
      });
    },
  });

  useEffect(() => {
    checkGraphStatus(resolvedApiUrl, apiKey, authScheme).then((ok) => {
      if (!ok) {
        toast.error("Failed to connect to LangGraph server", {
          description: () => (
            <p>
              Please ensure your graph is running at <code>{resolvedApiUrl}</code> and your API key
              is correctly set (if connecting to a deployed graph).
            </p>
          ),
          duration: 10000,
          richColors: true,
          closeButton: true,
        });
      }
    });
  }, [apiKey, authScheme, resolvedApiUrl]);

  return (
    <StreamContext.Provider value={{ ...streamValue, agentProgress }}>
      {children}
    </StreamContext.Provider>
  );
};

// Default values for the form
const DEFAULT_API_URL = "http://localhost:2024/api/v1";
const DEFAULT_ASSISTANT_ID = "agent";
const AGENT_BUILDER_AUTH_SCHEME = "langsmith-api-key";

export const StreamProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // Get environment variables
  const envApiUrl: string | undefined = process.env.NEXT_PUBLIC_API_URL;
  const envAssistantId: string | undefined = process.env.NEXT_PUBLIC_ASSISTANT_ID;
  const envAuthScheme: string | undefined = process.env.NEXT_PUBLIC_AUTH_SCHEME;

  // Use URL params with env var fallbacks
  const [apiUrl, setApiUrl] = useQueryState("apiUrl", {
    defaultValue: envApiUrl || "",
  });
  const [assistantId, setAssistantId] = useQueryState("assistantId", {
    defaultValue: envAssistantId || "",
  });
  const [authScheme, setAuthScheme] = useQueryState("authScheme", {
    defaultValue: envAuthScheme || "",
  });
  const [isAgentBuilder, setIsAgentBuilder] = useState(
    () => (authScheme || envAuthScheme || "").toLowerCase() === AGENT_BUILDER_AUTH_SCHEME,
  );

  // For API key, use localStorage with env var fallback
  const [apiKey, _setApiKey] = useState(() => {
    const storedKey = getApiKey();
    return storedKey || "";
  });

  const setApiKey = (key: string) => {
    window.localStorage.setItem("lg:chat:apiKey", key);
    _setApiKey(key);
  };

  // Determine final values to use, prioritizing URL params then env vars
  const finalApiUrl = apiUrl || envApiUrl;
  const finalAssistantId = assistantId || envAssistantId;
  const finalAuthScheme = authScheme || envAuthScheme || "";

  // Show the form if we: don't have an API URL, or don't have an assistant ID
  if (!finalApiUrl || !finalAssistantId) {
    return (
      <div className="flex min-h-screen w-full items-center justify-center p-4">
        <div className="animate-in fade-in-0 zoom-in-95 bg-background flex max-w-3xl flex-col rounded-lg border shadow-lg">
          <div className="mt-14 flex flex-col gap-2 border-b p-6">
            <div className="flex flex-col items-start gap-2">
              <LangGraphLogoSVG className="h-7" />
              <h1 className="text-xl font-semibold tracking-tight">Agent Chat</h1>
            </div>
            <p className="text-muted-foreground">
              Welcome to Agent Chat! Before you get started, you need to enter the URL of the
              deployment and the assistant / graph ID.
            </p>
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();

              const form = e.target as HTMLFormElement;
              const formData = new FormData(form);
              const apiUrl = formData.get("apiUrl") as string;
              const assistantId = formData.get("assistantId") as string;
              const apiKey = formData.get("apiKey") as string;

              setApiUrl(apiUrl);
              setApiKey(apiKey);
              setAssistantId(assistantId);
              setAuthScheme(isAgentBuilder ? AGENT_BUILDER_AUTH_SCHEME : "");

              form.reset();
            }}
            className="bg-muted/50 flex flex-col gap-6 p-6"
          >
            <div className="flex flex-col gap-2">
              <Label htmlFor="apiUrl">
                Deployment URL<span className="text-rose-500">*</span>
              </Label>
              <p className="text-muted-foreground text-sm">
                This is the URL of your LangGraph deployment. Can be a local, or production
                deployment.
              </p>
              <Input
                id="apiUrl"
                name="apiUrl"
                className="bg-background"
                defaultValue={apiUrl || DEFAULT_API_URL}
                required
              />
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="assistantId">
                Assistant / Graph ID<span className="text-rose-500">*</span>
              </Label>
              <p className="text-muted-foreground text-sm">
                This is the ID of the graph (can be the graph name), or assistant to fetch threads
                from, and invoke when actions are taken.
              </p>
              <Input
                id="assistantId"
                name="assistantId"
                className="bg-background"
                defaultValue={assistantId || DEFAULT_ASSISTANT_ID}
                required
              />
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="apiKey">LangSmith API Key</Label>
              <p className="text-muted-foreground text-sm">
                This is <strong>NOT</strong> required if using a local LangGraph server. This value
                is stored in your browser's local storage and is only used to authenticate requests
                sent to your LangGraph server.
              </p>
              <PasswordInput
                id="apiKey"
                name="apiKey"
                defaultValue={apiKey ?? ""}
                className="bg-background"
                placeholder="lsv2_pt_..."
              />
            </div>

            <div className="flex flex-col gap-3">
              <div className="flex items-center justify-between gap-4">
                <div className="flex flex-col gap-1">
                  <Label htmlFor="agentBuilderEnabled">Built with Agent Builder</Label>
                  <p className="text-muted-foreground text-sm">
                    Enable this for Agent Builder deployments.
                  </p>
                </div>
                <Switch
                  id="agentBuilderEnabled"
                  checked={isAgentBuilder}
                  onCheckedChange={setIsAgentBuilder}
                />
              </div>
            </div>

            <div className="mt-2 flex justify-end">
              <Button type="submit" size="lg">
                Continue
                <ArrowRight className="size-5" />
              </Button>
            </div>
          </form>
        </div>
      </div>
    );
  }

  return (
    <StreamSession
      apiKey={apiKey}
      apiUrl={finalApiUrl}
      assistantId={finalAssistantId}
      authScheme={finalAuthScheme || undefined}
    >
      {children}
    </StreamSession>
  );
};

// Create a custom hook to use the context
export const useStreamContext = (): StreamContextType => {
  const context = useContext(StreamContext);
  if (context === undefined) {
    throw new Error("useStreamContext must be used within a StreamProvider");
  }
  return context;
};

export default StreamContext;
