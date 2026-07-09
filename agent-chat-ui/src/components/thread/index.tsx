import { v4 as uuidv4 } from "uuid";
import { ReactNode, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import { getApiKey } from "@/lib/api-key";
import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { Checkpoint, Message } from "@langchain/langgraph-sdk";
import { AssistantMessage } from "./messages/ai";
import {
  AgentActivity,
  getRoutingIntent,
  isInternalNoiseMessage,
} from "./agent-activity";
import { AgentTrace } from "./agent-trace";
import { ThreadSearch } from "./thread-search";
import { ActivityPanel } from "./activity-panel";
import { getContentString } from "./utils";
import { HumanMessage } from "./messages/human";
import {
  DO_NOT_RENDER_ID_PREFIX,
  ensureToolCallsHaveResponses,
} from "@/lib/ensure-tool-responses";
import { LangGraphLogoSVG } from "../icons/langgraph";
import { TooltipIconButton } from "./tooltip-icon-button";
import {
  Activity,
  ArrowDown,
  ArrowUp,
  Clock,
  PanelRightOpen,
  PanelRightClose,
  Search,
  Square,
  SquarePen,
  XIcon,
  Plus,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { useChatHistoryOpen } from "@/hooks/use-chat-history-open";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import ThreadHistory from "./history";
import { ChatHeaderTitle } from "./chat-header-title";
import { toast } from "sonner";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Label } from "../ui/label";
import { GitHubSVG } from "../icons/github";
import { ThemeToggle } from "./theme-toggle";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import { useFileUpload } from "@/hooks/use-file-upload";
import { ContentBlocksPreview } from "./ContentBlocksPreview";
import ModelSelector, { ModeSelector } from "@/components/model-selector";
import { useModelSelection } from "@/providers/ModelSelection";
import {
  useArtifactOpen,
  ArtifactContent,
  ArtifactTitle,
  useArtifactContext,
} from "./artifact";

// Commerce tool results render as answer cards (ShoppingCards / BookingCards),
// so they stay visible instead of collapsing into the activity trace.
const COMMERCE_TOOLS = new Set(["product_prices", "find_bookings"]);

/** Intermediate "activity" messages that fold into the collapsible trace:
 *  routing markers, tool-call requests, and plain tool-result dumps. The final
 *  text answer and commerce cards are kept out so they render normally. */
function isTraceMessage(m: Message): boolean {
  if (m.type === "tool") {
    return !COMMERCE_TOOLS.has((m as { name?: string }).name ?? "");
  }
  if (m.type === "ai") {
    if (getRoutingIntent(m)) return true;
    const hasToolCalls =
      ((m as { tool_calls?: unknown[] }).tool_calls?.length ?? 0) > 0;
    if (hasToolCalls) return true;
    // No visible text yet (thinking-only / still streaming) → activity.
    return getContentString(m.content).trim().length === 0;
  }
  return false;
}

type RenderItem =
  | { kind: "message"; message: Message }
  | { kind: "trace"; messages: Message[] };

/** Fold each turn's consecutive activity messages into one trace item so the
 *  transcript shows a clean answer with a collapsible "Thought process". */
function groupTurns(messages: Message[]): RenderItem[] {
  const items: RenderItem[] = [];
  let buffer: Message[] = [];
  const flush = () => {
    if (buffer.length) {
      items.push({ kind: "trace", messages: buffer });
      buffer = [];
    }
  };
  for (const m of messages) {
    if (m.type !== "human" && isTraceMessage(m)) {
      buffer.push(m);
    } else {
      flush();
      items.push({ kind: "message", message: m });
    }
  }
  flush();
  return items;
}

function StickyToBottomContent(props: {
  content: ReactNode;
  footer?: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const context = useStickToBottomContext();
  return (
    <div
      ref={context.scrollRef}
      style={{ width: "100%", height: "100%" }}
      className={props.className}
    >
      <div
        ref={context.contentRef}
        className={props.contentClassName}
      >
        {props.content}
      </div>

      {props.footer}
    </div>
  );
}

function ScrollToBottom(props: { className?: string }) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();

  if (isAtBottom) return null;
  return (
    <Button
      variant="outline"
      className={props.className}
      onClick={() => scrollToBottom()}
    >
      <ArrowDown className="h-4 w-4" />
      <span>Scroll to bottom</span>
    </Button>
  );
}

function OpenGitHubRepo() {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <a
            href="https://github.com/langchain-ai/agent-chat-ui"
            target="_blank"
            className="flex items-center justify-center"
          >
            <GitHubSVG
              width="24"
              height="24"
            />
          </a>
        </TooltipTrigger>
        <TooltipContent side="left">
          <p>Open GitHub repo</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

/** Time-of-day greeting shown on an empty thread. Computed after mount to
 *  avoid a server/client hydration mismatch on the hour. */
function Greeting() {
  const [greeting, setGreeting] = useState("Hello");
  useEffect(() => {
    const h = new Date().getHours();
    setGreeting(
      h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening",
    );
  }, []);
  return (
    <div className="flex flex-col items-center gap-1.5 text-center">
      <h1 className="text-2xl font-semibold tracking-tight">{greeting}!</h1>
      <p className="text-muted-foreground">How can I help you today?</p>
    </div>
  );
}

export function Thread() {
  const [artifactContext, setArtifactContext] = useArtifactContext();
  const [artifactOpen, closeArtifact] = useArtifactOpen();

  const [threadId, _setThreadId] = useQueryState("threadId");
  const [apiUrl] = useQueryState("apiUrl", {
    defaultValue: process.env.NEXT_PUBLIC_API_URL || "",
  });
  const [authScheme] = useQueryState("authScheme", {
    defaultValue: process.env.NEXT_PUBLIC_AUTH_SCHEME || "",
  });
  const [chatHistoryOpen, setChatHistoryOpen] = useChatHistoryOpen();
  useEffect(() => {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    if (url.searchParams.has("chatHistoryOpen")) {
      url.searchParams.delete("chatHistoryOpen");
      window.history.replaceState({}, "", url.toString());
    }
  }, []);
  const [hideToolCalls, setHideToolCalls] = useQueryState(
    "hideToolCalls",
    parseAsBoolean.withDefault(false),
  );
  const [input, setInput] = useState("");
  const {
    contentBlocks,
    setContentBlocks,
    handleFileUpload,
    dropRef,
    removeBlock,
    resetBlocks: _resetBlocks,
    dragOver,
    handlePaste,
  } = useFileUpload();
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const stream = useStreamContext();
  const messages = stream.messages;
  const { selection, setSelection, buildConfigurable } = useModelSelection();
  const isLoading = stream.isLoading;

  const lastError = useRef<string | undefined>(undefined);

  const setThreadId = (id: string | null) => {
    _setThreadId(id);

    // close artifact and reset artifact context
    closeArtifact();
    setArtifactContext({});
  };

  useEffect(() => {
    if (!stream.error) {
      lastError.current = undefined;
      return;
    }
    try {
      const message = (stream.error as any).message;
      if (!message || lastError.current === message) {
        // Message has already been logged. do not modify ref, return early.
        return;
      }

      // Message is defined, and it has not been logged yet. Save it, and send the error
      lastError.current = message;
      toast.error("An error occurred. Please try again.", {
        description: (
          <p>
            <strong>Error:</strong> <code>{message}</code>
          </p>
        ),
        richColors: true,
        closeButton: true,
      });
    } catch {
      // no-op
    }
  }, [stream.error]);

  const [pending, setPending] = useState<
    { text: string; blocks: typeof contentBlocks } | null
  >(null);
  const lastSubmitRef = useRef<{ text: string; at: number } | null>(null);

  const submitMessage = (text: string, blocks: typeof contentBlocks) => {
    const trimmed = text.trim();
    if (trimmed.length === 0 && blocks.length === 0) return;

    const newHumanMessage: Message = {
      id: uuidv4(),
      type: "human",
      content: [
        ...(trimmed.length > 0 ? [{ type: "text", text }] : []),
        ...blocks,
      ] as Message["content"],
    };

    const toolMessages = threadId
      ? ensureToolCallsHaveResponses(stream.messages)
      : [];

    const context =
      Object.keys(artifactContext).length > 0 ? artifactContext : undefined;

    stream.submit(
      { messages: [...toolMessages, newHumanMessage], context },
      {
        streamMode: ["values"],
        streamSubgraphs: true,
        streamResumable: true,
        config: { configurable: buildConfigurable() },
        optimisticValues: (prev) => ({
          ...prev,
          context,
          messages: [
            ...(threadId ? (prev.messages ?? []) : []),
            ...toolMessages,
            newHumanMessage,
          ],
        }),
      },
    );

    lastSubmitRef.current = { text: trimmed, at: Date.now() };
    setInput("");
    setContentBlocks([]);
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (trimmed.length === 0 && contentBlocks.length === 0) return;

    const lastHuman = [...messages].reverse().find((m) => m.type === "human");
    const lastHumanText = lastHuman
      ? getContentString(lastHuman.content).trim()
      : "";
    const isRepeat = trimmed.length > 0 && trimmed === lastHumanText;

    // Swallow accidental duplicates (dup of the running turn, or rapid re-send).
    if (isRepeat && isLoading) {
      toast.info("That message is already being answered.");
      return;
    }
    const recent = lastSubmitRef.current;
    if (
      isRepeat &&
      recent &&
      recent.text === trimmed &&
      Date.now() - recent.at < 3000
    ) {
      return;
    }

    // While a run streams, queue this message instead of dropping it.
    if (isLoading) {
      setPending({ text: input, blocks: contentBlocks });
      setInput("");
      setContentBlocks([]);
      toast.message("Queued, sends when the current reply finishes.");
      return;
    }

    submitMessage(input, contentBlocks);
  };

  // Flush a queued message once the in-flight run finishes.
  useEffect(() => {
    if (isLoading || !pending) return;
    const p = pending;
    setPending(null);
    submitMessage(p.text, p.blocks);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoading, pending]);

  const handleRegenerate = (
    parentCheckpoint: Checkpoint | null | undefined,
  ) => {
    stream.submit(undefined, {
      checkpoint: parentCheckpoint,
      streamMode: ["values"],
      streamSubgraphs: true,
      streamResumable: true,
      config: { configurable: buildConfigurable() },
    });
  };

  const handleCancel = () => {
    stream.stop();
    setPending(null);
    // stop() only detaches the stream; cancel the detached server run too.
    if (threadId && apiUrl) {
      const headers: Record<string, string> = {};
      const key = getApiKey();
      if (key) headers["X-Api-Key"] = key;
      if (authScheme) headers["X-Auth-Scheme"] = authScheme;
      fetch(`${apiUrl}/threads/${threadId}/runs/cancel`, {
        method: "POST",
        headers,
      }).catch(() => {});
    }
  };

  const chatStarted = !!threadId || !!messages.length;
  const hasNoAIOrToolMessages = !messages.find(
    (m) => m.type === "ai" || m.type === "tool",
  );

  const [threadSearchOpen, setThreadSearchOpen] = useState(false);
  const [activityOpen, setActivityOpen] = useState(false);
  const messagesScopeRef = useRef<HTMLDivElement>(null);

  // ⌘/Ctrl+F searches the open conversation instead of the browser page.
  useEffect(() => {
    if (!chatStarted) return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "f") {
        e.preventDefault();
        setThreadSearchOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [chatStarted]);

  return (
    <div className="flex h-screen w-full overflow-hidden">
      {chatStarted && (
        <ActivityPanel
          messages={messages}
          live={isLoading}
          open={activityOpen}
          onClose={() => setActivityOpen(false)}
        />
      )}
      <div className="relative hidden lg:flex">
        <motion.div
          className="absolute z-20 h-full overflow-hidden border-r bg-background"
          style={{ width: 300 }}
          animate={
            isLargeScreen
              ? { x: chatHistoryOpen ? 0 : -300 }
              : { x: chatHistoryOpen ? 0 : -300 }
          }
          initial={{ x: -300 }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          <div
            className="relative h-full"
            style={{ width: 300 }}
          >
            <ThreadHistory />
          </div>
        </motion.div>
      </div>

      <div
        className={cn(
          "grid w-full grid-cols-[1fr_0fr] transition-all duration-500",
          artifactOpen && "grid-cols-[3fr_2fr]",
        )}
      >
        <motion.div
          className={cn(
            "relative flex min-w-0 flex-1 flex-col overflow-hidden",
            !chatStarted && "grid-rows-[1fr]",
          )}
          layout={isLargeScreen}
          animate={{
            marginLeft: chatHistoryOpen ? (isLargeScreen ? 300 : 0) : 0,
            width: chatHistoryOpen
              ? isLargeScreen
                ? "calc(100% - 300px)"
                : "100%"
              : "100%",
          }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          {!chatStarted && (
            <div className="absolute top-0 left-0 z-10 flex w-full items-center justify-between gap-3 p-2 pl-4">
              <div>
                {(!chatHistoryOpen || !isLargeScreen) && (
                  <Button
                    className="hover:bg-muted"
                    variant="ghost"
                    onClick={() => setChatHistoryOpen((p) => !p)}
                  >
                    {chatHistoryOpen ? (
                      <PanelRightOpen className="size-5" />
                    ) : (
                      <PanelRightClose className="size-5" />
                    )}
                  </Button>
                )}
              </div>
              <div className="absolute top-2 right-4 flex items-center">
                <OpenGitHubRepo />
              </div>
            </div>
          )}
          {chatStarted && (
            <div className="relative z-10 flex items-center justify-between gap-3 p-2">
              <div className="relative flex items-center justify-start gap-2">
                <div className="absolute left-0 z-10">
                  {(!chatHistoryOpen || !isLargeScreen) && (
                    <Button
                      className="hover:bg-muted"
                      variant="ghost"
                      onClick={() => setChatHistoryOpen((p) => !p)}
                    >
                      {chatHistoryOpen ? (
                        <PanelRightOpen className="size-5" />
                      ) : (
                        <PanelRightClose className="size-5" />
                      )}
                    </Button>
                  )}
                </div>
                <motion.button
                  className="flex cursor-pointer items-center gap-2"
                  onClick={() => setThreadId(null)}
                  animate={{
                    marginLeft: !chatHistoryOpen ? 48 : 0,
                  }}
                  transition={{
                    type: "spring",
                    stiffness: 300,
                    damping: 30,
                  }}
                >
                  <LangGraphLogoSVG
                    width={32}
                    height={32}
                  />
                  <span className="text-xl font-semibold tracking-tight">
                    Cortex
                  </span>
                </motion.button>
                <div className="bg-border mx-1 hidden h-5 w-px sm:block" />
                <ChatHeaderTitle />
              </div>

              <div className="flex items-center gap-4">
                <div className="flex items-center">
                  <OpenGitHubRepo />
                </div>
                <TooltipIconButton
                  size="lg"
                  className="p-4"
                  tooltip="Search conversation (⌘F)"
                  variant="ghost"
                  onClick={() => setThreadSearchOpen((o) => !o)}
                >
                  <Search className="size-5" />
                </TooltipIconButton>
                <TooltipIconButton
                  size="lg"
                  className="p-4"
                  tooltip="Activity & sources"
                  variant="ghost"
                  onClick={() => setActivityOpen((o) => !o)}
                >
                  <Activity className="size-5" />
                </TooltipIconButton>
                <ThemeToggle />
                <TooltipIconButton
                  size="lg"
                  className="p-4"
                  tooltip="New thread"
                  variant="ghost"
                  onClick={() => setThreadId(null)}
                >
                  <SquarePen className="size-5" />
                </TooltipIconButton>
              </div>

              <div className="from-background to-background/0 absolute inset-x-0 top-full h-5 bg-gradient-to-b" />
            </div>
          )}

          <StickToBottom className="relative flex-1 overflow-hidden">
            {chatStarted && (
              <ThreadSearch
                scopeRef={messagesScopeRef}
                messages={messages}
                open={threadSearchOpen}
                onClose={() => setThreadSearchOpen(false)}
              />
            )}
            <StickyToBottomContent
              className={cn(
                "absolute inset-0 overflow-y-scroll px-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border [&::-webkit-scrollbar-track]:bg-transparent",
                !chatStarted && "mt-[25vh] flex flex-col items-stretch",
                chatStarted && "grid grid-rows-[1fr_auto]",
              )}
              contentClassName="pt-8 pb-16 max-w-3xl mx-auto flex flex-col gap-4 w-full"
              content={
                (() => {
                  const visible = messages.filter(
                    (m) =>
                      !m.id?.startsWith(DO_NOT_RENDER_ID_PREFIX) &&
                      !isInternalNoiseMessage(m),
                  );
                  const items = groupTurns(visible);
                  const lastVisible = visible[visible.length - 1];
                  return (
                    <div ref={messagesScopeRef} style={{ display: "contents" }}>
                      {items.map((item, index) =>
                        item.kind === "trace" ? (
                          <AgentTrace
                            key={item.messages[0]?.id || `trace-${index}`}
                            messages={item.messages}
                            live={isLoading && index === items.length - 1}
                          />
                        ) : item.message.type === "human" ? (
                          <HumanMessage
                            key={item.message.id || `human-${index}`}
                            message={item.message}
                            isLoading={isLoading}
                          />
                        ) : (
                          <AssistantMessage
                            key={item.message.id || `ai-${index}`}
                            message={item.message}
                            isLoading={isLoading}
                            handleRegenerate={handleRegenerate}
                          />
                        ),
                      )}
                      {/* Special rendering case where there are no AI/tool
                        messages, but there is an interrupt, render it outside
                        the messages list since there are no messages to render */}
                      {hasNoAIOrToolMessages && !!stream.interrupt && (
                        <AssistantMessage
                          key="interrupt-msg"
                          message={undefined}
                          isLoading={isLoading}
                          handleRegenerate={handleRegenerate}
                        />
                      )}
                      {/* Initial "routing" beat, before the first activity
                        message arrives; the live trace covers the rest. */}
                      {isLoading && lastVisible?.type === "human" && (
                        <AgentActivity messages={messages} />
                      )}
                    </div>
                  );
                })()
              }
              footer={
                <div className="sticky bottom-0 flex flex-col items-center gap-8 bg-background">
                  {!chatStarted && <Greeting />}

                  <ScrollToBottom className="animate-in fade-in-0 zoom-in-95 absolute bottom-full left-1/2 mb-4 -translate-x-1/2" />

                  <div
                    ref={dropRef}
                    data-prompt-composer
                    className={cn(
                      "bg-muted relative z-10 mx-auto mb-8 w-full max-w-[46rem] rounded-3xl shadow-sm transition-all",
                      dragOver
                        ? "border-primary border-2 border-dotted"
                        : "border border-solid",
                    )}
                  >
                    <form
                      onSubmit={handleSubmit}
                      className="mx-auto grid max-w-[46rem] grid-rows-[1fr_auto] gap-2"
                    >
                      {pending && (
                        <div className="mx-3.5 mt-3 flex items-center gap-2 rounded-lg border border-dashed border-border bg-muted/50 px-3 py-1.5 text-xs text-muted-foreground">
                          <Clock className="size-3.5 shrink-0 animate-pulse" />
                          <span className="min-w-0 flex-1 truncate">
                            Queued:{" "}
                            {pending.text.trim() ||
                              `${pending.blocks.length} attachment(s)`}
                          </span>
                          <button
                            type="button"
                            onClick={() => setPending(null)}
                            className="shrink-0 hover:text-foreground"
                            title="Cancel queued message"
                          >
                            <XIcon className="size-3.5" />
                          </button>
                        </div>
                      )}
                      <ContentBlocksPreview
                        blocks={contentBlocks}
                        onRemove={removeBlock}
                      />
                      <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onPaste={handlePaste}
                        onKeyDown={(e) => {
                          if (
                            e.key === "Enter" &&
                            !e.shiftKey &&
                            !e.metaKey &&
                            !e.nativeEvent.isComposing
                          ) {
                            e.preventDefault();
                            const el = e.target as HTMLElement | undefined;
                            const form = el?.closest("form");
                            form?.requestSubmit();
                          }
                        }}
                        placeholder="Type your message..."
                        className="field-sizing-content max-h-[280px] min-h-[52px] resize-none border-none bg-transparent p-4 pb-0 shadow-none ring-0 outline-none focus:ring-0 focus:outline-none"
                      />

                      <div className="flex flex-wrap items-center gap-x-2 gap-y-2 px-2.5 pb-2.5 pt-1">
                        <Label
                          htmlFor="file-input"
                          title="Attach a PDF or image"
                          className="flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                        >
                          <Plus className="size-5" />
                        </Label>
                        <input
                          id="file-input"
                          type="file"
                          onChange={handleFileUpload}
                          multiple
                          accept="image/jpeg,image/png,image/gif,image/webp,application/pdf"
                          className="hidden"
                        />
                        <ModelSelector
                          selection={selection}
                          onChange={setSelection}
                          hideToolCalls={hideToolCalls ?? false}
                          onHideToolCallsChange={(v) => setHideToolCalls(v)}
                        />
                        <ModeSelector
                          className="ml-auto"
                          mode={selection.mode}
                          onModeChange={(m) =>
                            setSelection({ ...selection, mode: m })
                          }
                        />
                        {stream.isLoading ? (
                          <Button
                            key="stop"
                            type="button"
                            size="icon"
                            title="Stop generating"
                            onClick={handleCancel}
                            className="size-9 rounded-full"
                          >
                            <Square className="size-3.5 fill-current" />
                          </Button>
                        ) : (
                          <Button
                            type="submit"
                            size="icon"
                            title="Send"
                            className="size-9 rounded-full shadow-sm transition-all"
                            disabled={
                              isLoading ||
                              (!input.trim() && contentBlocks.length === 0)
                            }
                          >
                            <ArrowUp className="size-5" />
                          </Button>
                        )}
                      </div>
                    </form>
                  </div>
                </div>
              }
            />
          </StickToBottom>
        </motion.div>
        <div className="relative flex flex-col border-l">
          <div className="absolute inset-0 flex min-w-[30vw] flex-col">
            <div className="grid grid-cols-[1fr_auto] border-b p-4">
              <ArtifactTitle className="truncate overflow-hidden" />
              <button
                onClick={closeArtifact}
                className="cursor-pointer"
              >
                <XIcon className="size-5" />
              </button>
            </div>
            <ArtifactContent className="relative flex-grow" />
          </div>
        </div>
      </div>
    </div>
  );
}
