import { Button } from "@/components/ui/button";
import { useThreads } from "@/providers/Thread";
import { Thread } from "@langchain/langgraph-sdk";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { getContentString } from "../utils";
import { useQueryState, parseAsBoolean } from "nuqs";
import { useChatHistoryOpen } from "@/hooks/use-chat-history-open";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import {
  PanelRightOpen,
  PanelRightClose,
  Pencil,
  Search,
  Trash2,
  Plus,
  Check,
  X,
  MoreHorizontal,
  Pin,
  PinOff,
} from "lucide-react";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { createClient } from "@/providers/client";
import { getApiKey } from "@/lib/api-key";
import { cn } from "@/lib/utils";
import { useDropdown } from "@/hooks/use-dropdown";

function getThreadLabel(t: Thread): string {
  const metaTitle = (t.metadata as Record<string, unknown> | undefined)?.title;
  if (typeof metaTitle === "string" && metaTitle.trim()) return metaTitle;
  if (
    typeof t.values === "object" &&
    t.values &&
    "messages" in t.values &&
    Array.isArray((t.values as { messages: unknown[] }).messages) &&
    (t.values as { messages: { content: unknown }[] }).messages.length > 0
  ) {
    const firstMessage = (t.values as { messages: { content: unknown }[] })
      .messages[0];
    const text = getContentString(firstMessage.content as never);
    if (text) return text.length > 80 ? text.slice(0, 80) + "…" : text;
  }
  return "New chat";
}

/** Concatenated text of every message in a thread (for cross-thread search). */
function threadText(t: Thread): string {
  const msgs = (t.values as { messages?: { content: unknown }[] } | undefined)
    ?.messages;
  if (!Array.isArray(msgs)) return "";
  return msgs.map((m) => getContentString(m.content as never)).join("\n");
}

/** `q` is expected already lower-cased. */
function threadMatches(t: Thread, q: string): boolean {
  return (
    getThreadLabel(t).toLowerCase().includes(q) ||
    threadText(t).toLowerCase().includes(q)
  );
}

/** A short …context… window around the first match, for the search result row. */
function matchSnippet(t: Thread, q: string): string | null {
  const text = threadText(t);
  const idx = text.toLowerCase().indexOf(q);
  if (idx === -1) return null;
  const start = Math.max(0, idx - 30);
  const end = Math.min(text.length, idx + q.length + 50);
  return (
    (start > 0 ? "…" : "") +
    text.slice(start, end).replace(/\s+/g, " ").trim() +
    (end < text.length ? "…" : "")
  );
}

function isPinned(t: Thread): boolean {
  return Boolean((t.metadata as Record<string, unknown> | undefined)?.pinned);
}

/** ChatGPT-style per-thread "⋯" menu: Rename / Pin / Delete. */
function ThreadMenu({
  pinned,
  onRename,
  onTogglePin,
  onDelete,
}: {
  pinned: boolean;
  onRename: () => void;
  onTogglePin: () => void;
  onDelete: () => void;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const { open, setOpen, mounted, openUp } = useDropdown(rootRef, {
    estimatedHeight: 140,
  });

  const item =
    "flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-sm hover:bg-muted";

  return (
    <div ref={rootRef} className="relative">
      <Button
        size="icon"
        variant="ghost"
        className={cn(
          "size-7 bg-background/70 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-background focus:opacity-100",
          open && "opacity-100",
        )}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
        title="More"
      >
        <MoreHorizontal className="size-3.5" />
      </Button>
      {mounted && (
        <div
          role="menu"
          onClick={(e) => e.stopPropagation()}
          className={cn(
            "animate-in fade-in-0 zoom-in-95 absolute right-0 z-50 w-40 rounded-md border border-border bg-background p-1 shadow-md transition-opacity",
            openUp ? "bottom-full mb-1" : "top-full mt-1",
            !open && "pointer-events-none opacity-0",
          )}
        >
          <button
            className={item}
            onClick={() => {
              setOpen(false);
              onRename();
            }}
          >
            <Pencil className="size-3.5" /> Rename
          </button>
          <button
            className={item}
            onClick={() => {
              setOpen(false);
              onTogglePin();
            }}
          >
            {pinned ? (
              <PinOff className="size-3.5" />
            ) : (
              <Pin className="size-3.5" />
            )}
            {pinned ? "Unpin" : "Pin"}
          </button>
          <button
            className={cn(item, "text-destructive hover:text-destructive")}
            onClick={() => {
              setOpen(false);
              onDelete();
            }}
          >
            <Trash2 className="size-3.5" /> Delete
          </button>
        </div>
      )}
    </div>
  );
}

function ThreadRow({
  thread,
  snippet,
  onClick,
  onRename,
  onTogglePin,
  onDelete,
}: {
  thread: Thread;
  snippet?: string | null;
  onClick: () => void;
  onRename: (newTitle: string) => Promise<void>;
  onTogglePin: () => Promise<void>;
  onDelete: () => Promise<void>;
}) {
  const [threadId] = useQueryState("threadId");
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(getThreadLabel(thread));
  const isActive = threadId === thread.thread_id;
  const pinned = isPinned(thread);

  const commit = async () => {
    const trimmed = draft.trim();
    if (!trimmed) {
      setDraft(getThreadLabel(thread));
      setEditing(false);
      return;
    }
    await onRename(trimmed);
    setEditing(false);
  };

  return (
    <div
      className={`group relative flex w-full items-center gap-1 rounded-md px-1 ${
        isActive ? "bg-muted" : ""
      }`}
    >
      {editing ? (
        <div className="flex w-full items-center gap-1">
          <Input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") commit();
              if (e.key === "Escape") {
                setDraft(getThreadLabel(thread));
                setEditing(false);
              }
            }}
            className="h-8 text-sm"
          />
          <Button
            size="icon"
            variant="ghost"
            className="size-7 shrink-0"
            onClick={commit}
            title="Save"
          >
            <Check className="size-4" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            className="size-7 shrink-0"
            onClick={() => {
              setDraft(getThreadLabel(thread));
              setEditing(false);
            }}
            title="Cancel"
          >
            <X className="size-4" />
          </Button>
        </div>
      ) : (
        <>
          <Button
            variant="ghost"
            className="h-auto min-h-9 w-full flex-1 flex-col items-start justify-center gap-0.5 truncate py-1 pr-7 text-left font-normal"
            onClick={onClick}
          >
            <p className="flex w-full items-center gap-1 truncate text-sm">
              {pinned && (
                <Pin className="size-3 shrink-0 text-muted-foreground" />
              )}
              <span className="truncate text-ellipsis">
                {getThreadLabel(thread)}
              </span>
            </p>
            {snippet && (
              <p className="w-full truncate text-xs font-normal text-muted-foreground">
                {snippet}
              </p>
            )}
          </Button>
          <div className="absolute right-1 top-1/2 -translate-y-1/2">
            <ThreadMenu
              pinned={pinned}
              onRename={() => {
                setDraft(getThreadLabel(thread));
                setEditing(true);
              }}
              onTogglePin={onTogglePin}
              onDelete={onDelete}
            />
          </div>
        </>
      )}
    </div>
  );
}

function ThreadList({
  threads,
  onThreadClick,
}: {
  threads: Thread[];
  onThreadClick?: (threadId: string) => void;
}) {
  const [threadId, setThreadId] = useQueryState("threadId");
  const [apiUrl] = useQueryState("apiUrl", {
    defaultValue: process.env.NEXT_PUBLIC_API_URL || "",
  });
  const [authScheme] = useQueryState("authScheme", {
    defaultValue: process.env.NEXT_PUBLIC_AUTH_SCHEME || "",
  });
  const { setThreads, getThreads } = useThreads();
  const [pendingDelete, setPendingDelete] = useState<Thread | null>(null);
  const [query, setQuery] = useState("");

  const client = () =>
    createClient(
      apiUrl || process.env.NEXT_PUBLIC_API_URL || "",
      getApiKey() ?? undefined,
      authScheme || undefined,
    );

  const renameThread = async (t: Thread, title: string) => {
    try {
      const c = client();
      const newMetadata = { ...(t.metadata ?? {}), title };
      await c.threads.update(t.thread_id, { metadata: newMetadata });
      setThreads((prev) =>
        prev.map((x) =>
          x.thread_id === t.thread_id ? { ...x, metadata: newMetadata } : x,
        ),
      );
      toast.success("Thread renamed");
    } catch (e) {
      console.error(e);
      toast.error("Failed to rename thread");
    }
  };

  const deleteThread = async (t: Thread) => {
    try {
      const c = client();
      await c.threads.delete(t.thread_id);
      setThreads((prev) => prev.filter((x) => x.thread_id !== t.thread_id));
      if (threadId === t.thread_id) setThreadId(null);
      toast.success("Thread deleted");
    } catch (e) {
      console.error(e);
      toast.error("Failed to delete thread");
      // Best-effort refresh in case it actually deleted but errored.
      getThreads().then(setThreads).catch(() => {});
    }
  };

  const togglePin = async (t: Thread) => {
    try {
      const c = client();
      const pinned = !isPinned(t);
      const newMetadata = { ...(t.metadata ?? {}), pinned };
      await c.threads.update(t.thread_id, { metadata: newMetadata });
      setThreads((prev) =>
        prev.map((x) =>
          x.thread_id === t.thread_id ? { ...x, metadata: newMetadata } : x,
        ),
      );
      toast.success(pinned ? "Thread pinned" : "Thread unpinned");
    } catch (e) {
      console.error(e);
      toast.error("Failed to update thread");
    }
  };

  const q = query.trim().toLowerCase();
  const filtered = q ? threads.filter((t) => threadMatches(t, q)) : threads;
  // Pinned threads float to the top (stable sort preserves recency within).
  const shown = [...filtered].sort(
    (a, b) => Number(isPinned(b)) - Number(isPinned(a)),
  );

  return (
    <>
      <div className="flex h-full w-full flex-col items-start justify-start gap-1 overflow-y-auto pb-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border [&::-webkit-scrollbar-track]:bg-transparent">
        <div className="w-full px-2 pb-2">
          <Button
            variant="outline"
            className="w-full justify-start gap-2"
            onClick={() => {
              setThreadId(null);
              onThreadClick?.("");
            }}
          >
            <Plus className="size-4" />
            New chat
          </Button>
        </div>
        <div className="w-full px-2 pb-2">
          <div className="flex items-center gap-2 rounded-full border border-border bg-muted/40 px-3 py-1.5 transition-colors focus-within:border-ring focus-within:bg-background">
            <Search className="size-4 shrink-0 text-muted-foreground" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search chats…"
              className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
            />
            {query && (
              <button
                type="button"
                onClick={() => setQuery("")}
                className="shrink-0 text-muted-foreground hover:text-foreground"
                title="Clear"
              >
                <X className="size-3.5" />
              </button>
            )}
          </div>
        </div>
        {q && shown.length === 0 ? (
          <p className="w-full px-4 py-6 text-center text-sm text-muted-foreground">
            No chats match “{query.trim()}”.
          </p>
        ) : (
          shown.map((t) => (
            <div
              key={t.thread_id}
              className="w-full px-2"
            >
              <ThreadRow
                thread={t}
                snippet={q ? matchSnippet(t, q) : null}
                onClick={() => {
                  onThreadClick?.(t.thread_id);
                  if (t.thread_id === threadId) return;
                  setThreadId(t.thread_id);
                }}
                onRename={(title) => renameThread(t, title)}
                onTogglePin={() => togglePin(t)}
                onDelete={async () => setPendingDelete(t)}
              />
            </div>
          ))
        )}
      </div>

      <Dialog
        open={!!pendingDelete}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete this chat?</DialogTitle>
            <DialogDescription>
              {pendingDelete
                ? `"${getThreadLabel(pendingDelete)}" will be permanently deleted. This action cannot be undone.`
                : ""}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="ghost"
              onClick={() => setPendingDelete(null)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={async () => {
                if (pendingDelete) {
                  await deleteThread(pendingDelete);
                  setPendingDelete(null);
                }
              }}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

function ThreadHistoryLoading() {
  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border [&::-webkit-scrollbar-track]:bg-transparent">
      {Array.from({ length: 30 }).map((_, i) => (
        <Skeleton
          key={`skeleton-${i}`}
          className="h-10 w-[280px]"
        />
      ))}
    </div>
  );
}

export default function ThreadHistory() {
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const [chatHistoryOpen, setChatHistoryOpen] = useChatHistoryOpen();

  const { getThreads, threads, setThreads, threadsLoading, setThreadsLoading } =
    useThreads();

  useEffect(() => {
    if (typeof window === "undefined") return;
    setThreadsLoading(true);
    getThreads()
      .then(setThreads)
      .catch(console.error)
      .finally(() => setThreadsLoading(false));
  }, []);

  return (
    <>
      <div className="shadow-inner-right hidden h-screen w-[300px] shrink-0 flex-col items-start justify-start gap-6 border-r-[1px] border-border lg:flex">
        <div className="flex w-full items-center justify-between px-4 pt-1.5">
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
          <h1 className="text-xl font-semibold tracking-tight">
            Thread History
          </h1>
        </div>
        {threadsLoading ? (
          <ThreadHistoryLoading />
        ) : (
          <ThreadList threads={threads} />
        )}
      </div>
      <div className="lg:hidden">
        <Sheet
          open={!!chatHistoryOpen && !isLargeScreen}
          onOpenChange={(open) => {
            if (isLargeScreen) return;
            setChatHistoryOpen(open);
          }}
        >
          <SheetContent
            side="left"
            className="flex lg:hidden"
          >
            <SheetHeader>
              <SheetTitle>Thread History</SheetTitle>
            </SheetHeader>
            <ThreadList
              threads={threads}
              onThreadClick={() => setChatHistoryOpen((o) => !o)}
            />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
