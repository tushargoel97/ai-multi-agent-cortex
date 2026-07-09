import { Button } from "@/components/ui/button";
import { useThreads } from "@/providers/Thread";
import { Thread } from "@langchain/langgraph-sdk";
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";

import { getContentString } from "../utils";
import { useQueryState } from "nuqs";
import { useChatHistoryOpen } from "@/hooks/use-chat-history-open";
import { setSidebarResizing, setSidebarWidth } from "@/hooks/use-sidebar-width";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ChevronLeft,
  Search,
  Plus,
  Check,
  X,
  Star,
  MoreVertical,
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
import { cn } from "@/lib/utils";
import {
  getThreadLabel,
  isPinned,
  useThreadActions,
  ThreadActionsMenu,
} from "./thread-actions";

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
      className={cn(
        "group relative flex w-full items-center gap-1 rounded-md px-1",
        isActive && "bg-muted",
      )}
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
                <Star className="size-3 shrink-0 fill-amber-400 text-amber-400" />
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
          <div
            className={cn(
              "absolute right-1 top-1/2 -translate-y-1/2 transition-opacity",
              isActive
                ? "opacity-100"
                : "opacity-0 group-hover:opacity-100 focus-within:opacity-100",
            )}
          >
            <ThreadActionsMenu
              pinned={pinned}
              placement="beside"
              triggerTitle="More"
              triggerClassName="inline-flex size-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-background hover:text-foreground"
              triggerActiveClassName="bg-background text-foreground"
              trigger={<MoreVertical className="size-3.5" />}
              onStar={onTogglePin}
              onRename={() => {
                setDraft(getThreadLabel(thread));
                setEditing(true);
              }}
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
  query = "",
}: {
  threads: Thread[];
  onThreadClick?: (threadId: string) => void;
  query?: string;
}) {
  const [threadId, setThreadId] = useQueryState("threadId");
  const { renameThread, deleteThread, togglePin } = useThreadActions();
  const [pendingDelete, setPendingDelete] = useState<Thread | null>(null);

  const q = query.trim().toLowerCase();
  const filtered = q ? threads.filter((t) => threadMatches(t, q)) : threads;
  // Pinned threads float to the top (stable sort preserves recency within).
  const shown = [...filtered].sort(
    (a, b) => Number(isPinned(b)) - Number(isPinned(a)),
  );

  return (
    <>
      <div className="flex h-full w-full flex-col items-start justify-start gap-1 overflow-y-auto pb-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border [&::-webkit-scrollbar-track]:bg-transparent">
        <div className="w-full px-2 pb-1 pt-1">
          <button
            type="button"
            onClick={() => {
              setThreadId(null);
              onThreadClick?.("");
            }}
            className="inline-flex items-center gap-1.5 rounded-md px-2 py-1.5 text-sm font-medium text-foreground transition-colors hover:bg-muted"
          >
            <Plus className="size-4" />
            New chat
          </button>
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
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    setThreadsLoading(true);
    getThreads()
      .then(setThreads)
      .catch(console.error)
      .finally(() => setThreadsLoading(false));
  }, []);

  // Drag the right edge to resize (pointer x == width; sidebar hugs the left edge).
  const startResize = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    setSidebarResizing(true);
    const onMove = (ev: PointerEvent) => setSidebarWidth(ev.clientX);
    const onUp = () => {
      setSidebarResizing(false);
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  };

  const closeSearch = () => {
    setSearchOpen(false);
    setQuery("");
  };

  // Collapse the search field back to its icon on any outside click.
  useEffect(() => {
    if (!searchOpen) return;
    const onDown = (e: PointerEvent) => {
      const t = e.target as HTMLElement;
      if (searchRef.current?.contains(t)) return;
      if (t.closest("[data-search-trigger]")) return;
      setSearchOpen(false);
      setQuery("");
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [searchOpen]);

  return (
    <>
      <div className="shadow-inner-right relative hidden h-screen w-full shrink-0 flex-col items-start justify-start gap-2 border-r-[1px] border-border lg:flex">
        <div className="flex w-full items-center justify-between gap-2 px-3 pt-2">
          <h1 className="truncate text-lg font-semibold tracking-tight">
            History
          </h1>
          <div className="flex shrink-0 items-center gap-0.5">
            <Button
              size="icon"
              variant="ghost"
              className="size-8 hover:bg-muted"
              data-search-trigger
              onClick={() => (searchOpen ? closeSearch() : setSearchOpen(true))}
              title="Search chats"
            >
              <Search className="size-4" />
            </Button>
            <button
              type="button"
              onClick={() => setChatHistoryOpen((p) => !p)}
              title="Collapse sidebar"
              className="inline-flex size-8 items-center justify-center rounded-full border border-border text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <ChevronLeft className="size-4" />
            </button>
          </div>
        </div>

        {threadsLoading ? (
          <ThreadHistoryLoading />
        ) : (
          <ThreadList
            threads={threads}
            query={query}
          />
        )}

        {searchOpen && (
          <div
            ref={searchRef}
            className="animate-in fade-in-0 slide-in-from-right-6 absolute inset-x-2 top-2 z-30"
          >
            <div className="flex items-center gap-2 rounded-full border border-black/10 bg-popover/65 px-3.5 py-2 shadow-lg backdrop-blur-xl backdrop-saturate-150 dark:border-white/10">
              <Search className="size-4 shrink-0 text-muted-foreground" />
              <input
                autoFocus
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Escape" && closeSearch()}
                placeholder="Search chats…"
                className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
              />
              <button
                type="button"
                onClick={closeSearch}
                className="shrink-0 text-muted-foreground hover:text-foreground"
                title="Close search"
              >
                <X className="size-4" />
              </button>
            </div>
          </div>
        )}

        <div
          onPointerDown={startResize}
          className="absolute right-0 top-0 z-40 h-full w-1.5 cursor-col-resize touch-none transition-colors hover:bg-primary/30"
          title="Drag to resize"
        />
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
            className="flex flex-col lg:hidden"
          >
            <SheetHeader>
              <SheetTitle>History</SheetTitle>
            </SheetHeader>
            <div className="flex items-center gap-2 rounded-full border border-border bg-muted/40 px-3 py-1.5">
              <Search className="size-4 shrink-0 text-muted-foreground" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search chats…"
                className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
              />
            </div>
            <ThreadList
              threads={threads}
              query={query}
              onThreadClick={() => setChatHistoryOpen((o) => !o)}
            />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
