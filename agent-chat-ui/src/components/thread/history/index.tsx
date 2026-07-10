import { Button } from "@/components/ui/button";
import { useThreads } from "@/providers/Thread";
import { Thread } from "@langchain/langgraph-sdk";
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";

import { getContentString } from "../utils";
import { useQueryState } from "nuqs";
import { useChatHistoryOpen } from "@/hooks/use-chat-history-open";
import { setSidebarResizing, setSidebarWidth } from "@/hooks/use-sidebar-width";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { ChevronLeft, Search, Plus, Check, X, Star, MoreVertical } from "lucide-react";
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
import { getThreadLabel, isPinned, useThreadActions, ThreadActionsMenu } from "./thread-actions";

/** Concatenated text of every message in a thread (for cross-thread search). */
function threadText(t: Thread): string {
  const msgs = (t.values as { messages?: { content: unknown }[] } | undefined)?.messages;
  if (!Array.isArray(msgs)) return "";
  return msgs.map((m) => getContentString(m.content as never)).join("\n");
}

/** `q` is expected already lower-cased. */
function threadMatches(t: Thread, q: string): boolean {
  return getThreadLabel(t).toLowerCase().includes(q) || threadText(t).toLowerCase().includes(q);
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
        "group hover:bg-muted/60 relative flex w-full items-center gap-1 rounded-full px-1 transition-colors",
        isActive && "bg-muted hover:bg-muted",
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
            className={cn(
              "h-auto min-h-9 w-full flex-1 flex-col items-start justify-center gap-0.5 truncate py-1 pr-7 text-left hover:bg-transparent",
              isActive ? "font-medium" : "font-normal",
            )}
            onClick={onClick}
          >
            <p className="flex w-full items-center gap-1 truncate text-sm">
              {pinned && <Star className="size-3 shrink-0 fill-amber-400 text-amber-400" />}
              <span className="truncate text-ellipsis">{getThreadLabel(thread)}</span>
            </p>
            {snippet && (
              <p className="text-muted-foreground w-full truncate text-xs font-normal">{snippet}</p>
            )}
          </Button>
          <div
            className={cn(
              "absolute top-1/2 right-1 -translate-y-1/2 transition-opacity",
              isActive
                ? "opacity-100"
                : "opacity-0 group-hover:opacity-100 focus-within:opacity-100",
            )}
          >
            <ThreadActionsMenu
              pinned={pinned}
              placement="beside"
              triggerTitle="More"
              triggerClassName="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-background hover:text-foreground"
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
  const shown = [...filtered].sort((a, b) => Number(isPinned(b)) - Number(isPinned(a)));

  return (
    <>
      <div className="[&::-webkit-scrollbar-thumb]:bg-border flex h-full w-full flex-col items-start justify-start gap-1 overflow-y-auto pb-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-track]:bg-transparent">
        <div className="w-full px-2 pt-1 pb-1">
          <button
            type="button"
            onClick={() => {
              setThreadId(null);
              onThreadClick?.("");
            }}
            className="text-foreground hover:bg-muted ml-2 inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm font-medium transition-colors"
          >
            <Plus className="size-4" />
            New chat
          </button>
        </div>
        {q && shown.length === 0 ? (
          <p className="text-muted-foreground w-full px-4 py-6 text-center text-sm">
            No chats match “{query.trim()}”.
          </p>
        ) : (
          shown.map((t) => (
            <div key={t.thread_id} className="w-full px-2">
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
            <Button variant="ghost" onClick={() => setPendingDelete(null)}>
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
    <div className="[&::-webkit-scrollbar-thumb]:bg-border flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-track]:bg-transparent">
      {Array.from({ length: 30 }).map((_, i) => (
        <Skeleton key={`skeleton-${i}`} className="h-10 w-[280px]" />
      ))}
    </div>
  );
}

export default function ThreadHistory() {
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const [chatHistoryOpen, setChatHistoryOpen] = useChatHistoryOpen();

  const { getThreads, threads, setThreads, threadsLoading, setThreadsLoading } = useThreads();
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

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

  // Focus the field once it has expanded open.
  useEffect(() => {
    if (searchOpen) searchInputRef.current?.focus();
  }, [searchOpen]);

  return (
    <>
      <div className="shadow-inner-right border-border relative hidden h-screen w-full shrink-0 flex-col items-start justify-start gap-2 border-r-[1px] lg:flex">
        <div
          className={cn(
            "flex w-full items-center justify-between gap-2 pt-2 pr-3 pl-7",
            searchOpen && "invisible",
          )}
        >
          <h1 className="truncate text-lg font-semibold tracking-tight">History</h1>
          <div className="flex shrink-0 items-center gap-0.5">
            <Button
              size="icon"
              variant="ghost"
              className="hover:bg-muted size-8"
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
              className="text-muted-foreground hover:bg-muted hover:text-foreground inline-flex size-8 items-center justify-center rounded-full transition-colors"
            >
              <ChevronLeft className="size-4" />
            </button>
          </div>
        </div>

        {threadsLoading ? <ThreadHistoryLoading /> : <ThreadList threads={threads} query={query} />}

        <div className="pointer-events-none absolute inset-x-2 top-2 z-30 flex justify-end">
          <div
            ref={searchRef}
            className={cn(
              "glass flex items-center gap-2 overflow-hidden rounded-full border px-3.5 py-2 shadow-lg transition-all duration-300 ease-out",
              searchOpen
                ? "pointer-events-auto w-full opacity-100"
                : "pointer-events-none w-8 opacity-0",
            )}
          >
            <Search className="text-muted-foreground size-4 shrink-0" />
            <input
              ref={searchInputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Escape" && closeSearch()}
              placeholder="Search chats…"
              tabIndex={searchOpen ? 0 : -1}
              className="placeholder:text-muted-foreground min-w-0 flex-1 bg-transparent text-sm outline-none"
            />
            <button
              type="button"
              onClick={closeSearch}
              tabIndex={searchOpen ? 0 : -1}
              className="text-muted-foreground hover:text-foreground shrink-0"
              title="Close search"
            >
              <X className="size-4" />
            </button>
          </div>
        </div>

        <div
          onPointerDown={startResize}
          className="hover:bg-primary/30 absolute top-0 right-0 z-40 h-full w-1.5 cursor-col-resize touch-none transition-colors"
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
          <SheetContent side="left" className="flex flex-col lg:hidden">
            <SheetHeader>
              <SheetTitle>History</SheetTitle>
            </SheetHeader>
            <div className="border-border bg-muted/40 flex items-center gap-2 rounded-full border px-3 py-1.5">
              <Search className="text-muted-foreground size-4 shrink-0" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search chats…"
                className="placeholder:text-muted-foreground min-w-0 flex-1 bg-transparent text-sm outline-none"
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
