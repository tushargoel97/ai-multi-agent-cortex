import { Button } from "@/components/ui/button";
import { useThreads } from "@/providers/Thread";
import { Thread } from "@langchain/langgraph-sdk";
import { useEffect, useState } from "react";
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
  Trash2,
  Plus,
  Check,
  X,
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
  return `Chat ${t.thread_id.slice(0, 8)}`;
}

function ThreadRow({
  thread,
  onClick,
  onRename,
  onDelete,
}: {
  thread: Thread;
  onClick: () => void;
  onRename: (newTitle: string) => Promise<void>;
  onDelete: () => Promise<void>;
}) {
  const [threadId] = useQueryState("threadId");
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(getThreadLabel(thread));
  const isActive = threadId === thread.thread_id;

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
        isActive ? "bg-gray-100" : ""
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
            className="h-9 w-full flex-1 items-start justify-start truncate text-left font-normal"
            onClick={onClick}
          >
            <p className="truncate text-ellipsis">{getThreadLabel(thread)}</p>
          </Button>
          <div className="absolute right-1 hidden gap-0.5 group-hover:flex">
            <Button
              size="icon"
              variant="ghost"
              className="size-7 bg-white/70 hover:bg-white"
              onClick={(e) => {
                e.stopPropagation();
                setDraft(getThreadLabel(thread));
                setEditing(true);
              }}
              title="Rename"
            >
              <Pencil className="size-3.5" />
            </Button>
            <Button
              size="icon"
              variant="ghost"
              className="size-7 bg-white/70 text-red-600 hover:bg-white hover:text-red-700"
              onClick={(e) => {
                e.stopPropagation();
                onDelete();
              }}
              title="Delete"
            >
              <Trash2 className="size-3.5" />
            </Button>
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

  return (
    <>
      <div className="flex h-full w-full flex-col items-start justify-start gap-1 overflow-y-auto pb-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
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
        {threads.map((t) => (
          <div
            key={t.thread_id}
            className="w-full px-2"
          >
            <ThreadRow
              thread={t}
              onClick={() => {
                onThreadClick?.(t.thread_id);
                if (t.thread_id === threadId) return;
                setThreadId(t.thread_id);
              }}
              onRename={(title) => renameThread(t, title)}
              onDelete={async () => setPendingDelete(t)}
            />
          </div>
        ))}
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
    <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
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
      <div className="shadow-inner-right hidden h-screen w-[300px] shrink-0 flex-col items-start justify-start gap-6 border-r-[1px] border-slate-300 lg:flex">
        <div className="flex w-full items-center justify-between px-4 pt-1.5">
          <Button
            className="hover:bg-gray-100"
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
