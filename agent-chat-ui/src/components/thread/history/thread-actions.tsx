"use client";

import { Thread } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Pencil, Star, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { createClient } from "@/providers/client";
import { useThreads } from "@/providers/Thread";
import { getApiKey } from "@/lib/api-key";
import { cn } from "@/lib/utils";
import { getContentString } from "../utils";

/** Human title for a thread: an explicit metadata title, else its first
 *  message trimmed, else a friendly default. */
export function getThreadLabel(t: Thread): string {
  const metaTitle = (t.metadata as Record<string, unknown> | undefined)?.title;
  if (typeof metaTitle === "string" && metaTitle.trim()) return metaTitle;
  if (
    typeof t.values === "object" &&
    t.values &&
    "messages" in t.values &&
    Array.isArray((t.values as { messages: unknown[] }).messages) &&
    (t.values as { messages: { content: unknown }[] }).messages.length > 0
  ) {
    const firstMessage = (t.values as { messages: { content: unknown }[] }).messages[0];
    const text = getContentString(firstMessage.content as never);
    if (text) return text.length > 80 ? text.slice(0, 80) + "…" : text;
  }
  return "New chat";
}

export function isPinned(t: Thread): boolean {
  return Boolean((t.metadata as Record<string, unknown> | undefined)?.pinned);
}

/** Rename / delete / star (pin) operations shared by the sidebar row menu and
 *  the chat-header title menu, with optimistic list updates + toasts. */
export function useThreadActions() {
  const [threadId, setThreadId] = useQueryState("threadId");
  const [apiUrl] = useQueryState("apiUrl", {
    defaultValue: process.env.NEXT_PUBLIC_API_URL || "",
  });
  const [authScheme] = useQueryState("authScheme", {
    defaultValue: process.env.NEXT_PUBLIC_AUTH_SCHEME || "",
  });
  const { setThreads, getThreads } = useThreads();

  const client = () =>
    createClient(
      apiUrl || process.env.NEXT_PUBLIC_API_URL || "",
      getApiKey() ?? undefined,
      authScheme || undefined,
    );

  const renameThread = async (t: Thread, title: string) => {
    try {
      const newMetadata = { ...(t.metadata ?? {}), title };
      await client().threads.update(t.thread_id, { metadata: newMetadata });
      setThreads((prev) =>
        prev.map((x) => (x.thread_id === t.thread_id ? { ...x, metadata: newMetadata } : x)),
      );
      toast.success("Chat renamed");
    } catch (e) {
      console.error(e);
      toast.error("Failed to rename chat");
    }
  };

  const deleteThread = async (t: Thread) => {
    try {
      await client().threads.delete(t.thread_id);
      setThreads((prev) => prev.filter((x) => x.thread_id !== t.thread_id));
      if (threadId === t.thread_id) setThreadId(null);
      toast.success("Chat deleted");
    } catch (e) {
      console.error(e);
      toast.error("Failed to delete chat");
      getThreads()
        .then(setThreads)
        .catch(() => {});
    }
  };

  const togglePin = async (t: Thread) => {
    try {
      const pinned = !isPinned(t);
      const newMetadata = { ...(t.metadata ?? {}), pinned };
      await client().threads.update(t.thread_id, { metadata: newMetadata });
      setThreads((prev) =>
        prev.map((x) => (x.thread_id === t.thread_id ? { ...x, metadata: newMetadata } : x)),
      );
      toast.success(pinned ? "Chat starred" : "Chat unstarred");
    } catch (e) {
      console.error(e);
      toast.error("Failed to update chat");
    }
  };

  return { renameThread, deleteThread, togglePin };
}

/** Thread actions menu: frosted panel of icon+label+shortcut rows with a
 *  divider before a red Delete. Self-contained trigger + portaled menu. */
export function ThreadActionsMenu({
  pinned,
  onStar,
  onRename,
  onDelete,
  trigger,
  triggerClassName,
  triggerActiveClassName,
  triggerTitle,
  placement = "beside",
}: {
  pinned: boolean;
  onStar: () => void;
  onRename: () => void;
  onDelete: () => void;
  trigger: ReactNode;
  triggerClassName?: string;
  triggerActiveClassName?: string;
  triggerTitle?: string;
  /** "beside" opens to the right of the trigger (sidebar row); "below" opens
   *  underneath it (header title dropdown). */
  placement?: "beside" | "below";
}) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);

  const W = 176; // w-44
  const H = 150; // approx height for clamping

  useEffect(() => {
    if (!open) {
      setPos(null);
      return;
    }
    const el = triggerRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    let left = placement === "below" ? r.left : r.right + 6;
    let top = placement === "below" ? r.bottom + 6 : r.top - 4;
    if (left + W > window.innerWidth - 8) {
      left = placement === "below" ? r.right - W : r.left - W - 6;
    }
    if (top + H > window.innerHeight - 8) top = window.innerHeight - H - 8;
    setPos({ top: Math.max(8, top), left: Math.max(8, left) });
  }, [open, placement]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (triggerRef.current?.contains(t) || menuRef.current?.contains(t)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        return;
      }
      const ae = document.activeElement;
      if (
        ae instanceof HTMLInputElement ||
        ae instanceof HTMLTextAreaElement ||
        (ae as HTMLElement | null)?.isContentEditable
      )
        return;
      const k = e.key.toLowerCase();
      if (k === "p") {
        e.preventDefault();
        setOpen(false);
        onStar();
      } else if (k === "r") {
        e.preventDefault();
        setOpen(false);
        onRename();
      } else if (k === "d") {
        e.preventDefault();
        setOpen(false);
        onDelete();
      }
    };
    const onScroll = () => setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    document.addEventListener("scroll", onScroll, true);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("scroll", onScroll, true);
    };
  }, [open, onStar, onRename, onDelete]);

  const item =
    "flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm transition-colors hover:bg-accent/70";
  const kbd = "ml-auto text-xs tabular-nums text-muted-foreground";

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        title={triggerTitle}
        aria-haspopup="menu"
        aria-expanded={open}
        className={cn(triggerClassName, open && triggerActiveClassName)}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
      >
        {trigger}
      </button>
      {open &&
        pos &&
        createPortal(
          <div
            ref={menuRef}
            role="menu"
            onClick={(e) => e.stopPropagation()}
            style={{ position: "fixed", top: pos.top, left: pos.left, width: W }}
            className="glass animate-in fade-in-0 zoom-in-95 text-popover-foreground z-[100] rounded-xl border p-1.5 shadow-xl"
          >
            <button
              className={item}
              onClick={() => {
                setOpen(false);
                onStar();
              }}
            >
              <Star className={cn("size-4 shrink-0", pinned && "fill-amber-400 text-amber-400")} />
              {pinned ? "Unstar" : "Star"}
              <span className={kbd}>P</span>
            </button>
            <button
              className={item}
              onClick={() => {
                setOpen(false);
                onRename();
              }}
            >
              <Pencil className="size-4 shrink-0" />
              Rename
              <span className={kbd}>R</span>
            </button>
            <div className="bg-border mx-1 my-1 h-px" />
            <button
              className={cn(item, "text-red-500 hover:bg-red-500/10")}
              onClick={() => {
                setOpen(false);
                onDelete();
              }}
            >
              <Trash2 className="size-4 shrink-0" />
              Delete
              <span className={cn(kbd, "text-red-500/70")}>D</span>
            </button>
          </div>,
          document.body,
        )}
    </>
  );
}
