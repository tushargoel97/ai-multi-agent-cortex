"use client";

import { Thread } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import { useState } from "react";
import { Check, ChevronDown, Star, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useThreads } from "@/providers/Thread";
import {
  getThreadLabel,
  isPinned,
  useThreadActions,
  ThreadActionsMenu,
} from "./history/thread-actions";

/** Chat header title: the active thread's name + chevron opening the shared
 *  star / rename / delete menu. Renames inline; delete confirms. */
export function ChatHeaderTitle() {
  const [threadId] = useQueryState("threadId");
  const { threads } = useThreads();
  const { renameThread, deleteThread, togglePin } = useThreadActions();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);

  const thread: Thread | undefined = threads.find(
    (t) => t.thread_id === threadId,
  );
  if (!threadId || !thread) return null;

  const label = getThreadLabel(thread);
  const pinned = isPinned(thread);

  const commit = async () => {
    const trimmed = draft.trim();
    setEditing(false);
    if (trimmed && trimmed !== label) await renameThread(thread, trimmed);
  };

  if (editing) {
    return (
      <div className="flex items-center gap-1">
        <Input
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
            if (e.key === "Escape") setEditing(false);
          }}
          onBlur={commit}
          className="h-8 w-64 text-sm"
        />
        <Button
          size="icon"
          variant="ghost"
          className="size-7"
          onClick={commit}
          title="Save"
        >
          <Check className="size-4" />
        </Button>
        <Button
          size="icon"
          variant="ghost"
          className="size-7"
          onClick={() => setEditing(false)}
          title="Cancel"
        >
          <X className="size-4" />
        </Button>
      </div>
    );
  }

  return (
    <>
      <ThreadActionsMenu
        pinned={pinned}
        placement="below"
        triggerTitle="Chat options"
        triggerClassName="flex max-w-[16rem] items-center gap-1.5 rounded-lg px-2 py-1 text-sm font-medium text-foreground transition-colors hover:bg-muted sm:max-w-[22rem]"
        triggerActiveClassName="bg-muted"
        trigger={
          <>
            {pinned && (
              <Star className="size-3.5 shrink-0 fill-amber-400 text-amber-400" />
            )}
            <span className="truncate">{label}</span>
            <ChevronDown className="size-4 shrink-0 text-muted-foreground" />
          </>
        }
        onStar={() => togglePin(thread)}
        onRename={() => {
          setDraft(label);
          setEditing(true);
        }}
        onDelete={() => setConfirmDelete(true)}
      />

      <Dialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete this chat?</DialogTitle>
            <DialogDescription>
              “{label}” will be permanently deleted. This action cannot be
              undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="ghost"
              onClick={() => setConfirmDelete(false)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={async () => {
                setConfirmDelete(false);
                await deleteThread(thread);
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
