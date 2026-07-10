import {
  XIcon,
  SendHorizontal,
  RefreshCcw,
  Pencil,
  Copy,
  Check,
  ChevronLeft,
  ChevronRight,
  ThumbsUp,
  ThumbsDown,
} from "lucide-react";
import { TooltipIconButton } from "../tooltip-icon-button";
import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { cn, copyTextToClipboard } from "@/lib/utils";
import { toast } from "sonner";

function ContentCopyable({ content, disabled }: { content: string; disabled: boolean }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async (e: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
    e.stopPropagation();
    const ok = await copyTextToClipboard(content);
    if (!ok) {
      toast.error("Couldn't copy to clipboard");
      return;
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <TooltipIconButton
      onClick={(e) => handleCopy(e)}
      variant="ghost"
      tooltip="Copy content"
      disabled={disabled}
    >
      <AnimatePresence mode="wait" initial={false}>
        {copied ? (
          <motion.div
            key="check"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.15 }}
          >
            <Check className="text-green-500" />
          </motion.div>
        ) : (
          <motion.div
            key="copy"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.15 }}
          >
            <Copy />
          </motion.div>
        )}
      </AnimatePresence>
    </TooltipIconButton>
  );
}

export function BranchSwitcher({
  branch,
  branchOptions,
  onSelect,
  isLoading,
}: {
  branch: string | undefined;
  branchOptions: string[] | undefined;
  onSelect: (branch: string) => void;
  isLoading: boolean;
}) {
  if (!branchOptions || !branch) return null;
  const index = branchOptions.indexOf(branch);

  return (
    <div className="flex items-center gap-2">
      <Button
        variant="ghost"
        size="icon"
        className="size-6 p-1"
        onClick={() => {
          const prevBranch = branchOptions[index - 1];
          if (!prevBranch) return;
          onSelect(prevBranch);
        }}
        disabled={isLoading}
      >
        <ChevronLeft />
      </Button>
      <span className="text-sm">
        {index + 1} / {branchOptions.length}
      </span>
      <Button
        variant="ghost"
        size="icon"
        className="size-6 p-1"
        onClick={() => {
          const nextBranch = branchOptions[index + 1];
          if (!nextBranch) return;
          onSelect(nextBranch);
        }}
        disabled={isLoading}
      >
        <ChevronRight />
      </Button>
    </div>
  );
}

/** Thumbs up/down feedback for an assistant message (local acknowledgement). */
function MessageFeedback({ disabled }: { disabled: boolean }) {
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null);

  const pick = (v: "up" | "down") => {
    const next = feedback === v ? null : v;
    setFeedback(next);
    if (next === "up") toast.success("Thanks for the feedback");
    if (next === "down") toast.success("Thanks, we'll try to do better");
  };

  return (
    <>
      <TooltipIconButton
        disabled={disabled}
        tooltip="Good response"
        variant="ghost"
        onClick={() => pick("up")}
      >
        <ThumbsUp className={cn(feedback === "up" && "fill-current text-green-500")} />
      </TooltipIconButton>
      <TooltipIconButton
        disabled={disabled}
        tooltip="Bad response"
        variant="ghost"
        onClick={() => pick("down")}
      >
        <ThumbsDown className={cn(feedback === "down" && "fill-current text-red-500")} />
      </TooltipIconButton>
    </>
  );
}

export function CommandBar({
  content,
  isHumanMessage,
  isAiMessage,
  isEditing,
  setIsEditing,
  handleSubmitEdit,
  handleRegenerate,
  isLoading,
}: {
  content: string;
  isHumanMessage?: boolean;
  isAiMessage?: boolean;
  isEditing?: boolean;
  setIsEditing?: React.Dispatch<React.SetStateAction<boolean>>;
  handleSubmitEdit?: () => void;
  handleRegenerate?: () => void;
  isLoading: boolean;
}) {
  if (isHumanMessage && isAiMessage) {
    throw new Error("Can only set one of isHumanMessage or isAiMessage to true, not both.");
  }

  if (!isHumanMessage && !isAiMessage) {
    throw new Error("One of isHumanMessage or isAiMessage must be set to true.");
  }

  if (
    isHumanMessage &&
    (isEditing === undefined || setIsEditing === undefined || handleSubmitEdit === undefined)
  ) {
    throw new Error(
      "If isHumanMessage is true, all of isEditing, setIsEditing, and handleSubmitEdit must be set.",
    );
  }

  const showEdit =
    isHumanMessage && isEditing !== undefined && !!setIsEditing && !!handleSubmitEdit;

  if (isHumanMessage && isEditing && !!setIsEditing && !!handleSubmitEdit) {
    return (
      <div className="flex items-center gap-2">
        <TooltipIconButton
          disabled={isLoading}
          tooltip="Cancel edit"
          variant="ghost"
          onClick={() => {
            setIsEditing(false);
          }}
        >
          <XIcon />
        </TooltipIconButton>
        <TooltipIconButton
          disabled={isLoading}
          tooltip="Submit"
          variant="secondary"
          onClick={handleSubmitEdit}
        >
          <SendHorizontal />
        </TooltipIconButton>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <ContentCopyable content={content} disabled={isLoading} />
      {isAiMessage && !!handleRegenerate && (
        <TooltipIconButton
          disabled={isLoading}
          tooltip="Refresh"
          variant="ghost"
          onClick={handleRegenerate}
        >
          <RefreshCcw />
        </TooltipIconButton>
      )}
      {isAiMessage && <MessageFeedback disabled={isLoading} />}
      {showEdit && (
        <TooltipIconButton
          disabled={isLoading}
          tooltip="Edit"
          variant="ghost"
          onClick={() => {
            setIsEditing?.(true);
          }}
        >
          <Pencil />
        </TooltipIconButton>
      )}
    </div>
  );
}
