"use client";

import { Trash2, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/**
 * Standard destructive delete control: a red bin icon button. Use this for
 * every delete/remove action so the affordance is consistent across the app.
 */
export function DeleteButton({
  onClick,
  disabled,
  busy,
  title = "Delete",
  className,
}: {
  onClick: () => void;
  disabled?: boolean;
  busy?: boolean;
  title?: string;
  className?: string;
}) {
  return (
    <Button
      type="button"
      size="icon"
      variant="ghost"
      disabled={disabled}
      title={title}
      onClick={onClick}
      className={cn(
        "size-8 shrink-0 text-destructive hover:bg-destructive/10 hover:text-destructive",
        className,
      )}
    >
      {busy ? (
        <Loader2 className="size-4 animate-spin" />
      ) : (
        <Trash2 className="size-4" />
      )}
      <span className="sr-only">{title}</span>
    </Button>
  );
}
