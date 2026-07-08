"use client";

import * as React from "react";

const useIsoLayoutEffect =
  typeof window !== "undefined" ? React.useLayoutEffect : React.useEffect;

export interface DropdownState {
  open: boolean;
  setOpen: (v: boolean | ((o: boolean) => boolean)) => void;
  /** Kept true briefly after close so the exit animation can play. */
  mounted: boolean;
  /** True when the panel should open upward (not enough room below). */
  openUp: boolean;
}

/**
 * Shared open/close behavior for the in-house dropdowns:
 * - optional controlled `open` (so a parent can coordinate sibling menus),
 * - click-outside + Escape to close,
 * - auto-flip: open downward when there's room, upward near the screen edge,
 * - a short mount delay so the close animation can finish before unmount.
 */
export function useDropdown(
  rootRef: React.RefObject<HTMLElement | null>,
  opts: {
    controlledOpen?: boolean;
    onOpenChange?: (o: boolean) => void;
    estimatedHeight?: number;
  } = {},
): DropdownState {
  const { controlledOpen, onOpenChange, estimatedHeight = 300 } = opts;
  const [internalOpen, setInternalOpen] = React.useState(false);
  const open = controlledOpen ?? internalOpen;

  const setOpen = React.useCallback(
    (v: boolean | ((o: boolean) => boolean)) => {
      const next = typeof v === "function" ? v(open) : v;
      if (onOpenChange) onOpenChange(next);
      else setInternalOpen(next);
    },
    [open, onOpenChange],
  );

  const [mounted, setMounted] = React.useState(open);
  const [openUp, setOpenUp] = React.useState(false);

  // Mount immediately on open; delay unmount so the exit animation plays.
  React.useEffect(() => {
    if (open) {
      setMounted(true);
      return;
    }
    if (!mounted) return;
    const t = setTimeout(() => setMounted(false), 150);
    return () => clearTimeout(t);
  }, [open, mounted]);

  // Decide direction before paint so the panel never flashes on the wrong side.
  useIsoLayoutEffect(() => {
    if (!open || !rootRef.current) return;
    const rect = rootRef.current.getBoundingClientRect();
    const below = window.innerHeight - rect.bottom;
    const above = rect.top;
    setOpenUp(below < estimatedHeight && above > below);
  }, [open]);

  React.useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, setOpen, rootRef]);

  return { open, setOpen, mounted, openUp };
}
