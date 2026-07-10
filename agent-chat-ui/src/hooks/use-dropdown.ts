"use client";

import * as React from "react";

export interface DropdownState {
  open: boolean;
  setOpen: (v: boolean | ((o: boolean) => boolean)) => void;
  /** Kept true briefly after close so the exit animation can play. */
  mounted: boolean;
}

/**
 * Shared open/close behavior for the in-house dropdowns:
 * - optional controlled `open` (so a parent can coordinate sibling menus),
 * - click-outside, outside scroll and Escape to close,
 * - a short mount delay so the close animation can finish before unmount.
 */
export function useDropdown(
  rootRef: React.RefObject<HTMLElement | null>,
  opts: {
    controlledOpen?: boolean;
    onOpenChange?: (o: boolean) => void;
    /** Portaled panel to also treat as "inside" for click-outside. */
    insideRef?: React.RefObject<HTMLElement | null>;
  } = {},
): DropdownState {
  const { controlledOpen, onOpenChange, insideRef } = opts;
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

  React.useEffect(() => {
    if (!open) return;
    const inside = (t: Node) => !!rootRef.current?.contains(t) || !!insideRef?.current?.contains(t);
    const onDown = (e: MouseEvent) => {
      if (!inside(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    // A portaled panel is fixed-positioned, so it can't follow its trigger:
    // close when anything else scrolls (the panel's own scroll stays open).
    const onScroll = (e: Event) => {
      if (!inside(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    window.addEventListener("scroll", onScroll, true);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", onScroll, true);
    };
  }, [open, setOpen, rootRef, insideRef]);

  return { open, setOpen, mounted };
}
