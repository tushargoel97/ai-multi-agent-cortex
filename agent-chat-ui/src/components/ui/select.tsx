"use client";

import * as React from "react";
import { createPortal } from "react-dom";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { useDropdown } from "@/hooks/use-dropdown";

export interface SelectOption {
  value: string;
  label: React.ReactNode;
  disabled?: boolean;
  /** Optional muted text shown right-aligned after the label. */
  hint?: React.ReactNode;
}

interface SelectProps {
  value: string;
  onValueChange: (value: string) => void;
  options: SelectOption[];
  placeholder?: React.ReactNode;
  /** Overrides the trigger text (defaults to the selected option's label). */
  triggerLabel?: React.ReactNode;
  disabled?: boolean;
  /** Extra classes for the trigger button (sizing / shape / colors). */
  className?: string;
  /** Extra classes for the popup menu. */
  menuClassName?: string;
  /** Leading icon rendered inside the trigger. */
  icon?: React.ReactNode;
  /** Stretch the trigger to fill its container (for form fields). */
  fullWidth?: boolean;
  ariaLabel?: string;
  /** Controlled open state (lets a parent coordinate sibling menus). */
  open?: boolean;
  onOpenChange?: (o: boolean) => void;
  /** Fired when the trigger is hovered (for hover-to-switch menubars). */
  onTriggerMouseEnter?: () => void;
}

/**
 * Dependency-free, theme-aware dropdown that replaces the native <select> so
 * menus match the app instead of the OS. Keyboard + click-outside supported.
 */
export function Select({
  value,
  onValueChange,
  options,
  placeholder = "Select…",
  triggerLabel,
  disabled,
  className,
  menuClassName,
  icon,
  fullWidth,
  ariaLabel,
  open: controlledOpen,
  onOpenChange,
  onTriggerMouseEnter,
}: SelectProps) {
  const [highlight, setHighlight] = React.useState(-1);
  const rootRef = React.useRef<HTMLDivElement>(null);
  const listRef = React.useRef<HTMLDivElement>(null);
  const { open, setOpen, mounted } = useDropdown(rootRef, {
    controlledOpen,
    onOpenChange,
    insideRef: listRef,
  });

  // The list renders in a body portal so no ancestor's backdrop-filter can
  // disable its own blur (nested backdrop-filters don't compose). Fixed
  // coordinates come from the trigger; hidden until the first placement,
  // which measures the real list to decide downward vs upward (upward only
  // when it truly doesn't fit below and there's more room above).
  const MAX_H = 320;
  const [pos, setPos] = React.useState<{
    left: number;
    minWidth: number;
    maxHeight: number;
    up: boolean;
    top?: number;
    bottom?: number;
  } | null>(null);

  React.useLayoutEffect(() => {
    if (!mounted || !open) return;
    const r = rootRef.current?.getBoundingClientRect();
    const lr = listRef.current?.getBoundingClientRect();
    if (!r || !lr) return;
    const gap = 4;
    const margin = 8;
    const spaceBelow = window.innerHeight - r.bottom - gap - margin;
    const spaceAbove = r.top - gap - margin;
    const h = Math.min(lr.height, MAX_H);
    const up = h > spaceBelow && spaceAbove > spaceBelow;
    const maxHeight = Math.max(
      96,
      Math.min(up ? spaceAbove : spaceBelow, MAX_H),
    );
    const left = Math.max(
      margin,
      Math.min(r.left, window.innerWidth - lr.width - margin),
    );
    setPos({
      left,
      minWidth: r.width,
      maxHeight,
      up,
      ...(up
        ? { bottom: window.innerHeight - r.top + gap }
        : { top: r.bottom + gap }),
    });
  }, [mounted, open]);

  React.useEffect(() => {
    if (!mounted) setPos(null);
  }, [mounted]);

  const selectedIndex = options.findIndex((o) => o.value === value);
  const selected = selectedIndex >= 0 ? options[selectedIndex] : undefined;

  React.useEffect(() => {
    if (!open) return;
    const start =
      selectedIndex >= 0
        ? selectedIndex
        : options.findIndex((o) => !o.disabled);
    setHighlight(start);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  React.useEffect(() => {
    if (!open || highlight < 0) return;
    listRef.current
      ?.querySelector<HTMLElement>(`[data-index="${highlight}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [highlight, open]);

  const move = (dir: 1 | -1) =>
    setHighlight((cur) => {
      let i = cur;
      for (let n = 0; n < options.length; n++) {
        i = (i + dir + options.length) % options.length;
        if (!options[i]?.disabled) return i;
      }
      return cur;
    });

  const choose = (i: number) => {
    const opt = options[i];
    if (!opt || opt.disabled) return;
    onValueChange(opt.value);
    setOpen(false);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (disabled) return;
    if (!open) {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(e.key)) {
        e.preventDefault();
        setOpen(true);
      }
      return;
    }
    switch (e.key) {
      case "Escape":
        e.preventDefault();
        setOpen(false);
        break;
      case "ArrowDown":
        e.preventDefault();
        move(1);
        break;
      case "ArrowUp":
        e.preventDefault();
        move(-1);
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        if (highlight >= 0) choose(highlight);
        break;
    }
  };

  return (
    <div
      ref={rootRef}
      className={cn("relative", fullWidth ? "flex w-full" : "inline-flex")}
    >
      <button
        type="button"
        role="combobox"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={ariaLabel}
        disabled={disabled}
        onClick={() => !disabled && setOpen((o) => !o)}
        onMouseEnter={onTriggerMouseEnter}
        onKeyDown={onKeyDown}
        className={cn(
          "border-border bg-background/60 text-foreground hover:bg-muted/60 focus-visible:ring-ring inline-flex h-9 w-full items-center gap-2 rounded-md border px-3 text-sm transition-colors outline-none focus-visible:ring-2 disabled:cursor-not-allowed disabled:opacity-50",
          className,
        )}
      >
        {icon}
        <span
          className={cn(
            "min-w-0 flex-1 truncate text-left",
            !selected && !triggerLabel && "text-muted-foreground",
          )}
        >
          {triggerLabel ?? selected?.label ?? placeholder}
        </span>
        <ChevronDown
          className={cn(
            "text-muted-foreground size-4 shrink-0 transition-transform",
            open && "rotate-180",
          )}
        />
      </button>

      {mounted &&
        createPortal(
          <div
            ref={listRef}
            role="listbox"
            data-state={open ? "open" : "closed"}
            style={{
              position: "fixed",
              left: pos?.left,
              top: pos?.top,
              bottom: pos?.bottom,
              minWidth: pos?.minWidth,
              maxHeight: pos?.maxHeight ?? MAX_H,
              visibility: pos ? undefined : "hidden",
            }}
            className={cn(
              "glass text-popover-foreground z-50 w-max max-w-[min(24rem,90vw)] overflow-y-auto rounded-md border p-1 shadow-lg duration-150",
              open
                ? cn(
                    "animate-in fade-in-0 zoom-in-95",
                    pos?.up ? "slide-in-from-bottom-1" : "slide-in-from-top-1",
                  )
                : cn(
                    "animate-out fade-out-0 zoom-out-95",
                    pos?.up ? "slide-out-to-bottom-1" : "slide-out-to-top-1",
                  ),
              menuClassName,
            )}
          >
            {options.length === 0 && (
              <div className="text-muted-foreground px-2 py-1.5 text-xs">
                No options
              </div>
            )}
            {options.map((o, i) => {
              const active = o.value === value;
              return (
                <button
                  key={`${o.value}-${i}`}
                  type="button"
                  role="option"
                  aria-selected={active}
                  data-index={i}
                  disabled={o.disabled}
                  onMouseEnter={() => !o.disabled && setHighlight(i)}
                  onClick={() => choose(i)}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-sm outline-none",
                    i === highlight &&
                      !o.disabled &&
                      "bg-accent text-accent-foreground",
                    o.disabled && "cursor-not-allowed opacity-50",
                  )}
                >
                  <Check
                    className={cn(
                      "size-4 shrink-0",
                      active ? "opacity-100" : "opacity-0",
                    )}
                  />
                  <span className="min-w-0 flex-1 truncate">{o.label}</span>
                  {o.hint != null && (
                    <span className="text-muted-foreground shrink-0 text-xs">
                      {o.hint}
                    </span>
                  )}
                </button>
              );
            })}
          </div>,
          document.body,
        )}
    </div>
  );
}

export default Select;
