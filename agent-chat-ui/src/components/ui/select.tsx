"use client";

import * as React from "react";
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
  align?: "start" | "end";
  /** Stretch the trigger to fill its container (for form fields). */
  fullWidth?: boolean;
  ariaLabel?: string;
  /** Controlled open state (lets a parent coordinate sibling menus). */
  open?: boolean;
  onOpenChange?: (o: boolean) => void;
  /** Fired when the trigger is hovered (for hover-to-switch menubars). */
  onTriggerMouseEnter?: () => void;
  /** Approx panel height used to choose the auto-flip direction. */
  estimatedHeight?: number;
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
  align = "start",
  fullWidth,
  ariaLabel,
  open: controlledOpen,
  onOpenChange,
  onTriggerMouseEnter,
  estimatedHeight = 300,
}: SelectProps) {
  const [highlight, setHighlight] = React.useState(-1);
  const rootRef = React.useRef<HTMLDivElement>(null);
  const listRef = React.useRef<HTMLDivElement>(null);
  const { open, setOpen, mounted, openUp } = useDropdown(rootRef, {
    controlledOpen,
    onOpenChange,
    estimatedHeight,
  });

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
          "inline-flex h-9 w-full items-center gap-2 rounded-md border border-border bg-background px-3 text-sm text-foreground outline-none transition-colors hover:bg-muted/60 focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
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
            "size-4 shrink-0 text-muted-foreground transition-transform",
            open && "rotate-180",
          )}
        />
      </button>

      {mounted && (
        <div
          ref={listRef}
          role="listbox"
          data-state={open ? "open" : "closed"}
          className={cn(
            "absolute z-50 max-h-64 w-max min-w-full max-w-[min(24rem,90vw)] overflow-y-auto rounded-md border border-border bg-popover p-1 text-popover-foreground shadow-lg duration-150",
            openUp ? "bottom-full mb-1" : "top-full mt-1",
            align === "end" ? "right-0" : "left-0",
            open
              ? cn(
                  "animate-in fade-in-0 zoom-in-95",
                  openUp ? "slide-in-from-bottom-1" : "slide-in-from-top-1",
                )
              : cn(
                  "animate-out fade-out-0 zoom-out-95",
                  openUp ? "slide-out-to-bottom-1" : "slide-out-to-top-1",
                ),
            menuClassName,
          )}
        >
          {options.length === 0 && (
            <div className="px-2 py-1.5 text-xs text-muted-foreground">
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
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {o.hint}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default Select;
