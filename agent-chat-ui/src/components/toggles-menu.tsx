"use client";

import * as React from "react";
import { ChevronDown, SlidersHorizontal } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

export interface ToggleDef {
  id: string;
  /** Title-cased display name (e.g. "Hide Tool Calls"). */
  name: string;
  description?: string;
  active: boolean;
  onToggle: (v: boolean) => void;
  /** "warn" tints the control amber to flag a safety-relaxing toggle. */
  tone?: "default" | "warn";
}

const MODES = [
  { value: "general", label: "General" },
  { value: "thinking", label: "Thinking" },
  { value: "research", label: "Research" },
];

const MODE_HINT: Record<string, string> = {
  general: "Fast, direct answers with normal routing.",
  thinking: "Forces the reasoner: quality tier + extended thinking.",
  research: "Deep web + KB research; asks clarifying questions first.",
};

function ModeSlider({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const activeIdx = Math.max(
    0,
    MODES.findIndex((m) => m.value === value),
  );
  return (
    <div>
      <div className="relative mx-2 h-1.5 rounded-full bg-muted">
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-primary transition-all"
          style={{ width: `${(activeIdx / (MODES.length - 1)) * 100}%` }}
        />
        {MODES.map((m, i) => (
          <button
            key={m.value}
            type="button"
            aria-label={m.label}
            onClick={() => onChange(m.value)}
            className="absolute top-1/2 flex size-5 -translate-x-1/2 -translate-y-1/2 items-center justify-center"
            style={{ left: `${(i / (MODES.length - 1)) * 100}%` }}
          >
            <span
              className={cn(
                "block size-3 rounded-full border-2 transition-colors",
                i <= activeIdx
                  ? "border-primary bg-primary"
                  : "border-muted-foreground/40 bg-background",
              )}
            />
          </button>
        ))}
      </div>
      <div className="mt-2 flex justify-between">
        {MODES.map((m, i) => (
          <button
            key={m.value}
            type="button"
            onClick={() => onChange(m.value)}
            className={cn(
              "text-[11px] transition-colors",
              i === activeIdx
                ? "font-semibold text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {m.label}
          </button>
        ))}
      </div>
      <p className="mt-1.5 text-[11px] text-muted-foreground">
        {MODE_HINT[value] ?? ""}
      </p>
    </div>
  );
}

/**
 * A pill dropdown that groups boolean options (mirrors the model selector's
 * look), so per-message toggles live in one menu instead of floating in the
 * toolbar. Opens upward for the chat input at the bottom of the screen.
 */
export function TogglesMenu({
  toggles,
  mode,
  onModeChange,
  label = "Options",
}: {
  toggles: ToggleDef[];
  mode?: string;
  onModeChange?: (m: string) => void;
  label?: string;
}) {
  const [open, setOpen] = React.useState(false);
  const rootRef = React.useRef<HTMLDivElement>(null);
  const activeCount = toggles.filter((t) => t.active).length;
  const warnActive = toggles.some((t) => t.tone === "warn" && t.active);

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
  }, [open]);

  return (
    <div ref={rootRef} className="relative inline-flex">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className={cn(
          "inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-xs font-medium transition-colors",
          warnActive
            ? "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400"
            : activeCount > 0
              ? "border-primary/40 bg-primary/10 text-primary"
              : "border-border bg-muted/50 text-muted-foreground hover:bg-muted hover:text-foreground",
        )}
      >
        <SlidersHorizontal className="size-3.5" />
        {label}
        {activeCount > 0 && (
          <span className="rounded-full bg-current/15 px-1.5 text-[10px] tabular-nums">
            {activeCount}
          </span>
        )}
        <ChevronDown
          className={cn("size-3.5 transition-transform", open && "rotate-180")}
        />
      </button>

      {open && (
        <div
          role="menu"
          className="absolute bottom-full right-0 z-50 mb-1 w-72 overflow-hidden rounded-lg border border-border bg-popover p-1 text-popover-foreground shadow-lg"
        >
          {mode !== undefined && onModeChange && (
            <>
              <div className="px-2.5 pb-1 pt-2">
                <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Mode
                </div>
                <ModeSlider value={mode} onChange={onModeChange} />
              </div>
              {toggles.length > 0 && <div className="my-1 h-px bg-border" />}
            </>
          )}
          {toggles.map((t) => (
            <div
              key={t.id}
              className="flex items-start gap-3 rounded-md px-2.5 py-2 hover:bg-accent/50"
            >
              <label
                htmlFor={`tgl-${t.id}`}
                className="min-w-0 flex-1 cursor-pointer select-none"
              >
                <span
                  className={cn(
                    "block text-sm font-medium",
                    t.tone === "warn" &&
                      t.active &&
                      "text-amber-600 dark:text-amber-400",
                  )}
                >
                  {t.name}
                </span>
                {t.description && (
                  <span className="mt-0.5 block text-xs font-normal text-muted-foreground">
                    {t.description}
                  </span>
                )}
              </label>
              <Switch
                id={`tgl-${t.id}`}
                checked={t.active}
                onCheckedChange={t.onToggle}
                aria-label={t.name}
                className="mt-0.5 shrink-0"
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default TogglesMenu;
