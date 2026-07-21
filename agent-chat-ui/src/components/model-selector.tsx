"use client";

import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type ReactNode,
} from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Brain,
  Check,
  Wrench,
  ChevronDown,
  ChevronRight,
  Globe,
  GripVertical,
  Pin,
  Server,
  ShieldOff,
  Zap,
  type LucideIcon,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import { createPortal } from "react-dom";
import { cn } from "@/lib/utils";

export interface AvailableModel {
  id: string;
  model_id: string;
  display_name: string;
  provider_name: string;
  provider_kind: string;
  is_default: boolean;
}

export interface ModelSelection {
  model_id: string | null;
  use_local: boolean;
  local_base_url: string;
  local_api_key: string;
  local_model_name: string;
  /** Relax the image safety pre-screen + configurable provider thresholds. */
  unrestricted: boolean;
  /** Response mode: general/instant (default) | thinking | research | engineer. */
  mode: "general" | "thinking" | "research" | "engineer";
  /** Model ids pinned in the picker (UI-only, never sent to the graph). */
  pinned_models: string[];
}

interface ToggleDef {
  id: string;
  name: string;
  description?: string;
  active: boolean;
  onToggle: (v: boolean) => void;
  /** "warn" tints the control amber to flag a safety-relaxing toggle. */
  tone?: "default" | "warn";
}

const STORAGE_KEY = "cortex:model-selection";

/** Sentinel understood by the graph: pick the model per intent (auto mode). */
const AUTO_MODEL_ID = "auto";

export const DEFAULT_SELECTION: ModelSelection = {
  model_id: AUTO_MODEL_ID,
  use_local: false,
  local_base_url: "http://host.docker.internal:1234/v1",
  local_api_key: "",
  local_model_name: "local-model",
  unrestricted: false,
  mode: "general",
  pinned_models: [],
};

export function loadModelSelection(): ModelSelection {
  if (typeof window === "undefined") return DEFAULT_SELECTION;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SELECTION;
    return { ...DEFAULT_SELECTION, ...JSON.parse(raw) };
  } catch {
    return DEFAULT_SELECTION;
  }
}

export function saveModelSelection(sel: ModelSelection) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(sel));
}

export function selectionToConfigurable(sel: ModelSelection): Record<string, unknown> {
  if (sel.use_local) {
    return {
      local_base_url: sel.local_base_url,
      local_api_key: sel.local_api_key,
      model_id: null,
      local_model_name: sel.local_model_name,
      unrestricted: sel.unrestricted,
      mode: sel.mode,
    };
  }
  return {
    model_id: sel.model_id,
    unrestricted: sel.unrestricted,
    mode: sel.mode,
  };
}

/** Browser locale + timezone, sent with each run so agents default to the
 *  user's country/region (shopping, booking, local results). */
export function browserContext(): Record<string, unknown> {
  if (typeof window === "undefined") return {};
  try {
    return {
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      locale: navigator.language,
    };
  } catch {
    return {};
  }
}

const MODEL_NAME_ACRONYMS: Record<string, string> = {
  gpt: "GPT",
  ai: "AI",
  llm: "LLM",
  xai: "xAI",
};

/** Turn a raw model id ("gpt-4o-mini") into a readable name ("GPT 4o Mini").
 *  Names that already read naturally (contain a space) are left untouched, so
 *  an admin-set display name is never mangled. */
function formatModelName(raw: string): string {
  const name = (raw || "").trim();
  if (!name || /\s/.test(name)) return name || "model";
  return name
    .split(/[-_]/)
    .filter(Boolean)
    .map((part) => {
      const low = part.toLowerCase();
      if (MODEL_NAME_ACRONYMS[low]) return MODEL_NAME_ACRONYMS[low];
      if (/\d/.test(part)) return part; // version tokens: 4o, 3.5, 5
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(" ");
}

type Mode = ModelSelection["mode"];

/** Pin the composer's hover zoom while a toolbar menu is open: the portaled
 *  menu is positioned from the zoomed rect, so shrinking would misalign it. */
function pinComposerZoom(el: HTMLElement | null, open: boolean) {
  const composer = el?.closest("[data-prompt-composer]");
  if (!composer) return;
  if (open) composer.setAttribute("data-menu-open", "");
  else composer.removeAttribute("data-menu-open");
}

const MODE_META: Record<Mode, { label: string; hint: string; icon: LucideIcon }> = {
  general: { label: "Instant", hint: "Fastest, direct answers.", icon: Zap },
  thinking: {
    label: "Thinking",
    hint: "Reasoner + extended thinking.",
    icon: Brain,
  },
  research: {
    label: "Research",
    hint: "Deep web/KB research; clarifies first.",
    icon: Globe,
  },
  engineer: {
    label: "Engineer",
    hint: "Top coding models + debugger agent.",
    icon: Wrench,
  },
};

function ModelRow({
  label,
  hint,
  selected,
  pinned,
  onSelect,
  onTogglePin,
}: {
  label: string;
  hint?: string;
  selected: boolean;
  pinned: boolean;
  onSelect: () => void;
  onTogglePin: () => void;
}) {
  return (
    <div className="group hover:bg-accent/60 flex w-full items-center gap-0.5 rounded-md pr-1 transition-colors">
      <button
        type="button"
        onClick={onSelect}
        className="flex min-w-0 flex-1 items-center py-1.5 pl-2 text-left"
      >
        <span className="flex min-w-0 flex-col">
          <span className="truncate text-sm">{label}</span>
          {hint && <span className="text-muted-foreground truncate text-[11px]">{hint}</span>}
        </span>
      </button>
      <button
        type="button"
        title={pinned ? "Unpin" : "Pin"}
        onClick={(e) => {
          e.stopPropagation();
          onTogglePin();
        }}
        className={cn(
          "hover:bg-muted shrink-0 rounded p-1 opacity-0 transition-colors group-hover:opacity-100 focus:opacity-100",
          pinned ? "text-primary" : "text-muted-foreground hover:text-foreground",
        )}
      >
        <Pin className={cn("size-3.5", pinned && "fill-current")} />
      </button>
      <Check
        className={cn("text-primary size-3.5 shrink-0", selected ? "opacity-100" : "opacity-0")}
      />
    </div>
  );
}

function MenuRow({
  checked,
  chevron,
  active,
  onClick,
  onMouseEnter,
  children,
}: {
  checked?: boolean;
  chevron?: boolean;
  active?: boolean;
  onClick?: (e: ReactMouseEvent<HTMLButtonElement>) => void;
  onMouseEnter?: (e: ReactMouseEvent<HTMLButtonElement>) => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      className={cn(
        "hover:bg-accent/60 flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors",
        active && "bg-accent/60",
      )}
    >
      <span className="min-w-0 flex-1">{children}</span>
      {chevron ? (
        <ChevronRight className="text-muted-foreground size-3.5 shrink-0" />
      ) : (
        <Check
          className={cn("text-primary size-3.5 shrink-0", checked ? "opacity-100" : "opacity-0")}
        />
      )}
    </button>
  );
}

/**
 * Claude-style consolidated prompt-box menu: a single pill that opens the top
 * models, provider submenus, and a nested "Options" panel (Local LLM / Hide
 * tools / Unrestricted). Response mode lives in its own toolbar dropdown.
 */
function PromptToolbarMenu({
  triggerLabel,
  useLocal,
  autoSelected,
  pinnedModels,
  providers,
  selectedId,
  isPinned,
  onSelectModel,
  onTogglePin,
  onReorderPinned,
  toggles,
}: {
  triggerLabel: string;
  useLocal: boolean;
  autoSelected: boolean;
  pinnedModels: AvailableModel[];
  providers: { name: string; models: AvailableModel[] }[];
  selectedId: string | null;
  isPinned: (id: string) => boolean;
  onSelectModel: (id: string) => void;
  onTogglePin: (id: string) => void;
  onReorderPinned: (source: string, target: string, after: boolean) => void;
  toggles: ToggleDef[];
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const subRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [menuBox, setMenuBox] = useState<{
    left: number;
    top?: number;
    bottom?: number;
    maxH: number;
  }>({ left: 0, maxH: 400 });
  // Range the menu + submenus stay within, so they never cover the composer.
  const [band, setBand] = useState({ top: 8, bottom: 600 });
  const [sub, setSub] = useState<
    | { kind: "provider"; name: string; anchorTop: number; left: number }
    | { kind: "mode"; anchorTop: number; left: number }
    | null
  >(null);
  const [subTop, setSubTop] = useState<number | null>(null);
  const [draggedPinned, setDraggedPinned] = useState<string | null>(null);
  const hoverTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearHoverTimer = () => {
    if (hoverTimer.current) clearTimeout(hoverTimer.current);
    hoverTimer.current = null;
  };

  useEffect(() => {
    if (!open) {
      clearHoverTimer();
      setSub(null);
    }
    pinComposerZoom(rootRef.current, open);
    return () => pinComposerZoom(rootRef.current, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Top-align the submenu to its row, nudged up only enough to fit the band.
  useLayoutEffect(() => {
    if (!sub) {
      setSubTop(null);
      return;
    }
    const el = subRef.current;
    if (!el) return;
    const h = el.getBoundingClientRect().height;
    const top = Math.max(band.top, Math.min(sub.anchorTop, band.bottom - h));
    setSubTop((t) => (t !== null && Math.abs(t - top) < 1 ? t : top));
  }, [sub, band]);

  // Close on outside click / Escape / resize (menu + submenu are portaled).
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (
        rootRef.current?.contains(t) ||
        menuRef.current?.contains(t) ||
        subRef.current?.contains(t)
      )
        return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    const onResize = () => setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", onResize);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", onResize);
    };
  }, [open]);

  const panel = "glass rounded-xl border p-1 text-popover-foreground shadow-xl";
  const sectionLabel =
    "px-2 pb-0.5 pt-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground";
  const MW = 256; // menu / submenu width (w-64)

  // Open toward whichever side of the composer has room; that side is the band.
  const toggle = () => {
    if (open) {
      setOpen(false);
      return;
    }
    const el = rootRef.current;
    if (!el) {
      setOpen(true);
      return;
    }
    const composer = (el.closest("[data-prompt-composer]") as HTMLElement | null) ?? el;
    const cr = composer.getBoundingClientRect();
    const pr = el.getBoundingClientRect();
    const gap = 8;
    const above = cr.top - gap;
    const below = window.innerHeight - cr.bottom - gap;
    const left = Math.max(8, Math.min(pr.left, window.innerWidth - MW - 8));
    if (below >= 260 || below >= above) {
      setMenuBox({ left, top: cr.bottom + gap, maxH: Math.max(180, below) });
      setBand({ top: cr.bottom + gap, bottom: window.innerHeight - 8 });
    } else {
      setMenuBox({
        left,
        bottom: window.innerHeight - cr.top + gap,
        maxH: Math.max(180, above),
      });
      setBand({ top: 8, bottom: cr.top - gap });
    }
    setOpen(true);
  };

  const choose = (id: string) => {
    onSelectModel(id);
    setOpen(false);
  };

  const openSubNow = (
    s: { kind: "provider"; name: string } | { kind: "mode" },
    anchor: HTMLElement,
  ) => {
    clearHoverTimer();
    const mr = menuRef.current?.getBoundingClientRect();
    if (!mr) return;
    let left = mr.right + 4;
    if (left + MW > window.innerWidth - 8) left = mr.left - MW - 4;
    setSub({
      ...s,
      anchorTop: anchor.getBoundingClientRect().top,
      left: Math.max(8, left),
    });
  };

  // Hover intent: delay the switch so a diagonal move toward the panel holds.
  const openSubSoon = (
    s: { kind: "provider"; name: string } | { kind: "mode" },
    anchor: HTMLElement,
  ) => {
    const same =
      sub &&
      sub.kind === s.kind &&
      (sub.kind !== "provider" || (s.kind === "provider" && sub.name === s.name));
    if (same) {
      clearHoverTimer();
      return;
    }
    if (!sub) {
      openSubNow(s, anchor);
      return;
    }
    clearHoverTimer();
    hoverTimer.current = setTimeout(() => openSubNow(s, anchor), 140);
  };

  const closeSubSoon = () => {
    clearHoverTimer();
    if (!sub) return;
    hoverTimer.current = setTimeout(() => setSub(null), 240);
  };

  const modelRow = (m: AvailableModel, hint?: string) => (
    <ModelRow
      key={m.id}
      label={formatModelName(m.display_name)}
      hint={hint ?? (m.is_default ? "default" : undefined)}
      selected={selectedId === m.id}
      pinned={isPinned(m.id)}
      onSelect={() => choose(m.id)}
      onTogglePin={() => onTogglePin(m.id)}
    />
  );

  const subProvider =
    sub?.kind === "provider" ? providers.find((p) => p.name === sub.name) : undefined;

  return (
    <div ref={rootRef} className="relative inline-flex">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={toggle}
        className="hover:bg-muted inline-flex h-8 max-w-[240px] items-center gap-1.5 rounded-full px-3 text-xs font-medium transition-colors"
      >
        {useLocal && <Server className="size-3.5 shrink-0 text-emerald-500" />}
        <span className="truncate">{triggerLabel}</span>
        <ChevronDown
          className={cn(
            "text-muted-foreground size-3.5 shrink-0 transition-transform",
            open && "rotate-180",
          )}
        />
      </button>

      {open &&
        createPortal(
          <div
            ref={menuRef}
            role="menu"
            onScroll={() => {
              clearHoverTimer();
              setSub(null);
            }}
            style={{
              position: "fixed",
              left: menuBox.left,
              top: menuBox.top,
              bottom: menuBox.bottom,
              maxHeight: menuBox.maxH,
            }}
            className={cn(
              "animate-in fade-in-0 zoom-in-95 z-[90] flex w-64 flex-col overflow-y-auto",
              panel,
            )}
          >
            {/* Auto */}
            <div className="shrink-0" onMouseEnter={closeSubSoon}>
              <MenuRow checked={autoSelected} onClick={() => choose(AUTO_MODEL_ID)}>
                <span className="flex flex-col">
                  <span className="truncate">Auto</span>
                  <span className="text-muted-foreground truncate text-[11px]">
                    Best model per task
                  </span>
                </span>
              </MenuRow>
            </div>

            {pinnedModels.length > 0 && (
              <div className="flex min-h-0 flex-col" onMouseEnter={closeSubSoon}>
                <div className={cn(sectionLabel, "shrink-0")}>Pinned</div>
                <div className="min-h-0 overflow-y-auto">
                  {pinnedModels.map((m) => (
                    <div
                      key={m.id}
                      draggable
                      onDragStart={(event) => {
                        setDraggedPinned(m.id);
                        event.dataTransfer.effectAllowed = "move";
                      }}
                      onDragOver={(event) => {
                        event.preventDefault();
                        event.dataTransfer.dropEffect = "move";
                      }}
                      onDrop={(event) => {
                        event.preventDefault();
                        if (draggedPinned && draggedPinned !== m.id) {
                          const box = event.currentTarget.getBoundingClientRect();
                          onReorderPinned(
                            draggedPinned,
                            m.id,
                            event.clientY > box.top + box.height / 2,
                          );
                        }
                        setDraggedPinned(null);
                      }}
                      onDragEnd={() => setDraggedPinned(null)}
                      className={cn(
                        "flex cursor-grab items-center active:cursor-grabbing",
                        draggedPinned === m.id && "opacity-40",
                      )}
                    >
                      <GripVertical className="text-muted-foreground size-3.5 shrink-0" />
                      <div className="min-w-0 flex-1">{modelRow(m, m.provider_name)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Models grouped by provider */}
            {providers.length > 0 && (
              <div className="shrink-0">
                <div className="bg-border my-1 h-px" />
                <div className={sectionLabel} onMouseEnter={closeSubSoon}>
                  Providers
                </div>
                {providers.map((p) => (
                  <MenuRow
                    key={p.name}
                    chevron
                    active={sub?.kind === "provider" && sub.name === p.name}
                    onClick={(e) => openSubNow({ kind: "provider", name: p.name }, e.currentTarget)}
                    onMouseEnter={(e) =>
                      openSubSoon({ kind: "provider", name: p.name }, e.currentTarget)
                    }
                  >
                    {p.name}
                  </MenuRow>
                ))}
              </div>
            )}

            <div className="shrink-0">
              <div className="bg-border my-1 h-px" />
              <MenuRow
                chevron
                active={sub?.kind === "mode"}
                onClick={(e) => openSubNow({ kind: "mode" }, e.currentTarget)}
                onMouseEnter={(e) => openSubSoon({ kind: "mode" }, e.currentTarget)}
              >
                Options
              </MenuRow>
            </div>
          </div>,
          document.body,
        )}

      {open &&
        sub &&
        createPortal(
          <div
            ref={subRef}
            role="menu"
            onClick={(e) => e.stopPropagation()}
            onMouseEnter={clearHoverTimer}
            style={{
              position: "fixed",
              top: subTop ?? band.top,
              left: sub.left,
              maxHeight: band.bottom - band.top,
              visibility: subTop === null ? "hidden" : undefined,
            }}
            className={cn(
              "animate-in fade-in-0 zoom-in-95 z-[100] w-64 overflow-y-auto",
              panel,
              sub.kind === "mode" && "p-1.5",
            )}
          >
            {sub.kind === "provider" ? (
              (subProvider?.models ?? []).map((m) => modelRow(m))
            ) : (
              <>
                <div className="text-muted-foreground px-1 pb-1 text-[10px] font-semibold tracking-wide uppercase">
                  Options
                </div>
                {toggles.map((t) => (
                  <div key={t.id} className="flex items-start gap-2 rounded-md px-2 py-1.5">
                    <label
                      htmlFor={`nm-${t.id}`}
                      className="min-w-0 flex-1 cursor-pointer select-none"
                    >
                      <span
                        className={cn(
                          "block text-sm font-medium",
                          t.tone === "warn" && t.active && "text-amber-600 dark:text-amber-400",
                        )}
                      >
                        {t.name}
                      </span>
                      {t.description && (
                        <span className="text-muted-foreground block text-[11px]">
                          {t.description}
                        </span>
                      )}
                    </label>
                    <Switch
                      id={`nm-${t.id}`}
                      checked={t.active}
                      onCheckedChange={t.onToggle}
                      className="mt-0.5"
                    />
                  </div>
                ))}
              </>
            )}
          </div>,
          document.body,
        )}
    </div>
  );
}

/**
 * Response-mode dropdown for the composer toolbar (left of Send): shows the
 * active mode and opens a small menu to switch General / Thinking / Research.
 */
export function ModeSelector({
  mode,
  onModeChange,
  className,
}: {
  mode: Mode;
  onModeChange: (m: Mode) => void;
  className?: string;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [box, setBox] = useState<{
    left: number;
    top?: number;
    bottom?: number;
  } | null>(null);

  const current = MODE_META[mode] ?? MODE_META.general;
  const CurrentIcon = current.icon;
  const W = 248;

  const toggle = () => {
    if (open) {
      setOpen(false);
      return;
    }
    const el = rootRef.current;
    if (!el) {
      setOpen(true);
      return;
    }
    const composer = (el.closest("[data-prompt-composer]") as HTMLElement | null) ?? el;
    const cr = composer.getBoundingClientRect();
    const pr = el.getBoundingClientRect();
    const gap = 8;
    const above = cr.top - gap;
    const below = window.innerHeight - cr.bottom - gap;
    const left = Math.max(8, Math.min(pr.left, cr.right - W, window.innerWidth - W - 8));
    if (below >= 260 || below >= above) {
      setBox({ left, top: cr.bottom + gap });
    } else {
      setBox({ left, bottom: window.innerHeight - cr.top + gap });
    }
    setOpen(true);
  };

  useEffect(() => {
    pinComposerZoom(rootRef.current, open);
    return () => pinComposerZoom(rootRef.current, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (rootRef.current?.contains(t) || menuRef.current?.contains(t)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    const onResize = () => setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", onResize);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", onResize);
    };
  }, [open]);

  return (
    <div ref={rootRef} className={cn("relative inline-flex", className)}>
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={toggle}
        title="Response mode"
        className="hover:bg-muted inline-flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium transition-colors"
      >
        <CurrentIcon className="text-muted-foreground size-3.5 shrink-0" />
        <span className="truncate">{current.label}</span>
        <ChevronDown
          className={cn(
            "text-muted-foreground size-3.5 shrink-0 transition-transform",
            open && "rotate-180",
          )}
        />
      </button>

      {open &&
        box &&
        createPortal(
          <div
            ref={menuRef}
            role="menu"
            style={{
              position: "fixed",
              left: box.left,
              top: box.top,
              bottom: box.bottom,
              width: W,
            }}
            className="glass animate-in fade-in-0 zoom-in-95 text-popover-foreground z-[90] rounded-xl border p-1.5 shadow-xl"
          >
            {(Object.keys(MODE_META) as Mode[]).map((k) => {
              const m = MODE_META[k];
              const Icon = m.icon;
              return (
                <button
                  key={k}
                  type="button"
                  onClick={() => {
                    onModeChange(k);
                    setOpen(false);
                  }}
                  className="hover:bg-accent/60 flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors"
                >
                  <Icon className="text-muted-foreground mt-0.5 size-4 shrink-0" />
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-medium">{m.label}</span>
                    <span className="text-muted-foreground block text-[11px]">{m.hint}</span>
                  </span>
                  <Check
                    className={cn(
                      "text-primary mt-0.5 size-4 shrink-0",
                      mode === k ? "opacity-100" : "opacity-0",
                    )}
                  />
                </button>
              );
            })}
          </div>,
          document.body,
        )}
    </div>
  );
}

/**
 * Model picker pill for the chat input toolbar; the Local LLM toggle opens an
 * endpoint config dialog.
 */
export default function ModelSelector({
  selection,
  onChange,
  hideToolCalls = false,
  onHideToolCallsChange,
}: {
  selection: ModelSelection;
  onChange: (sel: ModelSelection) => void;
  hideToolCalls?: boolean;
  onHideToolCallsChange?: (v: boolean) => void;
}) {
  const [models, setModels] = useState<AvailableModel[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);

  // Local draft so editing in the dialog doesn't immediately mutate live state.
  const [draft, setDraft] = useState({
    local_base_url: selection.local_base_url,
    local_api_key: selection.local_api_key,
    local_model_name: selection.local_model_name,
  });

  useEffect(() => {
    fetch("/api/v1/models")
      .then((r) => r.json())
      .then((data) => {
        setModels(data);
        setLoaded(true);
        if (!selection.model_id) {
          onChange({ ...selection, model_id: AUTO_MODEL_ID });
        }
      })
      .catch(() => setLoaded(true));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-sync draft when dialog opens
  useEffect(() => {
    if (dialogOpen) {
      setDraft({
        local_base_url: selection.local_base_url,
        local_api_key: selection.local_api_key,
        local_model_name: selection.local_model_name,
      });
    }
  }, [dialogOpen, selection]);

  const activeLabel = selection.use_local
    ? `Local · ${selection.local_model_name || "model"}`
    : selection.model_id === AUTO_MODEL_ID
      ? "Auto"
      : (() => {
          const m = models.find((x) => x.id === selection.model_id);
          return m
            ? formatModelName(m.display_name)
            : loaded && models.length === 0
              ? "No models"
              : "Select model";
        })();

  const pinnedIds = selection.pinned_models ?? [];
  const selectedId = selection.use_local ? null : (selection.model_id ?? AUTO_MODEL_ID);
  const isPinned = (id: string) => pinnedIds.includes(id);

  const pinnedModels = loaded
    ? pinnedIds.map((id) => models.find((m) => m.id === id)).filter((m): m is AvailableModel => !!m)
    : [];
  const byProvider = new Map<string, AvailableModel[]>();
  if (loaded) {
    for (const m of models) {
      if (isPinned(m.id)) continue;
      byProvider.set(m.provider_name, [...(byProvider.get(m.provider_name) ?? []), m]);
    }
  }
  const providers = [...byProvider.entries()]
    .map(([name, ms]) => ({ name, models: ms }))
    .sort((a, b) => a.name.localeCompare(b.name));

  const togglePin = (id: string) => {
    const cur = selection.pinned_models ?? [];
    onChange({
      ...selection,
      pinned_models: cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id],
    });
  };

  const reorderPinned = (source: string, target: string, after: boolean) => {
    const next = pinnedIds.filter((id) => id !== source);
    next.splice(next.indexOf(target) + Number(after), 0, source);
    onChange({ ...selection, pinned_models: next });
  };

  const toggles: ToggleDef[] = [
    {
      id: "local",
      name: "Local LLM",
      description: "Route to your own OpenAI-compatible endpoint",
      active: selection.use_local,
      onToggle: (v) => {
        onChange({ ...selection, use_local: v });
        if (v) setDialogOpen(true);
      },
    },
    {
      id: "hide-tools",
      name: "Hide Tool Calls",
      description: "Collapse tool activity in the transcript",
      active: hideToolCalls,
      onToggle: (v) => onHideToolCallsChange?.(v),
    },
    {
      id: "unrestricted",
      name: "Unrestricted Mode",
      description: "Take the gloves off",
      active: selection.unrestricted,
      onToggle: (v) => onChange({ ...selection, unrestricted: v }),
      tone: "warn",
    },
  ];

  return (
    <>
      <div className="flex items-center gap-1.5">
        <PromptToolbarMenu
          triggerLabel={activeLabel}
          useLocal={selection.use_local}
          autoSelected={selectedId === AUTO_MODEL_ID}
          pinnedModels={pinnedModels}
          providers={providers}
          selectedId={selectedId}
          isPinned={isPinned}
          onSelectModel={(value) =>
            onChange({
              ...selection,
              model_id: value || null,
              use_local: false,
            })
          }
          onTogglePin={togglePin}
          onReorderPinned={reorderPinned}
          toggles={toggles}
        />
        {selection.unrestricted && (
          <button
            type="button"
            onClick={() => onChange({ ...selection, unrestricted: false })}
            title="Unrestricted mode is on — click to turn it off"
            className="text-muted-foreground hover:text-foreground inline-flex shrink-0 items-center gap-1 text-[11px] transition-colors"
          >
            <ShieldOff className="size-3" />
            Unrestricted
          </button>
        )}
      </div>

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Configure Local LLM</DialogTitle>
            <DialogDescription>
              Connect to any OpenAI-compatible endpoint (LM Studio, llama.cpp, vLLM, Ollama via{" "}
              <code>/v1</code>).
            </DialogDescription>
          </DialogHeader>

          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="lc-url" className="text-xs">
                Base URL
              </Label>
              <Input
                id="lc-url"
                value={draft.local_base_url}
                onChange={(e) => setDraft((d) => ({ ...d, local_base_url: e.target.value }))}
                placeholder="http://host.docker.internal:1234/v1"
              />
              <p className="text-muted-foreground text-[11px]">
                When the chat runs in Docker, use <code>host.docker.internal</code> instead of{" "}
                <code>localhost</code>.
              </p>
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="lc-model" className="text-xs">
                Model name
              </Label>
              <Input
                id="lc-model"
                value={draft.local_model_name}
                onChange={(e) => setDraft((d) => ({ ...d, local_model_name: e.target.value }))}
                placeholder="llama-3.1-8b-instruct"
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="lc-key" className="text-xs">
                API key (optional)
              </Label>
              <Input
                id="lc-key"
                type="password"
                value={draft.local_api_key}
                onChange={(e) => setDraft((d) => ({ ...d, local_api_key: e.target.value }))}
                placeholder="leave blank if not required"
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              type="button"
              onClick={() => {
                onChange({ ...selection, use_local: false });
                setDialogOpen(false);
              }}
            >
              Disable
            </Button>
            <Button
              type="button"
              onClick={() => {
                onChange({ ...selection, ...draft, use_local: true });
                setDialogOpen(false);
              }}
            >
              Save & use
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
