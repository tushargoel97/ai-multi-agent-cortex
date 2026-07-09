"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Check, ChevronDown, ChevronRight, Server } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import { useDropdown } from "@/hooks/use-dropdown";
import { type ToggleDef } from "@/components/toggles-menu";
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
  /** Response mode: general (default) | thinking | research. */
  mode: "general" | "thinking" | "research";
}

const STORAGE_KEY = "cortex:model-selection";

/** Sentinel understood by the graph: pick the model per intent (auto mode). */
export const AUTO_MODEL_ID = "auto";

export const DEFAULT_SELECTION: ModelSelection = {
  model_id: AUTO_MODEL_ID,
  use_local: false,
  local_base_url: "http://host.docker.internal:1234/v1",
  local_api_key: "",
  local_model_name: "local-model",
  unrestricted: false,
  mode: "general",
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

export function selectionToConfigurable(
  sel: ModelSelection,
): Record<string, unknown> {
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
  return { model_id: sel.model_id, unrestricted: sel.unrestricted, mode: sel.mode };
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
export function formatModelName(raw: string): string {
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

const NESTED_MODES = [
  { value: "general", label: "General", hint: "Fast, direct answers." },
  { value: "thinking", label: "Thinking", hint: "Reasoner + extended thinking." },
  {
    value: "research",
    label: "Research",
    hint: "Deep web/KB research; clarifies first.",
  },
];

interface MenuOption {
  value: string;
  label: ReactNode;
  hint?: string;
  checked?: boolean;
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
  onClick?: () => void;
  onMouseEnter?: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      className={cn(
        "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors hover:bg-accent/60",
        active && "bg-accent/60",
      )}
    >
      <span className="min-w-0 flex-1">{children}</span>
      {chevron ? (
        <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" />
      ) : (
        <Check
          className={cn(
            "size-3.5 shrink-0 text-primary",
            checked ? "opacity-100" : "opacity-0",
          )}
        />
      )}
    </button>
  );
}

/**
 * Claude-style consolidated prompt-box menu: a single pill that opens the top
 * models, a nested "More models" list, and a nested "Mode & options" panel
 * (General / Thinking / Research + Local LLM / Hide tools / Unrestricted).
 */
function PromptToolbarMenu({
  triggerLabel,
  modeLabel,
  useLocal,
  topOptions,
  moreOptions,
  onSelectModel,
  mode,
  onModeChange,
  toggles,
}: {
  triggerLabel: string;
  modeLabel: string | null;
  useLocal: boolean;
  topOptions: MenuOption[];
  moreOptions: MenuOption[];
  onSelectModel: (value: string) => void;
  mode: string;
  onModeChange: (m: string) => void;
  toggles: ToggleDef[];
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const { open, setOpen, mounted, openUp } = useDropdown(rootRef, {
    estimatedHeight: 360,
  });
  const [sub, setSub] = useState<null | "more" | "mode">(null);

  useEffect(() => {
    if (!open) setSub(null);
  }, [open]);

  const panel =
    "rounded-xl border border-border bg-popover p-1 text-popover-foreground shadow-lg";
  const subPos = openUp ? "bottom-0" : "top-0";

  const choose = (value: string) => {
    onSelectModel(value);
    setOpen(false);
  };

  const renderOption = (o: MenuOption) => (
    <MenuRow key={o.value} checked={o.checked} onClick={() => choose(o.value)}>
      <span className="flex flex-col">
        <span className="truncate">{o.label}</span>
        {o.hint && (
          <span className="truncate text-[11px] text-muted-foreground">
            {o.hint}
          </span>
        )}
      </span>
    </MenuRow>
  );

  return (
    <div ref={rootRef} className="relative inline-flex">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="inline-flex h-8 max-w-[240px] items-center gap-1.5 rounded-full border border-border bg-muted/50 px-3 text-xs font-medium transition-colors hover:bg-muted"
      >
        {useLocal && <Server className="size-3.5 shrink-0 text-emerald-500" />}
        <span className="truncate">{triggerLabel}</span>
        {modeLabel && (
          <span className="shrink-0 text-muted-foreground">{modeLabel}</span>
        )}
        <ChevronDown
          className={cn(
            "size-3.5 shrink-0 text-muted-foreground transition-transform",
            open && "rotate-180",
          )}
        />
      </button>

      {mounted && (
        <div
          role="menu"
          className={cn(
            "absolute right-0 z-50 w-60 duration-150",
            panel,
            openUp ? "bottom-full mb-1.5" : "top-full mt-1.5",
            open
              ? "animate-in fade-in-0 zoom-in-95"
              : "animate-out fade-out-0 zoom-out-95 pointer-events-none",
          )}
        >
          <div onMouseEnter={() => setSub(null)}>
            {topOptions.map(renderOption)}
          </div>

          {moreOptions.length > 0 && (
            <div className="relative">
              <MenuRow
                chevron
                active={sub === "more"}
                onMouseEnter={() => setSub("more")}
              >
                More models
              </MenuRow>
              {sub === "more" && (
                <div
                  className={cn(
                    "absolute right-full z-50 mr-1 max-h-[320px] w-56 overflow-y-auto",
                    panel,
                    subPos,
                  )}
                >
                  {moreOptions.map(renderOption)}
                </div>
              )}
            </div>
          )}

          <div className="my-1 h-px bg-border" />

          <div className="relative">
            <MenuRow
              chevron
              active={sub === "mode"}
              onMouseEnter={() => setSub("mode")}
            >
              <span className="flex items-center justify-between gap-2">
                <span>Mode &amp; options</span>
                <span className="text-[11px] text-muted-foreground">
                  {modeLabel ?? "General"}
                </span>
              </span>
            </MenuRow>
            {sub === "mode" && (
              <div
                className={cn(
                  "absolute right-full z-50 mr-1 w-64 p-1.5",
                  panel,
                  subPos,
                )}
              >
                <div className="px-1 pb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Mode
                </div>
                {NESTED_MODES.map((m) => (
                  <button
                    key={m.value}
                    type="button"
                    onClick={() => onModeChange(m.value)}
                    className="flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left hover:bg-accent/60"
                  >
                    <span className="min-w-0 flex-1">
                      <span className="block text-sm font-medium">
                        {m.label}
                      </span>
                      <span className="block text-[11px] text-muted-foreground">
                        {m.hint}
                      </span>
                    </span>
                    <Check
                      className={cn(
                        "mt-0.5 size-3.5 shrink-0 text-primary",
                        mode === m.value ? "opacity-100" : "opacity-0",
                      )}
                    />
                  </button>
                ))}
                <div className="my-1 h-px bg-border" />
                {toggles.map((t) => (
                  <div
                    key={t.id}
                    className="flex items-start gap-2 rounded-md px-2 py-1.5"
                  >
                    <label
                      htmlFor={`nm-${t.id}`}
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
                        <span className="block text-[11px] text-muted-foreground">
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
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Compact in-line model picker for the chat input toolbar.
 * - Cloud mode: shows a small select.
 * - Local mode: shows a "Local LLM" pill that opens a config dialog.
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
    fetch("/api/models")
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

  const isSel = (id: string) =>
    !selection.use_local && (selection.model_id ?? AUTO_MODEL_ID) === id;

  const toOption = (m: AvailableModel): MenuOption => ({
    value: m.id,
    label: formatModelName(m.display_name),
    hint: `${m.provider_name}${m.is_default ? " · default" : ""}`,
    checked: isSel(m.id),
  });

  // Top level shows Auto + the default + a couple more; the rest nest under
  // "More models". An explicitly-selected model is always surfaced up top.
  const topModels: AvailableModel[] = [];
  const defaultModel = models.find((m) => m.is_default);
  if (defaultModel) topModels.push(defaultModel);
  for (const m of models) {
    if (topModels.length >= 4) break;
    if (!topModels.some((t) => t.id === m.id)) topModels.push(m);
  }
  if (
    !selection.use_local &&
    selection.model_id &&
    selection.model_id !== AUTO_MODEL_ID
  ) {
    const sel = models.find((m) => m.id === selection.model_id);
    if (sel && !topModels.some((t) => t.id === sel.id)) {
      topModels.unshift(sel);
      if (topModels.length > 5) topModels.pop();
    }
  }
  const topIds = new Set(topModels.map((m) => m.id));
  const topOptions: MenuOption[] = [
    {
      value: AUTO_MODEL_ID,
      label: "✨ Auto",
      hint: "Best model per task",
      checked: isSel(AUTO_MODEL_ID),
    },
    ...(loaded ? topModels.map(toOption) : []),
  ];
  const moreOptions: MenuOption[] = loaded
    ? models.filter((m) => !topIds.has(m.id)).map(toOption)
    : [];

  const modeLabel =
    selection.mode === "thinking"
      ? "Thinking"
      : selection.mode === "research"
        ? "Research"
        : null;

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
      description:
        "Direct answers, no PII redaction, and a relaxed image pre-screen. Providers still enforce their own limits.",
      active: selection.unrestricted,
      onToggle: (v) => onChange({ ...selection, unrestricted: v }),
      tone: "warn",
    },
  ];

  return (
    <>
      <PromptToolbarMenu
        triggerLabel={activeLabel}
        modeLabel={modeLabel}
        useLocal={selection.use_local}
        topOptions={topOptions}
        moreOptions={moreOptions}
        onSelectModel={(value) =>
          onChange({ ...selection, model_id: value || null, use_local: false })
        }
        mode={selection.mode}
        onModeChange={(m) =>
          onChange({ ...selection, mode: m as ModelSelection["mode"] })
        }
        toggles={toggles}
      />

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Configure Local LLM</DialogTitle>
            <DialogDescription>
              Connect to any OpenAI-compatible endpoint (LM Studio,
              llama.cpp, vLLM, Ollama via <code>/v1</code>).
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
                onChange={(e) =>
                  setDraft((d) => ({ ...d, local_base_url: e.target.value }))
                }
                placeholder="http://host.docker.internal:1234/v1"
              />
              <p className="text-[11px] text-muted-foreground">
                When the chat runs in Docker, use{" "}
                <code>host.docker.internal</code> instead of{" "}
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
                onChange={(e) =>
                  setDraft((d) => ({ ...d, local_model_name: e.target.value }))
                }
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
                onChange={(e) =>
                  setDraft((d) => ({ ...d, local_api_key: e.target.value }))
                }
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
