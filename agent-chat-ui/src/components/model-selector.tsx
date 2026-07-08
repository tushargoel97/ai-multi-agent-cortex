"use client";

import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Cpu, Server } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Select, type SelectOption } from "@/components/ui/select";
import { TogglesMenu } from "@/components/toggles-menu";
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

const DEFAULT_SELECTION: ModelSelection = {
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
  // Which prompt-box menu is open, so hovering the other trigger switches to it.
  const [menuOpen, setMenuOpen] = useState<"model" | "options" | null>(null);

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

  const modelOptions: SelectOption[] = [
    {
      value: AUTO_MODEL_ID,
      label: (
        <span>
          ✨ Auto{" "}
          <span className="text-muted-foreground">· best per task</span>
        </span>
      ),
    },
    ...(loaded
      ? models.map((m) => ({
          value: m.id,
          label: formatModelName(m.display_name),
          hint: `${m.provider_name}${m.is_default ? " · default" : ""}`,
        }))
      : [{ value: "__loading", label: "Loading…", disabled: true }]),
    ...(loaded && models.length === 0
      ? [{ value: "__empty", label: "No models (see /admin)", disabled: true }]
      : []),
  ];

  return (
    <>
      <div className="flex items-center gap-2">
        {!selection.use_local ? (
          <Select
            ariaLabel="Select model"
            value={selection.model_id ?? AUTO_MODEL_ID}
            onValueChange={(v) =>
              onChange({ ...selection, model_id: v || null })
            }
            options={modelOptions}
            triggerLabel={activeLabel}
            icon={<Cpu className="size-3.5 shrink-0 text-muted-foreground" />}
            open={menuOpen === "model"}
            onOpenChange={(o) => setMenuOpen(o ? "model" : null)}
            onTriggerMouseEnter={() => setMenuOpen((c) => (c ? "model" : c))}
            className="h-8 max-w-[240px] rounded-full border-border bg-muted/50 text-xs font-medium hover:bg-muted"
          />
        ) : (
          <button
            type="button"
            onClick={() => setDialogOpen(true)}
            className={cn(
              "inline-flex h-8 items-center gap-1.5 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 text-xs font-medium text-emerald-700 transition-colors hover:bg-emerald-500/20 dark:text-emerald-300",
            )}
            title={selection.local_base_url}
          >
            <Server className="size-3.5" />
            <span className="max-w-[160px] truncate">{activeLabel}</span>
          </button>
        )}

        <TogglesMenu
          open={menuOpen === "options"}
          onOpenChange={(o) => setMenuOpen(o ? "options" : null)}
          onTriggerMouseEnter={() => setMenuOpen((c) => (c ? "options" : c))}
          mode={selection.mode}
          onModeChange={(m) =>
            onChange({ ...selection, mode: m as ModelSelection["mode"] })
          }
          toggles={[
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
          ]}
        />
      </div>

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
