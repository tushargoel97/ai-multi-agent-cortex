"use client";

import { useEffect, useState } from "react";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { ChevronDown, Cpu, Server } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
    };
  }
  return { model_id: sel.model_id };
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

/**
 * Compact in-line model picker for the chat input toolbar.
 * - Cloud mode: shows a small select.
 * - Local mode: shows a "Local LLM" pill that opens a config dialog.
 */
export default function ModelSelector({
  selection,
  onChange,
}: {
  selection: ModelSelection;
  onChange: (sel: ModelSelection) => void;
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
            ? `${m.display_name}`
            : loaded && models.length === 0
              ? "No models"
              : "Select model";
        })();

  return (
    <>
      <div className="flex items-center gap-2">
        {!selection.use_local ? (
          <div className="relative inline-flex items-center">
            <Cpu className="pointer-events-none absolute left-3 size-3.5 text-muted-foreground" />
            <select
              aria-label="Select model"
              value={selection.model_id ?? ""}
              onChange={(e) =>
                onChange({ ...selection, model_id: e.target.value || null })
              }
              className="h-8 max-w-[220px] cursor-pointer appearance-none truncate rounded-full border border-border bg-muted/50 pl-8 pr-8 text-xs font-medium text-foreground transition-colors hover:bg-muted focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <option value={AUTO_MODEL_ID}>
                ✨ Auto, best model per task
              </option>
              {!loaded && <option value="">Loading…</option>}
              {loaded && models.length === 0 && (
                <option value="">No models (see /admin)</option>
              )}
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.display_name}, {m.provider_name}
                  {m.is_default ? " (default)" : ""}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-3 size-3.5 text-muted-foreground" />
          </div>
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

        <div className="flex items-center gap-1.5">
          <Switch
            id="use-local-llm"
            checked={selection.use_local}
            onCheckedChange={(v) => {
              onChange({ ...selection, use_local: v });
              if (v) setDialogOpen(true);
            }}
          />
          <Label
            htmlFor="use-local-llm"
            className="cursor-pointer text-xs text-muted-foreground"
          >
            Local LLM
          </Label>
        </div>
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
