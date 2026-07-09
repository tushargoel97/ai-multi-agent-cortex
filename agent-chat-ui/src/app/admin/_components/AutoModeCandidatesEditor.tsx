"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { getAdminToken } from "../token";
import { toast } from "sonner";
import {
  ArrowDown,
  ArrowUp,
  ChevronDown,
  ChevronRight,
  Plus,
  RotateCcw,
  Save,
  Sparkles,
  X,
} from "lucide-react";

// { profile: { intent: [model_id, ...] } }, mirrors cortex auto_mode.yaml.
type Profiles = Record<string, Record<string, string[]>>;

interface ModelOpt {
  model_id: string;
  display_name: string;
  provider_kind: string;
}

const PROFILES = ["balanced", "quality", "cost"] as const;

const INTENT_ORDER = [
  "fast",
  "general_chat",
  "knowledge_query",
  "reasoning_task",
  "coding_task",
  "prompt_caching",
  "product_specs",
  "shopping",
  "booking",
  "image_generation",
];

const INTENT_LABELS: Record<string, string> = {
  fast: "Fast tier",
  general_chat: "General chat",
  knowledge_query: "Knowledge query",
  reasoning_task: "Reasoning task",
  coding_task: "Coding task",
  prompt_caching: "Prompt caching",
  product_specs: "Product specs",
  shopping: "Shopping",
  booking: "Booking",
  image_generation: "Image generation",
};

async function adminFetch(url: string, init: RequestInit = {}) {
  const token = getAdminToken();
  return fetch(url, {
    ...init,
    headers: {
      ...(init.headers || {}),
      "Content-Type": "application/json",
      "X-Admin-Token": token || "",
    },
  });
}

export default function AutoModeCandidatesEditor({
  refreshKey = 0,
}: {
  refreshKey?: number;
}) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activeProfile, setActiveProfile] = useState("balanced");
  const [profile, setProfile] = useState<string>("balanced");
  const [defaults, setDefaults] = useState<Profiles>({});
  const [overrides, setOverrides] = useState<Profiles>({});
  const [models, setModels] = useState<ModelOpt[]>([]);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);

  async function load() {
    setLoading(true);
    try {
      const r = await adminFetch("/api/admin/auto-mode");
      if (!r.ok) {
        toast.error("Could not load auto-mode config");
        return;
      }
      const d = await r.json();
      setActiveProfile(d.activeProfile || "balanced");
      setProfile((p) => (d.defaults?.[p] ? p : d.activeProfile || "balanced"));
      setDefaults(d.defaults || {});
      setOverrides(d.overrides || {});
      setModels(d.models || []);
      setDirty(false);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (open) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, refreshKey]);

  // Canonical intent set for the selected profile (defaults ∪ overrides).
  const intents = (() => {
    const keys = new Set<string>([
      ...Object.keys(defaults[profile] || {}),
      ...Object.keys(overrides[profile] || {}),
    ]);
    const ordered = INTENT_ORDER.filter((i) => keys.has(i));
    const extra = [...keys].filter((k) => !INTENT_ORDER.includes(k));
    return [...ordered, ...extra];
  })();

  function effectiveList(p: string, intent: string): string[] {
    return overrides[p]?.[intent] ?? defaults[p]?.[intent] ?? [];
  }
  function isOverridden(p: string, intent: string): boolean {
    return overrides[p]?.[intent] !== undefined;
  }

  function mutate(p: string, intent: string, next: string[] | null) {
    setOverrides((prev) => {
      const copy: Profiles = { ...prev, [p]: { ...(prev[p] || {}) } };
      if (next === null) {
        delete copy[p][intent];
        if (Object.keys(copy[p]).length === 0) delete copy[p];
      } else {
        copy[p][intent] = next;
      }
      return copy;
    });
    setDirty(true);
  }

  function addCandidate(p: string, intent: string) {
    const key = `${p}::${intent}`;
    const value = (drafts[key] || "").trim();
    if (!value) return;
    const list = effectiveList(p, intent);
    if (list.includes(value)) {
      toast.error(`${value} is already listed`);
      return;
    }
    mutate(p, intent, [...list, value]);
    setDrafts((d) => ({ ...d, [key]: "" }));
  }
  function removeCandidate(p: string, intent: string, idx: number) {
    const list = [...effectiveList(p, intent)];
    list.splice(idx, 1);
    mutate(p, intent, list);
  }
  function moveCandidate(p: string, intent: string, idx: number, dir: -1 | 1) {
    const list = [...effectiveList(p, intent)];
    const j = idx + dir;
    if (j < 0 || j >= list.length) return;
    [list[idx], list[j]] = [list[j], list[idx]];
    mutate(p, intent, list);
  }

  async function save() {
    setSaving(true);
    try {
      const r = await adminFetch("/api/admin/auto-mode", {
        method: "PUT",
        body: JSON.stringify({ overrides }),
      });
      if (!r.ok) {
        const e = await r.json().catch(() => ({}));
        toast.error(e.error || "Save failed");
        return;
      }
      toast.success("Auto-mode candidates saved");
      load();
    } finally {
      setSaving(false);
    }
  }

  function labelFor(id: string): string {
    if (id === "finetuned") return "finetuned · newest fine-tuned model";
    const m = models.find((x) => x.model_id === id);
    return m ? `${id} · ${m.display_name}` : id;
  }
  function isKnown(id: string): boolean {
    return id === "finetuned" || models.some((m) => m.model_id === id);
  }

  return (
    <div className="rounded-lg border bg-muted/30">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-3 p-4 text-left"
      >
        <div>
          <h3 className="flex items-center gap-1.5 text-sm font-semibold">
            <Sparkles className="size-4 text-amber-500" />
            Auto-mode candidates
          </h3>
          <p className="text-muted-foreground text-xs">
            Per-intent model order for each profile, the first enabled model
            wins. Your edits layer over the shipped defaults.
          </p>
        </div>
        {open ? (
          <ChevronDown className="size-4 shrink-0" />
        ) : (
          <ChevronRight className="size-4 shrink-0" />
        )}
      </button>

      {open && (
        <div className="space-y-4 border-t p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground text-xs">
                Editing profile
              </span>
              <div className="flex items-center gap-1 rounded-full border border-border bg-background/60 p-1">
                {PROFILES.map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => setProfile(p)}
                    className={
                      "rounded-full px-3 py-1 text-xs font-medium capitalize transition-colors " +
                      (profile === p
                        ? "bg-primary text-primary-foreground"
                        : "text-muted-foreground hover:bg-muted")
                    }
                  >
                    {p}
                    {p === activeProfile && (
                      <span className="ml-1 opacity-70">(active)</span>
                    )}
                  </button>
                ))}
              </div>
            </div>
            <Button onClick={save} disabled={!dirty || saving} size="sm">
              <Save className="mr-2 size-4" />
              {saving ? "Saving…" : "Save changes"}
            </Button>
          </div>

          {loading ? (
            <p className="text-muted-foreground text-sm">Loading…</p>
          ) : intents.length === 0 ? (
            <p className="text-muted-foreground text-sm">
              No defaults published yet, send one chat in Auto mode so the
              graph mirrors its shipped candidate lists here.
            </p>
          ) : (
            <div className="space-y-3">
              {intents.map((intent) => {
                const list = effectiveList(profile, intent);
                const overridden = isOverridden(profile, intent);
                const key = `${profile}::${intent}`;
                return (
                  <div
                    key={intent}
                    className="rounded-md border bg-background/60 p-3"
                  >
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">
                          {INTENT_LABELS[intent] || intent}
                        </span>
                        <code className="text-muted-foreground text-[11px]">
                          {intent}
                        </code>
                        {overridden && (
                          <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] font-medium text-amber-600">
                            customized
                          </span>
                        )}
                      </div>
                      {overridden && (
                        <button
                          type="button"
                          onClick={() => mutate(profile, intent, null)}
                          className="text-muted-foreground hover:text-foreground flex items-center gap-1 text-[11px]"
                          title="Revert to shipped default"
                        >
                          <RotateCcw className="size-3" /> reset
                        </button>
                      )}
                    </div>

                    <ol className="space-y-1.5">
                      {list.map((id, idx) => (
                        <li
                          key={`${id}-${idx}`}
                          className="flex items-center gap-2"
                        >
                          <span className="text-muted-foreground w-5 text-right text-xs tabular-nums">
                            {idx + 1}.
                          </span>
                          <span
                            className={
                              "flex-1 truncate text-xs " +
                              (isKnown(id) ? "" : "text-amber-600")
                            }
                            title={
                              isKnown(id)
                                ? id
                                : `${id}, not an enabled registry model`
                            }
                          >
                            {labelFor(id)}
                          </span>
                          <div className="flex items-center gap-0.5">
                            <button
                              type="button"
                              onClick={() =>
                                moveCandidate(profile, intent, idx, -1)
                              }
                              disabled={idx === 0}
                              className="text-muted-foreground rounded p-1 hover:bg-muted disabled:opacity-30"
                              title="Move up"
                            >
                              <ArrowUp className="size-3.5" />
                            </button>
                            <button
                              type="button"
                              onClick={() =>
                                moveCandidate(profile, intent, idx, 1)
                              }
                              disabled={idx === list.length - 1}
                              className="text-muted-foreground rounded p-1 hover:bg-muted disabled:opacity-30"
                              title="Move down"
                            >
                              <ArrowDown className="size-3.5" />
                            </button>
                            <button
                              type="button"
                              onClick={() =>
                                removeCandidate(profile, intent, idx)
                              }
                              className="rounded p-1 text-rose-500 hover:bg-rose-500/10"
                              title="Remove"
                            >
                              <X className="size-3.5" />
                            </button>
                          </div>
                        </li>
                      ))}
                      {list.length === 0 && (
                        <li className="text-muted-foreground text-xs italic">
                          No candidates, this intent falls back to the fast
                          tier.
                        </li>
                      )}
                    </ol>

                    <div className="mt-2 flex items-center gap-2">
                      <input
                        list={`models-${key}`}
                        value={drafts[key] || ""}
                        onChange={(e) =>
                          setDrafts((d) => ({ ...d, [key]: e.target.value }))
                        }
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            addCandidate(profile, intent);
                          }
                        }}
                        placeholder="model id (or 'finetuned')"
                        className="h-8 flex-1 rounded-md border bg-background/60 px-2 text-xs"
                      />
                      <datalist id={`models-${key}`}>
                        <option value="finetuned" />
                        {models.map((m) => (
                          <option key={m.model_id} value={m.model_id}>
                            {m.display_name}
                          </option>
                        ))}
                      </datalist>
                      <button
                        type="button"
                        onClick={() => addCandidate(profile, intent)}
                        className="flex items-center gap-1 rounded-md border px-2 py-1 text-xs hover:bg-muted"
                      >
                        <Plus className="size-3.5" /> add
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
