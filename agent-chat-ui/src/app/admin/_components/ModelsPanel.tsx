"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { DeleteButton } from "@/components/ui/delete-button";
import { getAdminToken } from "../token";
import { useDropdown } from "@/hooks/use-dropdown";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { ListFilter, RefreshCw, Search, X } from "lucide-react";
import AutoModeCandidatesEditor from "./AutoModeCandidatesEditor";

interface Provider {
  id: string;
  name: string;
  kind: string;
  enabled: boolean;
  api_key_set?: boolean;
}

interface Model {
  id: string;
  provider_id: string;
  provider_name: string;
  provider_kind: string;
  model_id: string;
  display_name: string;
  enabled: boolean;
  is_default: boolean;
}

const FAMILY_RULES: [RegExp, string][] = [
  [/\bfinetuned\b/i, "Fine-tuned"],
  [/\bclaude\b/i, "Claude"],
  [/\bchatgpt\b|\bgpt(?:[-\s]|$)/i, "GPT"],
  [/(?:^|[-\s])o[134](?:[-\s]|$)/i, "OpenAI o-series"],
  [/\bgemini\b/i, "Gemini"],
  [/\bgemma\b/i, "Gemma"],
  [/\bllama\b/i, "Llama"],
  [/\bmixtral\b/i, "Mixtral"],
  [/\bmistral\b|\bcodestral\b/i, "Mistral"],
  [/\bqwen\b/i, "Qwen"],
  [/\bdeepseek\b/i, "DeepSeek"],
  [/\bgrok\b/i, "Grok"],
  [/\bphi\b/i, "Phi"],
  [/\bcommand[-\s]?r\b|\bcohere\b/i, "Command"],
  [/\bnova\b/i, "Nova"],
  [/\btitan\b/i, "Titan"],
  [/\bgranite\b/i, "Granite"],
];

const modelFamily = (model: Model) =>
  FAMILY_RULES.find(([pattern]) => pattern.test(`${model.model_id} ${model.display_name}`))?.[1] ??
  "Other";

const toggleValue = (values: string[], value: string) =>
  values.includes(value) ? values.filter((item) => item !== value) : [...values, value];

function CheckboxFilterGroup({
  label,
  options,
  selected,
  onToggle,
}: {
  label: string;
  options: { value: string; label: string }[];
  selected: string[];
  onToggle: (value: string) => void;
}) {
  return (
    <fieldset className="space-y-1">
      <legend className="text-muted-foreground px-2 pb-1 text-xs font-medium">{label}</legend>
      {options.map((option) => (
        <label
          key={option.value}
          className="hover:bg-muted/60 flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm"
        >
          <input
            type="checkbox"
            className="accent-primary size-4"
            checked={selected.includes(option.value)}
            onChange={() => onToggle(option.value)}
          />
          <span className="truncate">{option.label}</span>
        </label>
      ))}
    </fieldset>
  );
}

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

export default function ModelsPanel({ refreshKey = 0 }: { refreshKey?: number }) {
  const confirm = useConfirm();
  const [models, setModels] = useState<Model[]>([]);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [syncing, setSyncing] = useState(false);
  const [autoProfile, setAutoProfile] = useState("balanced");
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [providerFilters, setProviderFilters] = useState<string[]>([]);
  const [familyFilters, setFamilyFilters] = useState<string[]>([]);
  const [statusFilters, setStatusFilters] = useState<string[]>([]);
  const searchRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const filterRef = useRef<HTMLDivElement>(null);
  const filterMenuRef = useRef<HTMLDivElement>(null);
  const {
    open: filtersOpen,
    setOpen: setFiltersOpen,
    mounted: filtersMounted,
  } = useDropdown(filterRef, { insideRef: filterMenuRef });
  const [filterPosition, setFilterPosition] = useState<{
    right: number;
    top?: number;
    bottom?: number;
    maxHeight: number;
  } | null>(null);
  const [form, setForm] = useState({
    provider_id: "",
    model_id: "",
    display_name: "",
    is_default: false,
  });

  async function load() {
    const [m, p] = await Promise.all([
      adminFetch("/api/admin/models"),
      adminFetch("/api/admin/providers"),
    ]);
    if (m.ok) setModels(await m.json());
    if (p.ok) setProviders(await p.json());
  }

  useEffect(() => {
    load();
  }, [refreshKey]);

  const modelRows = useMemo(
    () => models.map((model) => ({ ...model, family: modelFamily(model) })),
    [models],
  );
  const providerOptions = useMemo(
    () =>
      [
        ...new Map(
          models.map((model) => [
            model.provider_id,
            { value: model.provider_id, label: `${model.provider_name} (${model.provider_kind})` },
          ]),
        ).values(),
      ].sort((a, b) => a.label.localeCompare(b.label)),
    [models],
  );
  const familyOptions = useMemo(
    () =>
      [...new Set(modelRows.map((model) => model.family))]
        .sort()
        .map((family) => ({ value: family, label: family })),
    [modelRows],
  );
  const filteredModels = useMemo(() => {
    const term = query.trim().toLowerCase();
    return modelRows.filter(
      (model) =>
        (!term ||
          [
            model.display_name,
            model.model_id,
            model.provider_name,
            model.provider_kind,
            model.family,
          ]
            .join(" ")
            .toLowerCase()
            .includes(term)) &&
        (!providerFilters.length || providerFilters.includes(model.provider_id)) &&
        (!familyFilters.length || familyFilters.includes(model.family)) &&
        (!statusFilters.some((status) => status === "enabled" || status === "disabled") ||
          statusFilters.includes(model.enabled ? "enabled" : "disabled")) &&
        (!statusFilters.includes("default") || model.is_default),
    );
  }, [familyFilters, modelRows, providerFilters, query, statusFilters]);
  const activeFilterCount = providerFilters.length + familyFilters.length + statusFilters.length;

  const closeSearch = () => {
    setSearchOpen(false);
    setQuery("");
  };

  useEffect(() => {
    if (!searchOpen) return;
    searchInputRef.current?.focus();
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as HTMLElement;
      if (searchRef.current?.contains(target) || target.closest("[data-model-search-trigger]"))
        return;
      closeSearch();
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [searchOpen]);

  useLayoutEffect(() => {
    if (!filtersMounted || !filtersOpen) return;
    const trigger = filterRef.current?.getBoundingClientRect();
    if (!trigger) return;
    const gap = 6;
    const margin = 8;
    const below = window.innerHeight - trigger.bottom - gap - margin;
    const above = trigger.top - gap - margin;
    const up = above > below;
    setFilterPosition({
      right: Math.max(margin, window.innerWidth - trigger.right),
      maxHeight: Math.max(160, Math.min(512, up ? above : below)),
      ...(up ? { bottom: window.innerHeight - trigger.top + gap } : { top: trigger.bottom + gap }),
    });
  }, [filtersMounted, filtersOpen]);

  useEffect(() => {
    if (!filtersMounted) setFilterPosition(null);
  }, [filtersMounted]);

  const filterMenu =
    filtersMounted &&
    createPortal(
      <div
        ref={filterMenuRef}
        role="menu"
        style={{
          position: "fixed",
          right: filterPosition?.right,
          top: filterPosition?.top,
          bottom: filterPosition?.bottom,
          maxHeight: filterPosition?.maxHeight,
          visibility: filterPosition ? undefined : "hidden",
        }}
        className={cn(
          "glass z-50 w-72 space-y-3 overflow-y-auto rounded-xl border p-3 shadow-xl transition duration-150",
          filtersOpen
            ? "animate-in fade-in-0 zoom-in-95"
            : "animate-out fade-out-0 zoom-out-95 pointer-events-none",
        )}
      >
        <div className="flex items-center justify-between px-2">
          <span className="text-sm font-medium">Filters</span>
          {activeFilterCount > 0 && (
            <button
              type="button"
              className="text-muted-foreground hover:text-foreground text-xs"
              onClick={() => {
                setProviderFilters([]);
                setFamilyFilters([]);
                setStatusFilters([]);
              }}
            >
              Clear
            </button>
          )}
        </div>
        <CheckboxFilterGroup
          label="Provider"
          options={providerOptions}
          selected={providerFilters}
          onToggle={(value) => setProviderFilters((filters) => toggleValue(filters, value))}
        />
        <div className="bg-border h-px" />
        <CheckboxFilterGroup
          label="Model family"
          options={familyOptions}
          selected={familyFilters}
          onToggle={(value) => setFamilyFilters((filters) => toggleValue(filters, value))}
        />
        <div className="bg-border h-px" />
        <CheckboxFilterGroup
          label="Status"
          options={[
            { value: "enabled", label: "Enabled" },
            { value: "disabled", label: "Disabled" },
            { value: "default", label: "Default" },
          ]}
          selected={statusFilters}
          onToggle={(value) => setStatusFilters((filters) => toggleValue(filters, value))}
        />
      </div>,
      document.body,
    );

  async function create(e: React.FormEvent) {
    e.preventDefault();
    if (!form.provider_id) {
      toast.error("Pick a provider");
      return;
    }
    const r = await adminFetch("/api/admin/models", {
      method: "POST",
      body: JSON.stringify(form),
    });
    if (!r.ok) {
      toast.error("Failed to create model");
      return;
    }
    setForm({
      provider_id: form.provider_id,
      model_id: "",
      display_name: "",
      is_default: false,
    });
    load();
  }

  async function patch(id: string, body: Record<string, unknown>) {
    const r = await adminFetch(`/api/admin/models/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    });
    if (!r.ok) toast.error("Update failed");
    else load();
  }

  async function del(id: string) {
    if (!(await confirm({ title: "Delete this model?" }))) return;
    const r = await adminFetch(`/api/admin/models/${id}`, {
      method: "DELETE",
    });
    if (!r.ok) toast.error("Delete failed");
    else load();
  }

  async function syncAll() {
    const eligible = providers.filter((p) => p.kind !== "azure_openai");
    if (eligible.length === 0) {
      toast.error("No providers, add one in the Providers tab first");
      return;
    }
    setSyncing(true);
    const t = toast.loading(`Syncing ${eligible.length} provider(s)…`);
    try {
      let ins = 0;
      let upd = 0;
      let failed = 0;
      const failures: string[] = [];
      for (const p of eligible) {
        const r = await fetch(`/api/admin/providers/${p.id}/sync-models`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Admin-Token": getAdminToken() || "",
          },
        });
        const data = await r.json().catch(() => ({}));
        if (r.ok) {
          ins += data.inserted || 0;
          upd += data.updated || 0;
        } else {
          failed++;
          failures.push(`${p.name}: ${data.error || r.status}`);
        }
      }
      if (failed === eligible.length) {
        toast.error(`All syncs failed, ${failures[0]}`, { id: t });
      } else {
        toast.success(
          `Synced, ${ins} new, ${upd} updated${failed ? `, ${failed} failed (${failures.join("; ")})` : ""}`,
          { id: t },
        );
      }
      load();
    } catch (e) {
      toast.error(`Sync failed, ${e instanceof Error ? e.message : "network error"}`, { id: t });
    } finally {
      setSyncing(false);
    }
  }

  async function saveAutoProfile(profile: string) {
    setAutoProfile(profile);
    const r = await adminFetch("/api/admin/settings", {
      method: "PUT",
      body: JSON.stringify({ key: "auto_profile", value: profile }),
    });
    if (r.ok) toast.success(`Auto mode profile: ${profile}`);
    else toast.error("Could not save auto profile");
  }

  useEffect(() => {
    adminFetch("/api/admin/settings")
      .then((r) => (r.ok ? r.json() : { settings: {} }))
      .then((d) => setAutoProfile(d.settings?.auto_profile || "balanced"))
      .catch(() => {});
  }, [refreshKey]);

  return (
    <div className="space-y-6">
      <div className="bg-muted/30 rounded-lg border p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold">Auto mode profile</h3>
            <p className="text-muted-foreground text-xs">
              When chat users pick “Auto”, the router intent selects the model from this profile
              (only enabled models are eligible).
            </p>
          </div>
          <div className="border-border bg-background/60 flex items-center gap-1 rounded-full border p-1">
            {(["balanced", "quality", "cost"] as const).map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => saveAutoProfile(p)}
                className={
                  "rounded-full px-3 py-1 text-xs font-medium capitalize transition-colors " +
                  (autoProfile === p
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted")
                }
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      </div>

      <AutoModeCandidatesEditor refreshKey={refreshKey} />

      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold">Models</h2>
          <p className="text-muted-foreground text-sm">
            Models attached to providers. Click <em>Sync from providers</em> to fetch the latest
            model list from each provider&apos;s API. Mark exactly one as default, chat users get
            this when they don&apos;t pick one.
          </p>
        </div>
        <Button onClick={syncAll} disabled={providers.length === 0 || syncing} className="shrink-0">
          <RefreshCw className={`mr-2 size-4 ${syncing ? "animate-spin" : ""}`} />
          {syncing ? "Syncing…" : "Sync from providers"}
        </Button>
      </div>

      <form
        onSubmit={create}
        className="bg-muted/30 grid grid-cols-1 gap-4 rounded-lg border p-6 md:grid-cols-2"
      >
        <div className="md:col-span-2">
          <h3 className="mb-2 font-semibold">Add model manually</h3>
        </div>
        <div className="flex flex-col gap-2">
          <Label>Provider</Label>
          <Select
            fullWidth
            placeholder="Select provider"
            value={form.provider_id}
            onValueChange={(v) => setForm({ ...form, provider_id: v })}
            options={providers.map((p) => ({
              value: p.id,
              label: `${p.name} (${p.kind})`,
            }))}
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label>Model ID</Label>
          <Input
            value={form.model_id}
            onChange={(e) => setForm({ ...form, model_id: e.target.value })}
            placeholder="gpt-4o-mini"
            required
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label>Display name</Label>
          <Input
            value={form.display_name}
            onChange={(e) => setForm({ ...form, display_name: e.target.value })}
            placeholder="GPT-4o mini"
            required
          />
        </div>
        <div className="mt-6 flex items-center gap-2">
          <Switch
            checked={form.is_default}
            onCheckedChange={(v) => setForm({ ...form, is_default: v })}
          />
          <Label>Mark as default</Label>
        </div>
        <div className="md:col-span-2">
          <Button type="submit">Add Model</Button>
        </div>
      </form>

      <div className="relative">
        <div
          aria-hidden="true"
          className="bg-muted pointer-events-none absolute top-px right-px z-20 h-11 w-1.5 rounded-tr-lg"
        />
        <div
          ref={searchRef}
          className={cn(
            "glass absolute top-1.5 right-3 z-30 flex h-8 items-center gap-2 overflow-hidden rounded-full border px-3 shadow-lg transition-[width,opacity] duration-300 ease-out",
            searchOpen
              ? "pointer-events-auto w-[calc(100%-1.5rem)] opacity-100 sm:w-[28rem]"
              : "pointer-events-none w-8 opacity-0",
          )}
        >
          <Search className="text-muted-foreground size-4 shrink-0" />
          <input
            ref={searchInputRef}
            aria-label="Search models"
            className="placeholder:text-muted-foreground min-w-0 flex-1 bg-transparent text-sm outline-none"
            placeholder={`Search ${models.length} models`}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => event.key === "Escape" && closeSearch()}
            tabIndex={searchOpen ? 0 : -1}
          />
          <span className="text-muted-foreground shrink-0 text-[11px]">
            {filteredModels.length}
          </span>
          <button
            type="button"
            className="text-muted-foreground hover:text-foreground shrink-0"
            title="Close search"
            tabIndex={searchOpen ? 0 : -1}
            onClick={closeSearch}
          >
            <X className="size-4" />
          </button>
        </div>

        <div className="bg-background/60 hover-scrollbar header-offset-scrollbar max-h-[36rem] overflow-auto overscroll-contain rounded-lg border [contain:paint]">
          <table className="w-full text-sm">
            <thead className="bg-muted sticky top-0 z-10 text-left">
              <tr>
                <th className="p-3">Display</th>
                <th className="p-3">Model ID</th>
                <th className="p-3">Family</th>
                <th className="p-3">Provider</th>
                <th className="p-3">Default</th>
                <th className="p-3">Enabled</th>
                <th className="w-24 px-3 py-1.5">
                  <div className="flex items-center justify-end gap-0.5">
                    <Button
                      type="button"
                      size="icon"
                      variant="ghost"
                      className={cn("size-8", searchOpen && "bg-background")}
                      aria-label="Search models"
                      data-model-search-trigger
                      onClick={() => (searchOpen ? closeSearch() : setSearchOpen(true))}
                    >
                      <Search className="size-4" />
                    </Button>
                    <div ref={filterRef}>
                      <Button
                        type="button"
                        size="icon"
                        variant="ghost"
                        className={cn("relative size-8", filtersOpen && "bg-background")}
                        aria-label="Filter models"
                        aria-haspopup="menu"
                        aria-expanded={filtersOpen}
                        onClick={() => setFiltersOpen((open) => !open)}
                      >
                        <ListFilter className="size-4" />
                        {activeFilterCount > 0 && (
                          <span className="bg-primary text-primary-foreground absolute -top-1 -right-1 flex size-4 items-center justify-center rounded-full text-[10px]">
                            {activeFilterCount}
                          </span>
                        )}
                      </Button>
                    </div>
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredModels.map((m) => (
                <tr key={m.id} className="border-t">
                  <td className="p-3 font-medium">{m.display_name}</td>
                  <td className="p-3 font-mono text-xs">{m.model_id}</td>
                  <td className="p-3">
                    <span className="bg-muted text-muted-foreground rounded px-2 py-0.5 text-xs">
                      {m.family}
                    </span>
                  </td>
                  <td className="p-3">
                    {m.provider_name}{" "}
                    <span className="text-muted-foreground text-xs">({m.provider_kind})</span>
                  </td>
                  <td className="p-3">
                    <Switch
                      checked={m.is_default}
                      onCheckedChange={(v) => patch(m.id, { is_default: v })}
                    />
                  </td>
                  <td className="p-3">
                    <Switch
                      checked={m.enabled}
                      onCheckedChange={(v) => patch(m.id, { enabled: v })}
                    />
                  </td>
                  <td className="p-3 text-right">
                    <DeleteButton onClick={() => del(m.id)} title="Delete model" />
                  </td>
                </tr>
              ))}
              {filteredModels.length === 0 && (
                <tr>
                  <td colSpan={7} className="text-muted-foreground p-3">
                    {models.length === 0
                      ? "No models yet, add one above or sync from a provider."
                      : "No models match the current filters."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      {filterMenu}
    </div>
  );
}
