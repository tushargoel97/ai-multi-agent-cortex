"use client";

import { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { DeleteButton } from "@/components/ui/delete-button";
import { getAdminToken } from "../token";
import ModelDetails from "./ModelDetails";
import {
  Download,
  Search,
  Cpu,
  HardDriveDownload,
  Loader2,
  CheckCircle2,
  PlayCircle,
  Plus,
  Pause,
  X,
  Sparkles,
  FolderOpen,
  Route,
  MemoryStick,
  ChevronRight,
} from "lucide-react";

interface CatalogModel {
  name: string;
  repo_id?: string;
  description: string;
  size_mb: number;
  context_length: number;
  native_context_length?: number;
  architecture?: string;
  tool_use?: boolean;
  parameters: string;
  tags: string[];
  downloaded: boolean;
  active: boolean;
}

interface SearchHit {
  repo_id: string;
  filename: string;
  size_mb: number;
  downloads: number;
  likes: number;
  last_modified: string;
  tags: string[];
  in_catalog: boolean;
}

type SortKey = "downloads" | "latest" | "size_asc" | "size_desc";

const BASE_SORTS: { key: SortKey; label: string }[] = [
  { key: "downloads", label: "Popular" },
  { key: "latest", label: "Latest" },
];

const sortHits = (hits: SearchHit[], key: SortKey): SearchHit[] =>
  [...hits].sort((a, b) => {
    switch (key) {
      case "latest":
        return (b.last_modified || "").localeCompare(a.last_modified || "");
      case "size_asc":
        return (a.size_mb || Infinity) - (b.size_mb || Infinity);
      case "size_desc":
        return (b.size_mb || 0) - (a.size_mb || 0);
      default:
        return (b.downloads || 0) - (a.downloads || 0);
    }
  });

interface ProgressEntry {
  progress: number;
  downloaded_mb: number;
  total_mb: number;
  status: string;
}

interface RegistryRow {
  id: string;
  model_id: string;
  provider_kind: string;
  enabled: boolean;
  description: string | null;
}

interface LocalFile {
  filename: string;
  size_mb: number;
}

interface Profile {
  description: string;
}

const slugify = (f: string) =>
  f
    .replace(/\.gguf$/i, "")
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");

const fmtCompact = (n: number) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : `${n}`;

const fmtSize = (mb: number) => (mb >= 1024 ? `${(mb / 1024).toFixed(1)}GB` : `${mb}MB`);

const splitRepo = (r: string): [string, string] => {
  const i = r.indexOf("/");
  return i === -1 ? ["", r] : [r.slice(0, i), r.slice(i + 1)];
};

const quantOf = (f: string) => f.match(/(?:IQ|Q)\d(?:_[A-Z0-9]+)*/i)?.[0] ?? "";

const isDownloading = (s?: string) => s === "starting" || s === "downloading";
const isActiveDownload = (s?: string) =>
  !!s && ["starting", "downloading", "pausing", "cancelling"].includes(s);

const checked = async (response: Response, label: string): Promise<Response> => {
  if (response.ok) return response;
  const data = await response.json().catch(() => null);
  throw new Error(data?.error ?? data?.detail ?? `${label} ${response.status}`);
};

const Chip = ({ children }: { children: React.ReactNode }) => (
  <span className="bg-background/60 text-muted-foreground rounded-sm px-1.5 py-0.5 text-[10px]">
    {children}
  </span>
);

export default function LocalModelsPanel({ onChanged }: { onChanged?: () => void }) {
  const confirm = useConfirm();
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [loaded, setLoaded] = useState<string | null>(null);
  const [memory, setMemory] = useState<{ total_mb?: number; available_mb?: number }>({});
  const [idleTtl, setIdleTtl] = useState<number>(0);
  const [progress, setProgress] = useState<Record<string, ProgressEntry>>({});
  const [registry, setRegistry] = useState<Record<string, RegistryRow>>({});
  const [files, setFiles] = useState<LocalFile[]>([]);
  const [edits, setEdits] = useState<Record<string, Profile>>({});
  const [imports, setImports] = useState<Record<string, { name: string; description: string }>>({});
  const [describing, setDescribing] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [expandedHit, setExpandedHit] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>("downloads");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const headers = useCallback(() => {
    const t = getAdminToken();
    return { "Content-Type": "application/json", "X-Admin-Token": t || "" };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/admin/local/catalog", { headers: headers() });
      if (!res.ok) throw new Error(`catalog ${res.status}`);
      const data = await res.json();
      setCatalog(data.models ?? []);
      setLoaded(data.loaded ?? null);
      setMemory(data.memory ?? {});
      setIdleTtl(data.idle?.idle_ttl_minutes ?? 0);
      const p = await fetch("/api/v1/admin/local/progress", { headers: headers() });
      if (p.ok) setProgress(await p.json());
      const lf = await fetch("/api/v1/admin/local/local-files", { headers: headers() });
      if (lf.ok) setFiles((await lf.json()).files ?? []);
      const reg = await fetch("/api/v1/admin/models", { headers: headers() });
      if (reg.ok) {
        const rows: RegistryRow[] = await reg.json();
        setRegistry(
          Object.fromEntries(
            rows.filter((r) => r.provider_kind === "local").map((r) => [r.model_id, r]),
          ),
        );
      }
    } catch (e) {
      setError((e as Error).message);
    }
  }, [headers]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const hasActiveDownload = Object.values(progress).some((entry) => isActiveDownload(entry.status));

  useEffect(() => {
    if (!hasActiveDownload) return;
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
  }, [hasActiveDownload, refresh]);

  const localProviderId = useCallback(async () => {
    const response = await checked(
      await fetch("/api/v1/admin/providers", { headers: headers() }),
      "providers",
    );
    const provs = await response.json();
    return provs.find((p: { kind: string }) => p.kind === "local")?.id ?? null;
  }, [headers]);

  const doSearch = async () => {
    if (!query.trim()) return;
    setSearching(true);
    setError(null);
    try {
      const r = await fetch(`/api/v1/admin/local/search?q=${encodeURIComponent(query)}`, {
        headers: headers(),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.error ?? "search failed");
      setHits(data.results ?? []);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSearching(false);
    }
  };

  const startDownload = async (name: string, repo_id?: string, filename?: string) => {
    setBusy(name);
    setError(null);
    try {
      await checked(
        await fetch("/api/v1/admin/local/download", {
          method: "POST",
          headers: headers(),
          body: JSON.stringify({ name, repo_id, filename }),
        }),
        "download",
      );
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const controlDownload = async (name: string, action: "pause" | "cancel") => {
    setError(null);
    const optimistic = action === "pause" ? "pausing" : "cancelling";
    setProgress((p) => (p[name] ? { ...p, [name]: { ...p[name], status: optimistic } } : p));
    try {
      await checked(
        await fetch(`/api/v1/admin/local/download/${action}`, {
          method: "POST",
          headers: headers(),
          body: JSON.stringify({ name }),
        }),
        action,
      );
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const registerModel = useCallback(
    async (model_id: string, profile?: Partial<Profile>) => {
      const pid = await localProviderId();
      if (!pid) throw new Error("No local model provider is configured");
      await checked(
        await fetch("/api/v1/admin/models", {
          method: "POST",
          headers: headers(),
          body: JSON.stringify({
            provider_id: pid,
            model_id,
            display_name: `${model_id} (self-hosted)`,
            is_default: false,
            description: profile?.description || undefined,
          }),
        }),
        "register model",
      );
    },
    [headers, localProviderId],
  );

  const loadInto = async (name: string) => {
    setBusy(name);
    setError(null);
    try {
      const r = await fetch("/api/v1/admin/local/load", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({ name }),
      });
      if (!r.ok) throw new Error(`load ${r.status}`);
      await registerModel(name);
      await refresh();
      onChanged?.();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const remove = async (name: string) => {
    if (!(await confirm({ title: `Delete model "${name}"?` }))) return;
    setBusy(name);
    setError(null);
    try {
      const deleted = await fetch(`/api/v1/admin/local/models/${encodeURIComponent(name)}`, {
        method: "DELETE",
        headers: headers(),
      });
      if (!deleted.ok) throw new Error(`delete ${deleted.status}`);
      const row = registry[name];
      if (row) {
        const disabled = await fetch(`/api/v1/admin/models/${row.id}`, {
          method: "PATCH",
          headers: headers(),
          body: JSON.stringify({ enabled: false }),
        });
        if (!disabled.ok) throw new Error(`disable registry entry ${disabled.status}`);
      }
      await refresh();
      onChanged?.();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const profileFor = (m: CatalogModel): Profile =>
    edits[m.name] ?? {
      description: registry[m.name]?.description ?? "",
    };

  const setProfile = (name: string, patch: Partial<Profile>, base: Profile) =>
    setEdits((e) => ({ ...e, [name]: { ...base, ...e[name], ...patch } }));

  const saveProfile = async (m: CatalogModel) => {
    const profile = profileFor(m);
    setBusy(m.name);
    setError(null);
    try {
      const row = registry[m.name];
      if (row) {
        await checked(
          await fetch(`/api/v1/admin/models/${row.id}`, {
            method: "PATCH",
            headers: headers(),
            body: JSON.stringify({
              description: profile.description.trim() || null,
            }),
          }),
          "save model profile",
        );
      } else {
        await registerModel(m.name, profile);
      }
      if (profile.description.trim()) {
        await checked(
          await fetch(`/api/v1/admin/local/models/${encodeURIComponent(m.name)}`, {
            method: "PATCH",
            headers: headers(),
            body: JSON.stringify({ description: profile.description.trim() }),
          }),
          "save catalog description",
        );
      }
      setEdits((e) => {
        const next = { ...e };
        delete next[m.name];
        return next;
      });
      await refresh();
      onChanged?.();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const autoDescribe = async (m: CatalogModel) => {
    if (!m.repo_id) return;
    setDescribing(m.name);
    setError(null);
    try {
      const r = await fetch("/api/v1/admin/local/describe", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({ repo_id: m.repo_id, name: m.name }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.error ?? `auto-describe ${r.status}`);
      if (data?.description) {
        setProfile(m.name, { description: data.description }, profileFor(m));
      } else {
        setError(
          "Auto-describe returned nothing (model card missing or no OpenAI key configured).",
        );
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setDescribing(null);
    }
  };

  const importFile = async (f: LocalFile) => {
    const form = imports[f.filename] ?? { name: slugify(f.filename), description: "" };
    if (!form.name.trim()) return;
    setBusy(f.filename);
    setError(null);
    try {
      const r = await fetch("/api/v1/admin/local/import-local", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({
          name: form.name.trim(),
          filename: f.filename,
          description: form.description.trim() || undefined,
        }),
      });
      if (!r.ok) throw new Error(`import ${r.status}`);
      await registerModel(form.name.trim(), {
        description: form.description.trim() || undefined,
      });
      setImports((x) => {
        const next = { ...x };
        delete next[f.filename];
        return next;
      });
      await refresh();
      onChanged?.();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const fitBadge = (m: CatalogModel) => {
    const avail = memory.available_mb;
    if (!avail || !m.size_mb || m.active) return null;
    const label = m.size_mb * 1.15 <= avail ? "fits" : m.size_mb <= avail ? "tight" : "too large";
    const cls =
      label === "fits"
        ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
        : label === "tight"
          ? "bg-amber-500/10 text-amber-600 dark:text-amber-400"
          : "bg-red-500/10 text-red-600 dark:text-red-400";
    return (
      <span className={`rounded-sm px-1.5 py-0.5 text-[10px] font-medium uppercase ${cls}`}>
        {label}
      </span>
    );
  };

  const installed = catalog
    .filter((m) => m.downloaded)
    .sort((a, b) => Number(b.active) - Number(a.active));
  const available = catalog.filter((m) => !m.downloaded);

  const memTotal = memory.total_mb ?? 0;
  const memAvail = memory.available_mb ?? 0;
  const activeModel = catalog.find((m) => m.active);
  const activeMb = loaded && activeModel ? Math.min(activeModel.size_mb, memTotal - memAvail) : 0;
  const otherMb = Math.max(0, memTotal - memAvail - activeMb);
  const memPct = (v: number) => `${memTotal ? ((v / memTotal) * 100).toFixed(1) : 0}%`;

  const renderDownloadControls = (name: string, dl: ProgressEntry) => {
    if (["paused", "error"].includes(dl.status))
      return (
        <div className="flex flex-col gap-1.5 text-xs">
          <span className="text-muted-foreground">
            {dl.status === "paused" ? "Paused" : "Stopped"} at {dl.progress}% · {dl.downloaded_mb}/
            {dl.total_mb}MB
          </span>
          <div className="flex gap-1.5">
            <Button size="sm" onClick={() => startDownload(name)}>
              <PlayCircle className="mr-1 size-3.5" /> Resume
            </Button>
            <Button size="sm" variant="ghost" onClick={() => controlDownload(name, "cancel")}>
              <X className="mr-1 size-3.5" /> Cancel
            </Button>
          </div>
        </div>
      );
    if (dl.status === "complete") return null;
    const active = isDownloading(dl.status);
    return (
      <div className="flex flex-col gap-1.5 text-xs">
        <div className="flex items-center justify-between gap-2">
          <span className="text-muted-foreground capitalize">{dl.status}</span>
          <span className="tabular-nums">
            {dl.downloaded_mb}/{dl.total_mb}MB · {dl.progress}%
          </span>
        </div>
        <div className="bg-muted h-1.5 overflow-hidden rounded-full">
          <div
            className="h-full bg-emerald-500 transition-all"
            style={{ width: `${dl.progress}%` }}
          />
        </div>
        {active && (
          <div className="flex gap-1.5">
            <Button size="sm" variant="ghost" onClick={() => controlDownload(name, "pause")}>
              <Pause className="mr-1 size-3.5" /> Pause
            </Button>
            <Button size="sm" variant="ghost" onClick={() => controlDownload(name, "cancel")}>
              <X className="mr-1 size-3.5" /> Cancel
            </Button>
          </div>
        )}
      </div>
    );
  };

  const renderCard = (m: CatalogModel) => {
    const dl = progress[m.name];
    const profile = profileFor(m);
    const dirty = m.name in edits;
    const routed = !!registry[m.name]?.description?.trim() && registry[m.name]?.enabled;
    const open = expanded === m.name;
    return (
      <div
        key={m.name}
        className={`bg-muted/50 flex flex-col gap-2 rounded-lg border p-4 ${
          m.active ? "ring-1 ring-emerald-500/30" : ""
        }`}
      >
        <div className="flex items-start justify-between gap-2">
          <button
            type="button"
            onClick={() => setExpanded(open ? null : m.name)}
            className="group min-w-0 flex-1 text-left"
          >
            <div className="flex flex-wrap items-center gap-1.5 font-medium">
              <ChevronRight
                className={`text-muted-foreground size-3.5 shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
              />
              <span className="group-hover:text-foreground break-all">{m.name}</span>
              {m.active && (
                <span className="rounded-sm bg-emerald-100 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-600 uppercase dark:bg-emerald-500/10 dark:text-emerald-400">
                  in memory
                </span>
              )}
              {routed && (
                <span className="inline-flex items-center gap-0.5 rounded-sm bg-sky-500/10 px-1.5 py-0.5 text-[10px] font-semibold text-sky-600 uppercase dark:text-sky-400">
                  <Route className="size-2.5" /> routing
                </span>
              )}
            </div>
            <div className="text-muted-foreground ml-5 text-xs">{m.description}</div>
          </button>
          <div className="text-muted-foreground flex shrink-0 flex-col items-end gap-1 text-right text-xs">
            <span className="tabular-nums">
              {m.parameters ? `${m.parameters} · ` : ""}
              {fmtSize(m.size_mb)}
            </span>
            {fitBadge(m)}
          </div>
        </div>

        <div className="ml-5 flex flex-wrap gap-1">
          {(m.tags ?? []).map((t) => (
            <Chip key={t}>{t}</Chip>
          ))}
          {m.architecture && <Chip>{m.architecture}</Chip>}
          {m.tool_use && <Chip>tool template</Chip>}
          {m.native_context_length ? (
            <Chip>{Math.round(m.native_context_length / 1024)}k ctx</Chip>
          ) : null}
        </div>

        {dl && dl.status !== "complete" ? renderDownloadControls(m.name, dl) : null}

        {m.downloaded && (
          <div className="flex flex-col gap-1.5">
            <textarea
              className="bg-background/60 placeholder:text-muted-foreground/60 min-h-14 w-full resize-y rounded-md border px-2.5 py-1.5 text-xs transition-colors outline-none focus:border-sky-400/50"
              placeholder="Capabilities: what should the router send to this model?"
              value={profile.description}
              onChange={(e) => setProfile(m.name, { description: e.target.value }, profile)}
            />
            <div className="flex justify-end gap-1.5">
              {m.repo_id ? (
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={describing === m.name}
                  onClick={() => autoDescribe(m)}
                  title="Generate from the HuggingFace model card"
                >
                  {describing === m.name ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : (
                    <Sparkles className="size-3.5" />
                  )}
                  Auto-describe
                </Button>
              ) : null}
              <Button
                size="sm"
                variant="outline"
                disabled={!dirty || busy === m.name}
                onClick={() => saveProfile(m)}
              >
                Save
              </Button>
            </div>
          </div>
        )}

        <div className="flex gap-2">
          {!m.downloaded ? (
            !dl || dl.status === "complete" ? (
              <Button size="sm" onClick={() => startDownload(m.name)} disabled={busy === m.name}>
                <Download className="mr-1 size-3.5" />
                Download
              </Button>
            ) : null
          ) : (
            <>
              <Button
                size="sm"
                variant={m.active ? "outline" : "default"}
                disabled={busy === m.name}
                onClick={() => loadInto(m.name)}
              >
                {m.active ? (
                  <CheckCircle2 className="mr-1 size-3.5" />
                ) : (
                  <PlayCircle className="mr-1 size-3.5" />
                )}
                {m.active ? "Loaded" : "Load & enable"}
              </Button>
              <DeleteButton
                disabled={busy === m.name}
                onClick={() => remove(m.name)}
                title="Delete model"
              />
            </>
          )}
        </div>

        {open && <ModelDetails repoId={m.repo_id} local={m} headers={headers} />}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Local Models</h2>
        <p className="text-muted-foreground mt-1 text-sm">
          Self-hosted GGUF models running inside the <code>ai</code> container. Describe what a
          model is capable of and the router automatically sends in-domain queries to it (and lets
          agents consult it) at zero API cost. Idle models auto-unload and reload on demand.
        </p>
      </div>

      {error && (
        <div className="bg-destructive/10 text-destructive rounded-md border border-red-200 px-4 py-2 text-sm">
          {error}
        </div>
      )}

      {memTotal > 0 && (
        <section className="bg-background/60 rounded-xl border p-4 shadow-sm">
          <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 text-xs">
            <span className="flex items-center gap-1.5 font-medium">
              <MemoryStick className="size-3.5" /> Host memory
            </span>
            <span className="text-muted-foreground tabular-nums">
              {(memAvail / 1024).toFixed(1)}GB free of {(memTotal / 1024).toFixed(0)}GB
              {loaded && activeMb > 0 && (
                <>
                  {" "}
                  · <span className="text-foreground/80">{loaded}</span> holds ~
                  {(activeMb / 1024).toFixed(1)}
                  GB{idleTtl > 0 && <> · unloads after {idleTtl}m idle</>}
                </>
              )}
            </span>
          </div>
          <div className="bg-muted mt-2.5 flex h-2 overflow-hidden rounded-full">
            <div
              className="bg-foreground/25 transition-all duration-700"
              style={{ width: memPct(otherMb) }}
            />
            <div
              className="bg-emerald-500/80 transition-all duration-700"
              style={{ width: memPct(activeMb) }}
            />
          </div>
          <div className="text-muted-foreground/80 mt-1.5 flex gap-4 text-[10px]">
            <span className="flex items-center gap-1">
              <i className="bg-foreground/25 inline-block size-1.5 rounded-full" /> system
            </span>
            <span className="flex items-center gap-1">
              <i className="inline-block size-1.5 rounded-full bg-emerald-500/80" /> loaded model
            </span>
            <span className="flex items-center gap-1">
              <i className="bg-muted inline-block size-1.5 rounded-full border" /> free
            </span>
          </div>
        </section>
      )}

      <section className="bg-background/60 rounded-xl border p-5 shadow-sm">
        <h3 className="mb-3 flex items-center gap-2 font-semibold">
          <Search className="size-4" /> Search HuggingFace
        </h3>
        <div className="flex gap-2">
          <Input
            placeholder="e.g. qwen2.5, llama-3.3, gemma-3"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && doSearch()}
          />
          <Button onClick={doSearch} disabled={searching}>
            {searching ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Search className="size-4" />
            )}
            Search
          </Button>
        </div>
        {hits.length > 0 && (
          <div className="mt-3 flex items-center gap-1.5">
            <span className="text-muted-foreground mr-1 text-xs">Sort</span>
            {BASE_SORTS.map((s) => (
              <button
                key={s.key}
                onClick={() => setSortKey(s.key)}
                className={`rounded-full px-2.5 py-1 text-xs transition-colors ${
                  sortKey === s.key
                    ? "bg-muted text-foreground font-medium"
                    : "text-muted-foreground hover:bg-muted/50"
                }`}
              >
                {s.label}
              </button>
            ))}
            {(() => {
              const isSize = sortKey === "size_asc" || sortKey === "size_desc";
              return (
                <button
                  onClick={() => setSortKey(sortKey === "size_desc" ? "size_asc" : "size_desc")}
                  title={isSize ? "Toggle ascending / descending" : "Sort by size"}
                  className={`rounded-full px-2.5 py-1 text-xs transition-colors ${
                    isSize
                      ? "bg-muted text-foreground font-medium"
                      : "text-muted-foreground hover:bg-muted/50"
                  }`}
                >
                  Size {sortKey === "size_asc" ? "↑" : sortKey === "size_desc" ? "↓" : "↕"}
                </button>
              );
            })()}
          </div>
        )}
        {hits.length > 0 && (
          <div className="hover-scrollbar mt-3 max-h-[32rem] overflow-auto overscroll-contain rounded-lg border [contain:paint]">
            {sortHits(hits, sortKey).map((h) => {
              const name = h.repo_id.replace(/\//g, "_");
              const dl = progress[name];
              const [owner, repo] = splitRepo(h.repo_id);
              const quant = quantOf(h.filename);
              const open = expandedHit === h.repo_id;
              return (
                <div key={h.repo_id} className="border-t first:border-t-0">
                  <div className="hover:bg-muted/40 flex items-center gap-3 px-3.5 py-2.5 transition-colors">
                    <button
                      type="button"
                      onClick={() => setExpandedHit(open ? null : h.repo_id)}
                      className="group flex min-w-0 flex-1 items-start gap-1.5 text-left"
                    >
                      <ChevronRight
                        className={`text-muted-foreground mt-0.5 size-3.5 shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
                      />
                      <span className="min-w-0">
                        <span className="flex flex-wrap items-baseline gap-x-1">
                          {owner && (
                            <span className="text-muted-foreground text-xs">{owner} /</span>
                          )}
                          <span className="group-hover:text-foreground text-sm font-medium break-all">
                            {repo}
                          </span>
                          {h.in_catalog && (
                            <span className="bg-muted text-muted-foreground ml-1 rounded-sm px-1 py-px text-[10px] uppercase">
                              in library
                            </span>
                          )}
                        </span>
                        <span className="text-muted-foreground/80 mt-0.5 flex min-w-0 items-center gap-1.5 font-mono text-[10px]">
                          {quant && (
                            <span className="bg-muted shrink-0 rounded-sm px-1 py-px">{quant}</span>
                          )}
                          <span className="truncate" title={h.filename}>
                            {h.filename}
                          </span>
                        </span>
                      </span>
                    </button>
                    {h.size_mb ? (
                      <span
                        className="text-foreground/70 shrink-0 text-xs tabular-nums"
                        title="Download size"
                      >
                        {fmtSize(h.size_mb)}
                      </span>
                    ) : null}
                    <span
                      className="text-muted-foreground shrink-0 text-xs tabular-nums"
                      title={`${h.downloads.toLocaleString()} downloads`}
                    >
                      {fmtCompact(h.downloads)} ↓
                    </span>
                    {!dl || dl.status === "complete" ? (
                      <Button
                        size="sm"
                        variant="outline"
                        className="shrink-0"
                        disabled={busy === name}
                        onClick={() => startDownload(name, h.repo_id, h.filename)}
                      >
                        <Plus className="mr-1 size-3.5" />
                        Add
                      </Button>
                    ) : null}
                  </div>
                  {dl && dl.status !== "complete" && (
                    <div className="px-3.5 pb-2.5">{renderDownloadControls(name, dl)}</div>
                  )}
                  {open && (
                    <div className="px-3.5 pb-3">
                      <ModelDetails repoId={h.repo_id} local={{ tags: h.tags }} headers={headers} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>

      {files.length > 0 && (
        <section className="bg-background/60 rounded-xl border p-5 shadow-sm">
          <h3 className="mb-1 flex items-center gap-2 font-semibold">
            <FolderOpen className="size-4" /> Found in models folder
          </h3>
          <p className="text-muted-foreground mb-3 text-xs">
            GGUF files already on disk (drop yours into <code>./models</code>). Import to register,
            load, and make them describable.
          </p>
          <div className="flex flex-col gap-2">
            {files.map((f) => {
              const form = imports[f.filename] ?? { name: slugify(f.filename), description: "" };
              return (
                <div
                  key={f.filename}
                  className="bg-muted/50 flex flex-col gap-2 rounded-lg border p-3"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="min-w-0 truncate font-mono text-xs" title={f.filename}>
                      {f.filename}
                    </span>
                    <span className="text-muted-foreground shrink-0 text-xs tabular-nums">
                      {(f.size_mb / 1024).toFixed(1)}GB
                    </span>
                  </div>
                  <div className="flex flex-col gap-2 md:flex-row">
                    <Input
                      className="md:max-w-56"
                      placeholder="model name"
                      value={form.name}
                      onChange={(e) =>
                        setImports((x) => ({
                          ...x,
                          [f.filename]: { ...form, name: e.target.value },
                        }))
                      }
                    />
                    <Input
                      placeholder="What is this model capable of? (drives routing)"
                      value={form.description}
                      onChange={(e) =>
                        setImports((x) => ({
                          ...x,
                          [f.filename]: { ...form, description: e.target.value },
                        }))
                      }
                    />
                    <Button size="sm" disabled={busy === f.filename} onClick={() => importFile(f)}>
                      {busy === f.filename ? (
                        <Loader2 className="mr-1 size-3.5 animate-spin" />
                      ) : (
                        <Plus className="mr-1 size-3.5" />
                      )}
                      Import & load
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      <section className="bg-background/60 rounded-xl border p-5 shadow-sm">
        <h3 className="mb-1 flex items-center gap-2 font-semibold">
          <HardDriveDownload className="size-4" /> Your models
          <span className="text-muted-foreground text-xs font-normal">
            · {installed.length} on disk
          </span>
        </h3>
        <p className="text-muted-foreground mb-3 text-xs">
          Downloaded models. Describe each one to route queries to it; load makes it the active
          in-memory model. Click a model for full details.
        </p>
        {installed.length === 0 ? (
          <p className="text-muted-foreground text-sm">No models downloaded yet.</p>
        ) : (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">{installed.map(renderCard)}</div>
        )}
      </section>

      <section className="bg-background/60 rounded-xl border p-5 shadow-sm">
        <h3 className="mb-1 flex items-center gap-2 font-semibold">
          <Cpu className="size-4" /> Browse catalog
          <span className="text-muted-foreground text-xs font-normal">
            · {available.length} available
          </span>
        </h3>
        <p className="text-muted-foreground mb-3 text-xs">
          Curated GGUF models plus anything added from search. Click a model for full details.
        </p>
        <div className="hover-scrollbar grid max-h-[48rem] auto-rows-max grid-cols-1 gap-3 overflow-y-auto overscroll-contain pr-1 [contain:paint] md:grid-cols-2">
          {available.map(renderCard)}
        </div>
      </section>

      <div className="text-muted-foreground text-xs">
        Currently loaded: <code>{loaded ?? "(none, described models load on demand)"}</code>
      </div>
    </div>
  );
}
