"use client";

import { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { DeleteButton } from "@/components/ui/delete-button";
import { getAdminToken } from "../token";
import {
  Download,
  Search,
  Cpu,
  Loader2,
  CheckCircle2,
  PlayCircle,
  Plus,
} from "lucide-react";

interface CatalogModel {
  name: string;
  description: string;
  size_mb: number;
  context_length: number;
  parameters: string;
  tags: string[];
  downloaded: boolean;
  active: boolean;
}

interface SearchHit {
  repo_id: string;
  filename: string;
  downloads: number;
  likes: number;
  tags: string[];
  in_catalog: boolean;
}

interface ProgressEntry {
  progress: number;
  downloaded_mb: number;
  total_mb: number;
  status: string;
}

export default function LocalModelsPanel({
  onChanged,
}: {
  onChanged?: () => void;
}) {
  const confirm = useConfirm();
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [loaded, setLoaded] = useState<string | null>(null);
  const [progress, setProgress] = useState<Record<string, ProgressEntry>>({});
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const headers = useCallback(() => {
    const t = getAdminToken();
    return { "Content-Type": "application/json", "X-Admin-Token": t || "" };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/local/catalog", {
        headers: headers(),
      });
      if (!res.ok) throw new Error(`catalog ${res.status}`);
      const data = await res.json();
      setCatalog(data.models ?? []);
      setLoaded(data.loaded ?? null);
      const p = await fetch("/api/admin/local/progress", {
        headers: headers(),
      });
      if (p.ok) setProgress(await p.json());
    } catch (e) {
      setError((e as Error).message);
    }
  }, [headers]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, [refresh]);

  const doSearch = async () => {
    if (!query.trim()) return;
    setSearching(true);
    setError(null);
    try {
      const r = await fetch(
        `/api/admin/local/search?q=${encodeURIComponent(query)}`,
        { headers: headers() },
      );
      const data = await r.json();
      if (!r.ok) throw new Error(data?.error ?? "search failed");
      setHits(data.results ?? []);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSearching(false);
    }
  };

  const startDownload = async (
    name: string,
    repo_id?: string,
    filename?: string,
  ) => {
    setBusy(name);
    setError(null);
    try {
      const r = await fetch("/api/admin/local/download", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({ name, repo_id, filename }),
      });
      if (!r.ok) throw new Error(`download ${r.status}`);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const loadInto = async (name: string) => {
    setBusy(name);
    setError(null);
    try {
      const r = await fetch("/api/admin/local/load", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({ name }),
      });
      if (!r.ok) throw new Error(`load ${r.status}`);

      const provs = await fetch("/api/admin/providers", {
        headers: headers(),
      }).then((x) => x.json());
      const local = provs.find((p: { kind: string }) => p.kind === "local");
      if (local) {
        await fetch("/api/admin/models", {
          method: "POST",
          headers: headers(),
          body: JSON.stringify({
            provider_id: local.id,
            model_id: name,
            display_name: `${name} (self-hosted)`,
            is_default: false,
          }),
        });
      }

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
    try {
      await fetch(`/api/admin/local/models/${encodeURIComponent(name)}`, {
        method: "DELETE",
        headers: headers(),
      });
      await refresh();
      onChanged?.();
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Local Models</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Self-hosted GGUF models running inside the <code>ai</code> container.
          Search HuggingFace or pick from the curated catalog. Loading a model
          auto-registers it under the <em>Self-Hosted</em> provider so it
          appears in the chat dropdown.
        </p>
      </div>

      {error && (
        <div className="rounded-md border border-red-200 bg-destructive/10 px-4 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      <section className="rounded-xl border bg-background/60 p-5 shadow-sm">
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
          <div className="mt-4 max-h-96 overflow-auto rounded-md border">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 text-xs uppercase tracking-wider text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 text-left">Repo</th>
                  <th className="px-3 py-2 text-left">File</th>
                  <th className="px-3 py-2 text-right">Downloads</th>
                  <th className="px-3 py-2"></th>
                </tr>
              </thead>
              <tbody>
                {hits.map((h) => {
                  const name = h.repo_id.replace(/\//g, "_");
                  const dl = progress[name];
                  return (
                    <tr key={h.repo_id} className="border-t">
                      <td className="px-3 py-2 font-mono text-xs">
                        {h.repo_id}
                        {h.in_catalog && (
                          <span className="ml-2 rounded-sm bg-muted px-1 text-[10px] uppercase">
                            catalog
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">
                        {h.filename}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                        {h.downloads.toLocaleString()}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {dl ? (
                          <span className="text-xs text-muted-foreground">
                            {dl.status} {dl.progress}%
                          </span>
                        ) : (
                          <Button
                            size="sm"
                            variant="outline"
                            disabled={busy === name}
                            onClick={() =>
                              startDownload(name, h.repo_id, h.filename)
                            }
                          >
                            <Plus className="mr-1 size-3.5" />
                            Add
                          </Button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="rounded-xl border bg-background/60 p-5 shadow-sm">
        <h3 className="mb-3 flex items-center gap-2 font-semibold">
          <Cpu className="size-4" /> Catalog
        </h3>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {catalog.map((m) => {
            const dl = progress[m.name];
            return (
              <div
                key={m.name}
                className="flex flex-col gap-2 rounded-lg border bg-muted/50 p-4"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="font-medium">
                      {m.name}{" "}
                      {m.active && (
                        <span className="ml-1 rounded-sm bg-emerald-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-emerald-600 dark:text-emerald-400">
                          active
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {m.description}
                    </div>
                  </div>
                  <div className="text-right text-xs text-muted-foreground">
                    {m.parameters} · {m.size_mb}MB
                  </div>
                </div>
                <div className="flex flex-wrap gap-1">
                  {m.tags.map((t) => (
                    <span
                      key={t}
                      className="rounded-sm bg-background/60 px-1.5 py-0.5 text-[10px] text-muted-foreground"
                    >
                      {t}
                    </span>
                  ))}
                </div>

                {dl && dl.status !== "complete" ? (
                  <div className="flex flex-col gap-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">{dl.status}</span>
                      <span className="tabular-nums">
                        {dl.downloaded_mb}/{dl.total_mb}MB · {dl.progress}%
                      </span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full bg-emerald-500 transition-all"
                        style={{ width: `${dl.progress}%` }}
                      />
                    </div>
                  </div>
                ) : null}

                <div className="flex gap-2">
                  {!m.downloaded ? (
                    <Button
                      size="sm"
                      onClick={() => startDownload(m.name)}
                      disabled={!!dl || busy === m.name}
                    >
                      <Download className="mr-1 size-3.5" />
                      Download
                    </Button>
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
              </div>
            );
          })}
        </div>
      </section>

      <div className="text-xs text-muted-foreground">
        Currently loaded:{" "}
        <code>
          {loaded ?? "(none, load a model to enable self-hosted chat)"}
        </code>
      </div>
    </div>
  );
}
