"use client";

import { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { getAdminToken } from "../token";
import {
  Database,
  FlaskConical,
  Loader2,
  PackageCheck,
  PlayCircle,
  Square,
  CheckCircle2,
  AlertTriangle,
  FileText,
  FileImage,
  Table2,
  Link2,
  MessageSquarePlus,
  Trash2,
  Upload,
  Lightbulb,
  Globe,
  RefreshCw,
  Zap,
} from "lucide-react";

interface LossPoint {
  iter: number;
  train_loss: number;
}

interface TrainerStatus {
  phase: string;
  iter?: number;
  total_iters?: number;
  train_loss?: number | null;
  val_loss?: number | null;
  history?: LossPoint[];
  log_tail?: string[];
  error?: string;
  gguf_filename?: string;
  output_name?: string;
  sources_total?: number;
  sources_done?: number;
  chunks_total?: number;
  chunks_done?: number;
  pairs_generated?: number;
  train_count?: number;
  valid_count?: number;
  gaps_total?: number;
  gaps_done?: number;
  products_learned?: number;
  scrape_total?: number;
  scrape_done?: number;
  scrape_current?: string;
  scrape_saved?: string[];
  scrape_errors?: string[];
  scrape_outcomes?: {
    url: string;
    status: string;
    detail?: string;
    products?: string[];
  }[];
}

interface DatasetSplit {
  exists: boolean;
  count: number;
}

interface GapEntry {
  id: string;
  question: string;
  reason: string;
  status: string;
  researched_summary?: string | null;
  created_at: string;
}

interface SourceEntry {
  id: string;
  type: "pdf" | "excel" | "url" | "prompt" | "image";
  name: string;
  url?: string;
  size_kb?: number;
}

const SOURCE_ICONS = {
  pdf: FileText,
  excel: Table2,
  url: Link2,
  prompt: MessageSquarePlus,
  image: FileImage,
} as const;

// Per-URL outcome badge colors for the intelligent scrape agent.
const OUTCOME_STYLES: Record<string, string> = {
  extracted: "bg-emerald-500/15 text-emerald-600",
  index: "bg-sky-500/15 text-sky-600",
  blocked: "bg-amber-500/15 text-amber-600",
  skipped: "bg-amber-500/15 text-amber-600",
  error: "bg-rose-500/15 text-rose-600",
  empty: "bg-muted text-muted-foreground",
};

const BASE_MODELS = [
  {
    id: "unsloth/gemma-3-1b-it",
    label: "Gemma 3 1B — recommended (~2 GB, fast train + serve, no HF login)",
    output: "finetuned-gemma3-1b-hardware",
  },
  {
    id: "google/gemma-4-e2b-it",
    label: "Gemma 4 E2B — highest quality, 9.5 GB, slow on CPU",
    output: "finetuned-gemma4-e2b-hardware",
  },
];

const DEFAULT_MODEL_NAME = BASE_MODELS[0].output;
const BUSY_PHASES = [
  "training",
  "fusing",
  "converting",
  "extracting",
  "generating_dataset",
  "researching",
  "scraping",
];

function fmtParams(n?: number | null): string | null {
  if (!n || n <= 0) return null;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${Math.round(n / 1e6)}M`;
  return `${n}`;
}

function fmtBytes(n?: number | null): string | null {
  if (!n || n <= 0) return null;
  const gb = n / 1e9;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${Math.round(n / 1e6)} MB`;
}

function LossSparkline({ history }: { history: LossPoint[] }) {
  if (history.length < 2) return null;
  const w = 240;
  const h = 48;
  const losses = history.map((p) => p.train_loss);
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const span = max - min || 1;
  const points = history
    .map((p, i) => {
      const x = (i / (history.length - 1)) * w;
      const y = h - ((p.train_loss - min) / span) * (h - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      width={w}
      height={h}
      className="rounded border bg-muted/50"
      aria-label="training loss"
    >
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        className="text-foreground"
      />
    </svg>
  );
}

export default function FinetunePanel({
  onChanged,
}: {
  onChanged?: () => void;
}) {
  const [trainerUp, setTrainerUp] = useState<boolean | null>(null);
  const [status, setStatus] = useState<TrainerStatus>({ phase: "idle" });
  const [dataset, setDataset] = useState<
    Record<string, DatasetSplit> & {
      custom_pairs?: number;
      sources_count?: number;
      adapters_exist?: boolean;
    }
  >({});
  const [sources, setSources] = useState<SourceEntry[]>([]);
  const [gaps, setGaps] = useState<GapEntry[]>([]);
  const [urlInput, setUrlInput] = useState("");
  const [promptInput, setPromptInput] = useState("");
  const [includeBuiltin, setIncludeBuiltin] = useState(true);
  const [useSources, setUseSources] = useState(true);
  const [baseModel, setBaseModel] = useState(BASE_MODELS[0].id);
  const [iters, setIters] = useState(600);
  const [batchSize, setBatchSize] = useState(4);
  const [modelName, setModelName] = useState(DEFAULT_MODEL_NAME);
  const [busy, setBusy] = useState<string | null>(null);
  const [registered, setRegistered] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hfQuery, setHfQuery] = useState("");
  const [hfResults, setHfResults] = useState<
    {
      id: string;
      downloads: number;
      likes: number;
      gated: boolean;
      params?: number | null;
      size_bytes?: number | null;
    }[]
  >([]);
  const [hfSearching, setHfSearching] = useState(false);
  const [showHf, setShowHf] = useState(false);
  const [finetuned, setFinetuned] = useState<
    { id: string; model_id: string; display_name: string }[]
  >([]);

  const headers = useCallback(() => {
    const t = getAdminToken();
    return { "Content-Type": "application/json", "X-Admin-Token": t || "" };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/trainer/progress", {
        headers: headers(),
      });
      if (!res.ok) throw new Error(`progress ${res.status}`);
      setStatus(await res.json());
      setTrainerUp(true);
      const d = await fetch("/api/admin/trainer/dataset/status", {
        headers: headers(),
      });
      if (d.ok) setDataset(await d.json());
    } catch {
      setTrainerUp(false);
    }
  }, [headers]);

  const refreshSources = useCallback(async () => {
    try {
      const r = await fetch("/api/admin/trainer/sources", { headers: headers() });
      if (r.ok) setSources((await r.json()).sources ?? []);
    } catch {
      /* trainer down — handled by refresh() */
    }
  }, [headers]);

  const refreshGaps = useCallback(async () => {
    try {
      const r = await fetch("/api/admin/gaps", { headers: headers() });
      if (r.ok) setGaps((await r.json()).gaps ?? []);
    } catch {
      /* db down — non-fatal */
    }
  }, [headers]);

  useEffect(() => {
    refresh();
    refreshSources();
    refreshGaps();
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, [refresh, refreshSources, refreshGaps]);

  const post = useCallback(
    async (path: string, body?: object) => {
      const r = await fetch(`/api/admin/trainer/${path}`, {
        method: "POST",
        headers: headers(),
        body: body ? JSON.stringify(body) : undefined,
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data?.detail ?? data?.error ?? `${path} ${r.status}`);
      return data;
    },
    [headers],
  );

  // Dynamic spec import: every URL / uploaded document in the sources list
  // is dispatched by the trainer (AMD DB parser, Intel-chart parser, generic
  // LLM distillation; TechPowerUp only if it doesn't 403).
  const importSpecs = async () => {
    setBusy("scrape");
    setError(null);
    try {
      const items = sources.map((s) => (s.type === "url" ? s.url! : s.id));
      if (items.length === 0) {
        throw new Error("Add a URL or upload a document first.");
      }
      await post("scrape", { sources: items, max_products: 30 });
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const generateDataset = async () => {
    setBusy("dataset");
    setError(null);
    try {
      await post("dataset/generate", {
        include_builtin: includeBuiltin,
        use_sources: useSources,
        max_pairs: 500,
      });
      // Source-backed generation runs as a background job — progress arrives
      // via the polling; the built-in path returns counts instantly.
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const uploadFile = async (file: File) => {
    setBusy("sources");
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const t = getAdminToken();
      // No Content-Type header — the browser sets the multipart boundary.
      const r = await fetch("/api/admin/trainer/sources/upload", {
        method: "POST",
        headers: { "X-Admin-Token": t || "" },
        body: form,
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data?.detail ?? data?.error ?? `upload ${r.status}`);
      await refreshSources();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const addUrl = async () => {
    if (!urlInput.trim()) return;
    setBusy("sources");
    setError(null);
    try {
      await post("sources/url", { url: urlInput.trim() });
      setUrlInput("");
      await refreshSources();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const addPrompt = async () => {
    if (!promptInput.trim()) return;
    setBusy("sources");
    setError(null);
    try {
      await post("sources/prompt", { text: promptInput.trim() });
      setPromptInput("");
      await refreshSources();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const deleteSource = async (id: string) => {
    setError(null);
    try {
      const r = await fetch(`/api/admin/trainer/sources/${id}`, {
        method: "DELETE",
        headers: headers(),
      });
      if (!r.ok) throw new Error(`delete ${r.status}`);
      await refreshSources();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const researchGaps = async () => {
    const fresh = gaps.filter((g) => g.status === "new");
    if (fresh.length === 0) return;
    setBusy("research");
    setError(null);
    try {
      await post("gaps/research", {
        gaps: fresh.map((g) => ({ id: g.id, question: g.question })),
      });
      // Poll until the research job settles, then persist statuses.
      for (;;) {
        await new Promise((r) => setTimeout(r, 2500));
        const s: TrainerStatus & {
          research_results?: { id: string; status: string; summary: string }[];
        } = await fetch("/api/admin/trainer/progress", { headers: headers() }).then(
          (x) => x.json(),
        );
        setStatus(s);
        if (s.phase === "research_done") {
          for (const r of s.research_results ?? []) {
            await fetch("/api/admin/gaps", {
              method: "PATCH",
              headers: headers(),
              body: JSON.stringify({
                id: r.id,
                status: r.status === "researched" ? "researched" : "new",
                researched_summary: r.summary,
              }),
            });
          }
          break;
        }
        if (s.phase === "error") throw new Error(s.error ?? "research failed");
      }
      await refreshGaps();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const dismissGap = async (id: string) => {
    await fetch("/api/admin/gaps", {
      method: "PATCH",
      headers: headers(),
      body: JSON.stringify({ id, status: "dismissed" }),
    });
    await refreshGaps();
  };

  const startTraining = async () => {
    setBusy("train");
    setError(null);
    setRegistered(false);
    try {
      await post("train", { iters, batch_size: batchSize, base_model: baseModel });
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  // Quick top-up: warm-start from the current adapters (resume=true) and train
  // fewer iters — teaches newly-added facts without a full retrain. The base
  // is taken from the adapters' base_model.txt on the trainer side.
  const startTopUp = async () => {
    setBusy("train");
    setError(null);
    setRegistered(false);
    try {
      await post("train", {
        iters,
        batch_size: batchSize,
        base_model: baseModel,
        resume: true,
      });
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const stopTraining = async () => {
    setError(null);
    try {
      await post("train/stop");
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const convertAndRegister = async () => {
    setBusy("convert");
    setError(null);
    setRegistered(false);
    const name = modelName.startsWith("finetuned-")
      ? modelName
      : `finetuned-${modelName}`;
    try {
      await post("convert", { output_name: name });
      // Wait for fuse + GGUF conversion to finish (can take a few minutes).
      let filename = "";
      for (;;) {
        await new Promise((r) => setTimeout(r, 2000));
        const s: TrainerStatus = await fetch("/api/admin/trainer/progress", {
          headers: headers(),
        }).then((x) => x.json());
        setStatus(s);
        if (s.phase === "done") {
          filename = s.gguf_filename ?? `${name}.gguf`;
          break;
        }
        if (s.phase === "error") throw new Error(s.error ?? "conversion failed");
      }

      // Import + load the GGUF in the ai service.
      const imp = await fetch("/api/admin/local/import-local", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({
          name,
          filename,
          description: "Self-trained hardware specialist (MLX LoRA fine-tune)",
        }),
      });
      if (!imp.ok) {
        const data = await imp.json().catch(() => ({}));
        throw new Error(data?.detail ?? data?.error ?? `import ${imp.status}`);
      }

      // Register in the model registry so it appears in the chat dropdown.
      // The `finetuned-` model_id prefix is how the cortex specialist agent
      // discovers this model — do not strip it.
      const provs = await fetch("/api/admin/providers", {
        headers: headers(),
      }).then((x) => x.json());
      const local = provs.find((p: { kind: string }) => p.kind === "local");
      if (!local) throw new Error("No local provider registered");
      await fetch("/api/admin/models", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({
          provider_id: local.id,
          model_id: name,
          display_name: "Hardware Specialist (fine-tuned)",
          is_default: false,
        }),
      });

      setRegistered(true);
      onChanged?.();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const searchHf = async () => {
    const q = hfQuery.trim();
    if (!q) return;
    setHfSearching(true);
    setError(null);
    try {
      const r = await fetch(
        `/api/admin/trainer/hf/search?q=${encodeURIComponent(q)}`,
        { headers: headers() },
      );
      const data = await r.json();
      if (!r.ok) throw new Error(data?.detail ?? data?.error ?? `search ${r.status}`);
      setHfResults(data.results ?? []);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setHfSearching(false);
    }
  };

  const pickBase = (repoId: string) => {
    setBaseModel(repoId);
    const tail = repoId.split("/").pop() ?? repoId;
    const slug = tail
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
    setModelName(`finetuned-${slug}-hardware`);
    setShowHf(false);
  };

  const refreshFinetuned = useCallback(async () => {
    try {
      const r = await fetch("/api/admin/models", { headers: headers() });
      const rows = await r.json();
      if (Array.isArray(rows)) {
        setFinetuned(
          rows
            .filter((m: { model_id?: string }) =>
              m.model_id?.startsWith("finetuned-"),
            )
            .map((m: { id: string; model_id: string; display_name?: string }) => ({
              id: m.id,
              model_id: m.model_id,
              display_name: m.display_name || m.model_id,
            })),
        );
      }
    } catch {
      /* registry unavailable — ignore */
    }
  }, [headers]);

  useEffect(() => {
    void refreshFinetuned();
  }, [refreshFinetuned, registered]);

  const deleteFinetuned = async (m: {
    id: string;
    model_id: string;
    display_name: string;
  }) => {
    if (
      !window.confirm(
        `Delete "${m.display_name}" (${m.model_id})?\n\nRemoves it from the model registry and deletes the .gguf file.`,
      )
    )
      return;
    setBusy(`del-${m.id}`);
    setError(null);
    try {
      const r = await fetch(`/api/admin/models/${m.id}`, {
        method: "DELETE",
        headers: headers(),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(d?.detail ?? d?.error ?? `delete ${r.status}`);
      }
      // Best-effort GGUF removal from the ai service.
      await fetch(`/api/admin/local/models/${encodeURIComponent(m.model_id)}`, {
        method: "DELETE",
        headers: headers(),
      }).catch(() => {});
      await refreshFinetuned();
      onChanged?.();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const clearArtifacts = async () => {
    if (
      !window.confirm(
        "Clear the LoRA adapters + fused working files?\n\nThey're tied to the current base model and can't be reused for a different one. The next full training recreates them.",
      )
    )
      return;
    setBusy("clear-artifacts");
    setError(null);
    try {
      const r = await fetch("/api/admin/trainer/artifacts", {
        method: "DELETE",
        headers: headers(),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(d?.detail ?? `clear ${r.status}`);
      }
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const phase = status.phase ?? "idle";
  const training = phase === "training";
  const converting = phase === "fusing" || phase === "converting";
  const generating = phase === "extracting" || phase === "generating_dataset";
  const jobRunning = BUSY_PHASES.includes(phase) || busy !== null;
  const trainPct =
    training && status.total_iters
      ? Math.round(((status.iter ?? 0) / status.total_iters) * 100)
      : phase === "trained"
        ? 100
        : 0;
  const trainCount = dataset.train?.count ?? 0;
  const hasDataset = (dataset.train?.exists && dataset.valid?.exists) ?? false;
  const hasAdapters = !!dataset.adapters_exist;

  if (trainerUp === false) {
    return (
      <div className="space-y-4">
        <div className="flex items-start gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 text-sm text-amber-700 dark:text-amber-300">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <div>
            <p className="font-medium">Trainer service is not reachable.</p>
            <p className="mt-1">
              The fine-tuning service runs on the host (MLX needs the Apple
              Silicon GPU — it can’t run inside Docker). Start it with:
            </p>
            <pre className="mt-2 rounded bg-amber-500/15 p-2 text-xs">
              cd trainer && uv run uvicorn app.main:app --host 0.0.0.0 --port 8200
            </pre>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* 1 — Dataset */}
      <section className="rounded-lg border p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="size-4 text-muted-foreground" />
            <h2 className="font-medium">1 · Dataset</h2>
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={generateDataset}
            disabled={jobRunning}
          >
            {busy === "dataset" ? (
              <Loader2 className="mr-1 size-4 animate-spin" />
            ) : null}
            Generate dataset
          </Button>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          Builds the chat-format Q&amp;A training set. Combine the built-in
          hardware spec sheets with your own sources — PDFs, Excel files,
          website links, or pasted text — which are turned into Q&amp;A pairs
          by an LLM.
        </p>

        {/* Training sources */}
        <div className="mt-4 rounded-md border border-dashed p-3">
          <p className="text-sm font-medium text-foreground">Training sources</p>
          {sources.length > 0 ? (
            <ul className="mt-2 space-y-1">
              {sources.map((s) => {
                const Icon = SOURCE_ICONS[s.type] ?? FileText;
                return (
                  <li
                    key={s.id}
                    className="flex items-center gap-2 text-sm text-muted-foreground"
                  >
                    <Icon className="size-4 shrink-0 text-muted-foreground/70" />
                    <span className="min-w-0 flex-1 truncate" title={s.name}>
                      {s.name}
                    </span>
                    {s.size_kb != null && (
                      <span className="text-xs text-muted-foreground/70">
                        {s.size_kb} KB
                      </span>
                    )}
                    <button
                      onClick={() => deleteSource(s.id)}
                      className="text-muted-foreground/70 transition hover:text-destructive"
                      title="Remove source"
                    >
                      <Trash2 className="size-4" />
                    </button>
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="mt-1 text-sm text-muted-foreground/70">
              None yet — the built-in hardware dataset is used on its own.
            </p>
          )}

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <label className="inline-flex cursor-pointer items-center gap-1 rounded-md border px-3 py-1.5 text-sm text-muted-foreground transition hover:bg-muted/50">
              <Upload className="size-4" />
              Upload PDF / Excel / Image
              <input
                type="file"
                accept=".pdf,.xlsx,.xls,.png,.jpg,.jpeg,.webp"
                className="hidden"
                disabled={jobRunning}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) uploadFile(f);
                  e.target.value = "";
                }}
              />
            </label>
            <div className="flex min-w-64 flex-1 items-center gap-2">
              <Input
                placeholder="https://example.com/page.aspx"
                value={urlInput}
                disabled={jobRunning}
                onChange={(e) => setUrlInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && addUrl()}
              />
              <Button size="sm" variant="outline" onClick={addUrl} disabled={jobRunning || !urlInput.trim()}>
                <Link2 className="mr-1 size-4" /> Add URL
              </Button>
            </div>
          </div>
          <div className="mt-2 flex items-start gap-2">
            <textarea
              className="min-h-16 w-full rounded-md border px-3 py-2 text-sm shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring"
              placeholder="Paste seed text / knowledge to train on…"
              value={promptInput}
              disabled={jobRunning}
              onChange={(e) => setPromptInput(e.target.value)}
            />
            <Button
              size="sm"
              variant="outline"
              onClick={addPrompt}
              disabled={jobRunning || !promptInput.trim()}
            >
              <MessageSquarePlus className="mr-1 size-4" /> Add
            </Button>
          </div>
        </div>

        {/* Generation options */}
        <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-sm text-muted-foreground">
          <label className="inline-flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={includeBuiltin}
              disabled={jobRunning}
              onChange={(e) => setIncludeBuiltin(e.target.checked)}
            />
            Include built-in hardware dataset
          </label>
          <label className="inline-flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={useSources}
              disabled={jobRunning || sources.length === 0}
              onChange={(e) => setUseSources(e.target.checked)}
            />
            Use uploaded sources ({sources.length})
          </label>
          <button
            onClick={importSpecs}
            disabled={jobRunning || sources.length === 0}
            title="Import specs from every URL/document above: deterministic parsers for AMD's DB and Intel chart PDFs, and the intelligent scrape agent (crawls index/leaf pages, respects robots.txt and anti-bot 403s) for any other URL"
            className="rounded-full border border-border px-3 py-1 text-xs font-medium hover:bg-muted disabled:opacity-50"
          >
            Import specs from sources
          </button>
        </div>

        {phase === "scraping" && (
          <p className="mt-3 text-sm text-muted-foreground">
            <Loader2 className="mr-1 inline size-4 animate-spin" />
            Importing specs {status.scrape_done ?? 0}/{status.scrape_total ?? "…"} —{" "}
            <span className="font-mono">{status.scrape_current}</span>
          </p>
        )}
        {phase === "scrape_done" && (
          <p className="mt-3 text-sm text-emerald-600 dark:text-emerald-400">
            <CheckCircle2 className="mr-1 inline size-4" />
            Learned {status.products_learned ?? 0} product(s)
            {(status.scrape_errors?.length ?? 0) > 0 &&
              ` · skipped sources: ${status.scrape_errors!.join("; ").slice(0, 160)}`}
            {" — now Generate dataset."}
          </p>
        )}
        {phase === "scrape_done" &&
          (status.scrape_outcomes?.length ?? 0) > 0 && (
            <ul className="mt-2 max-h-56 space-y-1 overflow-y-auto text-xs">
              {status.scrape_outcomes!.map((o, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span
                    className={
                      "mt-0.5 shrink-0 rounded px-1.5 py-0.5 font-medium " +
                      (OUTCOME_STYLES[o.status] ?? "bg-muted text-muted-foreground")
                    }
                  >
                    {o.status}
                  </span>
                  <span className="min-w-0">
                    <span className="font-mono break-all text-muted-foreground">
                      {o.url}
                    </span>
                    {o.detail ? (
                      <span className="text-muted-foreground"> — {o.detail}</span>
                    ) : null}
                  </span>
                </li>
              ))}
            </ul>
          )}

        {generating && (
          <p className="mt-3 text-sm text-muted-foreground">
            <Loader2 className="mr-1 inline size-4 animate-spin" />
            {phase === "extracting"
              ? `Extracting sources… ${status.sources_done ?? 0}/${status.sources_total ?? 0}`
              : `Generating Q&A pairs… ${status.pairs_generated ?? 0} pairs from ${status.chunks_done ?? 0}/${status.chunks_total ?? 0} chunks`}
          </p>
        )}
        {phase === "dataset_ready" && (
          <p className="mt-3 text-sm text-emerald-600 dark:text-emerald-400">
            <CheckCircle2 className="mr-1 inline size-4" />
            Dataset built: {status.train_count} train / {status.valid_count}{" "}
            valid ({status.pairs_generated ?? 0} pairs from your sources)
          </p>
        )}
        <p className="mt-2 text-sm">
          {hasDataset ? (
            <span className="text-emerald-600 dark:text-emerald-400">
              <CheckCircle2 className="mr-1 inline size-4" />
              {trainCount} train / {dataset.valid?.count ?? 0} validation
              examples ready
              {(dataset.custom_pairs ?? 0) > 0 &&
                ` (incl. ${dataset.custom_pairs} from your sources)`}
            </span>
          ) : (
            <span className="text-muted-foreground">No dataset generated yet.</span>
          )}
        </p>
      </section>

      {/* Knowledge gaps — self-improvement loop */}
      <section className="rounded-lg border p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Lightbulb className="size-4 text-muted-foreground" />
            <h2 className="font-medium">Knowledge gaps</h2>
            {gaps.filter((g) => g.status === "new").length > 0 && (
              <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-xs font-medium text-amber-700 dark:text-amber-300">
                {gaps.filter((g) => g.status === "new").length} new
              </span>
            )}
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={researchGaps}
            disabled={jobRunning || gaps.every((g) => g.status !== "new")}
          >
            {busy === "research" || phase === "researching" ? (
              <Loader2 className="mr-1 size-4 animate-spin" />
            ) : (
              <Globe className="mr-1 size-4" />
            )}
            Research gaps (web)
          </Button>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          Questions the specialist couldn&apos;t answer are captured here
          automatically. Research looks up the missing specs on the web and
          adds them to the training data — then Generate dataset → Train bakes
          them into the model&apos;s weights. The model never browses at answer
          time.
        </p>
        {phase === "researching" && (
          <p className="mt-3 text-sm text-muted-foreground">
            <Loader2 className="mr-1 inline size-4 animate-spin" />
            Researching… {status.gaps_done ?? 0}/{status.gaps_total ?? 0} gaps,{" "}
            {status.products_learned ?? 0} product(s) learned
          </p>
        )}
        {gaps.length > 0 ? (
          <ul className="mt-3 space-y-2">
            {gaps.map((g) => (
              <li
                key={g.id}
                className="flex items-start gap-2 rounded-md border border-border/60 px-3 py-2 text-sm"
              >
                <span
                  className={
                    g.status === "researched"
                      ? "mt-0.5 rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] font-medium text-emerald-600 dark:text-emerald-400"
                      : g.status === "trained"
                        ? "mt-0.5 rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium text-muted-foreground"
                        : "mt-0.5 rounded-full bg-amber-500/15 px-2 py-0.5 text-[11px] font-medium text-amber-700 dark:text-amber-300"
                  }
                >
                  {g.status}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-foreground" title={g.question}>
                    {g.question}
                  </p>
                  {g.researched_summary && (
                    <p className="mt-0.5 truncate text-xs text-muted-foreground" title={g.researched_summary}>
                      {g.researched_summary}
                    </p>
                  )}
                </div>
                <button
                  onClick={() => dismissGap(g.id)}
                  className="text-muted-foreground/70 transition hover:text-destructive"
                  title="Dismiss gap"
                >
                  <Trash2 className="size-4" />
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-3 text-sm text-muted-foreground/70">
            No gaps captured yet — they appear when users ask about hardware
            the model doesn&apos;t know.
          </p>
        )}
        {gaps.some((g) => g.status === "researched") && (
          <p className="mt-3 text-sm text-emerald-600 dark:text-emerald-400">
            <RefreshCw className="mr-1 inline size-4" />
            Researched gaps ready — run Generate dataset, then Train, then
            Convert &amp; Register to teach the model.
          </p>
        )}
      </section>

      {/* 2 — Train */}
      <section className="rounded-lg border p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FlaskConical className="size-4 text-muted-foreground" />
            <h2 className="font-medium">2 · Train (MLX LoRA)</h2>
          </div>
          {training ? (
            <Button size="sm" variant="destructive" onClick={stopTraining}>
              <Square className="mr-1 size-4" /> Stop
            </Button>
          ) : (
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={startTopUp}
                disabled={jobRunning || !hasDataset || !hasAdapters}
                title={
                  hasAdapters
                    ? "Warm-start from the current adapters and train fewer iters (≈400 recommended) — teaches new facts without a full retrain"
                    : "Run a full training once before a quick top-up"
                }
              >
                {busy === "train" ? (
                  <Loader2 className="mr-1 size-4 animate-spin" />
                ) : (
                  <Zap className="mr-1 size-4" />
                )}
                Quick top-up
              </Button>
              <Button
                size="sm"
                onClick={startTraining}
                disabled={jobRunning || !hasDataset}
              >
                {busy === "train" ? (
                  <Loader2 className="mr-1 size-4 animate-spin" />
                ) : (
                  <PlayCircle className="mr-1 size-4" />
                )}
                Start training
              </Button>
            </div>
          )}
        </div>

        <div className="mt-3 flex flex-wrap items-end gap-4">
          <div className="text-sm">
            <span className="mb-1 block text-muted-foreground">Base model</span>
            <div className="flex items-center gap-2">
              <select
                value={baseModel}
                disabled={training}
                onChange={(e) => {
                  const v = e.target.value;
                  if (v === "__search__") {
                    setShowHf(true);
                    return;
                  }
                  setBaseModel(v);
                  const bm = BASE_MODELS.find((b) => b.id === v);
                  if (bm) setModelName(bm.output);
                }}
                className="h-9 cursor-pointer appearance-none rounded-full border border-border bg-muted/50 px-4 pr-8 text-sm text-foreground transition-colors hover:bg-muted focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {BASE_MODELS.map((b) => (
                  <option key={b.id} value={b.id}>
                    {b.label}
                  </option>
                ))}
                {!BASE_MODELS.some((b) => b.id === baseModel) && (
                  <option value={baseModel}>{baseModel} (custom)</option>
                )}
                <option value="__search__">Search Hugging Face…</option>
              </select>
              <button
                type="button"
                onClick={() => setShowHf((v) => !v)}
                disabled={training}
                className="inline-flex items-center gap-1 rounded-md border px-2 py-1.5 text-xs text-muted-foreground transition hover:bg-muted/50 disabled:opacity-50"
                title="Search Hugging Face for a base model"
              >
                <Globe className="size-3.5" /> HF
              </button>
            </div>
          </div>
          <label className="text-sm">
            <span className="mb-1 block text-muted-foreground">Iterations</span>
            <Input
              type="number"
              className="w-28"
              value={iters}
              min={50}
              disabled={training}
              onChange={(e) => setIters(Number(e.target.value))}
            />
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-muted-foreground">Batch size</span>
            <Input
              type="number"
              className="w-24"
              value={batchSize}
              min={1}
              disabled={training}
              onChange={(e) => setBatchSize(Number(e.target.value))}
            />
          </label>
        </div>

        <p className="mt-2 text-xs text-muted-foreground/70">
          {hasAdapters
            ? "Only change the base / search Hugging Face to SWITCH base models. Retraining the existing model on the same base — a full “Start training” (fresh adapters) or a “Quick top-up” (warm-start) — needs no change here."
            : "Choose the base model to train from — a preset, or “Search Hugging Face…” for any text-generation repo."}
        </p>

        {showHf && (
          <div className="mt-3 rounded-md border bg-muted/30 p-3">
            <div className="flex items-center gap-2">
              <Input
                placeholder="Search Hugging Face — e.g. Qwen3 4B instruct, Llama 3.2 3B"
                value={hfQuery}
                disabled={training}
                onChange={(e) => setHfQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") searchHf();
                }}
                className="h-9 flex-1"
              />
              <Button
                size="sm"
                variant="outline"
                onClick={searchHf}
                disabled={hfSearching || !hfQuery.trim()}
              >
                {hfSearching ? (
                  <Loader2 className="mr-1 size-4 animate-spin" />
                ) : (
                  <Globe className="mr-1 size-4" />
                )}
                Search
              </Button>
            </div>
            <p className="mt-1 text-xs text-muted-foreground/70">
              Any text-generation repo works as a training base. mlx-lm
              downloads it at train time; gated repos (e.g. google/gemma-*) need
              an HF_TOKEN on the trainer host.
            </p>
            {hfResults.length > 0 && (
              <ul className="mt-2 max-h-56 space-y-1 overflow-auto">
                {hfResults.map((m) => (
                  <li
                    key={m.id}
                    className="flex items-center justify-between gap-2 rounded border bg-background px-2 py-1.5"
                  >
                    <div className="min-w-0">
                      <div className="truncate font-mono text-xs text-foreground">
                        {m.id}
                      </div>
                      <div className="text-[11px] text-muted-foreground">
                        {fmtParams(m.params) && (
                          <span className="font-medium text-foreground/80">
                            {fmtParams(m.params)} params
                          </span>
                        )}
                        {fmtBytes(m.size_bytes) && (
                          <span>
                            {fmtParams(m.params) ? " · " : ""}~
                            {fmtBytes(m.size_bytes)}
                          </span>
                        )}
                        {(fmtParams(m.params) || fmtBytes(m.size_bytes)) && " · "}
                        ↓ {m.downloads.toLocaleString()}
                        {m.gated && (
                          <span className="ml-1 rounded bg-amber-500/15 px-1 text-amber-600">
                            gated
                          </span>
                        )}
                      </div>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => pickBase(m.id)}
                      disabled={training}
                    >
                      Use
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {(training || phase === "trained" || (status.history?.length ?? 0) > 0) && (
          <div className="mt-4 space-y-2">
            <div className="h-2 w-full overflow-hidden rounded bg-muted">
              <div
                className="h-full bg-primary transition-all"
                style={{ width: `${trainPct}%` }}
              />
            </div>
            <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm text-muted-foreground">
              <span>
                Iteration {status.iter ?? 0} / {status.total_iters ?? "—"}
              </span>
              <span>
                Train loss:{" "}
                {status.train_loss != null ? status.train_loss.toFixed(3) : "—"}
              </span>
              <span>
                Val loss:{" "}
                {status.val_loss != null ? status.val_loss.toFixed(3) : "—"}
              </span>
              {phase === "trained" && (
                <span className="text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 className="mr-1 inline size-4" />
                  Training complete
                </span>
              )}
            </div>
            {status.history && <LossSparkline history={status.history} />}
            {status.log_tail && status.log_tail.length > 0 && (
              <details className="text-xs">
                <summary className="cursor-pointer text-muted-foreground">
                  Training log
                </summary>
                <pre className="mt-1 max-h-48 overflow-auto rounded bg-muted/50 p-2">
                  {status.log_tail.join("\n")}
                </pre>
              </details>
            )}
          </div>
        )}
      </section>

      {/* 3 — Convert & register */}
      <section className="rounded-lg border p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <PackageCheck className="size-4 text-muted-foreground" />
            <h2 className="font-medium">3 · Convert to GGUF &amp; register</h2>
          </div>
          <Button
            size="sm"
            onClick={convertAndRegister}
            disabled={jobRunning || training}
          >
            {busy === "convert" || converting ? (
              <Loader2 className="mr-1 size-4 animate-spin" />
            ) : null}
            Convert &amp; Register
          </Button>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          Fuses the LoRA adapters, converts to GGUF (q8_0), imports it into the
          local llama.cpp service, and registers it in the model registry.
          Requires a completed training run.
        </p>
        <div className="mt-3">
          <label className="text-sm">
            <span className="mb-1 block text-muted-foreground">
              Model id (must keep the <code>finetuned-</code> prefix)
            </span>
            <Input
              className="w-full max-w-md"
              value={modelName}
              disabled={jobRunning}
              onChange={(e) => setModelName(e.target.value)}
            />
          </label>
        </div>
        {converting && (
          <p className="mt-3 text-sm text-muted-foreground">
            <Loader2 className="mr-1 inline size-4 animate-spin" />
            {phase === "fusing"
              ? "Fusing LoRA adapters into the base model…"
              : "Converting to GGUF (this can take a few minutes)…"}
          </p>
        )}
        {registered && (
          <p className="mt-3 text-sm text-emerald-600 dark:text-emerald-400">
            <CheckCircle2 className="mr-1 inline size-4" />
            Registered! The model now appears in the chat model dropdown, and
            hardware-spec questions will route to it automatically (no RAG, no
            web search).
          </p>
        )}
      </section>

      {/* 4 — Fine-tuned models */}
      <section className="rounded-lg border p-4">
        <div className="flex items-center gap-2">
          <Database className="size-4 text-muted-foreground" />
          <h2 className="font-medium">4 · Fine-tuned models</h2>
        </div>
        {finetuned.length > 0 ? (
          <ul className="mt-3 space-y-1">
            {finetuned.map((m) => (
              <li
                key={m.id}
                className="flex items-center justify-between gap-2 rounded border bg-background px-3 py-2 text-sm"
              >
                <div className="min-w-0">
                  <div className="truncate text-foreground">{m.display_name}</div>
                  <code className="text-xs text-muted-foreground">
                    {m.model_id}
                  </code>
                </div>
                <button
                  onClick={() => deleteFinetuned(m)}
                  disabled={jobRunning}
                  className="inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs text-muted-foreground transition hover:bg-destructive/10 hover:text-destructive disabled:opacity-50"
                  title="Delete from the registry and remove the .gguf file"
                >
                  {busy === `del-${m.id}` ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : (
                    <Trash2 className="size-3.5" />
                  )}
                  Delete
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-3 text-sm text-muted-foreground/70">
            No fine-tuned models registered yet — train one, then Convert &amp;
            Register.
          </p>
        )}
        <div className="mt-4 flex flex-wrap items-center gap-2 border-t pt-3">
          <Button
            size="sm"
            variant="outline"
            onClick={clearArtifacts}
            disabled={jobRunning}
          >
            {busy === "clear-artifacts" ? (
              <Loader2 className="mr-1 size-4 animate-spin" />
            ) : (
              <Trash2 className="mr-1 size-4" />
            )}
            Clear training artifacts
          </Button>
          <span className="text-xs text-muted-foreground/70">
            Removes the LoRA adapters + fused files (shared working state, tied
            to the current base). Do this before switching base models.
          </span>
        </div>
      </section>
    </div>
  );
}
