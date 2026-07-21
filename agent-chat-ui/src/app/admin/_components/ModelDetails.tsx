"use client";

import { useEffect, useState } from "react";
import { ChevronRight, Loader2, FileBox } from "lucide-react";

interface LocalMeta {
  filename?: string;
  size_mb?: number;
  context_length?: number;
  native_context_length?: number;
  architecture?: string;
  tool_use?: boolean;
  parameters?: string;
  tags?: string[];
  description?: string;
}

interface HfFile {
  path: string;
  size: number;
  is_gguf: boolean;
}

type Json = Record<string, unknown>;

const fmtBytes = (n: number) =>
  n >= 1e9
    ? `${(n / 1e9).toFixed(2)} GB`
    : n >= 1e6
      ? `${(n / 1e6).toFixed(1)} MB`
      : `${(n / 1e3).toFixed(0)} KB`;

const fmtDate = (s?: string) => {
  if (!s) return "";
  const d = new Date(s);
  return isNaN(+d)
    ? s
    : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
};

const isEmpty = (v: unknown) =>
  v == null ||
  v === "" ||
  (Array.isArray(v) && v.length === 0) ||
  (typeof v === "object" && Object.keys(v as object).length === 0);

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3 py-1 text-xs">
      <dt className="text-muted-foreground w-32 shrink-0">{label}</dt>
      <dd className="min-w-0 flex-1 break-words">{children}</dd>
    </div>
  );
}

function Mono({ children }: { children: React.ReactNode }) {
  return <span className="font-mono text-[11px]">{children}</span>;
}

function JsonTree({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value === null) return <Mono>null</Mono>;
  if (typeof value !== "object") {
    return <Mono>{typeof value === "string" ? value : JSON.stringify(value)}</Mono>;
  }
  const entries = Array.isArray(value)
    ? value.map((v, i) => [String(i), v] as const)
    : Object.entries(value as Json);
  if (entries.length === 0) return <Mono>{Array.isArray(value) ? "[]" : "{}"}</Mono>;
  return (
    <div className={depth > 0 ? "border-border/60 ml-1 border-l pl-2.5" : ""}>
      {entries.map(([k, v]) => {
        const nested = v !== null && typeof v === "object" && Object.keys(v as object).length > 0;
        return (
          <div key={k} className="py-0.5 text-[11px]">
            <span className="text-muted-foreground">{k}</span>
            {nested ? (
              <JsonTree value={v} depth={depth + 1} />
            ) : (
              <>
                <span className="text-muted-foreground">: </span>
                <Mono>{typeof v === "string" ? v : JSON.stringify(v)}</Mono>
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}

function Group({
  label,
  count,
  defaultOpen = false,
  children,
}: {
  label: string;
  count?: number;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-border/70 border-t first:border-t-0">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="hover:text-foreground text-muted-foreground flex w-full items-center gap-1.5 py-2 text-left text-xs font-medium transition-colors"
      >
        <ChevronRight className={`size-3.5 transition-transform ${open ? "rotate-90" : ""}`} />
        <span className="tracking-wide uppercase">{label}</span>
        {count != null && <span className="text-muted-foreground/60">· {count}</span>}
      </button>
      {open && <div className="pb-3 pl-5">{children}</div>}
    </div>
  );
}

export default function ModelDetails({
  repoId,
  local,
  headers,
}: {
  repoId?: string;
  local: LocalMeta;
  headers: () => Record<string, string>;
}) {
  const [hf, setHf] = useState<Json | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!repoId) return;
    let live = true;
    setLoading(true);
    setErr(null);
    fetch(`/api/v1/admin/local/hf-details?repo_id=${encodeURIComponent(repoId)}`, {
      headers: headers(),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`details ${r.status}`))))
      .then((d) => live && setHf(d))
      .catch((e) => live && setErr((e as Error).message))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [repoId, headers]);

  const card = (hf?.cardData as Json | undefined) ?? undefined;
  const license =
    (card?.license as string) ??
    (hf?.tags as string[] | undefined)?.find((t) => t.startsWith("license:"))?.slice(8);
  const files = (hf?.files as HfFile[] | undefined) ?? [];
  const gguf = hf?.gguf as Json | undefined;
  const config = hf?.config as Json | undefined;

  return (
    <div className="bg-background/40 mt-1 rounded-lg border px-3">
      <Group label="Overview" defaultOpen>
        <dl>
          {local.description && <Row label="Description">{local.description}</Row>}
          {repoId && (
            <Row label="Repository">
              <a
                href={`https://huggingface.co/${repoId}`}
                target="_blank"
                rel="noreferrer"
                className="text-sky-600 hover:underline dark:text-sky-400"
              >
                <Mono>{repoId}</Mono>
              </a>
            </Row>
          )}
          {(hf?.pipeline_tag as string) && <Row label="Task">{hf!.pipeline_tag as string}</Row>}
          {(hf?.library_name as string) && <Row label="Library">{hf!.library_name as string}</Row>}
          {license && <Row label="License">{license}</Row>}
          {typeof hf?.downloads === "number" && (
            <Row label="Downloads">{(hf.downloads as number).toLocaleString()}</Row>
          )}
          {typeof hf?.likes === "number" && (
            <Row label="Likes">{(hf.likes as number).toLocaleString()}</Row>
          )}
          {(hf?.gated as unknown) ? <Row label="Gated">{String(hf!.gated)}</Row> : null}
          {fmtDate(hf?.lastModified as string) && (
            <Row label="Updated">{fmtDate(hf!.lastModified as string)}</Row>
          )}
          {fmtDate(hf?.createdAt as string) && (
            <Row label="Created">{fmtDate(hf!.createdAt as string)}</Row>
          )}
          {(hf?.sha as string) && (
            <Row label="Commit">
              <Mono>{(hf!.sha as string).slice(0, 12)}</Mono>
            </Row>
          )}
        </dl>
        {loading && (
          <div className="text-muted-foreground flex items-center gap-1.5 py-1 text-xs">
            <Loader2 className="size-3.5 animate-spin" /> Loading HuggingFace details…
          </div>
        )}
        {err && (
          <div className="text-muted-foreground py-1 text-xs">HuggingFace details unavailable.</div>
        )}
      </Group>

      <Group label="On device">
        <dl>
          {local.filename && (
            <Row label="File">
              <Mono>{local.filename}</Mono>
            </Row>
          )}
          {local.parameters && <Row label="Parameters">{local.parameters}</Row>}
          {local.size_mb ? (
            <Row label="Size on disk">
              {local.size_mb >= 1024
                ? `${(local.size_mb / 1024).toFixed(2)} GB`
                : `${local.size_mb} MB`}
            </Row>
          ) : null}
          {local.architecture && <Row label="Architecture">{local.architecture}</Row>}
          {local.context_length ? (
            <Row label="Load context">{local.context_length.toLocaleString()} tokens</Row>
          ) : null}
          {local.native_context_length ? (
            <Row label="Native context">{local.native_context_length.toLocaleString()} tokens</Row>
          ) : null}
          <Row label="Tool template">{local.tool_use ? "detected" : "not advertised"}</Row>
        </dl>
      </Group>

      {files.length > 0 && (
        <Group label="Files" count={files.length}>
          <div className="flex flex-col">
            {files.map((f) => (
              <div
                key={f.path}
                className={`flex items-center gap-2 py-1 text-[11px] ${f.is_gguf ? "text-foreground" : "text-muted-foreground"}`}
              >
                {f.is_gguf && <FileBox className="size-3 shrink-0 text-emerald-500" />}
                <Mono>{f.path}</Mono>
                <span className="text-muted-foreground/70 ml-auto shrink-0 tabular-nums">
                  {f.size ? fmtBytes(f.size) : ""}
                </span>
              </div>
            ))}
          </div>
        </Group>
      )}

      {card && !isEmpty(card) && (
        <Group label="Model card" count={Object.keys(card).length}>
          <JsonTree value={card} />
        </Group>
      )}

      {gguf && !isEmpty(gguf) && (
        <Group label="GGUF metadata">
          <JsonTree value={gguf} />
        </Group>
      )}

      {config && !isEmpty(config) && (
        <Group label="Config">
          <JsonTree value={config} />
        </Group>
      )}

      {Array.isArray(hf?.tags) && (hf!.tags as string[]).length > 0 && (
        <Group label="Tags" count={(hf!.tags as string[]).length}>
          <div className="flex flex-wrap gap-1">
            {(hf!.tags as string[]).map((t) => (
              <span
                key={t}
                className="bg-muted text-muted-foreground rounded-sm px-1.5 py-0.5 font-mono text-[10px]"
              >
                {t}
              </span>
            ))}
          </div>
        </Group>
      )}
    </div>
  );
}
