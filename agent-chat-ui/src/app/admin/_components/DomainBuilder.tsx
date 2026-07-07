"use client";

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { DeleteButton } from "@/components/ui/delete-button";
import { getAdminToken } from "../token";
import { cn } from "@/lib/utils";
import {
  Loader2,
  Plus,
  Sparkles,
  Pencil,
  X,
  Layers,
  FolderPlus,
  ChevronRight,
} from "lucide-react";

export interface Field {
  key: string;
  label: string;
  questions?: string[];
  answer?: string;
}
export interface SubdomainInfo {
  name: string;
  label: string;
  description: string;
  builtin: boolean;
  render: string;
  fields?: string[];
}
export interface DomainInfo {
  name: string;
  description: string;
  builtin: boolean;
  subdomains: SubdomainInfo[];
}

interface EditState {
  domain: string;
  name: string;
  original: string | null; // slug being edited, or null for new
  description: string;
  render: string;
  fields: Field[];
  overview: string[];
  entities: Record<string, unknown>[];
  sample: string;
  builtin: boolean; // hardware: fixed schema, rows-only editing
  curated: string[]; // read-only built-in product names (hardware)
}

const blankEdit = (domain: string): EditState => ({
  domain,
  name: "",
  original: null,
  description: "",
  render: "prose",
  fields: [],
  overview: [],
  entities: [],
  sample: "",
  builtin: false,
  curated: [],
});

export default function DomainBuilder({
  domains,
  onChanged,
}: {
  domains: DomainInfo[];
  onChanged: () => void;
}) {
  const confirm = useConfirm();
  const [newDomain, setNewDomain] = useState("");
  const [edit, setEdit] = useState<EditState | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openDomains, setOpenDomains] = useState<Set<string>>(new Set());

  const toggleOpen = (name: string) =>
    setOpenDomains((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });

  const headers = useCallback(() => {
    const t = getAdminToken();
    return { "Content-Type": "application/json", "X-Admin-Token": t || "" };
  }, []);

  const api = useCallback(
    async (method: string, path: string, body?: object) => {
      const r = await fetch(`/api/admin/trainer/${path}`, {
        method,
        headers: headers(),
        body: body ? JSON.stringify(body) : undefined,
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok)
        throw new Error(data?.detail ?? data?.error ?? `${path} ${r.status}`);
      return data;
    },
    [headers],
  );

  const createDomain = async () => {
    if (!newDomain.trim()) return;
    setBusy("domain");
    setError(null);
    try {
      await api("POST", "domains", { name: newDomain.trim() });
      setNewDomain("");
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const deleteDomain = async (name: string) => {
    if (!(await confirm({
      title: `Delete domain "${name}"?`,
      description: "This removes the domain and all its subdomains and data.",
      confirmText: "Delete",
    })))
      return;
    setBusy(`del:${name}`);
    setError(null);
    try {
      await api("DELETE", `domains/${encodeURIComponent(name)}`);
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const openEdit = async (domain: string, sub: SubdomainInfo) => {
    setError(null);
    setBusy(`open:${domain}/${sub.name}`);
    try {
      const data = await api(
        "GET",
        `domains/${encodeURIComponent(domain)}/subdomains/${encodeURIComponent(sub.name)}`,
      );
      setEdit({
        domain,
        name: data.name ?? sub.name,
        original: sub.name,
        description: data.description ?? "",
        render: data.render ?? "prose",
        fields: (data.fields ?? []) as Field[],
        overview: (data.overview ?? []) as string[],
        entities: (data.entities ?? []) as Record<string, unknown>[],
        sample: "",
        builtin: !!data.builtin,
        curated: (data.curated ?? []) as string[],
      });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const deleteSubdomain = async (domain: string, name: string) => {
    if (!(await confirm({
      title: `Delete subdomain "${name}"?`,
      description: "This removes its schema and data.",
      confirmText: "Delete",
    })))
      return;
    setBusy(`del:${domain}/${name}`);
    setError(null);
    try {
      await api(
        "DELETE",
        `domains/${encodeURIComponent(domain)}/subdomains/${encodeURIComponent(name)}`,
      );
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const proposeSchema = async () => {
    if (!edit) return;
    setBusy("schema");
    setError(null);
    try {
      const data = await api("POST", "domains/propose-schema", {
        description: edit.description,
        sample_text: edit.sample,
      });
      setEdit({
        ...edit,
        render: data.render ?? edit.render,
        fields: (data.fields ?? []) as Field[],
      });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const proposeTemplates = async () => {
    if (!edit) return;
    setBusy("templates");
    setError(null);
    try {
      const data = await api("POST", "domains/propose-templates", {
        fields: edit.fields,
      });
      setEdit({
        ...edit,
        fields: (data.fields ?? edit.fields) as Field[],
        overview: (data.overview ?? edit.overview) as string[],
      });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const saveSubdomain = async () => {
    if (!edit) return;
    if (!edit.builtin && !edit.name.trim()) {
      setError("Subdomain name is required.");
      return;
    }
    if (!edit.builtin && edit.fields.length === 0) {
      setError("Add at least one field (or use Propose schema).");
      return;
    }
    setBusy("save");
    setError(null);
    try {
      let slug = edit.name.trim();
      if (!edit.builtin) {
        const saved = await api(
          "POST",
          `domains/${encodeURIComponent(edit.domain)}/subdomains`,
          {
            name: edit.name.trim(),
            description: edit.description,
            render: edit.render,
            fields: edit.fields,
            overview: edit.overview.length ? edit.overview : null,
          },
        );
        slug = saved?.name ?? slug;
      }
      await api(
        "POST",
        `domains/${encodeURIComponent(edit.domain)}/subdomains/${encodeURIComponent(slug)}/entities`,
        { entities: edit.entities },
      );
      setEdit(null);
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  // ── field + entity row helpers ─────────────────────────────────────────────
  const setField = (i: number, patch: Partial<Field>) =>
    edit &&
    setEdit({
      ...edit,
      fields: edit.fields.map((f, k) => (k === i ? { ...f, ...patch } : f)),
    });
  const addField = () =>
    edit && setEdit({ ...edit, fields: [...edit.fields, { key: "", label: "" }] });
  const removeField = (i: number) =>
    edit &&
    setEdit({ ...edit, fields: edit.fields.filter((_, k) => k !== i) });

  const setCell = (row: number, key: string, value: string) =>
    edit &&
    setEdit({
      ...edit,
      entities: edit.entities.map((e, k) =>
        k === row ? { ...e, [key]: value } : e,
      ),
    });
  const addRow = () =>
    edit && setEdit({ ...edit, entities: [...edit.entities, { name: "" }] });
  const removeRow = (i: number) =>
    edit &&
    setEdit({ ...edit, entities: edit.entities.filter((_, k) => k !== i) });

  return (
    <div className="mt-4 rounded-md border border-dashed p-3">
      <div className="flex items-center gap-2">
        <Layers className="size-4 text-muted-foreground" />
        <p className="text-sm font-medium text-foreground">
          Domains &amp; subdomains
        </p>
      </div>
      <p className="mt-1 text-sm text-muted-foreground">
        Create your own knowledge domains — define a subdomain&apos;s fields
        yourself or let the assistant propose a schema — add rows, and train. The
        built-in Hardware domain&apos;s product rows are editable too; its fields
        and Q&amp;A style are fixed.
      </p>

      {error && (
        <p className="mt-2 rounded bg-rose-500/10 px-2 py-1 text-xs text-rose-600 dark:text-rose-400">
          {error}
        </p>
      )}

      {/* New domain */}
      <div className="mt-3 flex items-center gap-2">
        <Input
          placeholder="New domain name (e.g. software, vehicles)"
          value={newDomain}
          onChange={(e) => setNewDomain(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && createDomain()}
          className="max-w-xs"
        />
        <Button
          size="sm"
          variant="secondary"
          onClick={createDomain}
          disabled={busy === "domain" || !newDomain.trim()}
        >
          {busy === "domain" ? (
            <Loader2 className="mr-1 size-4 animate-spin" />
          ) : (
            <FolderPlus className="mr-1 size-4" />
          )}
          Add domain
        </Button>
      </div>

      {/* Domains (hierarchical accordion) */}
      <div className="mt-3 space-y-2">
        {domains.map((d) => {
          const open = openDomains.has(d.name);
          return (
            <div key={d.name} className="overflow-hidden rounded-lg border">
              <div className="flex items-center gap-2 bg-muted/30 px-2 py-2">
                <button
                  type="button"
                  onClick={() => toggleOpen(d.name)}
                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                >
                  <ChevronRight
                    className={cn(
                      "size-4 shrink-0 text-muted-foreground transition-transform",
                      open && "rotate-90",
                    )}
                  />
                  <span className="text-sm font-medium capitalize text-foreground">
                    {d.name}
                  </span>
                  {d.builtin && (
                    <span className="rounded-full bg-muted px-1.5 text-[10px] text-muted-foreground">
                      built-in
                    </span>
                  )}
                  <span className="text-xs text-muted-foreground/70">
                    {d.subdomains.length} subdomain
                    {d.subdomains.length === 1 ? "" : "s"}
                  </span>
                </button>
                {!d.builtin && (
                  <div className="flex shrink-0 items-center gap-1">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        setOpenDomains((p) => new Set(p).add(d.name));
                        setEdit(blankEdit(d.name));
                      }}
                    >
                      <Plus className="mr-1 size-4" /> Subdomain
                    </Button>
                    <DeleteButton
                      onClick={() => deleteDomain(d.name)}
                      title="Delete domain"
                    />
                  </div>
                )}
              </div>
              {open && (
                <div className="space-y-1 border-t p-2">
                  {d.subdomains.length === 0 && (
                    <p className="px-1 py-1 text-xs text-muted-foreground/70">
                      No subdomains yet
                      {d.builtin ? "." : " — add one with the Subdomain button."}
                    </p>
                  )}
                  {d.subdomains.map((s) => (
                    <div
                      key={s.name}
                      className="flex items-center gap-2 rounded-md border bg-background px-2 py-1.5 text-sm"
                    >
                      <span className="capitalize text-foreground">{s.label}</span>
                      <span className="rounded-full bg-muted px-1.5 text-[10px] text-muted-foreground">
                        {s.render === "spec_table" ? "table" : "prose"}
                      </span>
                      <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground/70">
                        {(s.fields ?? []).join(", ")}
                      </span>
                      <div className="flex shrink-0 items-center gap-1">
                        <Button
                          size="icon"
                          variant="ghost"
                          className="size-8"
                          onClick={() => openEdit(d.name, s)}
                          disabled={busy === `open:${d.name}/${s.name}`}
                          title={d.builtin ? "Edit rows" : "Edit subdomain"}
                        >
                          {busy === `open:${d.name}/${s.name}` ? (
                            <Loader2 className="size-4 animate-spin" />
                          ) : (
                            <Pencil className="size-4" />
                          )}
                        </Button>
                        {!d.builtin && (
                          <DeleteButton
                            onClick={() => deleteSubdomain(d.name, s.name)}
                            title="Delete subdomain"
                          />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {edit && (
        <SubdomainEditor
          edit={edit}
          setEdit={setEdit}
          busy={busy}
          onClose={() => setEdit(null)}
          onSave={saveSubdomain}
          onProposeSchema={proposeSchema}
          onProposeTemplates={proposeTemplates}
          setField={setField}
          addField={addField}
          removeField={removeField}
          setCell={setCell}
          addRow={addRow}
          removeRow={removeRow}
        />
      )}
    </div>
  );
}

function SubdomainEditor({
  edit,
  setEdit,
  busy,
  onClose,
  onSave,
  onProposeSchema,
  onProposeTemplates,
  setField,
  addField,
  removeField,
  setCell,
  addRow,
  removeRow,
}: {
  edit: EditState;
  setEdit: (e: EditState) => void;
  busy: string | null;
  onClose: () => void;
  onSave: () => void;
  onProposeSchema: () => void;
  onProposeTemplates: () => void;
  setField: (i: number, patch: Partial<Field>) => void;
  addField: () => void;
  removeField: (i: number) => void;
  setCell: (row: number, key: string, value: string) => void;
  addRow: () => void;
  removeRow: (i: number) => void;
}) {
  const [showTemplates, setShowTemplates] = useState(false);
  return (
    <div className="mt-3 rounded-md border border-primary/30 bg-muted/30 p-3">
      <div className="flex items-center gap-2">
        <p className="text-sm font-medium text-foreground">
          {edit.builtin
            ? "Edit rows"
            : edit.original
              ? "Edit subdomain"
              : "New subdomain"}
          <span className="text-muted-foreground">
            {" · "}
            {edit.builtin ? `hardware / ${edit.name}` : edit.domain}
          </span>
        </p>
        <Button size="sm" variant="ghost" className="ml-auto" onClick={onClose}>
          <X className="size-4" />
        </Button>
      </div>

      {edit.builtin ? (
        <p className="mt-2 text-xs text-muted-foreground">
          Fixed hardware schema · answers render as a spec table.
          {edit.curated.length > 0 &&
            ` Built-in (edit these in facts.yaml): ${edit.curated.join(", ")}.`}
        </p>
      ) : (
        <>
      <div className="mt-2 flex flex-wrap items-center gap-3">
        <Input
          placeholder="Subdomain name (e.g. games)"
          value={edit.name}
          disabled={!!edit.original}
          onChange={(e) => setEdit({ ...edit, name: e.target.value })}
          className="max-w-xs"
        />
        <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
          <Switch
            checked={edit.render === "spec_table"}
            onCheckedChange={(v) =>
              setEdit({ ...edit, render: v ? "spec_table" : "prose" })
            }
          />
          Spec-table answers (hardware-style)
        </label>
      </div>
      <Input
        placeholder="Short description"
        value={edit.description}
        onChange={(e) => setEdit({ ...edit, description: e.target.value })}
        className="mt-2"
      />

      {/* Smart schema proposal */}
      <div className="mt-3 rounded border border-dashed p-2">
        <p className="text-xs font-medium text-foreground">
          Schema — define fields, or let the assistant propose them
        </p>
        <textarea
          className="mt-1 min-h-14 w-full rounded-md border px-2 py-1 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
          placeholder="Paste a sample of your data or describe it, then Propose schema…"
          value={edit.sample}
          onChange={(e) => setEdit({ ...edit, sample: e.target.value })}
        />
        <Button
          size="sm"
          variant="outline"
          className="mt-1"
          onClick={onProposeSchema}
          disabled={busy === "schema"}
        >
          {busy === "schema" ? (
            <Loader2 className="mr-1 size-4 animate-spin" />
          ) : (
            <Sparkles className="mr-1 size-4" />
          )}
          Propose schema
        </Button>
      </div>

      {/* Fields editor */}
      <div className="mt-3">
        <div className="flex items-center gap-2">
          <p className="text-xs font-medium text-foreground">Fields</p>
          <Button size="sm" variant="ghost" onClick={addField}>
            <Plus className="mr-1 size-4" /> Add field
          </Button>
          <button
            onClick={() => setShowTemplates((v) => !v)}
            className="ml-auto text-xs text-muted-foreground underline-offset-2 hover:underline"
          >
            {showTemplates ? "Hide" : "Edit"} question templates
          </button>
        </div>
        {edit.fields.length === 0 && (
          <p className="mt-1 text-xs text-muted-foreground/70">
            No fields yet — add some or propose a schema above.
          </p>
        )}
        <div className="mt-1 space-y-2">
          {edit.fields.map((f, i) => (
            <div key={i} className="rounded border p-2">
              <div className="flex items-center gap-2">
                <Input
                  placeholder="key (snake_case)"
                  value={f.key}
                  onChange={(e) => setField(i, { key: e.target.value })}
                  className="max-w-[10rem]"
                />
                <Input
                  placeholder="Label"
                  value={f.label}
                  onChange={(e) => setField(i, { label: e.target.value })}
                  className="max-w-[12rem]"
                />
                <DeleteButton
                  onClick={() => removeField(i)}
                  title="Remove field"
                />
              </div>
              {showTemplates && (
                <div className="mt-2 space-y-1 pl-1">
                  <Input
                    placeholder="Questions (one per line uses {name}) — comma separated"
                    value={(f.questions ?? []).join(" | ")}
                    onChange={(e) =>
                      setField(i, {
                        questions: e.target.value
                          .split("|")
                          .map((q) => q.trim())
                          .filter(Boolean),
                      })
                    }
                  />
                  <Input
                    placeholder="Answer template — uses {name} and {value}"
                    value={f.answer ?? ""}
                    onChange={(e) => setField(i, { answer: e.target.value })}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
        {edit.fields.length > 0 && (
          <Button
            size="sm"
            variant="outline"
            className="mt-2"
            onClick={onProposeTemplates}
            disabled={busy === "templates"}
          >
            {busy === "templates" ? (
              <Loader2 className="mr-1 size-4 animate-spin" />
            ) : (
              <Sparkles className="mr-1 size-4" />
            )}
            Auto-generate templates
          </Button>
        )}
        {edit.overview.length > 0 && (
          <p className="mt-1 text-xs text-muted-foreground/70">
            Overview sentences: {edit.overview.length} (auto)
          </p>
        )}
      </div>
        </>
      )}

      {/* Entries */}
      <div className="mt-3">
        <div className="flex items-center gap-2">
          <p className="text-xs font-medium text-foreground">
            Entries ({edit.entities.length})
          </p>
          <Button
            size="sm"
            variant="secondary"
            onClick={addRow}
            disabled={edit.fields.length === 0}
          >
            <Plus className="mr-1 size-4" /> Add entry
          </Button>
        </div>
        {edit.entities.length === 0 ? (
          <p className="mt-1 text-xs text-muted-foreground/70">
            No entries yet — add one, or use Smart import to fill them.
          </p>
        ) : (
          <div className="mt-2 max-h-[26rem] space-y-2 overflow-auto pr-1">
            {edit.entities.map((row, ri) => (
              <div key={ri} className="rounded-md border bg-background p-2">
                <div className="flex items-center gap-2">
                  <Input
                    value={String(row.name ?? "")}
                    onChange={(e) => setCell(ri, "name", e.target.value)}
                    placeholder="Name"
                    className="h-8 flex-1 font-medium"
                  />
                  <DeleteButton
                    onClick={() => removeRow(ri)}
                    title="Remove entry"
                  />
                </div>
                {edit.fields.length > 0 && (
                  <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {edit.fields.map((f) => (
                      <label key={f.key} className="flex flex-col gap-0.5">
                        <span className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                          {f.label || f.key}
                        </span>
                        <Input
                          value={String(row[f.key] ?? "")}
                          onChange={(e) => setCell(ri, f.key, e.target.value)}
                          className="h-8"
                        />
                      </label>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="mt-3 flex items-center gap-2">
        <Button size="sm" onClick={onSave} disabled={busy === "save"}>
          {busy === "save" && (
            <Loader2 className="mr-1 size-4 animate-spin" />
          )}
          Save
        </Button>
        <Button size="sm" variant="ghost" onClick={onClose}>
          Cancel
        </Button>
      </div>
    </div>
  );
}
