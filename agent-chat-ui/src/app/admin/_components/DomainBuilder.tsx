"use client";

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { DeleteButton } from "@/components/ui/delete-button";
import { getAdminToken } from "../token";
import {
  Loader2,
  Plus,
  Sparkles,
  Pencil,
  X,
  Layers,
  ChevronDown,
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
  selectedSubs,
  onToggleSub,
  onToggleDomain,
  selectDisabled,
}: {
  domains: DomainInfo[];
  onChanged: () => void;
  selectedSubs: string[];
  onToggleSub: (key: string) => void;
  onToggleDomain: (d: DomainInfo) => void;
  selectDisabled?: boolean;
}) {
  const confirm = useConfirm();
  const [newDomain, setNewDomain] = useState("");
  const [showNewDomain, setShowNewDomain] = useState(false);
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
      setShowNewDomain(false);
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
      description:
        "This permanently deletes the domain folder and every subdomain inside it, schemas, curated rows, and imported data.",
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
      description:
        "This permanently deletes the subdomain's files, its schema, curated rows, and imported data.",
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

  const editorEl = edit ? (
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
  ) : null;

  return (
    <div className="mt-4 space-y-3 rounded-md border p-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <Layers className="size-4 text-muted-foreground" />
            <h3 className="text-sm font-semibold text-foreground">
              Manage Domains
            </h3>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Create your own knowledge domains, define a subdomain&apos;s fields
            or let the assistant propose a schema, add rows, and train. The
            built-in Hardware domain&apos;s rows are editable too (its fields and
            answer style are fixed).
          </p>
        </div>
        <Button
          size="sm"
          className="shrink-0"
          onClick={() => setShowNewDomain((s) => !s)}
        >
          <Plus className="mr-1 size-4" /> New domain
        </Button>
      </div>

      {error && (
        <div className="rounded border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      {showNewDomain && (
        <div className="space-y-3 rounded-md border border-dashed p-4">
          <p className="text-sm font-medium">New domain</p>
          <div className="flex flex-wrap items-center gap-2">
            <Input
              className="w-56"
              placeholder="name (e.g. software, vehicles)"
              value={newDomain}
              onChange={(e) => setNewDomain(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && createDomain()}
            />
            <div className="ml-auto flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowNewDomain(false)}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={createDomain}
                disabled={busy === "domain" || !newDomain.trim()}
              >
                {busy === "domain" ? (
                  <Loader2 className="mr-1 size-4 animate-spin" />
                ) : (
                  <Plus className="mr-1 size-4" />
                )}
                Create
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Domains */}
      <ul className="space-y-2">
        {domains.map((d) => {
          const open = openDomains.has(d.name);
          const keys = d.subdomains.map((s) => `${d.name}/${s.name}`);
          const selCount = keys.filter((k) => selectedSubs.includes(k)).length;
          const allOn = keys.length > 0 && selCount === keys.length;
          return (
            <li key={d.name} className="rounded-md border">
              <div className="flex items-center gap-2 px-3 py-2">
                <button
                  type="button"
                  onClick={() => toggleOpen(d.name)}
                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                >
                  {open ? (
                    <ChevronDown className="size-4 shrink-0 text-muted-foreground" />
                  ) : (
                    <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
                  )}
                  <span className="shrink-0 text-sm font-medium capitalize">
                    {d.name}
                  </span>
                  {d.builtin && (
                    <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-[10px] uppercase text-muted-foreground">
                      built-in
                    </span>
                  )}
                  <span className="shrink-0 text-xs text-muted-foreground/70">
                    {d.subdomains.length} subdomain
                    {d.subdomains.length === 1 ? "" : "s"}
                  </span>
                </button>
                <div className="flex shrink-0 items-center gap-2">
                  {keys.length > 0 && (
                    <span
                      className="text-[10px] tabular-nums text-muted-foreground/70"
                      title="Subdomains selected for training"
                    >
                      {selCount}/{keys.length}
                    </span>
                  )}
                  <Switch
                    checked={allOn}
                    disabled={selectDisabled || keys.length === 0}
                    onCheckedChange={() => onToggleDomain(d)}
                    title="Include all subdomains in training"
                  />
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
                  {!d.builtin && (
                    <DeleteButton
                      onClick={() => deleteDomain(d.name)}
                      disabled={selCount > 0}
                      title={
                        selCount > 0
                          ? "Turn its training toggle off to delete this domain"
                          : "Delete domain"
                      }
                    />
                  )}
                </div>
              </div>
              {open && (
                <div className="space-y-1 border-t px-3 py-3">
                  {d.subdomains.length === 0 && (
                    <p className="text-xs text-muted-foreground/70">
                      No subdomains yet
                      {d.builtin ? "." : ", add one with the Subdomain button."}
                    </p>
                  )}
                  {d.subdomains.map((s) => (
                    <div key={s.name}>
                      <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5 text-sm">
                        <span className="capitalize text-foreground">
                          {s.label}
                        </span>
                        <span className="rounded-full bg-muted px-1.5 text-[10px] text-muted-foreground">
                          {s.render === "spec_table" ? "table" : "prose"}
                        </span>
                        <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground/70">
                          {(s.fields ?? []).join(", ")}
                        </span>
                        <div className="flex shrink-0 items-center gap-1">
                          <Switch
                            checked={selectedSubs.includes(
                              `${d.name}/${s.name}`,
                            )}
                            disabled={selectDisabled}
                            onCheckedChange={() =>
                              onToggleSub(`${d.name}/${s.name}`)
                            }
                            title="Include in training"
                          />
                          <Button
                            size="icon"
                            variant="ghost"
                            className="size-8"
                            onClick={() => openEdit(d.name, s)}
                            disabled={busy === `open:${d.name}/${s.name}`}
                            title={s.builtin ? "Edit rows" : "Edit subdomain"}
                          >
                            {busy === `open:${d.name}/${s.name}` ? (
                              <Loader2 className="size-4 animate-spin" />
                            ) : (
                              <Pencil className="size-4" />
                            )}
                          </Button>
                          {!s.builtin && (
                            <DeleteButton
                              onClick={() => deleteSubdomain(d.name, s.name)}
                              disabled={selectedSubs.includes(
                                `${d.name}/${s.name}`,
                              )}
                              title={
                                selectedSubs.includes(`${d.name}/${s.name}`)
                                  ? "Turn its training toggle off to delete"
                                  : "Delete subdomain"
                              }
                            />
                          )}
                        </div>
                      </div>
                      {edit &&
                        edit.domain === d.name &&
                        edit.original === s.name &&
                        editorEl}
                    </div>
                  ))}
                  {edit &&
                    edit.domain === d.name &&
                    edit.original === null &&
                    editorEl}
                </div>
              )}
            </li>
          );
        })}
      </ul>
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
          Schema, define fields, or let the assistant propose them
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
            No fields yet, add some or propose a schema above.
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
                    placeholder="Questions (one per line uses {name}), comma separated"
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
                    placeholder="Answer template, uses {name} and {value}"
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
            No entries yet, add one, or use Smart import to fill them.
          </p>
        ) : (
          <div className="mt-2 max-h-[26rem] space-y-2 overflow-auto pr-1">
            {edit.entities.map((row, ri) => (
              <div key={ri} className="rounded-md border bg-background/60 p-2">
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
