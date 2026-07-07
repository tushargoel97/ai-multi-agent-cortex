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
  Save,
  X,
  Layers,
  FolderPlus,
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
    if (!edit || !edit.name.trim()) {
      setError("Subdomain name is required.");
      return;
    }
    if (edit.fields.length === 0) {
      setError("Add at least one field (or use Propose schema).");
      return;
    }
    setBusy("save");
    setError(null);
    try {
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
      const slug = saved?.name ?? edit.name.trim();
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

  const userDomains = domains.filter((d) => !d.builtin);

  return (
    <div className="mt-4 rounded-md border border-dashed p-3">
      <div className="flex items-center gap-2">
        <Layers className="size-4 text-muted-foreground" />
        <p className="text-sm font-medium text-foreground">
          Custom domains &amp; subdomains
        </p>
      </div>
      <p className="mt-1 text-sm text-muted-foreground">
        Create your own knowledge domains. Define a subdomain&apos;s fields
        yourself or let the assistant propose a schema from a description/sample
        for you to review — then add rows and train on it.
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
          variant="outline"
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

      {/* Existing user domains */}
      <div className="mt-3 space-y-2">
        {userDomains.length === 0 && (
          <p className="text-sm text-muted-foreground/70">
            No custom domains yet.
          </p>
        )}
        {userDomains.map((d) => (
          <div key={d.name} className="rounded-md border p-2">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium capitalize text-foreground">
                {d.name}
              </span>
              <span className="text-xs text-muted-foreground/70">
                {d.subdomains.length} subdomain
                {d.subdomains.length === 1 ? "" : "s"}
              </span>
              <div className="ml-auto flex items-center gap-1">
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setEdit(blankEdit(d.name))}
                >
                  <Plus className="mr-1 size-4" /> Subdomain
                </Button>
                <DeleteButton
                  onClick={() => deleteDomain(d.name)}
                  title="Delete domain"
                />
              </div>
            </div>
            {d.subdomains.length > 0 && (
              <ul className="mt-1 space-y-1">
                {d.subdomains.map((s) => (
                  <li
                    key={s.name}
                    className="flex items-center gap-2 rounded px-2 py-1 text-sm hover:bg-muted/50"
                  >
                    <span className="capitalize text-foreground">{s.label}</span>
                    <span className="rounded bg-muted px-1 text-[10px] text-muted-foreground">
                      {s.render === "spec_table" ? "table" : "prose"}
                    </span>
                    <span className="truncate text-xs text-muted-foreground/70">
                      {(s.fields ?? []).join(", ")}
                    </span>
                    <div className="ml-auto flex items-center gap-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => openEdit(d.name, s)}
                        disabled={busy === `open:${d.name}/${s.name}`}
                      >
                        {busy === `open:${d.name}/${s.name}` ? (
                          <Loader2 className="size-4 animate-spin" />
                        ) : (
                          "Edit"
                        )}
                      </Button>
                      <DeleteButton
                        onClick={() => deleteSubdomain(d.name, s.name)}
                        title="Delete subdomain"
                      />
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))}
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
          {edit.original ? "Edit" : "New"} subdomain
          <span className="text-muted-foreground"> · {edit.domain}</span>
        </p>
        <Button size="sm" variant="ghost" className="ml-auto" onClick={onClose}>
          <X className="size-4" />
        </Button>
      </div>

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

      {/* Entities (rows) */}
      <div className="mt-3">
        <div className="flex items-center gap-2">
          <p className="text-xs font-medium text-foreground">
            Data rows ({edit.entities.length})
          </p>
          <Button
            size="sm"
            variant="ghost"
            onClick={addRow}
            disabled={edit.fields.length === 0}
          >
            <Plus className="mr-1 size-4" /> Add row
          </Button>
        </div>
        {edit.entities.length > 0 && (
          <div className="mt-1 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-muted-foreground">
                  <th className="p-1">name</th>
                  {edit.fields.map((f) => (
                    <th key={f.key} className="p-1">
                      {f.key || f.label}
                    </th>
                  ))}
                  <th className="p-1"></th>
                </tr>
              </thead>
              <tbody>
                {edit.entities.map((row, ri) => (
                  <tr key={ri} className="border-t">
                    <td className="p-1">
                      <Input
                        value={String(row.name ?? "")}
                        onChange={(e) => setCell(ri, "name", e.target.value)}
                        className="h-8 min-w-[8rem]"
                      />
                    </td>
                    {edit.fields.map((f) => (
                      <td key={f.key} className="p-1">
                        <Input
                          value={String(row[f.key] ?? "")}
                          onChange={(e) => setCell(ri, f.key, e.target.value)}
                          className="h-8 min-w-[7rem]"
                        />
                      </td>
                    ))}
                    <td className="p-1">
                      <DeleteButton
                        onClick={() => removeRow(ri)}
                        title="Remove row"
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="mt-3 flex items-center gap-2">
        <Button size="sm" onClick={onSave} disabled={busy === "save"}>
          {busy === "save" ? (
            <Loader2 className="mr-1 size-4 animate-spin" />
          ) : (
            <Save className="mr-1 size-4" />
          )}
          Save subdomain
        </Button>
        <Button size="sm" variant="ghost" onClick={onClose}>
          Cancel
        </Button>
      </div>
    </div>
  );
}
