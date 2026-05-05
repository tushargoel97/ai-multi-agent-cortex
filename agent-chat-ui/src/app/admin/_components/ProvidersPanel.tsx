"use client";

import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { getAdminToken } from "../token";
import { toast } from "sonner";
import { RefreshCw } from "lucide-react";

interface Provider {
  id: string;
  name: string;
  kind: string;
  api_key_set: boolean;
  base_url: string | null;
  azure_endpoint: string | null;
  azure_api_version: string | null;
  enabled: boolean;
}

const KINDS = ["openai", "azure_openai", "anthropic", "google", "local"];

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

export default function ProvidersPanel({
  onChanged,
}: {
  onChanged?: () => void;
}) {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [loading, setLoading] = useState(true);
  const [form, setForm] = useState({
    name: "",
    kind: "openai",
    api_key: "",
    base_url: "",
    azure_endpoint: "",
    azure_api_version: "",
  });

  async function load() {
    setLoading(true);
    const r = await adminFetch("/api/admin/providers");
    if (!r.ok) {
      toast.error("Failed to load providers");
      setLoading(false);
      return;
    }
    setProviders(await r.json());
    setLoading(false);
  }

  useEffect(() => {
    load();
  }, []);

  async function create(e: React.FormEvent) {
    e.preventDefault();
    const r = await adminFetch("/api/admin/providers", {
      method: "POST",
      body: JSON.stringify({
        ...form,
        base_url: form.base_url || null,
        azure_endpoint: form.azure_endpoint || null,
        azure_api_version: form.azure_api_version || null,
      }),
    });
    if (!r.ok) {
      toast.error("Failed to create provider");
      return;
    }
    setForm({
      name: "",
      kind: "openai",
      api_key: "",
      base_url: "",
      azure_endpoint: "",
      azure_api_version: "",
    });
    load();
    onChanged?.();
  }

  async function patchProvider(id: string, body: Record<string, unknown>) {
    const r = await adminFetch(`/api/admin/providers/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    });
    if (!r.ok) toast.error("Update failed");
    else {
      load();
      onChanged?.();
    }
  }

  async function syncModels(id: string) {
    const t = toast.loading("Syncing models from provider…");
    const r = await adminFetch(`/api/admin/providers/${id}/sync-models`, {
      method: "POST",
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      toast.error(data.error || "Sync failed", { id: t });
      return;
    }
    toast.success(
      `Synced ${data.total} models (${data.inserted} new, ${data.updated} updated)`,
      { id: t },
    );
    onChanged?.();
  }

  async function deleteProvider(id: string) {
    if (!confirm("Delete this provider and all its models?")) return;
    const r = await adminFetch(`/api/admin/providers/${id}`, {
      method: "DELETE",
    });
    if (!r.ok) toast.error("Delete failed");
    else {
      load();
      onChanged?.();
    }
  }

  async function syncAll() {
    const eligible = providers.filter((p) => p.kind !== "azure_openai");
    if (eligible.length === 0) {
      toast.error("No providers to sync");
      return;
    }
    const t = toast.loading(`Syncing ${eligible.length} provider(s)…`);
    let inserted = 0;
    let updated = 0;
    let failed = 0;
    const failures: string[] = [];
    for (const p of eligible) {
      const r = await adminFetch(`/api/admin/providers/${p.id}/sync-models`, {
        method: "POST",
      });
      const data = await r.json().catch(() => ({}));
      if (r.ok) {
        inserted += data.inserted || 0;
        updated += data.updated || 0;
      } else {
        failed++;
        failures.push(`${p.name}: ${data.error || r.status}`);
      }
    }
    if (failed === eligible.length) {
      toast.error(`All syncs failed — ${failures[0]}`, { id: t });
    } else {
      toast.success(
        `Synced — ${inserted} new, ${updated} updated${failed ? `, ${failed} failed (${failures.join("; ")})` : ""}`,
        { id: t },
      );
    }
    onChanged?.();
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold">LLM Providers</h2>
          <p className="text-muted-foreground text-sm">
            Add a provider, set its API key, then click <em>Sync models</em> to
            pull the latest list directly from the provider.
          </p>
        </div>
        <Button
          onClick={syncAll}
          disabled={loading || providers.length === 0}
          className="shrink-0"
        >
          <RefreshCw className="mr-2 size-4" />
          Sync all providers
        </Button>
      </div>

      {!loading &&
        providers.length > 0 &&
        providers.filter(
          (p) => !p.api_key_set && p.kind !== "local" && p.kind !== "azure_openai",
        ).length > 0 && (
          <div className="rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
            <strong>API keys missing</strong> on{" "}
            {
              providers.filter(
                (p) =>
                  !p.api_key_set &&
                  p.kind !== "local" &&
                  p.kind !== "azure_openai",
              ).length
            }{" "}
            provider(s). Click <em>Not set — click to add</em> in the API Key
            column below to add your key, then use <em>Sync models</em> to pull
            the latest list.
          </div>
        )}

      <form
        onSubmit={create}
        className="grid grid-cols-1 md:grid-cols-2 gap-4 rounded-lg border p-6 bg-muted/30"
      >
        <div className="md:col-span-2">
          <h3 className="font-semibold mb-2">Add new provider</h3>
        </div>
        <div className="flex flex-col gap-2">
          <Label>Name</Label>
          <Input
            placeholder="My OpenAI"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            required
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label>Kind</Label>
          <select
            value={form.kind}
            onChange={(e) => setForm({ ...form, kind: e.target.value })}
            className="border rounded-md h-9 px-3 bg-background"
          >
            {KINDS.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </div>
        <div className="flex flex-col gap-2">
          <Label>API Key</Label>
          <Input
            type="password"
            value={form.api_key}
            onChange={(e) => setForm({ ...form, api_key: e.target.value })}
            placeholder={form.kind === "local" ? "Optional" : "Required"}
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label>Base URL</Label>
          <Input
            value={form.base_url}
            onChange={(e) => setForm({ ...form, base_url: e.target.value })}
            placeholder={
              form.kind === "local"
                ? "http://ai:8100/v1"
                : "Optional override"
            }
          />
        </div>
        {form.kind === "azure_openai" && (
          <>
            <div className="flex flex-col gap-2">
              <Label>Azure Endpoint</Label>
              <Input
                value={form.azure_endpoint}
                onChange={(e) =>
                  setForm({ ...form, azure_endpoint: e.target.value })
                }
                placeholder="https://<resource>.openai.azure.com"
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label>API Version</Label>
              <Input
                value={form.azure_api_version}
                onChange={(e) =>
                  setForm({ ...form, azure_api_version: e.target.value })
                }
                placeholder="2024-12-01-preview"
              />
            </div>
          </>
        )}
        <div className="md:col-span-2">
          <Button type="submit">Add Provider</Button>
        </div>
      </form>

      <div className="rounded-lg border bg-white">
        <table className="w-full text-sm">
          <thead className="bg-muted/50 text-left">
            <tr>
              <th className="p-3">Name</th>
              <th className="p-3">Kind</th>
              <th className="p-3">API Key</th>
              <th className="p-3">Base URL</th>
              <th className="p-3">Enabled</th>
              <th className="p-3"></th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={6} className="p-3 text-muted-foreground">
                  Loading…
                </td>
              </tr>
            )}
            {!loading && providers.length === 0 && (
              <tr>
                <td colSpan={6} className="p-3 text-muted-foreground">
                  No providers configured yet.
                </td>
              </tr>
            )}
            {providers.map((p) => (
              <tr key={p.id} className="border-t">
                <td className="p-3 font-medium">{p.name}</td>
                <td className="p-3">
                  <span className="rounded bg-slate-100 px-2 py-0.5 text-xs">
                    {p.kind}
                  </span>
                </td>
                <td className="p-3">
                  <SetKey
                    onSave={(key) => patchProvider(p.id, { api_key: key })}
                    isSet={p.api_key_set}
                  />
                </td>
                <td className="p-3 text-xs text-muted-foreground">
                  {p.base_url || "—"}
                </td>
                <td className="p-3">
                  <Switch
                    checked={p.enabled}
                    onCheckedChange={(v) =>
                      patchProvider(p.id, { enabled: v })
                    }
                  />
                </td>
                <td className="p-3 text-right">
                  <div className="flex items-center justify-end gap-2">
                    {p.kind !== "azure_openai" && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => syncModels(p.id)}
                        title="Fetch latest models from this provider's API"
                      >
                        <RefreshCw className="mr-1 size-3.5" />
                        Sync models
                      </Button>
                    )}
                    <button
                      onClick={() => deleteProvider(p.id)}
                      className="text-xs text-rose-500 hover:underline"
                    >
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SetKey({
  onSave,
  isSet,
}: {
  onSave: (key: string) => void;
  isSet: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState("");
  if (!editing) {
    return (
      <button
        onClick={() => setEditing(true)}
        className="text-xs text-muted-foreground hover:underline"
      >
        {isSet ? "✓ Set — click to update" : "Not set — click to add"}
      </button>
    );
  }
  return (
    <div className="flex gap-2">
      <Input
        type="password"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="sk-..."
        className="h-7 text-xs"
      />
      <Button
        size="sm"
        onClick={() => {
          onSave(value);
          setValue("");
          setEditing(false);
        }}
      >
        Save
      </Button>
      <Button
        size="sm"
        variant="ghost"
        onClick={() => setEditing(false)}
      >
        Cancel
      </Button>
    </div>
  );
}
