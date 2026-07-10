"use client";

import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { DeleteButton } from "@/components/ui/delete-button";
import { getAdminToken } from "../token";
import { toast } from "sonner";
import { RefreshCw } from "lucide-react";
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
            <h3 className="text-sm font-semibold">✨ Auto mode profile</h3>
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

      <div className="bg-background/60 rounded-lg border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50 text-left">
            <tr>
              <th className="p-3">Display</th>
              <th className="p-3">Model ID</th>
              <th className="p-3">Provider</th>
              <th className="p-3">Default</th>
              <th className="p-3">Enabled</th>
              <th className="p-3"></th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.id} className="border-t">
                <td className="p-3 font-medium">{m.display_name}</td>
                <td className="p-3 font-mono text-xs">{m.model_id}</td>
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
            {models.length === 0 && (
              <tr>
                <td colSpan={6} className="text-muted-foreground p-3">
                  No models yet, add one above or sync from a provider.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
