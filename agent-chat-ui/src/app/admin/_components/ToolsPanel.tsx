"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { DeleteButton } from "@/components/ui/delete-button";
import { getAdminToken } from "../token";
import {
  Wrench,
  Plug,
  Trash2,
  Plus,
  Loader2,
  RefreshCw,
} from "lucide-react";

interface ToolRow {
  id: string;
  name: string;
  kind: string;
  description: string;
  enabled: boolean;
  config: Record<string, unknown>;
  mcp_server_id: string | null;
}

interface McpRow {
  id: string;
  name: string;
  transport: string;
  url: string | null;
  command: string | null;
  args: unknown;
  env: unknown;
  enabled: boolean;
  last_error: string | null;
}

interface CatalogEntry {
  id: string;
  label: string;
  description: string;
  config_fields: string[];
  available: boolean;
}

const KIND_LABEL: Record<string, string> = {
  builtin: "Built-in",
  langchain: "LangChain",
  mcp: "MCP",
};

export default function ToolsPanel() {
  const confirm = useConfirm();
  const [tools, setTools] = useState<ToolRow[]>([]);
  const [mcpServers, setMcpServers] = useState<McpRow[]>([]);
  const [catalog, setCatalog] = useState<CatalogEntry[]>([]);
  const [suppressed, setSuppressed] = useState<string[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Add-LangChain-tool form
  const [catalogId, setCatalogId] = useState("");
  const [catalogCfg, setCatalogCfg] = useState<Record<string, string>>({});

  // Add-MCP-server form
  const [mcpName, setMcpName] = useState("");
  const [mcpTransport, setMcpTransport] = useState("streamable_http");
  const [mcpUrl, setMcpUrl] = useState("");
  const [mcpCommand, setMcpCommand] = useState("");
  const [mcpArgs, setMcpArgs] = useState("");
  const [mcpHeaders, setMcpHeaders] = useState("");

  const headers = useCallback(() => {
    const t = getAdminToken();
    return { "Content-Type": "application/json", "X-Admin-Token": t || "" };
  }, []);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const res = await fetch("/api/admin/tools", { headers: headers() });
      if (!res.ok) throw new Error(`load ${res.status}`);
      const data = await res.json();
      setTools(data.tools ?? []);
      setMcpServers(data.mcpServers ?? []);
      setCatalog(data.catalog ?? []);
      setSuppressed(data.suppressed ?? []);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [headers]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const selectedCatalog = catalog.find((c) => c.id === catalogId);

  const toggleTool = async (t: ToolRow) => {
    setBusy(t.id);
    try {
      await fetch(`/api/admin/tools/${t.id}`, {
        method: "PATCH",
        headers: headers(),
        body: JSON.stringify({ enabled: !t.enabled }),
      });
      await refresh();
    } finally {
      setBusy(null);
    }
  };

  const deleteTool = async (t: ToolRow) => {
    if (
      !(await confirm({
        title: `Delete "${t.name}"?`,
        description:
          "It won't be re-added on restart — you can Restore it later.",
      }))
    )
      return;
    setBusy(t.id);
    setError(null);
    try {
      const r = await fetch(`/api/admin/tools/${t.id}`, {
        method: "DELETE",
        headers: headers(),
      });
      if (!r.ok) throw new Error((await r.json()).error || `delete ${r.status}`);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const restoreTool = async (name: string) => {
    setBusy(`restore-${name}`);
    setError(null);
    try {
      const r = await fetch("/api/admin/tools/restore", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({ name }),
      });
      if (!r.ok) throw new Error(`restore ${r.status}`);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const addCatalogTool = async () => {
    if (!catalogId) return;
    setBusy("add-catalog");
    setError(null);
    try {
      const r = await fetch("/api/admin/tools", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({
          catalog: catalogId,
          name: catalogId,
          description: selectedCatalog?.description ?? "",
          config: catalogCfg,
        }),
      });
      if (!r.ok) throw new Error((await r.json()).error || `add ${r.status}`);
      setCatalogId("");
      setCatalogCfg({});
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const addMcp = async () => {
    if (!mcpName.trim()) return;
    setBusy("add-mcp");
    setError(null);
    let env: Record<string, string> = {};
    if (mcpHeaders.trim()) {
      try {
        env = JSON.parse(mcpHeaders);
      } catch {
        setError("Headers/env must be valid JSON");
        setBusy(null);
        return;
      }
    }
    try {
      const r = await fetch("/api/admin/mcp-servers", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({
          name: mcpName.trim(),
          transport: mcpTransport,
          url: mcpUrl.trim() || null,
          command: mcpCommand.trim() || null,
          args: mcpArgs.trim() ? mcpArgs.trim().split(/\s+/) : [],
          env,
        }),
      });
      if (!r.ok) throw new Error((await r.json()).error || `add ${r.status}`);
      setMcpName("");
      setMcpUrl("");
      setMcpCommand("");
      setMcpArgs("");
      setMcpHeaders("");
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const toggleMcp = async (s: McpRow) => {
    setBusy(s.id);
    try {
      await fetch(`/api/admin/mcp-servers/${s.id}`, {
        method: "PATCH",
        headers: headers(),
        body: JSON.stringify({ enabled: !s.enabled }),
      });
      await refresh();
    } finally {
      setBusy(null);
    }
  };

  const deleteMcp = async (s: McpRow) => {
    if (
      !(await confirm({
        title: `Remove MCP server "${s.name}"?`,
        description: "Its discovered tools will be removed.",
      }))
    )
      return;
    setBusy(s.id);
    try {
      await fetch(`/api/admin/mcp-servers/${s.id}`, {
        method: "DELETE",
        headers: headers(),
      });
      await refresh();
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-lg font-semibold">Tools & MCP</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Manage built-in and third-party tools and register MCP servers.
            Grant tools to agents in the Agents tab. Changes apply on the next
            message.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => void refresh()}>
          <RefreshCw className="mr-1 size-4" /> Refresh
        </Button>
      </div>

      {error && (
        <div className="rounded border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Tools list */}
      <section className="space-y-2">
        <h3 className="flex items-center gap-2 text-sm font-medium">
          <Wrench className="size-4" /> Available tools ({tools.length})
        </h3>
        <ul className="divide-y rounded-md border">
          {tools.map((t) => (
            <li
              key={t.id}
              className="flex items-center justify-between gap-3 px-3 py-2"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="truncate font-mono text-xs text-foreground">
                    {t.name}
                  </span>
                  <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] uppercase text-muted-foreground">
                    {KIND_LABEL[t.kind] ?? t.kind}
                  </span>
                </div>
                {t.description && (
                  <p className="truncate text-[11px] text-muted-foreground">
                    {t.description}
                  </p>
                )}
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <label className="flex cursor-pointer items-center gap-1 text-xs text-muted-foreground">
                  <input
                    type="checkbox"
                    checked={t.enabled}
                    disabled={busy === t.id}
                    onChange={() => void toggleTool(t)}
                  />
                  Enabled
                </label>
                <DeleteButton
                  disabled={busy === t.id || t.enabled}
                  title={
                    t.enabled
                      ? "Disable this tool first (uncheck Enabled) to remove it from all agents, then delete"
                      : "Delete tool"
                  }
                  onClick={() => void deleteTool(t)}
                />
              </div>
            </li>
          ))}
          {tools.length === 0 && (
            <li className="px-3 py-4 text-sm text-muted-foreground">
              No tools yet — the langgraph server seeds built-ins on startup.
            </li>
          )}
        </ul>
      </section>

      {/* Removed tools */}
      {suppressed.length > 0 && (
        <section className="space-y-2">
          <h3 className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
            <Trash2 className="size-4" /> Removed tools ({suppressed.length})
          </h3>
          <ul className="flex flex-wrap gap-2">
            {suppressed.map((name) => (
              <li
                key={name}
                className="flex items-center gap-1 rounded border bg-muted/40 px-2 py-1"
              >
                <span className="font-mono text-xs text-muted-foreground">
                  {name}
                </span>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={busy === `restore-${name}`}
                  onClick={() => void restoreTool(name)}
                >
                  Restore
                </Button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Add LangChain tool */}
      <section className="space-y-2 rounded-md border border-dashed p-3">
        <h3 className="flex items-center gap-2 text-sm font-medium">
          <Plus className="size-4" /> Add a prebuilt LangChain tool
        </h3>
        <div className="flex flex-wrap items-end gap-2">
          <select
            className="h-9 rounded-md border bg-background px-2 text-sm"
            value={catalogId}
            onChange={(e) => {
              setCatalogId(e.target.value);
              setCatalogCfg({});
            }}
          >
            <option value="">Select a tool…</option>
            {catalog.map((c) => (
              <option key={c.id} value={c.id} disabled={!c.available}>
                {c.label}
                {c.available ? "" : " (package not installed)"}
              </option>
            ))}
          </select>
          {selectedCatalog?.config_fields.map((field) => (
            <Input
              key={field}
              className="h-9 w-48"
              placeholder={field}
              value={catalogCfg[field] ?? ""}
              onChange={(e) =>
                setCatalogCfg((p) => ({ ...p, [field]: e.target.value }))
              }
            />
          ))}
          <Button
            size="sm"
            disabled={!catalogId || busy === "add-catalog"}
            onClick={() => void addCatalogTool()}
          >
            {busy === "add-catalog" ? (
              <Loader2 className="mr-1 size-4 animate-spin" />
            ) : (
              <Plus className="mr-1 size-4" />
            )}
            Add
          </Button>
        </div>
        {selectedCatalog && (
          <p className="text-[11px] text-muted-foreground">
            {selectedCatalog.description}
          </p>
        )}
      </section>

      {/* MCP servers */}
      <section className="space-y-2">
        <h3 className="flex items-center gap-2 text-sm font-medium">
          <Plug className="size-4" /> MCP servers ({mcpServers.length})
        </h3>
        <ul className="divide-y rounded-md border">
          {mcpServers.map((s) => (
            <li
              key={s.id}
              className="flex items-center justify-between gap-3 px-3 py-2"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="truncate text-sm font-medium">{s.name}</span>
                  <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] uppercase text-muted-foreground">
                    {s.transport}
                  </span>
                </div>
                <p className="truncate font-mono text-[11px] text-muted-foreground">
                  {s.url || s.command || ""}
                </p>
                {s.last_error && (
                  <p className="truncate text-[11px] text-destructive">
                    {s.last_error}
                  </p>
                )}
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <label className="flex cursor-pointer items-center gap-1 text-xs text-muted-foreground">
                  <input
                    type="checkbox"
                    checked={s.enabled}
                    disabled={busy === s.id}
                    onChange={() => void toggleMcp(s)}
                  />
                  Enabled
                </label>
                <DeleteButton
                  disabled={busy === s.id}
                  onClick={() => void deleteMcp(s)}
                  title="Remove MCP server"
                />
              </div>
            </li>
          ))}
          {mcpServers.length === 0 && (
            <li className="px-3 py-4 text-sm text-muted-foreground">
              No MCP servers registered.
            </li>
          )}
        </ul>

        <div className="space-y-2 rounded-md border border-dashed p-3">
          <p className="text-sm font-medium">Register an MCP server</p>
          <div className="flex flex-wrap items-center gap-2">
            <Input
              className="h-9 w-40"
              placeholder="name"
              value={mcpName}
              onChange={(e) => setMcpName(e.target.value)}
            />
            <select
              className="h-9 rounded-md border bg-background px-2 text-sm"
              value={mcpTransport}
              onChange={(e) => setMcpTransport(e.target.value)}
            >
              <option value="streamable_http">streamable_http</option>
              <option value="sse">sse</option>
              <option value="stdio">stdio</option>
            </select>
            {mcpTransport === "stdio" ? (
              <>
                <Input
                  className="h-9 w-40"
                  placeholder="command"
                  value={mcpCommand}
                  onChange={(e) => setMcpCommand(e.target.value)}
                />
                <Input
                  className="h-9 w-48"
                  placeholder="args (space-separated)"
                  value={mcpArgs}
                  onChange={(e) => setMcpArgs(e.target.value)}
                />
              </>
            ) : (
              <Input
                className="h-9 w-72"
                placeholder="https://server/mcp"
                value={mcpUrl}
                onChange={(e) => setMcpUrl(e.target.value)}
              />
            )}
            <Input
              className="h-9 w-56"
              placeholder='headers/env JSON (optional)'
              value={mcpHeaders}
              onChange={(e) => setMcpHeaders(e.target.value)}
            />
            <Button
              size="sm"
              disabled={!mcpName.trim() || busy === "add-mcp"}
              onClick={() => void addMcp()}
            >
              {busy === "add-mcp" ? (
                <Loader2 className="mr-1 size-4 animate-spin" />
              ) : (
                <Plus className="mr-1 size-4" />
              )}
              Add
            </Button>
          </div>
          <p className="text-[11px] text-muted-foreground">
            Discovered MCP tools appear in the list above after the next
            langgraph restart, then can be granted to agents in the Agents tab.
          </p>
        </div>
      </section>
    </div>
  );
}
