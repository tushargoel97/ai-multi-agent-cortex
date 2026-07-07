"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { getAdminToken } from "../token";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { DeleteButton } from "@/components/ui/delete-button";
import { Switch } from "@/components/ui/switch";
import {
  Bot,
  Plus,
  Loader2,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  RotateCcw,
} from "lucide-react";

interface AgentRow {
  id: string;
  name: string;
  kind: string;
  description: string;
  system_prompt: string;
  enabled: boolean;
  edited: boolean;
  tools: string[];
  hasGrants: boolean;
  subagents: string[];
  defaultPrompt: string;
}

interface Edit {
  name: string;
  system_prompt: string;
  description: string;
  tools: string[];
  subagents: string[];
}

export default function AgentsPanel() {
  const [agents, setAgents] = useState<AgentRow[]>([]);
  const [tools, setTools] = useState<string[]>([]);
  const [edits, setEdits] = useState<Record<string, Edit>>({});
  const [expanded, setExpanded] = useState<string | null>(null);
  const [openCard, setOpenCard] = useState<string>("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const confirm = useConfirm();

  const [showNew, setShowNew] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [newPrompt, setNewPrompt] = useState("");
  const [newTools, setNewTools] = useState<string[]>([]);

  const headers = useCallback(
    () => ({
      "Content-Type": "application/json",
      "X-Admin-Token": getAdminToken() || "",
    }),
    [],
  );

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const r = await fetch("/api/admin/agents", { headers: headers() });
      if (!r.ok) throw new Error(`load ${r.status}`);
      const data = await r.json();
      setAgents(data.agents ?? []);
      setTools(data.tools ?? []);
      const e: Record<string, Edit> = {};
      for (const a of data.agents ?? []) {
        e[a.name] = {
          name: a.name,
          system_prompt: a.system_prompt,
          description: a.description,
          tools: [...a.tools],
          subagents: [...(a.subagents ?? [])],
        };
      }
      setEdits(e);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [headers]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const dirty = (a: AgentRow) => {
    const e = edits[a.name];
    if (!e) return false;
    return (
      e.name !== a.name ||
      e.system_prompt !== a.system_prompt ||
      e.description !== a.description ||
      [...e.tools].sort().join(",") !== [...a.tools].sort().join(",") ||
      [...e.subagents].sort().join(",") !==
        [...(a.subagents ?? [])].sort().join(",")
    );
  };

  const patch = async (name: string, body: unknown, key = name) => {
    setBusy(key);
    setError(null);
    try {
      const r = await fetch(`/api/admin/agents/${encodeURIComponent(name)}`, {
        method: "PATCH",
        headers: headers(),
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error((await r.json()).error || `save ${r.status}`);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const slugify = (s: string) =>
    s
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 60);

  const saveAgent = async (a: AgentRow) => {
    const e = edits[a.name];
    const renamed =
      a.kind === "custom" && !!e.name.trim() && slugify(e.name) !== a.name;
    const body: Record<string, unknown> = {
      system_prompt: e.system_prompt,
      description: e.description,
      tools: e.tools,
      subagents: e.subagents,
    };
    if (renamed) body.new_name = e.name.trim();
    await patch(a.name, body);
    if (renamed) setExpanded(slugify(e.name));
  };

  const toggleEnabled = (a: AgentRow) =>
    patch(a.name, { enabled: !a.enabled }, `en-${a.name}`);

  const resetAgent = async (a: AgentRow) => {
    if (
      !(await confirm({
        title: `Reset "${a.name}" to default?`,
        description:
          "Restores the packaged system prompt and clears custom tool grants.",
        confirmText: "Reset",
      }))
    )
      return;
    await patch(a.name, { reset: true }, `reset-${a.name}`);
  };

  const deleteAgent = async (a: AgentRow) => {
    if (
      !(await confirm({
        title: `Delete agent "${a.name}"?`,
        description: "This custom agent will be removed permanently.",
      }))
    )
      return;
    setBusy(`del-${a.name}`);
    setError(null);
    try {
      const r = await fetch(`/api/admin/agents/${encodeURIComponent(a.name)}`, {
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

  const createAgent = async () => {
    if (!newName.trim() || !newDesc.trim() || !newPrompt.trim()) return;
    setBusy("create");
    setError(null);
    try {
      const r = await fetch("/api/admin/agents", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({
          name: newName.trim(),
          description: newDesc.trim(),
          system_prompt: newPrompt.trim(),
          tools: newTools,
        }),
      });
      if (!r.ok) throw new Error((await r.json()).error || `create ${r.status}`);
      setNewName("");
      setNewDesc("");
      setNewPrompt("");
      setNewTools([]);
      setShowNew(false);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const toggleEditTool = (name: string, tool: string) =>
    setEdits((prev) => {
      const cur = new Set(prev[name]?.tools ?? []);
      if (cur.has(tool)) cur.delete(tool);
      else cur.add(tool);
      return { ...prev, [name]: { ...prev[name], tools: [...cur] } };
    });

  const toggleEditSubagent = (name: string, sub: string) =>
    setEdits((prev) => {
      const cur = new Set(prev[name]?.subagents ?? []);
      if (cur.has(sub)) cur.delete(sub);
      else cur.add(sub);
      return { ...prev, [name]: { ...prev[name], subagents: [...cur] } };
    });

  const setEdit = (name: string, patchObj: Partial<Edit>) =>
    setEdits((prev) => ({ ...prev, [name]: { ...prev[name], ...patchObj } }));

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-lg font-semibold">Agents</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Edit each agent&apos;s system prompt, tool access, and subagents, or
            create custom agents. Custom agents auto-route via the router by
            their description — no restart. Changes apply on the next message.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => void refresh()}>
            <RefreshCw className="mr-1 size-4" /> Refresh
          </Button>
          <Button size="sm" onClick={() => setShowNew((s) => !s)}>
            <Plus className="mr-1 size-4" /> New agent
          </Button>
        </div>
      </div>

      {error && (
        <div className="rounded border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Create custom agent */}
      {showNew && (
        <div className="space-y-3 rounded-md border border-dashed p-4">
          <p className="text-sm font-medium">New custom agent</p>
          <div className="flex flex-wrap gap-2">
            <Input
              className="w-48"
              placeholder="name (e.g. translator)"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
            />
            <Input
              className="min-w-64 flex-1"
              placeholder="description — when should the router pick this agent?"
              value={newDesc}
              onChange={(e) => setNewDesc(e.target.value)}
            />
          </div>
          <textarea
            className="min-h-28 w-full rounded-md border px-3 py-2 font-mono text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring"
            placeholder="System prompt — how this agent should behave…"
            value={newPrompt}
            onChange={(e) => setNewPrompt(e.target.value)}
          />
          <div>
            <p className="mb-1 text-xs font-medium text-muted-foreground">
              Tools
            </p>
            <ul className="max-h-56 space-y-0.5 overflow-y-auto rounded-md border p-2">
              {tools.map((t) => (
                <li
                  key={t}
                  className="flex items-center justify-between gap-2 rounded px-2 py-1.5 hover:bg-muted/50"
                >
                  <span className="truncate font-mono text-xs text-muted-foreground">
                    {t}
                  </span>
                  <Switch
                    checked={newTools.includes(t)}
                    onCheckedChange={() =>
                      setNewTools((prev) =>
                        prev.includes(t)
                          ? prev.filter((x) => x !== t)
                          : [...prev, t],
                      )
                    }
                  />
                </li>
              ))}
              {tools.length === 0 && (
                <li className="px-2 py-1.5 text-xs text-muted-foreground/70">
                  No enabled tools.
                </li>
              )}
            </ul>
          </div>
          <div className="flex justify-end gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowNew(false)}
            >
              Cancel
            </Button>
            <Button
              size="sm"
              disabled={
                busy === "create" ||
                !newName.trim() ||
                !newDesc.trim() ||
                !newPrompt.trim()
              }
              onClick={() => void createAgent()}
            >
              {busy === "create" ? (
                <Loader2 className="mr-1 size-4 animate-spin" />
              ) : (
                <Plus className="mr-1 size-4" />
              )}
              Create
            </Button>
          </div>
        </div>
      )}

      {/* Agent list */}
      <ul className="space-y-2">
        {agents.map((a) => {
          const e = edits[a.name] ?? {
            name: a.name,
            system_prompt: a.system_prompt,
            description: a.description,
            tools: a.tools,
            subagents: a.subagents ?? [],
          };
          const open = expanded === a.name;
          return (
            <li key={a.id} className="rounded-md border">
              <div className="flex items-center gap-2 px-3 py-2">
                <button
                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                  onClick={() => setExpanded(open ? null : a.name)}
                >
                  {open ? (
                    <ChevronDown className="size-4 shrink-0 text-muted-foreground" />
                  ) : (
                    <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
                  )}
                  <span className="shrink-0 text-sm font-medium capitalize">
                    {a.name}
                  </span>
                  <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-[10px] uppercase text-muted-foreground">
                    {a.kind}
                  </span>
                  {a.edited && a.kind === "builtin" && (
                    <span className="shrink-0 rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-600">
                      edited
                    </span>
                  )}
                  <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground">
                    {a.description}
                  </span>
                </button>
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Switch
                    checked={a.enabled}
                    disabled={busy === `en-${a.name}`}
                    onCheckedChange={() => void toggleEnabled(a)}
                  />
                  Enabled
                </div>
                {a.kind === "custom" ? (
                  <DeleteButton
                    busy={busy === `del-${a.name}`}
                    onClick={() => void deleteAgent(a)}
                    title="Delete agent"
                  />
                ) : (
                  <Button
                    size="icon"
                    variant="ghost"
                    className="size-8"
                    disabled={busy === `reset-${a.name}`}
                    title="Reset to default"
                    onClick={() => void resetAgent(a)}
                  >
                    <RotateCcw className="size-4" />
                  </Button>
                )}
              </div>

              {open && (
                <div className="space-y-3 border-t px-3 py-3">
                  {a.kind === "custom" && (
                    <div>
                      <p className="mb-1 text-xs font-medium text-muted-foreground">
                        Name
                      </p>
                      <Input
                        value={e.name}
                        onChange={(ev) =>
                          setEdit(a.name, { name: ev.target.value })
                        }
                      />
                      <p className="mt-1 text-[11px] text-muted-foreground/70">
                        Renaming re-slugs (spaces/symbols → underscores) and
                        updates every tool grant + subagent link.
                      </p>
                    </div>
                  )}
                  <div>
                    <p className="mb-1 text-xs font-medium text-muted-foreground">
                      Description (routing hint)
                    </p>
                    <Input
                      value={e.description}
                      onChange={(ev) =>
                        setEdit(a.name, { description: ev.target.value })
                      }
                    />
                  </div>
                  <div>
                    <p className="mb-1 text-xs font-medium text-muted-foreground">
                      System prompt
                    </p>
                    <textarea
                      className="min-h-40 w-full rounded-md border px-3 py-2 font-mono text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      value={e.system_prompt}
                      onChange={(ev) =>
                        setEdit(a.name, { system_prompt: ev.target.value })
                      }
                    />
                  </div>
                  <div className="rounded-md border">
                    <button
                      type="button"
                      onClick={() =>
                        setOpenCard((c) =>
                          c === `${a.name}:tools` ? "" : `${a.name}:tools`,
                        )
                      }
                      className="flex w-full items-center justify-between px-3 py-2 text-left"
                    >
                      <span className="text-xs font-medium text-muted-foreground">
                        Tool access{" "}
                        <span className="text-muted-foreground/60">
                          ({e.tools.length}/{tools.length})
                        </span>
                      </span>
                      {openCard === `${a.name}:tools` ? (
                        <ChevronDown className="size-4 text-muted-foreground" />
                      ) : (
                        <ChevronRight className="size-4 text-muted-foreground" />
                      )}
                    </button>
                    {openCard === `${a.name}:tools` && (
                      <ul className="max-h-64 space-y-0.5 overflow-y-auto border-t p-2">
                        {tools.map((t) => (
                          <li
                            key={t}
                            className="flex items-center justify-between gap-2 rounded px-2 py-1.5 hover:bg-muted/50"
                          >
                            <span className="truncate font-mono text-xs text-muted-foreground">
                              {t}
                            </span>
                            <Switch
                              checked={e.tools.includes(t)}
                              onCheckedChange={() => toggleEditTool(a.name, t)}
                            />
                          </li>
                        ))}
                        {tools.length === 0 && (
                          <li className="px-2 py-1.5 text-xs text-muted-foreground/70">
                            No enabled tools.
                          </li>
                        )}
                      </ul>
                    )}
                  </div>
                  <div className="rounded-md border">
                    <button
                      type="button"
                      onClick={() =>
                        setOpenCard((c) =>
                          c === `${a.name}:subs` ? "" : `${a.name}:subs`,
                        )
                      }
                      className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left"
                    >
                      <span className="min-w-0 text-xs font-medium text-muted-foreground">
                        Subagents{" "}
                        <span className="text-muted-foreground/60">
                          ({e.subagents.length})
                        </span>
                        <span className="ml-1 font-normal text-muted-foreground/60">
                          — delegated subtasks; shared memory read-only
                        </span>
                      </span>
                      {openCard === `${a.name}:subs` ? (
                        <ChevronDown className="size-4 shrink-0 text-muted-foreground" />
                      ) : (
                        <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
                      )}
                    </button>
                    {openCard === `${a.name}:subs` && (
                      <ul className="max-h-64 space-y-0.5 overflow-y-auto border-t p-2">
                        {agents
                          .filter((x) => x.name !== a.name && x.enabled)
                          .map((x) => (
                            <li
                              key={x.id}
                              className="flex items-center justify-between gap-2 rounded px-2 py-1.5 hover:bg-muted/50"
                            >
                              <span className="min-w-0">
                                <span className="font-mono text-xs capitalize text-foreground">
                                  {x.name}
                                </span>
                                {x.description && (
                                  <span className="block truncate text-[11px] text-muted-foreground">
                                    {x.description}
                                  </span>
                                )}
                              </span>
                              <Switch
                                checked={e.subagents.includes(x.name)}
                                onCheckedChange={() =>
                                  toggleEditSubagent(a.name, x.name)
                                }
                              />
                            </li>
                          ))}
                        {agents.filter((x) => x.name !== a.name && x.enabled)
                          .length === 0 && (
                          <li className="px-2 py-1.5 text-xs text-muted-foreground/70">
                            No other agents to delegate to.
                          </li>
                        )}
                      </ul>
                    )}
                  </div>
                  <div className="flex justify-end">
                    <Button
                      size="sm"
                      disabled={busy === a.name || !dirty(a)}
                      onClick={() => void saveAgent(a)}
                    >
                      {busy === a.name ? (
                        <Loader2 className="mr-1 size-4 animate-spin" />
                      ) : null}
                      Save
                    </Button>
                  </div>
                </div>
              )}
            </li>
          );
        })}
        {agents.length === 0 && (
          <li className="rounded-md border px-3 py-4 text-sm text-muted-foreground">
            <Bot className="mr-1 inline size-4" /> Agents appear after the
            langgraph server starts (it seeds the built-ins).
          </li>
        )}
      </ul>
    </div>
  );
}
