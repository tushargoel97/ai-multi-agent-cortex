import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";
import { ensureAgentsTable } from "@/lib/tool-tables";

interface AgentRow {
  id: string;
  name: string;
  kind: string;
  description: string;
  system_prompt: string;
  enabled: boolean;
  edited: boolean;
}

function parseJson<T>(value: string | undefined, fallback: T): T {
  if (!value) return fallback;
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

function slug(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 60);
}

export async function GET(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await ensureAgentsTable();

  const { rows: agents } = await query<AgentRow>(
    `SELECT id, name, kind, description, system_prompt, enabled, edited
       FROM agents ORDER BY kind, name`,
  );
  const { rows: grants } = await query<{ agent_name: string; tool_name: string }>(
    `SELECT agent_name, tool_name FROM agent_tools`,
  );
  const { rows: subRows } = await query<{
    agent_name: string;
    subagent_name: string;
  }>(`SELECT agent_name, subagent_name FROM agent_subagents`);
  const { rows: toolRows } = await query<{ name: string; enabled: boolean }>(
    `SELECT name, enabled FROM tools ORDER BY kind, name`,
  );
  const { rows: settings } = await query<{ key: string; value: string }>(
    `SELECT key, value FROM app_settings WHERE key = ANY($1)`,
    [["agent_defaults", "agent_tool_defaults"]],
  );

  const map: Record<string, string> = {};
  for (const r of settings) map[r.key] = r.value;
  const agentDefaults = parseJson<Record<string, { description: string; system_prompt: string }>>(
    map["agent_defaults"],
    {},
  );
  const toolDefaults = parseJson<Record<string, string[]>>(map["agent_tool_defaults"], {});

  const grantsByAgent: Record<string, string[]> = {};
  for (const g of grants) (grantsByAgent[g.agent_name] ??= []).push(g.tool_name);

  const subsByAgent: Record<string, string[]> = {};
  for (const s of subRows) (subsByAgent[s.agent_name] ??= []).push(s.subagent_name);

  const out = agents.map((a) => ({
    id: a.id,
    name: a.name,
    kind: a.kind,
    description: a.description,
    system_prompt: a.system_prompt,
    enabled: a.enabled,
    edited: a.edited,
    tools: grantsByAgent[a.name] ?? toolDefaults[a.name] ?? [],
    hasGrants: a.name in grantsByAgent,
    subagents: subsByAgent[a.name] ?? [],
    defaultPrompt: agentDefaults[a.name]?.system_prompt ?? "",
  }));

  return NextResponse.json({
    agents: out,
    tools: toolRows.filter((t) => t.enabled).map((t) => t.name),
  });
}

export async function POST(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await ensureAgentsTable();

  const body = (await req.json().catch(() => null)) ?? {};
  const name = slug(String(body.name ?? ""));
  const description = String(body.description ?? "").trim();
  const systemPrompt = String(body.system_prompt ?? "").trim();
  if (!name || !description || !systemPrompt) {
    return NextResponse.json(
      { error: "name, description and system prompt are all required" },
      { status: 400 },
    );
  }

  const { rows } = await query<{ id: string }>(
    `INSERT INTO agents (id, name, kind, description, system_prompt, enabled, edited)
       VALUES (gen_random_uuid(), $1, 'custom', $2, $3, true, true)
     ON CONFLICT (name) DO NOTHING
     RETURNING id`,
    [name, description, systemPrompt],
  );
  if (!rows.length) {
    return NextResponse.json({ error: `an agent named "${name}" already exists` }, { status: 409 });
  }
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      await query(
        `INSERT INTO agent_tools (id, agent_name, tool_name)
           VALUES (gen_random_uuid(), $1, $2)
         ON CONFLICT (agent_name, tool_name) DO NOTHING`,
        [name, String(t)],
      );
    }
  }
  return NextResponse.json({ ok: true, name });
}
