import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";
import { ensureToolTables } from "@/lib/tool-tables";

interface ToolRow {
  id: string;
  name: string;
  kind: string;
  description: string;
  enabled: boolean;
  config: unknown;
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

function parseJson<T>(value: string | undefined, fallback: T): T {
  if (!value) return fallback;
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

/** Everything the Tools admin panel needs in one call. */
export async function GET(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await ensureToolTables();

  const { rows: tools } = await query<ToolRow>(
    `SELECT id, name, kind, description, enabled, config, mcp_server_id
       FROM tools ORDER BY kind, name`,
  );
  const { rows: mcpServers } = await query<McpRow>(
    `SELECT id, name, transport, url, command, args, env, enabled, last_error
       FROM mcp_servers ORDER BY name`,
  );
  const { rows: grants } = await query<{ agent_name: string; tool_name: string }>(
    `SELECT agent_name, tool_name FROM agent_tools`,
  );
  const { rows: settings } = await query<{ key: string; value: string }>(
    `SELECT key, value FROM app_settings WHERE key = ANY($1)`,
    [["tool_catalog", "agent_tool_defaults"]],
  );

  const map: Record<string, string> = {};
  for (const r of settings) map[r.key] = r.value;
  const catalog = parseJson<CatalogEntry[]>(map["tool_catalog"], []);
  const defaults = parseJson<Record<string, string[]>>(
    map["agent_tool_defaults"],
    {},
  );

  const grantsByAgent: Record<string, string[]> = {};
  for (const g of grants) (grantsByAgent[g.agent_name] ??= []).push(g.tool_name);

  const agentNames = Array.from(
    new Set([...Object.keys(defaults), ...Object.keys(grantsByAgent)]),
  ).sort();
  const agents = agentNames.map((name) => ({
    name,
    defaultTools: defaults[name] ?? [],
    tools: grantsByAgent[name] ?? defaults[name] ?? [],
    customized: name in grantsByAgent,
  }));

  return NextResponse.json({ tools, mcpServers, catalog, agents });
}

/** Add (or re-enable) a prebuilt LangChain catalog tool. */
export async function POST(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await ensureToolTables();

  const body = (await req.json().catch(() => null)) ?? {};
  const catalog = String(body.catalog ?? "").trim();
  if (!catalog) {
    return NextResponse.json({ error: "catalog id required" }, { status: 400 });
  }
  const name = String(body.name ?? catalog).trim();
  const config = { catalog, config: body.config ?? {} };

  try {
    const { rows } = await query<{ id: string }>(
      `INSERT INTO tools (name, kind, description, enabled, config)
         VALUES ($1, 'langchain', $2, true, $3::jsonb)
       ON CONFLICT (name) DO UPDATE
         SET config = EXCLUDED.config, enabled = true,
             description = EXCLUDED.description
       RETURNING id`,
      [name, String(body.description ?? ""), JSON.stringify(config)],
    );
    return NextResponse.json({ ok: true, id: rows[0]?.id });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "insert failed" },
      { status: 500 },
    );
  }
}
