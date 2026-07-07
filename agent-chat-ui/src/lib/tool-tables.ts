import { query } from "@/lib/db";

let ensured = false;

/** Create the tool-control tables if they don't exist yet (idempotent).
 * Mirrors cortex/db/models/tool.py so the admin panel works before the
 * langgraph server ever runs its own create_all. */
export async function ensureToolTables(): Promise<void> {
  if (ensured) return;
  await query(`
    CREATE TABLE IF NOT EXISTS app_settings (
      key varchar(100) PRIMARY KEY,
      value text NOT NULL
    )`);
  await query(`
    CREATE TABLE IF NOT EXISTS mcp_servers (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      name varchar(80) NOT NULL UNIQUE,
      transport varchar(20) NOT NULL DEFAULT 'streamable_http',
      url text,
      command text,
      args jsonb NOT NULL DEFAULT '[]'::jsonb,
      env jsonb NOT NULL DEFAULT '{}'::jsonb,
      enabled boolean NOT NULL DEFAULT true,
      last_error text,
      created_at timestamptz NOT NULL DEFAULT now()
    )`);
  await query(`
    CREATE TABLE IF NOT EXISTS tools (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      name varchar(120) NOT NULL UNIQUE,
      kind varchar(20) NOT NULL DEFAULT 'builtin',
      description text NOT NULL DEFAULT '',
      enabled boolean NOT NULL DEFAULT true,
      config jsonb NOT NULL DEFAULT '{}'::jsonb,
      mcp_server_id uuid REFERENCES mcp_servers(id) ON DELETE CASCADE,
      created_at timestamptz NOT NULL DEFAULT now()
    )`);
  await query(`
    CREATE TABLE IF NOT EXISTS agent_tools (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      agent_name varchar(60) NOT NULL,
      tool_name varchar(120) NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      CONSTRAINT uq_agent_tool UNIQUE (agent_name, tool_name)
    )`);
  ensured = true;
}

const SUPPRESSED_KEY = "suppressed_tools";

/** Names of tools the admin deleted, never re-seeded, bound, or granted. */
export async function getSuppressedTools(): Promise<string[]> {
  const { rows } = await query<{ value: string }>(
    `SELECT value FROM app_settings WHERE key = $1`,
    [SUPPRESSED_KEY],
  );
  if (!rows.length) return [];
  try {
    const v = JSON.parse(rows[0].value);
    return Array.isArray(v) ? v.map((x) => String(x)) : [];
  } catch {
    return [];
  }
}

export async function setSuppressedTools(names: string[]): Promise<void> {
  const unique = Array.from(new Set(names));
  await query(
    `INSERT INTO app_settings (key, value) VALUES ($1, $2)
     ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value`,
    [SUPPRESSED_KEY, JSON.stringify(unique)],
  );
}

let agentsEnsured = false;

/** Create the agents table if missing (mirrors cortex/db/models/agent.py). */
export async function ensureAgentsTable(): Promise<void> {
  if (agentsEnsured) return;
  await ensureToolTables();
  await query(`
    CREATE TABLE IF NOT EXISTS agents (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      name varchar(60) NOT NULL UNIQUE,
      kind varchar(20) NOT NULL DEFAULT 'custom',
      description text NOT NULL DEFAULT '',
      system_prompt text NOT NULL DEFAULT '',
      enabled boolean NOT NULL DEFAULT true,
      edited boolean NOT NULL DEFAULT false,
      created_at timestamptz NOT NULL DEFAULT now()
    )`);
  await query(`
    CREATE TABLE IF NOT EXISTS agent_subagents (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      agent_name varchar(60) NOT NULL,
      subagent_name varchar(60) NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      CONSTRAINT uq_agent_subagent UNIQUE (agent_name, subagent_name)
    )`);
  agentsEnsured = true;
}
