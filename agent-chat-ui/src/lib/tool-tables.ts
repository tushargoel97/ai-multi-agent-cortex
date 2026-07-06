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
