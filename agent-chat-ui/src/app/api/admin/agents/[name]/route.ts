import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";
import { ensureToolTables } from "@/lib/tool-tables";

/** Set an agent's tool grants (replaces the YAML whitelist for that agent).
 * An empty/absent list reverts the agent to its YAML default. */
export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ name: string }> },
) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await ensureToolTables();
  const { name } = await params;
  const body = (await req.json().catch(() => null)) ?? {};
  const tools = Array.isArray(body.tools)
    ? Array.from(new Set(body.tools.map((t: unknown) => String(t))))
    : null;
  if (tools === null) {
    return NextResponse.json({ error: "tools[] required" }, { status: 400 });
  }

  await query(`DELETE FROM agent_tools WHERE agent_name = $1`, [name]);
  for (const tool of tools) {
    await query(
      `INSERT INTO agent_tools (agent_name, tool_name) VALUES ($1, $2)
       ON CONFLICT (agent_name, tool_name) DO NOTHING`,
      [name, tool],
    );
  }
  return NextResponse.json({ ok: true, count: tools.length });
}

/** Reset an agent's tools back to its YAML default (clear grants). */
export async function DELETE(
  req: Request,
  { params }: { params: Promise<{ name: string }> },
) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await ensureToolTables();
  const { name } = await params;
  await query(`DELETE FROM agent_tools WHERE agent_name = $1`, [name]);
  return NextResponse.json({ ok: true });
}
