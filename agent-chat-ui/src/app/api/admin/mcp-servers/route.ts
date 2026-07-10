import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";
import { ensureToolTables } from "@/lib/tool-tables";

/** Register (or update) an external MCP server whose tools become grantable. */
export async function POST(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await ensureToolTables();

  const body = (await req.json().catch(() => null)) ?? {};
  const name = String(body.name ?? "").trim();
  const transport = String(body.transport ?? "streamable_http").trim();
  if (!name) {
    return NextResponse.json({ error: "name required" }, { status: 400 });
  }
  if ((transport === "streamable_http" || transport === "sse") && !body.url) {
    return NextResponse.json({ error: "url required for http transports" }, { status: 400 });
  }
  if (transport === "stdio" && !body.command) {
    return NextResponse.json({ error: "command required for stdio transport" }, { status: 400 });
  }

  try {
    const { rows } = await query<{ id: string }>(
      `INSERT INTO mcp_servers (id, name, transport, url, command, args, env, enabled)
         VALUES (gen_random_uuid(), $1, $2, $3, $4, $5::jsonb, $6::jsonb, true)
       ON CONFLICT (name) DO UPDATE
         SET transport = EXCLUDED.transport, url = EXCLUDED.url,
             command = EXCLUDED.command, args = EXCLUDED.args, env = EXCLUDED.env
       RETURNING id`,
      [
        name,
        transport,
        body.url ?? null,
        body.command ?? null,
        JSON.stringify(Array.isArray(body.args) ? body.args : []),
        JSON.stringify(body.env ?? {}),
      ],
    );
    return NextResponse.json({ ok: true, id: rows[0]?.id });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "insert failed" },
      { status: 500 },
    );
  }
}
