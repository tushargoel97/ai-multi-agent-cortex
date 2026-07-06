import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";
import { ensureToolTables } from "@/lib/tool-tables";

/** Enable/disable a tool or update a LangChain tool's config. */
export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await ensureToolTables();
  const { id } = await params;
  const body = (await req.json().catch(() => null)) ?? {};

  const fields: string[] = [];
  const vals: unknown[] = [];
  let i = 1;
  if (typeof body.enabled === "boolean") {
    fields.push(`enabled = $${i++}`);
    vals.push(body.enabled);
  }
  if (body.config !== undefined) {
    fields.push(`config = $${i++}::jsonb`);
    vals.push(JSON.stringify(body.config));
  }
  if (!fields.length) {
    return NextResponse.json({ error: "nothing to update" }, { status: 400 });
  }
  vals.push(id);
  await query(`UPDATE tools SET ${fields.join(", ")} WHERE id = $${i}`, vals);
  return NextResponse.json({ ok: true });
}

/** Remove a LangChain/MCP tool (built-ins can only be disabled). */
export async function DELETE(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await ensureToolTables();
  const { id } = await params;

  const { rows } = await query<{ kind: string; name: string }>(
    `SELECT kind, name FROM tools WHERE id = $1`,
    [id],
  );
  if (!rows.length) {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }
  if (rows[0].kind === "builtin") {
    return NextResponse.json(
      { error: "built-in tools can't be deleted — disable it instead" },
      { status: 400 },
    );
  }
  await query(`DELETE FROM tools WHERE id = $1`, [id]);
  await query(`DELETE FROM agent_tools WHERE tool_name = $1`, [rows[0].name]);
  return NextResponse.json({ ok: true });
}
