import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";
import { ensureToolTables } from "@/lib/tool-tables";

/** Enable/disable or edit an MCP server. */
export async function PATCH(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await ensureToolTables();
  const { id } = await params;
  const body = (await req.json().catch(() => null)) ?? {};

  const fields: string[] = [];
  const vals: unknown[] = [];
  let i = 1;
  for (const key of ["transport", "url", "command"] as const) {
    if (body[key] !== undefined) {
      fields.push(`${key} = $${i++}`);
      vals.push(body[key]);
    }
  }
  if (body.args !== undefined) {
    fields.push(`args = $${i++}::jsonb`);
    vals.push(JSON.stringify(Array.isArray(body.args) ? body.args : []));
  }
  if (body.env !== undefined) {
    fields.push(`env = $${i++}::jsonb`);
    vals.push(JSON.stringify(body.env ?? {}));
  }
  if (typeof body.enabled === "boolean") {
    fields.push(`enabled = $${i++}`);
    vals.push(body.enabled);
  }
  if (!fields.length) {
    return NextResponse.json({ error: "nothing to update" }, { status: 400 });
  }
  vals.push(id);
  await query(`UPDATE mcp_servers SET ${fields.join(", ")} WHERE id = $${i}`, vals);
  return NextResponse.json({ ok: true });
}

/** Remove an MCP server (its discovered tool rows cascade-delete). */
export async function DELETE(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await ensureToolTables();
  const { id } = await params;
  await query(`DELETE FROM mcp_servers WHERE id = $1`, [id]);
  return NextResponse.json({ ok: true });
}
