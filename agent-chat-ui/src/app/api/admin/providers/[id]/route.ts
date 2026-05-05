import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";

export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  const { id } = await params;
  const body = await req.json();
  const fields: string[] = [];
  const values: unknown[] = [];
  let i = 1;
  for (const k of [
    "name",
    "kind",
    "api_key",
    "base_url",
    "azure_endpoint",
    "azure_api_version",
    "enabled",
  ]) {
    if (k in body) {
      fields.push(`${k} = $${i++}`);
      values.push(body[k]);
    }
  }
  if (!fields.length) {
    return NextResponse.json({ error: "no fields to update" }, { status: 400 });
  }
  fields.push(`updated_at = now()`);
  values.push(id);
  await query(
    `UPDATE llm_providers SET ${fields.join(", ")} WHERE id = $${i}`,
    values,
  );
  return NextResponse.json({ ok: true });
}

export async function DELETE(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  const { id } = await params;
  await query("DELETE FROM llm_providers WHERE id = $1", [id]);
  return NextResponse.json({ ok: true });
}
