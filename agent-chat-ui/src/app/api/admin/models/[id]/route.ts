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

  if (body.is_default === true) {
    await query("UPDATE llm_models SET is_default = false WHERE is_default = true");
  }

  for (const k of ["model_id", "display_name", "enabled", "is_default"]) {
    if (k in body) {
      fields.push(`${k} = $${i++}`);
      values.push(body[k]);
    }
  }
  if (!fields.length) {
    return NextResponse.json({ error: "no fields" }, { status: 400 });
  }
  values.push(id);
  await query(
    `UPDATE llm_models SET ${fields.join(", ")} WHERE id = $${i}`,
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
  await query("DELETE FROM llm_models WHERE id = $1", [id]);
  return NextResponse.json({ ok: true });
}
