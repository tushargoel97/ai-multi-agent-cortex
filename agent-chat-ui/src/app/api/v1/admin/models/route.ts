import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";

interface ModelRow {
  id: string;
  provider_id: string;
  provider_name: string;
  provider_kind: string;
  model_id: string;
  display_name: string;
  enabled: boolean;
  is_default: boolean;
  description: string | null;
}

export async function GET(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  const { rows } = await query<ModelRow>(
    `SELECT m.id, m.provider_id, p.name AS provider_name, p.kind AS provider_kind,
            m.model_id, m.display_name, m.enabled, m.is_default,
            m.description
     FROM llm_models m JOIN llm_providers p ON m.provider_id = p.id
     ORDER BY p.name, m.display_name`,
  );
  return NextResponse.json(rows);
}

export async function POST(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  const body = await req.json();
  const { provider_id, model_id, display_name, is_default, description } = body ?? {};
  if (!provider_id || !model_id || !display_name) {
    return NextResponse.json(
      { error: "provider_id, model_id, display_name required" },
      { status: 400 },
    );
  }
  if (is_default) {
    await query("UPDATE llm_models SET is_default = false WHERE is_default = true");
  }
  const { rows } = await query<{ id: string }>(
    `INSERT INTO llm_models (id, provider_id, model_id, display_name, enabled, is_default, description, created_at)
     VALUES (gen_random_uuid(), $1, $2, $3, true, $4, $5, now())
     ON CONFLICT (provider_id, model_id) DO UPDATE
       SET display_name = EXCLUDED.display_name, enabled = true,
           is_default = EXCLUDED.is_default,
           description = COALESCE($5, llm_models.description)
     RETURNING id`,
    [
      provider_id,
      model_id,
      display_name,
      !!is_default,
      typeof description === "string" && description.trim() ? description.trim() : null,
    ],
  );
  return NextResponse.json({ id: rows[0].id }, { status: 201 });
}
