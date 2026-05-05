import { NextResponse } from "next/server";
import { query } from "@/lib/db";

interface PublicModelRow {
  id: string;
  model_id: string;
  display_name: string;
  provider_name: string;
  provider_kind: string;
  is_default: boolean;
}

export async function GET() {
  try {
    const { rows } = await query<PublicModelRow>(
      `SELECT m.id, m.model_id, m.display_name, p.name AS provider_name,
              p.kind AS provider_kind, m.is_default
       FROM llm_models m JOIN llm_providers p ON m.provider_id = p.id
       WHERE m.enabled = true AND p.enabled = true
       ORDER BY m.is_default DESC, p.name, m.display_name`,
    );
    return NextResponse.json(rows);
  } catch (e) {
    console.error("[/api/models] DB error:", e);
    return NextResponse.json([], { status: 200 });
  }
}
