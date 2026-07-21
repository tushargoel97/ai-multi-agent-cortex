import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";

const VALID_KINDS = ["openai", "azure_openai", "anthropic", "google", "local"];

interface ProviderRow {
  id: string;
  name: string;
  kind: string;
  api_key: string;
  base_url: string | null;
  azure_endpoint: string | null;
  azure_api_version: string | null;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export async function GET(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  const { rows } = await query<ProviderRow>(
    "SELECT id, name, kind, api_key, base_url, azure_endpoint, azure_api_version, enabled, created_at, updated_at FROM llm_providers ORDER BY created_at",
  );
  return NextResponse.json(
    rows.map((r) => ({
      ...r,
      api_key_set: !!r.api_key,
      api_key: undefined,
    })),
  );
}

export async function POST(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  const body = await req.json();
  const { name, kind, api_key, base_url, azure_endpoint, azure_api_version } = body ?? {};
  if (!name || !kind) {
    return NextResponse.json({ error: "name and kind are required" }, { status: 400 });
  }
  if (!VALID_KINDS.includes(kind)) {
    return NextResponse.json(
      { error: `kind must be one of ${VALID_KINDS.join(", ")}` },
      { status: 400 },
    );
  }
  const { rows } = await query<{ id: string }>(
    `INSERT INTO llm_providers (id, name, kind, api_key, base_url, azure_endpoint, azure_api_version, enabled, created_at, updated_at)
     VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, true, now(), now())
     RETURNING id`,
    [
      name,
      kind,
      api_key ?? "",
      base_url ?? null,
      azure_endpoint ?? null,
      azure_api_version ?? null,
    ],
  );
  return NextResponse.json({ id: rows[0].id }, { status: 201 });
}
