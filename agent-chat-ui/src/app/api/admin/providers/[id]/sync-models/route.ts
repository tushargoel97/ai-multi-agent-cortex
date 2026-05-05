import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";

interface ProviderRow {
  id: string;
  name: string;
  kind: string;
  api_key: string;
  base_url: string | null;
  azure_endpoint: string | null;
  azure_api_version: string | null;
}

interface SyncedModel {
  model_id: string;
  display_name: string;
}

async function listOpenAIModels(p: ProviderRow): Promise<SyncedModel[]> {
  const base = p.base_url?.replace(/\/$/, "") || "https://api.openai.com/v1";
  const r = await fetch(`${base}/models`, {
    headers: { Authorization: `Bearer ${p.api_key}` },
  });
  if (!r.ok) throw new Error(`OpenAI list models failed: ${r.status}`);
  const data = (await r.json()) as { data: { id: string }[] };
  return data.data
    .filter(
      (m) =>
        /^(gpt-|o\d|chatgpt|text-|omni)/i.test(m.id) &&
        !/embed|whisper|tts|moderation|dall|image/i.test(m.id),
    )
    .map((m) => ({ model_id: m.id, display_name: m.id }));
}

async function listAnthropicModels(p: ProviderRow): Promise<SyncedModel[]> {
  const base = p.base_url?.replace(/\/$/, "") || "https://api.anthropic.com/v1";
  const r = await fetch(`${base}/models`, {
    headers: {
      "x-api-key": p.api_key,
      "anthropic-version": "2023-06-01",
    },
  });
  if (!r.ok) throw new Error(`Anthropic list models failed: ${r.status}`);
  const data = (await r.json()) as {
    data: { id: string; display_name?: string }[];
  };
  return data.data.map((m) => ({
    model_id: m.id,
    display_name: m.display_name || m.id,
  }));
}

async function listGoogleModels(p: ProviderRow): Promise<SyncedModel[]> {
  const base =
    p.base_url?.replace(/\/$/, "") ||
    "https://generativelanguage.googleapis.com/v1beta";
  const r = await fetch(`${base}/models?key=${encodeURIComponent(p.api_key)}`);
  if (!r.ok) throw new Error(`Google list models failed: ${r.status}`);
  const data = (await r.json()) as {
    models: {
      name: string;
      displayName?: string;
      supportedGenerationMethods?: string[];
    }[];
  };
  return data.models
    .filter((m) =>
      (m.supportedGenerationMethods || []).includes("generateContent"),
    )
    .map((m) => {
      const id = m.name.replace(/^models\//, "");
      return { model_id: id, display_name: m.displayName || id };
    });
}

async function listLocalModels(p: ProviderRow): Promise<SyncedModel[]> {
  const base = p.base_url?.replace(/\/$/, "");
  if (!base) throw new Error("Local provider has no base_url");
  const headers: Record<string, string> = {};
  if (p.api_key) headers.Authorization = `Bearer ${p.api_key}`;
  const r = await fetch(`${base}/models`, { headers });
  if (!r.ok) throw new Error(`Local list models failed: ${r.status}`);
  const data = (await r.json()) as { data: { id: string }[] };
  return data.data.map((m) => ({ model_id: m.id, display_name: m.id }));
}

export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  const { id } = await params;
  const { rows } = await query<ProviderRow>(
    `SELECT id, name, kind, api_key, base_url, azure_endpoint, azure_api_version
     FROM llm_providers WHERE id = $1`,
    [id],
  );
  if (!rows.length) {
    return NextResponse.json({ error: "Provider not found" }, { status: 404 });
  }
  const provider = rows[0];

  if (!provider.api_key && provider.kind !== "local") {
    return NextResponse.json(
      { error: "Provider has no API key set" },
      { status: 400 },
    );
  }

  let models: SyncedModel[];
  try {
    switch (provider.kind) {
      case "openai":
        models = await listOpenAIModels(provider);
        break;
      case "anthropic":
        models = await listAnthropicModels(provider);
        break;
      case "google":
        models = await listGoogleModels(provider);
        break;
      case "local":
        models = await listLocalModels(provider);
        break;
      case "azure_openai":
        return NextResponse.json(
          {
            error:
              "Azure OpenAI does not expose a generic list-models API; add deployments manually.",
          },
          { status: 400 },
        );
      default:
        return NextResponse.json(
          { error: `Unsupported provider kind: ${provider.kind}` },
          { status: 400 },
        );
    }
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Sync failed" },
      { status: 502 },
    );
  }

  let inserted = 0;
  let updated = 0;
  for (const m of models) {
    const r = await query<{ inserted: boolean }>(
      `INSERT INTO llm_models (id, provider_id, model_id, display_name, enabled, is_default, created_at)
       VALUES (gen_random_uuid(), $1, $2, $3, true, false, now())
       ON CONFLICT (provider_id, model_id) DO UPDATE
         SET display_name = EXCLUDED.display_name
       RETURNING (xmax = 0) AS inserted`,
      [provider.id, m.model_id, m.display_name],
    );
    if (r.rows[0]?.inserted) inserted++;
    else updated++;
  }

  return NextResponse.json({
    ok: true,
    total: models.length,
    inserted,
    updated,
  });
}
