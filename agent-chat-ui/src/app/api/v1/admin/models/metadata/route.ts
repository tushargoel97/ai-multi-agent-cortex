import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";

/** Cross-provider model metadata for sorting the registry: release date, cost,
 *  context, reasoning — sourced from the open models.dev catalog (75+
 *  providers, community-maintained TOML → api.json). Capability is a derived
 *  0–100 score, not a benchmark: reasoning support, price tier (providers
 *  price their strongest models highest), context size, and recency. */

const CATALOG_URL = "https://models.dev/api.json";
const TTL_MS = 24 * 60 * 60 * 1000;

interface Meta {
  release_date?: string;
  last_updated?: string;
  input_cost?: number;
  output_cost?: number;
  context?: number;
  reasoning?: boolean;
  tool_call?: boolean;
  capability?: number;
}

let catalogCache: { index: Map<string, Meta>; at: number } | null = null;

// Prefer first-party entries when the same model id appears under resellers.
const PRIMARY = ["anthropic", "openai", "google", "google-vertex", "mistral", "xai", "deepseek"];

function normalize(id: string): string {
  return id
    .toLowerCase()
    .replace(/-latest$/, "")
    .replace(/[-.](\d{8}|\d{4}-\d{2}-\d{2})$/, "");
}

function toMeta(model: Record<string, any>): Meta {
  const cost = model?.cost ?? {};
  const limit = model?.limit ?? {};
  return {
    release_date: model?.release_date || undefined,
    last_updated: model?.last_updated || undefined,
    input_cost: typeof cost.input === "number" ? cost.input : undefined,
    output_cost: typeof cost.output === "number" ? cost.output : undefined,
    context: typeof limit.context === "number" ? limit.context : undefined,
    reasoning: typeof model?.reasoning === "boolean" ? model.reasoning : undefined,
    tool_call: typeof model?.tool_call === "boolean" ? model.tool_call : undefined,
  };
}

async function catalogIndex(): Promise<Map<string, Meta>> {
  if (catalogCache && Date.now() - catalogCache.at < TTL_MS) return catalogCache.index;
  try {
    // models.dev 403s default library user agents; a browser-ish UA is accepted.
    const res = await fetch(CATALOG_URL, {
      signal: AbortSignal.timeout(10_000),
      headers: { "User-Agent": "Mozilla/5.0 (compatible; cortex-admin/1.0)" },
    });
    if (!res.ok) throw new Error(`catalog ${res.status}`);
    const data: Record<string, any> = await res.json();
    const index = new Map<string, Meta>();
    const providerIds = Object.keys(data).sort(
      (a, b) =>
        (PRIMARY.includes(a) ? PRIMARY.indexOf(a) : PRIMARY.length) -
        (PRIMARY.includes(b) ? PRIMARY.indexOf(b) : PRIMARY.length),
    );
    for (const pid of providerIds) {
      for (const [mid, model] of Object.entries(data[pid]?.models ?? {})) {
        const key = normalize(mid);
        if (!index.has(key)) index.set(key, toMeta(model as Record<string, any>));
      }
    }
    catalogCache = { index, at: Date.now() };
    return index;
  } catch {
    return catalogCache?.index ?? new Map();
  }
}

function lookup(index: Map<string, Meta>, modelId: string): Meta | undefined {
  const key = normalize(modelId);
  return index.get(key) ?? index.get(key.replace(/-(\d{5,})$/, ""));
}

function scoreCapability(matched: Record<string, Meta>): void {
  const metas = Object.values(matched);
  if (!metas.length) return;
  const logCost = (m: Meta) => Math.log1p(m.output_cost ?? 0);
  const logCtx = (m: Meta) => Math.log1p(m.context ?? 0);
  const time = (m: Meta) => (m.release_date ? Date.parse(m.release_date) || 0 : 0);
  const span = (values: number[]) => {
    const min = Math.min(...values);
    const range = Math.max(...values) - min;
    return (v: number) => (range > 0 ? (v - min) / range : 0.5);
  };
  const nCost = span(metas.map(logCost));
  const nCtx = span(metas.map(logCtx));
  const nTime = span(metas.map(time));
  // Recency outweighs price so a dated-but-expensive model (o1-pro) can't
  // outrank current flagships; price still separates tiers within a generation.
  for (const meta of metas) {
    meta.capability = Math.round(
      (meta.reasoning ? 30 : 0) +
        25 * nCost(logCost(meta)) +
        10 * nCtx(logCtx(meta)) +
        35 * nTime(time(meta)),
    );
  }
}

export async function GET(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  const { rows } = await query<{ model_id: string }>("SELECT DISTINCT model_id FROM llm_models");
  const index = await catalogIndex();
  const metadata: Record<string, Meta> = {};
  for (const { model_id } of rows) {
    const meta = lookup(index, model_id);
    if (meta) metadata[model_id] = { ...meta };
  }
  scoreCapability(metadata);
  return NextResponse.json({ metadata, source: "models.dev" });
}
