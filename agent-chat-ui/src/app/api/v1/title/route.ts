import { NextResponse } from "next/server";
import { query } from "@/lib/db";

interface ModelRow {
  model_id: string;
  kind: string;
  api_key: string;
  base_url: string | null;
  azure_endpoint: string | null;
  azure_api_version: string | null;
}

const SYSTEM =
  "You write a short, specific title summarizing a chat from the user's first message. " +
  "Rules: 3-6 words, Title Case, no quotes, no trailing punctuation, no emojis. " +
  "Reply with ONLY the title.";

async function pickModel(): Promise<ModelRow | null> {
  const { rows } = await query<ModelRow>(
    `SELECT m.model_id, p.kind, p.api_key, p.base_url,
            p.azure_endpoint, p.azure_api_version
       FROM llm_models m JOIN llm_providers p ON m.provider_id = p.id
      WHERE m.enabled = true AND p.enabled = true AND p.kind <> 'local'
      ORDER BY
        CASE WHEN m.model_id ~* '(mini|haiku|flash|nano|small|lite|8b)'
             THEN 0 ELSE 1 END,
        m.is_default DESC,
        p.name
      LIMIT 1`,
  );
  return rows[0] ?? null;
}

function cleanTitle(raw: string): string {
  let t = (raw || "").replace(/\s+/g, " ").trim();
  t = t.replace(/^["'`\u201c\u201d\u2018\u2019]+|["'`\u201c\u201d\u2018\u2019]+$/g, "");
  t = t.replace(/^(title|chat)\s*[:\-\u2013\u2014]\s*/i, "");
  t = t.replace(/[.!?,;:]+$/g, "").trim();
  const words = t.split(" ");
  if (words.length > 8) t = words.slice(0, 8).join(" ");
  if (t.length > 64) t = t.slice(0, 64).trimEnd();
  return t;
}

async function generate(row: ModelRow, text: string): Promise<string> {
  const prompt = text.slice(0, 2000);

  if (row.kind === "anthropic") {
    const base = row.base_url?.replace(/\/$/, "") || "https://api.anthropic.com/v1";
    const r = await fetch(`${base}/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": row.api_key,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: row.model_id,
        max_tokens: 24,
        system: SYSTEM,
        messages: [{ role: "user", content: prompt }],
      }),
    });
    if (!r.ok) throw new Error(`anthropic ${r.status}`);
    const d = (await r.json()) as { content?: { text?: string }[] };
    return d.content?.[0]?.text ?? "";
  }

  if (row.kind === "google") {
    const base =
      row.base_url?.replace(/\/$/, "") || "https://generativelanguage.googleapis.com/v1beta";
    const r = await fetch(
      `${base}/models/${row.model_id}:generateContent?key=${encodeURIComponent(row.api_key)}`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          systemInstruction: { parts: [{ text: SYSTEM }] },
          contents: [{ role: "user", parts: [{ text: prompt }] }],
          generationConfig: { maxOutputTokens: 24, temperature: 0.3 },
        }),
      },
    );
    if (!r.ok) throw new Error(`google ${r.status}`);
    const d = (await r.json()) as {
      candidates?: { content?: { parts?: { text?: string }[] } }[];
    };
    return d.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  }

  const headers: Record<string, string> = { "content-type": "application/json" };
  let url: string;
  if (row.kind === "azure_openai" && row.azure_endpoint) {
    const ep = row.azure_endpoint.replace(/\/$/, "");
    const ver = row.azure_api_version || "2024-06-01";
    url = `${ep}/openai/deployments/${row.model_id}/chat/completions?api-version=${ver}`;
    headers["api-key"] = row.api_key;
  } else {
    const base = row.base_url?.replace(/\/$/, "") || "https://api.openai.com/v1";
    url = `${base}/chat/completions`;
    if (row.api_key) headers.Authorization = `Bearer ${row.api_key}`;
  }
  const r = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: row.model_id,
      max_tokens: 24,
      temperature: 0.3,
      messages: [
        { role: "system", content: SYSTEM },
        { role: "user", content: prompt },
      ],
    }),
  });
  if (!r.ok) throw new Error(`openai ${r.status}`);
  const d = (await r.json()) as {
    choices?: { message?: { content?: string } }[];
  };
  return d.choices?.[0]?.message?.content ?? "";
}

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as { text?: unknown };
    const text = typeof body.text === "string" ? body.text.trim() : "";
    if (!text) return NextResponse.json({ title: "" });

    const row = await pickModel();
    if (!row) return NextResponse.json({ title: "" });

    const title = cleanTitle(await generate(row, text));
    return NextResponse.json({ title });
  } catch (e) {
    console.error("[/api/v1/title] error:", e);
    return NextResponse.json({ title: "" }, { status: 200 });
  }
}
