import { NextRequest, NextResponse } from "next/server";
import { query } from "@/lib/db";

const MODEL = process.env.SUGGESTIONS_MODEL || "gpt-4.1-nano";

const SYSTEM = [
  "You write example chat prompts. Given recent conversation topics, reply",
  "with exactly 10 varied short example prompts (each under 10 words) the",
  "user might plausibly ask next, mixing follow-ups with fresh related",
  "ideas. One per line. No numbering, no bullets, no quotes,",
  "no commentary.",
].join(" ");

const cache = new Map<string, string[]>();
const CACHE_MAX = 100;

let keyCache: { key: string; at: number } | null = null;

async function openaiKey(): Promise<string> {
  if (keyCache && Date.now() - keyCache.at < 60_000) return keyCache.key;
  const { rows } = await query<{ api_key: string }>(
    "SELECT api_key FROM llm_providers WHERE kind = 'openai' AND enabled LIMIT 1",
  );
  const key = rows[0]?.api_key?.trim() ?? "";
  keyCache = { key, at: Date.now() };
  return key;
}

function parseLines(text: string): string[] {
  return text
    .split("\n")
    .map((l) =>
      l
        .replace(/^[\s\d.\-*•)]+/, "")
        .replace(/^["'`]+|["'`]+$/g, "")
        .trim(),
    )
    .filter((l) => l.length > 4 && l.length <= 90)
    .slice(0, 12);
}

export async function POST(req: NextRequest) {
  let context: string[] = [];
  try {
    const body = await req.json();
    if (Array.isArray(body?.context)) {
      context = body.context
        .filter((c: unknown) => typeof c === "string" && c.trim())
        .slice(0, 12)
        .map((c: string) => c.replace(/\s+/g, " ").slice(0, 220));
    }
  } catch {}
  if (context.length === 0) return NextResponse.json({ suggestions: [] });

  const key = context.join("|");
  const hit = cache.get(key);
  if (hit) return NextResponse.json({ suggestions: hit });

  try {
    const apiKey = await openaiKey();
    if (!apiKey) return NextResponse.json({ suggestions: [] });
    const res = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          { role: "system", content: SYSTEM },
          {
            role: "user",
            content: `Recent topics:\n${context.map((c) => `- ${c}`).join("\n")}\n\nTen example prompts:`,
          },
        ],
        temperature: 0.9,
        max_tokens: 220,
      }),
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) return NextResponse.json({ suggestions: [] });
    const data = await res.json();
    const text: string = data?.choices?.[0]?.message?.content ?? "";
    const suggestions = parseLines(text);
    if (suggestions.length > 0) {
      if (cache.size >= CACHE_MAX) cache.delete(cache.keys().next().value as string);
      cache.set(key, suggestions);
    }
    return NextResponse.json({ suggestions });
  } catch {
    return NextResponse.json({ suggestions: [] });
  }
}
