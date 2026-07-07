import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";

// Mirrors cortex/db/models/app_setting.py, created here too so the panel
// works before the graph ever writes a setting.
const ENSURE_TABLE = `
  CREATE TABLE IF NOT EXISTS app_settings (
    key varchar(100) PRIMARY KEY,
    value text NOT NULL
  )`;

const KNOWN_KEYS = new Set(["auto_profile"]);

export async function GET(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await query(ENSURE_TABLE);
  const { rows } = await query(`SELECT key, value FROM app_settings`);
  const settings: Record<string, string> = {};
  for (const row of rows as { key: string; value: string }[]) {
    settings[row.key] = row.value;
  }
  return NextResponse.json({ settings });
}

export async function PUT(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  const { key, value } = (await req.json()) ?? {};
  if (!key || typeof value !== "string" || !KNOWN_KEYS.has(key)) {
    return NextResponse.json({ error: "unknown setting" }, { status: 400 });
  }
  await query(ENSURE_TABLE);
  await query(
    `INSERT INTO app_settings (key, value) VALUES ($1, $2)
     ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value`,
    [key, value],
  );
  return NextResponse.json({ ok: true });
}
