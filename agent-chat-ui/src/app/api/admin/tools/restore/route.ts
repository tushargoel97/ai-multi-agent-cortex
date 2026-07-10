import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";
import { ensureToolTables, getSuppressedTools, setSuppressedTools } from "@/lib/tool-tables";

/** Un-suppress a previously deleted tool and re-list it. Built-in tools live
 * in code, so the langgraph server refreshes the full description on its next
 * restart; the row is re-added here so it shows immediately. */
export async function POST(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await ensureToolTables();

  const body = (await req.json().catch(() => null)) ?? {};
  const name = String(body.name ?? "").trim();
  if (!name) {
    return NextResponse.json({ error: "name required" }, { status: 400 });
  }

  const suppressed = await getSuppressedTools();
  await setSuppressedTools(suppressed.filter((n) => n !== name));
  await query(
    `INSERT INTO tools (id, name, kind, enabled)
       VALUES (gen_random_uuid(), $1, 'builtin', true)
     ON CONFLICT (name) DO UPDATE SET enabled = true`,
    [name],
  );
  return NextResponse.json({ ok: true });
}
