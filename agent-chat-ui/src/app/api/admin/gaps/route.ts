import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";

// Mirrors cortex/db/models/knowledge_gap.py, created here too so the panel
// works before the first gap is ever logged by the graph.
const ENSURE_TABLE = `
  CREATE TABLE IF NOT EXISTS knowledge_gaps (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    question text NOT NULL,
    answer text,
    reason varchar(40) NOT NULL DEFAULT 'refusal',
    status varchar(20) NOT NULL DEFAULT 'new',
    researched_summary text,
    created_at timestamptz DEFAULT now()
  )`;

export async function GET(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await query(ENSURE_TABLE);
  const { rows } = await query(
    `SELECT id, question, answer, reason, status, researched_summary, created_at
     FROM knowledge_gaps
     WHERE status != 'dismissed'
     ORDER BY created_at DESC
     LIMIT 100`,
  );
  return NextResponse.json({ gaps: rows });
}

export async function PATCH(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  const body = await req.json();
  const { id, status, researched_summary } = body ?? {};
  if (!id || !status) {
    return NextResponse.json({ error: "id and status required" }, { status: 400 });
  }
  await query(ENSURE_TABLE);
  await query(
    `UPDATE knowledge_gaps
     SET status = $2, researched_summary = COALESCE($3, researched_summary)
     WHERE id = $1`,
    [id, status, researched_summary ?? null],
  );
  return NextResponse.json({ ok: true });
}
