import { NextResponse } from "next/server";
import { checkAdmin } from "@/lib/admin-auth";
import { query } from "@/lib/db";
import {
  getFinetunedLifecycle,
  promote,
  rollback,
  saveFinetunedLifecycle,
} from "@/lib/finetuned-lifecycle";
import { trainerUrl } from "@/lib/trainer";

const modelExists = async (modelId: string) =>
  (
    await query(
      `SELECT 1 FROM llm_models m JOIN llm_providers p ON p.id = m.provider_id
       WHERE m.model_id = $1 AND m.enabled = true AND p.enabled = true
         AND p.kind = 'local' AND m.model_id LIKE 'finetuned-%'`,
      [modelId],
    )
  ).rows.length > 0;

export async function GET(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  return NextResponse.json(await getFinetunedLifecycle());
}

export async function POST(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  const { action, model_id: modelId, run_id: runId } = (await req.json()) ?? {};
  const state = await getFinetunedLifecycle();
  if (action === "rollback") {
    const next = rollback(state);
    if (!next) return NextResponse.json({ error: "no previous model" }, { status: 409 });
    if (!(await modelExists(next.active!))) {
      return NextResponse.json({ error: "previous model is unavailable" }, { status: 409 });
    }
    await saveFinetunedLifecycle(next);
    return NextResponse.json(next);
  }
  if (action !== "promote" || !modelId || !runId) {
    return NextResponse.json({ error: "action, model_id, and run_id required" }, { status: 400 });
  }
  const [exists, runResponse] = await Promise.all([
    modelExists(modelId),
    fetch(`${trainerUrl}/admin/runs/${encodeURIComponent(runId)}`),
  ]);
  const run = runResponse.ok ? await runResponse.json() : null;
  if (!exists || run?.model_id !== modelId || !run?.evaluation?.passed) {
    return NextResponse.json(
      { error: "model must exist and pass its evaluation" },
      { status: 409 },
    );
  }
  const next = promote(state, modelId);
  await saveFinetunedLifecycle(next);
  return NextResponse.json(next);
}
