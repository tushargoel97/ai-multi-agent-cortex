import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";
import { ensureAgentsTable } from "@/lib/tool-tables";

async function setGrants(name: string, tools: string[]) {
  await query(`DELETE FROM agent_tools WHERE agent_name = $1`, [name]);
  for (const tool of Array.from(new Set(tools))) {
    await query(
      `INSERT INTO agent_tools (id, agent_name, tool_name)
         VALUES (gen_random_uuid(), $1, $2)
       ON CONFLICT (agent_name, tool_name) DO NOTHING`,
      [name, tool],
    );
  }
}

/** Update an agent's system prompt / description / enabled / tools, or reset a
 * built-in to its packaged defaults ({ reset: true }). */
export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ name: string }> },
) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await ensureAgentsTable();
  const { name } = await params;
  const body = (await req.json().catch(() => null)) ?? {};

  if (body.reset === true) {
    const { rows } = await query<{ value: string }>(
      `SELECT value FROM app_settings WHERE key = 'agent_defaults'`,
    );
    let prompt = "";
    let description = "";
    try {
      const d = JSON.parse(rows[0]?.value ?? "{}")[name];
      if (d) {
        prompt = String(d.system_prompt ?? "");
        description = String(d.description ?? "");
      }
    } catch {
      /* no defaults mirror yet */
    }
    await query(
      `UPDATE agents SET system_prompt = $1, description = $2, edited = false
        WHERE name = $3`,
      [prompt, description, name],
    );
    await query(`DELETE FROM agent_tools WHERE agent_name = $1`, [name]);
    return NextResponse.json({ ok: true });
  }

  const fields: string[] = [];
  const vals: unknown[] = [];
  let i = 1;
  if (typeof body.system_prompt === "string") {
    fields.push(`system_prompt = $${i++}`, `edited = true`);
    vals.push(body.system_prompt);
  }
  if (typeof body.description === "string") {
    fields.push(`description = $${i++}`);
    vals.push(body.description);
  }
  if (typeof body.enabled === "boolean") {
    fields.push(`enabled = $${i++}`);
    vals.push(body.enabled);
  }
  if (fields.length) {
    vals.push(name);
    await query(`UPDATE agents SET ${fields.join(", ")} WHERE name = $${i}`, vals);
  }
  if (Array.isArray(body.tools)) {
    await setGrants(name, body.tools.map((t: unknown) => String(t)));
  }
  return NextResponse.json({ ok: true });
}

/** Delete a custom agent (built-ins can only be reset). */
export async function DELETE(
  req: Request,
  { params }: { params: Promise<{ name: string }> },
) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  await ensureAgentsTable();
  const { name } = await params;

  const { rows } = await query<{ kind: string }>(
    `SELECT kind FROM agents WHERE name = $1`,
    [name],
  );
  if (!rows.length) {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }
  if (rows[0].kind !== "custom") {
    return NextResponse.json(
      { error: "built-in agents can't be deleted — reset it instead" },
      { status: 400 },
    );
  }
  await query(`DELETE FROM agents WHERE name = $1 AND kind = 'custom'`, [name]);
  await query(`DELETE FROM agent_tools WHERE agent_name = $1`, [name]);
  return NextResponse.json({ ok: true });
}

