import { NextResponse } from "next/server";
import { query, pool } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";
import { ensureAgentsTable } from "@/lib/tool-tables";

/** Slugify a name into a stable identifier (matches the create route). */
function slug(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 60);
}

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

async function setSubagents(name: string, subagents: string[]) {
  await query(`DELETE FROM agent_subagents WHERE agent_name = $1`, [name]);
  for (const sub of Array.from(new Set(subagents))) {
    if (sub && sub !== name) {
      await query(
        `INSERT INTO agent_subagents (id, agent_name, subagent_name)
           VALUES (gen_random_uuid(), $1, $2)
         ON CONFLICT (agent_name, subagent_name) DO NOTHING`,
        [name, sub],
      );
    }
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
  let { name } = await params;
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
    await query(`DELETE FROM agent_subagents WHERE agent_name = $1`, [name]);
    return NextResponse.json({ ok: true });
  }

  // Rename (custom agents only), cascades to tool grants + subagent links
  // (both directions) in one transaction, done before the other field updates
  // so they target the new name.
  if (typeof body.new_name === "string" && slug(body.new_name) !== name) {
    const newName = slug(body.new_name);
    if (!newName) {
      return NextResponse.json({ error: "invalid name" }, { status: 400 });
    }
    const { rows: cur } = await query<{ kind: string }>(
      `SELECT kind FROM agents WHERE name = $1`,
      [name],
    );
    if (!cur.length) {
      return NextResponse.json({ error: "not found" }, { status: 404 });
    }
    if (cur[0].kind !== "custom") {
      return NextResponse.json(
        { error: "built-in agents can't be renamed, only their prompt/tools" },
        { status: 400 },
      );
    }
    const { rows: taken } = await query(
      `SELECT 1 FROM agents WHERE name = $1`,
      [newName],
    );
    if (taken.length) {
      return NextResponse.json(
        { error: `an agent named "${newName}" already exists` },
        { status: 409 },
      );
    }
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      await client.query(`UPDATE agents SET name = $1 WHERE name = $2`, [
        newName,
        name,
      ]);
      await client.query(
        `UPDATE agent_tools SET agent_name = $1 WHERE agent_name = $2`,
        [newName, name],
      );
      await client.query(
        `UPDATE agent_subagents SET agent_name = $1 WHERE agent_name = $2`,
        [newName, name],
      );
      await client.query(
        `UPDATE agent_subagents SET subagent_name = $1 WHERE subagent_name = $2`,
        [newName, name],
      );
      await client.query("COMMIT");
    } catch (e) {
      await client.query("ROLLBACK");
      throw e;
    } finally {
      client.release();
    }
    name = newName;
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
  if (Array.isArray(body.subagents)) {
    await setSubagents(name, body.subagents.map((s: unknown) => String(s)));
  }
  return NextResponse.json({ ok: true, name });
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
      { error: "built-in agents can't be deleted, reset it instead" },
      { status: 400 },
    );
  }
  await query(`DELETE FROM agents WHERE name = $1 AND kind = 'custom'`, [name]);
  await query(`DELETE FROM agent_tools WHERE agent_name = $1`, [name]);
  await query(
    `DELETE FROM agent_subagents WHERE agent_name = $1 OR subagent_name = $1`,
    [name],
  );
  return NextResponse.json({ ok: true });
}

