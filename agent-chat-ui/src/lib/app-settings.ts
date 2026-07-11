import { query } from "@/lib/db";

const ENSURE_TABLE = `
  CREATE TABLE IF NOT EXISTS app_settings (
    key varchar(100) PRIMARY KEY,
    value text NOT NULL
  )`;

export async function getAppSettings(keys?: string[]) {
  await query(ENSURE_TABLE);
  const { rows } = await query<{ key: string; value: string }>(
    `SELECT key, value FROM app_settings${keys ? " WHERE key = ANY($1)" : ""}`,
    keys ? [keys] : [],
  );
  return Object.fromEntries(rows.map(({ key, value }) => [key, value]));
}

export async function setAppSetting(key: string, value: string) {
  await query(ENSURE_TABLE);
  await query(
    `INSERT INTO app_settings (key, value) VALUES ($1, $2)
     ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value`,
    [key, value],
  );
}
