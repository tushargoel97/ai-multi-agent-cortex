import { NextResponse } from "next/server";
import { checkAdmin } from "@/lib/admin-auth";
import { getAppSettings, setAppSetting } from "@/lib/app-settings";

const KNOWN_KEYS = new Set(["auto_profile"]);

export async function GET(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  return NextResponse.json({ settings: await getAppSettings() });
}

export async function PUT(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;
  const { key, value } = (await req.json()) ?? {};
  if (!key || typeof value !== "string" || !KNOWN_KEYS.has(key)) {
    return NextResponse.json({ error: "unknown setting" }, { status: 400 });
  }
  await setAppSetting(key, value);
  return NextResponse.json({ ok: true });
}
