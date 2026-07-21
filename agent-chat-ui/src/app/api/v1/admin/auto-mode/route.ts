import { NextResponse } from "next/server";
import { query } from "@/lib/db";
import { checkAdmin } from "@/lib/admin-auth";
import { getAppSettings, setAppSetting } from "@/lib/app-settings";

const DEFAULTS_KEY = "auto_mode_defaults";
const OVERRIDES_KEY = "auto_mode_overrides";
const PROFILE_KEY = "auto_profile";

type Profiles = Record<string, Record<string, string[]>>;

function safeParseProfiles(value: string | undefined): Profiles {
  if (!value) return {};
  try {
    const data = JSON.parse(value);
    return data && typeof data === "object" && !Array.isArray(data) ? (data as Profiles) : {};
  } catch {
    return {};
  }
}

function isValidOverrides(value: unknown): value is Profiles {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  for (const intents of Object.values(value)) {
    if (typeof intents !== "object" || intents === null || Array.isArray(intents)) {
      return false;
    }
    for (const list of Object.values(intents as Record<string, unknown>)) {
      if (!Array.isArray(list) || !list.every((x) => typeof x === "string")) {
        return false;
      }
    }
  }
  return true;
}

export async function GET(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;

  const settings = await getAppSettings([DEFAULTS_KEY, OVERRIDES_KEY, PROFILE_KEY]);

  const { rows: models } = await query<{
    model_id: string;
    display_name: string;
    provider_kind: string;
  }>(
    `SELECT m.model_id, m.display_name, p.kind AS provider_kind
       FROM llm_models m
       JOIN llm_providers p ON p.id = m.provider_id
      WHERE m.enabled = true AND p.enabled = true
      ORDER BY p.kind, m.display_name`,
  );

  return NextResponse.json({
    activeProfile: settings[PROFILE_KEY] || "balanced",
    defaults: safeParseProfiles(settings[DEFAULTS_KEY]),
    overrides: safeParseProfiles(settings[OVERRIDES_KEY]),
    models,
  });
}

export async function PUT(req: Request) {
  const unauthed = checkAdmin(req);
  if (unauthed) return unauthed;

  const body = (await req.json().catch(() => null)) ?? {};
  const overrides = body.overrides;
  if (!isValidOverrides(overrides)) {
    return NextResponse.json(
      { error: "overrides must be an object of { profile: { intent: string[] } }" },
      { status: 400 },
    );
  }

  const cleaned: Profiles = {};
  for (const [profile, intents] of Object.entries(overrides)) {
    const keptIntents: Record<string, string[]> = {};
    for (const [intent, list] of Object.entries(intents)) {
      if (list.length > 0) keptIntents[intent] = list;
    }
    if (Object.keys(keptIntents).length > 0) cleaned[profile] = keptIntents;
  }

  await setAppSetting(OVERRIDES_KEY, JSON.stringify(cleaned));
  return NextResponse.json({ ok: true });
}
