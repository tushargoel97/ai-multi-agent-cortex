import { NextResponse } from "next/server";

const ADMIN_TOKEN_HEADER = "x-admin-token";

export function checkAdmin(req: Request): NextResponse | null {
  const expected = process.env.ADMIN_TOKEN;
  if (!expected) {
    return NextResponse.json(
      {
        error:
          "ADMIN_TOKEN not configured on server. Set ADMIN_TOKEN env var to enable admin access.",
      },
      { status: 503 },
    );
  }
  const got = req.headers.get(ADMIN_TOKEN_HEADER);
  if (got !== expected) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  return null;
}

export const ADMIN_TOKEN_HEADER_NAME = ADMIN_TOKEN_HEADER;
