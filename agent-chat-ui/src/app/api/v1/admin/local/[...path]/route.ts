import { NextRequest, NextResponse } from "next/server";
import { checkAdmin } from "@/lib/admin-auth";

const AI_URL = process.env.LOCAL_AI_URL ?? "http://ai:8100/api/v1";

async function proxy(req: NextRequest, path: string) {
  const url = `${AI_URL}${path}${req.nextUrl.search}`;
  const init: RequestInit = {
    method: req.method,
    headers: { "Content-Type": "application/json" },
  };
  if (req.method !== "GET" && req.method !== "DELETE") {
    init.body = await req.text();
  }
  try {
    const res = await fetch(url, init);
    const text = await res.text();
    return new NextResponse(text, {
      status: res.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch (e) {
    return NextResponse.json(
      { error: `AI service unreachable at ${AI_URL}: ${(e as Error).message}` },
      { status: 502 },
    );
  }
}

export async function GET(req: NextRequest, { params }: { params: Promise<{ path?: string[] }> }) {
  const auth = checkAdmin(req);
  if (auth) return auth;
  const { path = [] } = await params;
  return proxy(req, "/admin/" + path.join("/"));
}

export async function POST(req: NextRequest, { params }: { params: Promise<{ path?: string[] }> }) {
  const auth = checkAdmin(req);
  if (auth) return auth;
  const { path = [] } = await params;
  return proxy(req, "/admin/" + path.join("/"));
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ path?: string[] }> },
) {
  const auth = checkAdmin(req);
  if (auth) return auth;
  const { path = [] } = await params;
  return proxy(req, "/admin/" + path.join("/"));
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ path?: string[] }> },
) {
  const auth = checkAdmin(req);
  if (auth) return auth;
  const { path = [] } = await params;
  return proxy(req, "/admin/" + path.join("/"));
}
