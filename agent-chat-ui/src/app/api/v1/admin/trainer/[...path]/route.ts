import { NextRequest, NextResponse } from "next/server";
import { checkAdmin } from "@/lib/admin-auth";
import { trainerUrl } from "@/lib/trainer";

async function proxy(req: NextRequest, path: string) {
  const url = `${trainerUrl}${path}${req.nextUrl.search}`;
  const contentType = req.headers.get("content-type") ?? "application/json";
  const init: RequestInit = {
    method: req.method,
    headers: { "Content-Type": contentType },
  };
  if (req.method !== "GET" && req.method !== "DELETE") {
    init.body = await req.arrayBuffer();
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
      {
        error:
          `Trainer unreachable at ${trainerUrl}: ${(e as Error).message}. ` +
          "Start it on the host: cd trainer && uv run uvicorn app.main:app --host 0.0.0.0 --port 8200",
      },
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

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ path?: string[] }> },
) {
  const auth = checkAdmin(req);
  if (auth) return auth;
  const { path = [] } = await params;
  return proxy(req, "/admin/" + path.join("/"));
}
