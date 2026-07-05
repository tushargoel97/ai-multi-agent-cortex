import { NextRequest, NextResponse } from "next/server";
import { checkAdmin } from "@/lib/admin-auth";

// Host-side MLX trainer (runs outside Docker — see trainer/README.md).
const TRAINER_URL = process.env.TRAINER_URL ?? "http://host.docker.internal:8200";

async function proxy(req: NextRequest, path: string) {
  const url = `${TRAINER_URL}${path}${req.nextUrl.search}`;
  // Pass the original content type through (multipart uploads carry their
  // boundary in it); default to JSON for body-less/plain requests.
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
          `Trainer unreachable at ${TRAINER_URL}: ${(e as Error).message}. ` +
          "Start it on the host: cd trainer && uv run uvicorn app.main:app --host 0.0.0.0 --port 8200",
      },
      { status: 502 },
    );
  }
}

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ path?: string[] }> },
) {
  const auth = checkAdmin(req);
  if (auth) return auth;
  const { path = [] } = await params;
  return proxy(req, "/admin/" + path.join("/"));
}

export async function POST(
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
