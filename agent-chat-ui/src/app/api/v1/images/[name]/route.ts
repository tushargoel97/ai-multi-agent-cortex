import { NextResponse } from "next/server";
import { readFile } from "fs/promises";
import path from "path";

const IMAGES_DIR = "/app/generated_images";

const SAFE_NAME = /^[a-zA-Z0-9_-]+\.png$/;

export async function GET(_req: Request, { params }: { params: Promise<{ name: string }> }) {
  const { name } = await params;
  if (!SAFE_NAME.test(name)) {
    return NextResponse.json({ error: "invalid image name" }, { status: 400 });
  }
  try {
    const data = await readFile(path.join(IMAGES_DIR, name));
    return new NextResponse(new Uint8Array(data), {
      headers: {
        "Content-Type": "image/png",
        "Cache-Control": "public, max-age=31536000, immutable",
      },
    });
  } catch {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }
}
