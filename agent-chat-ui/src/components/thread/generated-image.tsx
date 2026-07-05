"use client";

import { useState } from "react";
import { Download, ImageOff } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Renders a generated (or any markdown) image inside a rounded, responsive
 * frame that caps its size so images aren't huge, with a download button that
 * fades in on hover.
 */
export function GeneratedImage({ src, alt }: { src?: string; alt?: string }) {
  const [loaded, setLoaded] = useState(false);
  const [errored, setErrored] = useState(false);

  if (!src) return null;

  async function download() {
    if (!src) return;
    const name =
      (alt?.trim() ? alt.trim().replace(/\s+/g, "-").toLowerCase() : "generated-image") +
      ".png";
    try {
      const res = await fetch(src);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch {
      window.open(src, "_blank", "noopener,noreferrer");
    }
  }

  return (
    <div className="group relative my-3 w-full max-w-md overflow-hidden rounded-2xl border border-border bg-muted/30 shadow-sm">
      {errored ? (
        <div className="text-muted-foreground flex aspect-square w-full items-center justify-center gap-2 text-sm">
          <ImageOff className="size-5" /> image unavailable
        </div>
      ) : (
        <>
          {!loaded && (
            <div className="aspect-square w-full animate-pulse bg-muted" />
          )}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt={alt || "Generated image"}
            loading="lazy"
            onLoad={() => setLoaded(true)}
            onError={() => setErrored(true)}
            className={cn(
              "block h-auto w-full",
              loaded ? "opacity-100" : "absolute inset-0 opacity-0",
            )}
          />
          {loaded && (
            <button
              type="button"
              onClick={download}
              title="Download image"
              aria-label="Download image"
              className="absolute top-2 right-2 flex items-center gap-1 rounded-lg bg-black/55 px-2 py-1.5 text-xs font-medium text-white opacity-0 backdrop-blur-sm transition-opacity duration-150 group-hover:opacity-100 focus:opacity-100 hover:bg-black/70"
            >
              <Download className="size-3.5" /> Download
            </button>
          )}
        </>
      )}
    </div>
  );
}
