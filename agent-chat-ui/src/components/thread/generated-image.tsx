"use client";

import { useEffect, useRef, useState } from "react";
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
  const imgRef = useRef<HTMLImageElement>(null);

  // An image that finished loading (e.g. from cache) before React attached
  // onLoad won't fire it — detect that on mount so it doesn't stay blank.
  useEffect(() => {
    if (imgRef.current?.complete && imgRef.current.naturalWidth > 0) {
      setLoaded(true);
    }
  }, [src]);

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
    <div className="group relative my-3 w-full max-w-sm overflow-hidden rounded-2xl border border-border bg-muted/30 shadow-sm">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        ref={imgRef}
        src={src}
        alt={alt || "Generated image"}
        onLoad={() => setLoaded(true)}
        onError={() => setErrored(true)}
        className={cn("block h-auto w-full", !loaded && !errored && "opacity-0")}
        style={!loaded && !errored ? { minHeight: "14rem" } : undefined}
      />
      {!loaded && !errored && (
        <div className="pointer-events-none absolute inset-0 animate-pulse bg-muted" />
      )}
      {errored && (
        <div className="text-muted-foreground absolute inset-0 flex items-center justify-center gap-2 text-sm">
          <ImageOff className="size-5" /> image unavailable
        </div>
      )}
      {loaded && !errored && (
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
    </div>
  );
}
