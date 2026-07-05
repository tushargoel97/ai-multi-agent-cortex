"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
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
  const [zoomed, setZoomed] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  // An image that finished loading (e.g. from cache) before React attached
  // onLoad won't fire it — detect that on mount so it doesn't stay blank.
  useEffect(() => {
    if (imgRef.current?.complete && imgRef.current.naturalWidth > 0) {
      setLoaded(true);
    }
  }, [src]);

  // Close the zoom overlay on Escape.
  useEffect(() => {
    if (!zoomed) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setZoomed(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [zoomed]);

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
    <>
      <div className="group relative my-3 w-full max-w-sm overflow-hidden rounded-2xl border border-border bg-muted/30 shadow-sm">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          ref={imgRef}
          src={src}
          alt={alt || "Generated image"}
          onLoad={() => setLoaded(true)}
          onError={() => setErrored(true)}
          onClick={() => loaded && !errored && setZoomed(true)}
          className={cn(
            "block h-auto w-full",
            loaded && !errored && "cursor-zoom-in",
            !loaded && !errored && "opacity-0",
          )}
          style={!loaded && !errored ? { minHeight: "14rem" } : undefined}
        />
        {!loaded && !errored && (
          <div className="imggen-fill pointer-events-none">
            <div className="imggen-frame__aurora" />
            <div className="imggen-frame__shimmer" />
          </div>
        )}
        {errored && (
          <div className="text-muted-foreground absolute inset-0 flex items-center justify-center gap-2 text-sm">
            <ImageOff className="size-5" /> image unavailable
          </div>
        )}
        {loaded && !errored && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              download();
            }}
            title="Download image"
            aria-label="Download image"
            className="absolute top-2 right-2 rounded-lg bg-black/55 p-1.5 text-white opacity-0 backdrop-blur-sm transition-opacity duration-150 group-hover:opacity-100 hover:bg-black/70 focus-visible:opacity-100"
          >
            <Download className="size-4" />
          </button>
        )}
      </div>
      {zoomed &&
        typeof document !== "undefined" &&
        createPortal(
          <div
            className="fixed inset-0 z-50 flex cursor-zoom-out items-center justify-center bg-black/80 p-4 backdrop-blur-sm"
            onClick={() => setZoomed(false)}
            role="dialog"
            aria-modal="true"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={src}
              alt={alt || "Generated image"}
              className="max-h-[90vh] max-w-[90vw] rounded-lg object-contain shadow-2xl"
            />
          </div>,
          document.body,
        )}
    </>
  );
}
