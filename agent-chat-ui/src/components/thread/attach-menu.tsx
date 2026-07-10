"use client";

import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Camera, Paperclip, Plus } from "lucide-react";
import { toast } from "sonner";
import { useDropdown } from "@/hooks/use-dropdown";
import { cn } from "@/lib/utils";

/** Grab one frame of a user-picked screen/window/tab as a PNG File. The
 *  browser shows its own source picker; cancelling it is not an error. */
async function captureScreenshot(): Promise<File | null> {
  let stream: MediaStream | null = null;
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: 1 },
      audio: false,
    });
    const video = document.createElement("video");
    video.srcObject = stream;
    await video.play();
    // Give the browser's share-overlay a beat to fade from the frame.
    await new Promise((r) => setTimeout(r, 400));
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    const blob = await new Promise<Blob | null>((res) =>
      canvas.toBlob(res, "image/png"),
    );
    if (!blob) throw new Error("could not read a frame");
    return new File([blob], `screenshot-${Date.now()}.png`, {
      type: "image/png",
    });
  } catch (e) {
    // NotAllowedError = the user dismissed the picker; stay silent.
    if ((e as DOMException)?.name !== "NotAllowedError") {
      toast.error(`Screenshot failed: ${(e as Error).message}`);
    }
    return null;
  } finally {
    stream?.getTracks().forEach((t) => t.stop());
  }
}

/**
 * The composer's "+" button: a small frosted menu with attach-file and
 * take-screenshot actions. The menu portals to <body> so the composer's own
 * backdrop blur can't disable the menu's (nested backdrop-filter).
 */
export function AttachMenu({ onFiles }: { onFiles: (f: File[]) => void }) {
  const rootRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const { open, setOpen, mounted } = useDropdown(rootRef, {
    insideRef: menuRef,
  });
  const [box, setBox] = useState<{
    left: number;
    top?: number;
    bottom?: number;
    up: boolean;
  } | null>(null);

  // Same placement rule as the other composer menus: anchor to the composer
  // and open downward when there's room beneath it, upward otherwise.
  const toggle = () => {
    if (!open) {
      const el = rootRef.current;
      if (el) {
        const composer =
          (el.closest("[data-prompt-composer]") as HTMLElement | null) ?? el;
        const cr = composer.getBoundingClientRect();
        const r = el.getBoundingClientRect();
        const gap = 8;
        const W = 224; // w-56
        const below = window.innerHeight - cr.bottom - gap;
        const above = cr.top - gap;
        const left = Math.max(8, Math.min(r.left, window.innerWidth - W - 8));
        setBox(
          below >= 160 || below >= above
            ? { left, top: cr.bottom + gap, up: false }
            : { left, bottom: window.innerHeight - cr.top + gap, up: true },
        );
      }
    }
    setOpen((o) => !o);
  };

  const item =
    "flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-sm transition-colors hover:bg-accent/60";

  return (
    <div
      ref={rootRef}
      className="inline-flex"
    >
      <button
        type="button"
        title="Add files or a screenshot"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={toggle}
        className="text-muted-foreground hover:bg-muted hover:text-foreground flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-full transition-colors"
      >
        <Plus
          className={cn(
            "size-5 transition-transform duration-200",
            open && "rotate-45",
          )}
        />
      </button>
      <input
        ref={fileRef}
        type="file"
        multiple
        accept="image/jpeg,image/png,image/gif,image/webp,application/pdf"
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.length) onFiles(Array.from(e.target.files));
          e.target.value = "";
        }}
      />
      {mounted &&
        box &&
        createPortal(
          <div
            ref={menuRef}
            role="menu"
            data-state={open ? "open" : "closed"}
            style={{
              position: "fixed",
              left: box.left,
              top: box.top,
              bottom: box.bottom,
            }}
            className={cn(
              "glass z-[90] w-56 rounded-xl border p-1 shadow-xl duration-150",
              open
                ? cn(
                    "animate-in fade-in-0 zoom-in-95",
                    box.up ? "slide-in-from-bottom-1" : "slide-in-from-top-1",
                  )
                : cn(
                    "animate-out fade-out-0 zoom-out-95",
                    box.up ? "slide-out-to-bottom-1" : "slide-out-to-top-1",
                  ),
            )}
          >
            <button
              type="button"
              role="menuitem"
              className={item}
              onClick={() => {
                setOpen(false);
                fileRef.current?.click();
              }}
            >
              <Paperclip className="text-muted-foreground size-4 shrink-0" />
              <span className="flex min-w-0 flex-col">
                <span>Upload files</span>
                <span className="text-muted-foreground truncate text-[11px]">
                  Images (JPEG, PNG, GIF, WEBP) or PDF
                </span>
              </span>
            </button>
            <button
              type="button"
              role="menuitem"
              className={item}
              onClick={async () => {
                setOpen(false);
                const shot = await captureScreenshot();
                if (shot) onFiles([shot]);
              }}
            >
              <Camera className="text-muted-foreground size-4 shrink-0" />
              <span className="flex min-w-0 flex-col">
                <span>Take screenshot</span>
                <span className="text-muted-foreground truncate text-[11px]">
                  Capture a screen, window, or tab
                </span>
              </span>
            </button>
          </div>,
          document.body,
        )}
    </div>
  );
}
