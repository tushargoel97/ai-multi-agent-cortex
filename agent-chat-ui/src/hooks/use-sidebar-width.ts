"use client";

import { useEffect, useState } from "react";

const STORAGE_KEY = "cortex:sidebar-width";
const SIDEBAR_MIN = 240;
const SIDEBAR_MAX = 480;
const DEFAULT = 300;

const widthListeners = new Set<(v: number) => void>();
let currentWidth: number | null = null;

const resizeListeners = new Set<(v: boolean) => void>();

function clamp(n: number): number {
  return Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, Math.round(n)));
}

function read(): number {
  if (typeof window === "undefined") return DEFAULT;
  try {
    const v = window.localStorage.getItem(STORAGE_KEY);
    if (v === null) return DEFAULT;
    const n = Number(v);
    return Number.isFinite(n) ? clamp(n) : DEFAULT;
  } catch {
    return DEFAULT;
  }
}

/** Imperatively set the (persisted) sidebar width; notifies all consumers. */
export function setSidebarWidth(v: number) {
  currentWidth = clamp(v);
  try {
    window.localStorage.setItem(STORAGE_KEY, String(currentWidth));
  } catch {
    // ignore storage errors (private mode, quota, etc.)
  }
  widthListeners.forEach((l) => l(currentWidth as number));
}

/** Flag the drag so the layout switches to an instant (non-spring) transition. */
export function setSidebarResizing(v: boolean) {
  resizeListeners.forEach((l) => l(v));
}

export function useSidebarWidth(): number {
  const [value, setValue] = useState<number>(() =>
    currentWidth === null ? DEFAULT : currentWidth,
  );
  useEffect(() => {
    if (currentWidth === null) currentWidth = read();
    setValue(currentWidth);
    const l = (v: number) => setValue(v);
    widthListeners.add(l);
    return () => {
      widthListeners.delete(l);
    };
  }, []);
  return value;
}

export function useSidebarResizing(): boolean {
  const [value, setValue] = useState(false);
  useEffect(() => {
    const l = (v: boolean) => setValue(v);
    resizeListeners.add(l);
    return () => {
      resizeListeners.delete(l);
    };
  }, []);
  return value;
}
