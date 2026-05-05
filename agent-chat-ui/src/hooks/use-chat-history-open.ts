"use client";

import { useEffect, useState, useCallback } from "react";

const STORAGE_KEY = "chatHistoryOpen";
const listeners = new Set<(v: boolean) => void>();
let current: boolean | null = null;

function read(): boolean {
  if (typeof window === "undefined") return true;
  try {
    const v = window.localStorage.getItem(STORAGE_KEY);
    if (v === null) return true;
    return v === "true";
  } catch {
    return true;
  }
}

function write(v: boolean) {
  current = v;
  try {
    window.localStorage.setItem(STORAGE_KEY, String(v));
  } catch {
    // ignore storage errors (private mode, quota, etc.)
  }
  listeners.forEach((l) => l(v));
}

export function useChatHistoryOpen(): [
  boolean,
  (next: boolean | ((p: boolean) => boolean)) => void,
] {
  const [value, setValue] = useState<boolean>(() =>
    current === null ? true : current,
  );

  useEffect(() => {
    if (current === null) current = read();
    setValue(current);
    const listener = (v: boolean) => setValue(v);
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }, []);

  const set = useCallback(
    (next: boolean | ((p: boolean) => boolean)) => {
      const resolved =
        typeof next === "function"
          ? (next as (p: boolean) => boolean)(current ?? value)
          : next;
      write(resolved);
    },
    [value],
  );

  return [value, set];
}
