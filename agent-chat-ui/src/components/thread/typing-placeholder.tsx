"use client";

import { useEffect, useRef, useState } from "react";
import { Message } from "@langchain/langgraph-sdk";
import { useThreads } from "@/providers/Thread";
import { getThreadLabel } from "./history/thread-actions";
import { getContentString } from "./utils";

/** Context-aware prompt examples for the empty composer: thread titles seed a
 *  new chat, the last exchange seeds follow-ups. Empty = static placeholder. */
export function useSuggestions(messages: Message[], isLoading: boolean): string[] {
  const { threads } = useThreads();
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  let context: string[] = [];
  if (messages.length === 0) {
    context = threads
      .slice(0, 10)
      .map((t) => getThreadLabel(t))
      .filter((s) => s !== "New chat");
  } else if (!isLoading) {
    const lastHuman = [...messages].reverse().find((m) => m.type === "human");
    const lastAi = [...messages].reverse().find((m) => m.type === "ai");
    if (lastHuman) context.push(`User asked: ${getContentString(lastHuman.content)}`);
    if (lastAi) {
      const a = getContentString(lastAi.content).slice(0, 200);
      if (a) context.push(`Assistant answered: ${a}`);
    }
  }
  const key = context.join("|").slice(0, 800);

  useEffect(() => {
    if (isLoading || !key) return;
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      try {
        const res = await fetch("/api/suggestions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ context: key.split("|") }),
        });
        const data = await res.json();
        if (Array.isArray(data?.suggestions) && data.suggestions.length > 0) {
          setSuggestions(data.suggestions);
        }
      } catch {
        // keep previous suggestions
      }
    }, 1000);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [key, isLoading]);

  return suggestions;
}

/** Typewriter overlay shown instead of the static placeholder while the
 *  textarea is empty: types a suggestion, holds, deletes, cycles. */
export function TypingPlaceholder({ suggestions }: { suggestions: string[] }) {
  const [text, setText] = useState("");
  const state = useRef({
    i: 0,
    len: 0,
    phase: "type" as "type" | "hold" | "del",
  });
  const reduced =
    typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  useEffect(() => {
    if (suggestions.length === 0) return;
    if (reduced) {
      setText(suggestions[0]);
      return;
    }
    state.current = { i: 0, len: 0, phase: "type" };
    let t: ReturnType<typeof setTimeout>;
    const tick = () => {
      const s = state.current;
      const cur = suggestions[s.i % suggestions.length];
      let delay = 35;
      if (s.phase === "type") {
        s.len++;
        if (s.len >= cur.length) {
          s.phase = "hold";
          delay = 2500;
        }
      } else if (s.phase === "hold") {
        s.phase = "del";
        delay = 15;
      } else {
        s.len -= 2;
        delay = 15;
        if (s.len <= 0) {
          s.len = 0;
          s.i++;
          s.phase = "type";
          delay = 400;
        }
      }
      setText(cur.slice(0, Math.max(0, s.len)));
      t = setTimeout(tick, delay);
    };
    t = setTimeout(tick, 300);
    return () => clearTimeout(t);
  }, [suggestions, reduced]);

  if (suggestions.length === 0) return null;
  return (
    <span
      aria-hidden
      className="text-muted-foreground pointer-events-none absolute top-4 left-4 select-none"
    >
      {text}
      {!reduced && <span className="ml-px animate-[pulse_1.1s_ease-in-out_infinite]">|</span>}
    </span>
  );
}
