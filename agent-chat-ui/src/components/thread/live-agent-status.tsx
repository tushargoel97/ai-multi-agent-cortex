"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { LangGraphLogoSVG } from "../icons/langgraph";
import type { Activity } from "./agent-activity";

function useLivePhrase(activity: Activity) {
  const reducedMotion = useReducedMotion();
  const phrases = activity.phrases?.length ? activity.phrases : [activity.label];
  const key = activity.key ?? activity.label;
  const phraseKey = phrases.join("\u0000");
  const [position, setPosition] = useState({ key, index: 0 });
  const index = position.key === key ? position.index : 0;

  useEffect(() => {
    setPosition({ key, index: 0 });
    if (reducedMotion || phrases.length < 2) return;
    let interval: ReturnType<typeof setInterval> | undefined;
    const timeout = setTimeout(() => {
      setPosition({ key, index: 1 });
      interval = setInterval(
        () => setPosition((current) => ({ key, index: current.index + 1 })),
        2800,
      );
    }, 1200);
    return () => {
      clearTimeout(timeout);
      if (interval) clearInterval(interval);
    };
  }, [key, phraseKey, phrases.length, reducedMotion]);

  return { phrase: phrases[index % phrases.length], reducedMotion };
}

export function LiveAgentStatus({ activity }: { activity: Activity }) {
  const { phrase, reducedMotion } = useLivePhrase(activity);
  return (
    <>
      <span className="cortex-live-mark" aria-hidden="true">
        <LangGraphLogoSVG markOnly className="cortex-live-mark__glyph" width={14} height={14} />
      </span>
      <span className="live-status-viewport" aria-hidden="true">
        <AnimatePresence initial={false} mode="wait">
          <motion.span
            key={`${activity.key}:${phrase}`}
            initial={reducedMotion ? false : { y: 12, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={reducedMotion ? undefined : { y: -12, opacity: 0 }}
            transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
            className="shimmer-text block font-medium whitespace-nowrap"
          >
            {phrase}
          </motion.span>
        </AnimatePresence>
      </span>
      <span className="sr-only" role="status" aria-live="polite">
        {activity.label}
      </span>
    </>
  );
}
