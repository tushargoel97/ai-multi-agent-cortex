"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, Square } from "lucide-react";
import { ModeSelector, type ModelSelection } from "@/components/model-selector";
import { Button } from "@/components/ui/button";

export function ComposerActions({
  input,
  attachmentCount,
  isLoading,
  mode,
  onModeChange,
  onCancel,
}: {
  input: string;
  attachmentCount: number;
  isLoading: boolean;
  mode: ModelSelection["mode"];
  onModeChange: (mode: ModelSelection["mode"]) => void;
  onCancel: () => void;
}) {
  const canSend = input.trim().length > 0 || attachmentCount > 0;
  const showAction = isLoading || canSend;

  return (
    <div className="ml-auto flex items-center gap-2">
      <motion.div
        layout="position"
        data-composer-mode-position={showAction ? "inline" : "edge"}
        transition={{ type: "spring", stiffness: 440, damping: 34 }}
      >
        <ModeSelector mode={mode} onModeChange={onModeChange} />
      </motion.div>
      <AnimatePresence initial={false} mode="popLayout">
        {showAction && (
          <motion.div
            key={isLoading ? "stop" : "send"}
            initial={{ opacity: 0, scale: 0.72, x: 8 }}
            animate={{ opacity: 1, scale: 1, x: 0 }}
            exit={{ opacity: 0, scale: 0.72, x: 8 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
          >
            {isLoading ? (
              <Button
                type="button"
                size="icon"
                title="Stop generating"
                onClick={onCancel}
                className="size-8 rounded-full"
              >
                <Square className="size-3 fill-current" />
              </Button>
            ) : (
              <Button
                type="submit"
                size="icon"
                title="Send"
                className="size-8 rounded-full shadow-sm"
              >
                <ArrowUp className="size-4.5" />
              </Button>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
