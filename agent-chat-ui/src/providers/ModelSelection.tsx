"use client";

import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import {
  DEFAULT_SELECTION,
  loadModelSelection,
  saveModelSelection,
  selectionToConfigurable,
  browserContext,
  type ModelSelection,
} from "@/components/model-selector";

interface Ctx {
  selection: ModelSelection;
  setSelection: (s: ModelSelection) => void;
  buildConfigurable: () => Record<string, unknown>;
}

const ModelSelectionContext = createContext<Ctx | undefined>(undefined);

export function ModelSelectionProvider({ children }: { children: ReactNode }) {
  // Seed with the SSR default so the first client render matches the server;
  // the real (localStorage) selection loads in the mount effect below. Reading
  // localStorage in the useState initializer instead causes a hydration
  // mismatch (e.g. the TogglesMenu active-count badge) for returning users.
  const [selection, setSelectionState] = useState<ModelSelection>(
    DEFAULT_SELECTION,
  );

  // Re-read from localStorage after mount to avoid SSR mismatch
  useEffect(() => {
    setSelectionState(loadModelSelection());
  }, []);

  const setSelection = (s: ModelSelection) => {
    saveModelSelection(s);
    setSelectionState(s);
  };

  return (
    <ModelSelectionContext.Provider
      value={{
        selection,
        setSelection,
        buildConfigurable: () => ({
          ...selectionToConfigurable(selection),
          ...browserContext(),
        }),
      }}
    >
      {children}
    </ModelSelectionContext.Provider>
  );
}

export function useModelSelection(): Ctx {
  const ctx = useContext(ModelSelectionContext);
  if (!ctx) {
    throw new Error(
      "useModelSelection must be used inside ModelSelectionProvider",
    );
  }
  return ctx;
}
