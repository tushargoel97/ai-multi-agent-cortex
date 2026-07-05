"use client";

import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import {
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
  const [selection, setSelectionState] = useState<ModelSelection>(() =>
    loadModelSelection(),
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
