"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

export interface ConfirmOptions {
  title: string;
  description?: React.ReactNode;
  confirmText?: string;
  cancelText?: string;
  /** Style the confirm action as destructive. Defaults to true (most
   * confirmations are deletes). Set false for neutral confirmations. */
  destructive?: boolean;
}

type ConfirmFn = (options: ConfirmOptions) => Promise<boolean>;

const ConfirmContext = React.createContext<ConfirmFn | null>(null);

/**
 * App-wide in-UI confirmation dialog. Wrap the app once with this provider,
 * then call `const confirm = useConfirm()` and `await confirm({ title, ... })`
 *, a drop-in, promise-based replacement for the browser `confirm()`.
 *
 * Standard: prefer this over `window.confirm` for every confirmation.
 */
export function ConfirmDialogProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [options, setOptions] = React.useState<ConfirmOptions | null>(null);
  const resolverRef = React.useRef<((value: boolean) => void) | null>(null);

  const confirm = React.useCallback<ConfirmFn>((opts) => {
    setOptions(opts);
    return new Promise<boolean>((resolve) => {
      resolverRef.current = resolve;
    });
  }, []);

  const settle = React.useCallback((result: boolean) => {
    resolverRef.current?.(result);
    resolverRef.current = null;
    setOptions(null);
  }, []);

  return (
    <ConfirmContext.Provider value={confirm}>
      {children}
      <Dialog
        open={options !== null}
        onOpenChange={(open) => {
          if (!open) settle(false);
        }}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{options?.title}</DialogTitle>
            {options?.description != null && (
              <DialogDescription className="whitespace-pre-line">
                {options.description}
              </DialogDescription>
            )}
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => settle(false)}>
              {options?.cancelText ?? "Cancel"}
            </Button>
            <Button
              variant={options?.destructive === false ? "default" : "destructive"}
              onClick={() => settle(true)}
            >
              {options?.confirmText ?? "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </ConfirmContext.Provider>
  );
}

export function useConfirm(): ConfirmFn {
  const ctx = React.useContext(ConfirmContext);
  if (ctx === null) {
    throw new Error("useConfirm must be used within a ConfirmDialogProvider");
  }
  return ctx;
}
