"use client";

import { Thread } from "@/components/thread";
import { StreamProvider } from "@/providers/Stream";
import { ThreadProvider } from "@/providers/Thread";
import { ModelSelectionProvider } from "@/providers/ModelSelection";
import { ArtifactProvider } from "@/components/thread/artifact";
import { Toaster } from "@/components/ui/sonner";
import React from "react";

export default function DemoPage(): React.ReactNode {
  return (
    <React.Suspense fallback={<div>Loading (layout)...</div>}>
      <Toaster />
      <ThreadProvider>
        <StreamProvider>
          <ModelSelectionProvider>
            <ArtifactProvider>
              <Thread />
            </ArtifactProvider>
          </ModelSelectionProvider>
        </StreamProvider>
      </ThreadProvider>
    </React.Suspense>
  );
}
