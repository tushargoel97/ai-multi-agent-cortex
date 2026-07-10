"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Server, Boxes, Cpu, FlaskConical, Wrench, Bot } from "lucide-react";
import ProvidersPanel from "./_components/ProvidersPanel";
import ModelsPanel from "./_components/ModelsPanel";
import LocalModelsPanel from "./_components/LocalModelsPanel";
import FinetunePanel from "./_components/FinetunePanel";
import ToolsPanel from "./_components/ToolsPanel";
import AgentsPanel from "./_components/AgentsPanel";

type Tab = "providers" | "models" | "local" | "finetune" | "tools" | "agents";

const TABS: { id: Tab; label: string; icon: typeof Server }[] = [
  { id: "providers", label: "Providers", icon: Server },
  { id: "models", label: "Models", icon: Boxes },
  { id: "local", label: "Local Models", icon: Cpu },
  { id: "finetune", label: "Fine-Tuning", icon: FlaskConical },
  { id: "tools", label: "Tools & MCP", icon: Wrench },
  { id: "agents", label: "Agents", icon: Bot },
];

function AdminTabs() {
  const params = useSearchParams();
  const router = useRouter();
  const initial = (params.get("tab") as Tab) || "providers";
  const [tab, setTab] = useState<Tab>(initial);
  const [bump, setBump] = useState(0);

  useEffect(() => {
    const q = (params.get("tab") as Tab) || "providers";
    if (q !== tab) setTab(q);
  }, [params, tab]);

  const select = (t: Tab) => {
    setTab(t);
    router.replace(`/admin?tab=${t}`, { scroll: false });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Admin</h1>
        <p className="text-muted-foreground text-sm">
          Manage LLM providers, models, and self-hosted local models, all in one place.
        </p>
      </div>

      <div className="glass-tint inline-flex max-w-full items-center gap-1 overflow-x-auto rounded-full border p-1 shadow-sm">
        {TABS.map(({ id, label, icon: Icon }) => {
          const active = tab === id;
          return (
            <button
              key={id}
              onClick={() => select(id)}
              className={`flex shrink-0 items-center gap-2 rounded-full px-3.5 py-1.5 text-sm whitespace-nowrap transition-colors ${
                active
                  ? "bg-background/80 text-foreground font-medium shadow-sm"
                  : "text-muted-foreground hover:bg-background/40 hover:text-foreground"
              }`}
            >
              <Icon className="size-4" />
              {label}
            </button>
          );
        })}
      </div>

      <div className="glass-tint rounded-2xl border p-6 shadow-sm">
        {tab === "providers" && <ProvidersPanel onChanged={() => setBump((n) => n + 1)} />}
        {tab === "models" && <ModelsPanel refreshKey={bump} />}
        {tab === "local" && <LocalModelsPanel onChanged={() => setBump((n) => n + 1)} />}
        {tab === "finetune" && <FinetunePanel onChanged={() => setBump((n) => n + 1)} />}
        {tab === "tools" && <ToolsPanel />}
        {tab === "agents" && <AgentsPanel />}
      </div>
    </div>
  );
}

export default function AdminHome() {
  return (
    <Suspense fallback={<div className="text-muted-foreground p-4 text-sm">Loading…</div>}>
      <AdminTabs />
    </Suspense>
  );
}
