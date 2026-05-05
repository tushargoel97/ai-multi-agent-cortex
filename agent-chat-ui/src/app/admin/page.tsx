"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Server, Boxes, Cpu } from "lucide-react";
import ProvidersPanel from "./_components/ProvidersPanel";
import ModelsPanel from "./_components/ModelsPanel";
import LocalModelsPanel from "./_components/LocalModelsPanel";

type Tab = "providers" | "models" | "local";

const TABS: { id: Tab; label: string; icon: typeof Server }[] = [
  { id: "providers", label: "Providers", icon: Server },
  { id: "models", label: "Models", icon: Boxes },
  { id: "local", label: "Local Models", icon: Cpu },
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
        <p className="text-sm text-muted-foreground">
          Manage LLM providers, models, and self-hosted local models — all in
          one place.
        </p>
      </div>

      <div className="flex gap-1 border-b">
        {TABS.map(({ id, label, icon: Icon }) => {
          const active = tab === id;
          return (
            <button
              key={id}
              onClick={() => select(id)}
              className={`flex items-center gap-2 border-b-2 px-4 py-2 text-sm transition ${
                active
                  ? "border-slate-900 font-semibold text-slate-900"
                  : "border-transparent text-slate-500 hover:text-slate-800"
              }`}
            >
              <Icon className="size-4" />
              {label}
            </button>
          );
        })}
      </div>

      <div>
        {tab === "providers" && (
          <ProvidersPanel onChanged={() => setBump((n) => n + 1)} />
        )}
        {tab === "models" && <ModelsPanel refreshKey={bump} />}
        {tab === "local" && (
          <LocalModelsPanel onChanged={() => setBump((n) => n + 1)} />
        )}
      </div>
    </div>
  );
}

export default function AdminHome() {
  return (
    <Suspense fallback={<div className="p-4 text-sm text-slate-500">Loading…</div>}>
      <AdminTabs />
    </Suspense>
  );
}
