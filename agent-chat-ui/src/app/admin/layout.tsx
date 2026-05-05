"use client";

import { ReactNode, useState, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import Link from "next/link";
import { ADMIN_TOKEN_KEY, getAdminToken } from "./token";
import { Lock, LogOut } from "lucide-react";

export default function AdminLayout({ children }: { children: ReactNode }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [authed, setAuthed] = useState(false);

  useEffect(() => {
    if (getAdminToken()) setAuthed(true);
  }, []);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const res = await fetch("/api/admin/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data?.error ?? "Login failed");
        return;
      }
      window.localStorage.setItem(ADMIN_TOKEN_KEY, data.token);
      setAuthed(true);
    } catch {
      setError("Network error");
    } finally {
      setSubmitting(false);
    }
  };

  if (!authed) {
    return (
      <div className="flex min-h-screen w-full items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 p-4">
        <form
          onSubmit={onSubmit}
          className="flex w-full max-w-sm flex-col gap-5 rounded-xl border bg-white p-8 shadow-lg"
        >
          <div className="flex flex-col items-center gap-2">
            <div className="flex size-10 items-center justify-center rounded-full bg-slate-900 text-white">
              <Lock className="size-5" />
            </div>
            <h1 className="text-xl font-semibold">Cortex Admin</h1>
            <p className="text-center text-xs text-muted-foreground">
              Sign in to manage LLM providers and models.
            </p>
          </div>

          <div className="flex flex-col gap-2">
            <Label htmlFor="username">Username</Label>
            <Input
              id="username"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          {error && (
            <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-700">
              {error}
            </p>
          )}

          <Button type="submit" disabled={submitting}>
            {submitting ? "Signing in…" : "Sign in"}
          </Button>

          <Link
            href="/chat"
            className="text-center text-xs text-muted-foreground hover:underline"
          >
            ← Back to chat
          </Link>
        </form>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="flex items-center gap-6 border-b bg-white px-6 py-3 shadow-sm">
        <Link
          href="/admin"
          className="flex items-center gap-2 font-semibold tracking-tight"
        >
          <div className="flex size-7 items-center justify-center rounded-md bg-slate-900 text-white">
            <Lock className="size-4" />
          </div>
          Cortex Admin
        </Link>
        <div className="ml-auto flex items-center gap-3">
          <Link
            href="/chat"
            className="text-sm text-muted-foreground hover:underline"
          >
            ← Back to chat
          </Link>
          <button
            onClick={() => {
              window.localStorage.removeItem(ADMIN_TOKEN_KEY);
              setAuthed(false);
              setUsername("");
              setPassword("");
            }}
            className="flex items-center gap-1.5 rounded-md border bg-white px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-50"
          >
            <LogOut className="size-4" />
            Sign out
          </button>
        </div>
      </header>
      <main className="mx-auto max-w-6xl p-6">{children}</main>
    </div>
  );
}
