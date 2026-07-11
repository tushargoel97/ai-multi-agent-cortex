import { Client } from "@langchain/langgraph-sdk";

export function resolveApiUrl(apiUrl: string): string {
  if (!apiUrl || typeof window === "undefined") return apiUrl;
  return new URL(apiUrl, window.location.origin).toString().replace(/\/$/, "");
}

export function createClient(
  apiUrl: string,
  apiKey: string | undefined,
  authScheme: string | undefined,
) {
  return new Client({
    apiKey,
    apiUrl: resolveApiUrl(apiUrl),
    ...(authScheme && {
      defaultHeaders: {
        "X-Auth-Scheme": authScheme,
      },
    }),
  });
}
