import { initApiPassthrough } from "langgraph-nextjs-api-passthrough";

export const { GET, POST, PUT, PATCH, DELETE, OPTIONS, runtime } = initApiPassthrough({
  apiUrl: process.env.LANGGRAPH_API_URL ?? "remove-me",
  apiKey: process.env.LANGSMITH_API_KEY ?? "remove-me",
  baseRoute: "v1",
  runtime: "edge",
});
