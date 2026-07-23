export const AGENT_PROGRESS_PHASES = [
  "routing",
  "thinking",
  "researching",
  "collating",
  "refining",
  "generating_image",
] as const;

export type AgentProgressPhase = (typeof AGENT_PROGRESS_PHASES)[number];

export interface AgentProgressEvent {
  type: "agent_progress";
  phase: AgentProgressPhase;
  tool?: string;
}

export function isAgentProgressEvent(value: unknown): value is AgentProgressEvent {
  if (!value || typeof value !== "object") return false;
  const event = value as Partial<AgentProgressEvent>;
  return (
    event.type === "agent_progress" &&
    AGENT_PROGRESS_PHASES.includes(event.phase as AgentProgressPhase) &&
    (event.tool === undefined || typeof event.tool === "string")
  );
}
