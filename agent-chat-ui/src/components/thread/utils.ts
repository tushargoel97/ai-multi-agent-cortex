import type { Message } from "@langchain/langgraph-sdk";

/**
 * Extracts a string summary from a message's content, supporting multimodal (text, image, file, etc.).
 * - If text is present, returns the joined text.
 * - If not, returns a label for the first non-text modality (e.g., 'Image', 'Other').
 * - If unknown, returns 'Multimodal message'.
 */
export function getContentString(content: Message["content"]): string {
  if (typeof content === "string") return content;
  const texts = content
    .filter((c): c is { type: "text"; text: string } => c.type === "text")
    .map((c) => c.text);
  return texts.join(" ");
}

/**
 * Extracts streamed reasoning / "thinking" text from a message's content.
 * Anthropic emits `{ type: "thinking", thinking }` blocks; some providers use
 * `{ type: "reasoning" }`. Returns "" when there is no thinking content (the
 * common case, e.g. models that omit raw thinking).
 */
export function getThinkingString(content: Message["content"]): string {
  if (typeof content === "string" || !Array.isArray(content)) return "";
  return content
    .map((c) => {
      const block = c as { type?: string; thinking?: string; reasoning?: string };
      if (block?.type === "thinking" && typeof block.thinking === "string") return block.thinking;
      if (block?.type === "reasoning" && typeof block.reasoning === "string")
        return block.reasoning;
      return "";
    })
    .join("")
    .trim();
}
