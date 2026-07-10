import { AIMessage, ToolMessage } from "@langchain/langgraph-sdk";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, ChevronRight, Wrench, Sparkles } from "lucide-react";
import { ShoppingCards, BookingCards } from "./commerce-cards";

function isComplexValue(value: any): boolean {
  return Array.isArray(value) || (typeof value === "object" && value !== null);
}

export function ToolCalls({ toolCalls }: { toolCalls: AIMessage["tool_calls"] }) {
  if (!toolCalls || toolCalls.length === 0) return null;

  return (
    <div className="mx-auto grid max-w-3xl gap-2">
      {toolCalls.map((tc, idx) => (
        <ToolCallRow key={idx} name={tc.name} args={tc.args as Record<string, any>} />
      ))}
    </div>
  );
}

function ToolCallRow({ name, args }: { name: string; args: Record<string, any> }) {
  const [open, setOpen] = useState(false);
  const hasArgs = args && Object.keys(args).length > 0;

  return (
    <div className="border-border bg-background overflow-hidden rounded-lg border">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="text-foreground hover:bg-muted/50 flex w-full items-center gap-2 px-3 py-2 text-left text-sm transition-colors"
      >
        {open ? (
          <ChevronDown className="text-muted-foreground/70 h-4 w-4" />
        ) : (
          <ChevronRight className="text-muted-foreground/70 h-4 w-4" />
        )}
        <Sparkles className="text-muted-foreground/70 h-4 w-4" />
        <span className="text-muted-foreground">Thinking</span>
        <span className="text-muted-foreground/70"></span>
        <code className="bg-muted text-foreground rounded px-1.5 py-0.5 font-mono text-xs">
          {name}
        </code>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="border-border bg-muted/50 overflow-hidden border-t"
          >
            {hasArgs ? (
              <table className="divide-border min-w-full divide-y">
                <tbody className="divide-border divide-y">
                  {Object.entries(args).map(([key, value], argIdx) => (
                    <tr key={argIdx}>
                      <td className="text-foreground px-3 py-1.5 text-xs font-medium whitespace-nowrap">
                        {key}
                      </td>
                      <td className="text-muted-foreground px-3 py-1.5 text-xs">
                        {isComplexValue(value) ? (
                          <code className="bg-background rounded px-1.5 py-0.5 font-mono text-xs break-all">
                            {JSON.stringify(value, null, 2)}
                          </code>
                        ) : (
                          String(value)
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <code className="text-muted-foreground block px-3 py-2 text-xs">{"{}"}</code>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function ToolResult({ message }: { message: ToolMessage }) {
  const [open, setOpen] = useState(false);

  let parsedContent: any;
  let isJsonContent = false;

  try {
    if (typeof message.content === "string") {
      parsedContent = JSON.parse(message.content);
      isJsonContent = isComplexValue(parsedContent);
    }
  } catch {
    parsedContent = message.content;
  }

  // Rich cards for the commerce tools, the JSON comes straight from the tool
  // (not the model), so these render deterministically.
  if (isJsonContent && parsedContent && typeof parsedContent === "object") {
    if (message.name === "product_prices" && Array.isArray(parsedContent.offers)) {
      return <ShoppingCards data={parsedContent} />;
    }
    if (message.name === "find_bookings" && Array.isArray(parsedContent.options)) {
      return <BookingCards data={parsedContent} />;
    }
  }

  const contentStr = isJsonContent
    ? JSON.stringify(parsedContent, null, 2)
    : String(message.content);

  return (
    <div className="mx-auto grid max-w-3xl gap-2">
      <div className="border-border bg-background overflow-hidden rounded-lg border">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="text-foreground hover:bg-muted/50 flex w-full items-center gap-2 px-3 py-2 text-left text-sm transition-colors"
        >
          {open ? (
            <ChevronDown className="text-muted-foreground/70 h-4 w-4" />
          ) : (
            <ChevronRight className="text-muted-foreground/70 h-4 w-4" />
          )}
          <Wrench className="text-muted-foreground/70 h-4 w-4" />
          <span className="text-muted-foreground">Result from</span>
          <code className="bg-muted text-foreground rounded px-1.5 py-0.5 font-mono text-xs">
            {message.name ?? "tool"}
          </code>
        </button>
        <AnimatePresence initial={false}>
          {open && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.18 }}
              className="border-border bg-muted/50 overflow-hidden border-t"
            >
              <div className="max-h-96 overflow-auto p-3">
                {isJsonContent ? (
                  <pre className="text-foreground font-mono text-xs whitespace-pre-wrap">
                    {contentStr}
                  </pre>
                ) : (
                  <pre className="text-foreground font-mono text-xs whitespace-pre-wrap">
                    {contentStr}
                  </pre>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
