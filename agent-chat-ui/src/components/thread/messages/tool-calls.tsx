import { AIMessage, ToolMessage } from "@langchain/langgraph-sdk";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, ChevronRight, Wrench, Sparkles } from "lucide-react";

function isComplexValue(value: any): boolean {
  return Array.isArray(value) || (typeof value === "object" && value !== null);
}

export function ToolCalls({
  toolCalls,
}: {
  toolCalls: AIMessage["tool_calls"];
}) {
  if (!toolCalls || toolCalls.length === 0) return null;

  return (
    <div className="mx-auto grid max-w-3xl gap-2">
      {toolCalls.map((tc, idx) => (
        <ToolCallRow
          key={idx}
          name={tc.name}
          args={tc.args as Record<string, any>}
        />
      ))}
    </div>
  );
}

function ToolCallRow({
  name,
  args,
}: {
  name: string;
  args: Record<string, any>;
}) {
  const [open, setOpen] = useState(false);
  const hasArgs = args && Object.keys(args).length > 0;

  return (
    <div className="overflow-hidden rounded-lg border border-gray-200 bg-white">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-gray-700 transition-colors hover:bg-gray-50"
      >
        {open ? (
          <ChevronDown className="h-4 w-4 text-gray-400" />
        ) : (
          <ChevronRight className="h-4 w-4 text-gray-400" />
        )}
        <Sparkles className="h-4 w-4 text-gray-400" />
        <span className="text-gray-500">Thinking</span>
        <span className="text-gray-400"></span>
        <code className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-xs text-gray-700">
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
            className="overflow-hidden border-t border-gray-200 bg-gray-50"
          >
            {hasArgs ? (
              <table className="min-w-full divide-y divide-gray-200">
                <tbody className="divide-y divide-gray-200">
                  {Object.entries(args).map(([key, value], argIdx) => (
                    <tr key={argIdx}>
                      <td className="px-3 py-1.5 text-xs font-medium whitespace-nowrap text-gray-700">
                        {key}
                      </td>
                      <td className="px-3 py-1.5 text-xs text-gray-600">
                        {isComplexValue(value) ? (
                          <code className="rounded bg-white px-1.5 py-0.5 font-mono text-xs break-all">
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
              <code className="block px-3 py-2 text-xs text-gray-500">
                {"{}"}
              </code>
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

  const contentStr = isJsonContent
    ? JSON.stringify(parsedContent, null, 2)
    : String(message.content);

  return (
    <div className="mx-auto grid max-w-3xl gap-2">
      <div className="overflow-hidden rounded-lg border border-gray-200 bg-white">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-gray-700 transition-colors hover:bg-gray-50"
        >
          {open ? (
            <ChevronDown className="h-4 w-4 text-gray-400" />
          ) : (
            <ChevronRight className="h-4 w-4 text-gray-400" />
          )}
          <Wrench className="h-4 w-4 text-gray-400" />
          <span className="text-gray-500">Result from</span>
          <code className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-xs text-gray-700">
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
              className="overflow-hidden border-t border-gray-200 bg-gray-50"
            >
              <div className="max-h-96 overflow-auto p-3">
                {isJsonContent ? (
                  <pre className="font-mono text-xs whitespace-pre-wrap text-gray-700">
                    {contentStr}
                  </pre>
                ) : (
                  <pre className="font-mono text-xs whitespace-pre-wrap text-gray-700">
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
