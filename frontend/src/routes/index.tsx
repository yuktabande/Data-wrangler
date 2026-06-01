import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  FileSpreadsheet,
  Loader2,
  Paperclip,
  RefreshCw,
  Send,
} from "lucide-react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Data Wrangler" },
      { name: "description", content: "Run the Excel data wrangling agent." },
      { property: "og:title", content: "Data Wrangler" },
      { property: "og:description", content: "Run the Excel data wrangling agent." },
    ],
  }),
  component: Index,
});

type ApiStatus = {
  ok: boolean;
  inputExists: boolean;
  inputFile: string;
  outputDir: string;
};

type ApiResult = {
  ok: boolean;
  error?: string;
  filename?: string;
  summary?: Record<string, SheetSummary>;
  suggestions?: string;
  parsed_instruction?: {
    action?: string;
    sheets?: string[];
    columns?: string[];
  };
  execution_result?: {
    status: "success" | "error";
    message: string;
    output_path?: string;
    result_shape?: number[];
    chart_types?: string[];
    cleaning_steps?: string[];
  };
};

type SheetSummary = {
  rows: number;
  columns: number;
  column_names: string[];
  null_counts: Record<string, number>;
  dtypes: Record<string, string>;
};

type ChatMessage = {
  id: string;
  role: "user" | "agent";
  text: string;
  detail?: string;
  status?: "success" | "error";
};

const PROMPTS = [
  "Summarize every sheet",
  "Clean the Sales data and remove duplicates",
  "Merge Sales and Customers on Customer_ID",
  "Create visualizations for Sales data",
];

function formatSummary(summary?: Record<string, SheetSummary>) {
  if (!summary) return "No workbook summary loaded yet.";
  return Object.entries(summary)
    .map(([sheet, item]) => `${sheet}: ${item.rows} rows, ${item.columns} columns`)
    .join("\n");
}

function formatExecution(result: ApiResult) {
  if (result.error) return result.error;
  const exec = result.execution_result;
  if (!exec) return "Done.";

  const lines = [exec.message];
  if (result.parsed_instruction?.action) {
    lines.push(`Action: ${result.parsed_instruction.action}`);
  }
  if (result.parsed_instruction?.sheets?.length) {
    lines.push(`Sheets: ${result.parsed_instruction.sheets.join(", ")}`);
  }
  if (exec.output_path) {
    lines.push(`Output: ${exec.output_path}`);
  }
  if (exec.result_shape) {
    lines.push(`Shape: ${exec.result_shape.join(" x ")}`);
  }
  if (exec.chart_types?.length) {
    lines.push(`Charts: ${exec.chart_types.join(", ")}`);
  }
  if (exec.cleaning_steps?.length) {
    lines.push(`Cleaning: ${exec.cleaning_steps.join("; ")}`);
  }
  return lines.join("\n");
}

async function postJson(path: string, body: Record<string, unknown> = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = (await response.json()) as ApiResult;
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function Index() {
  const [status, setStatus] = useState<ApiStatus | null>(null);
  const [summary, setSummary] = useState<Record<string, SheetSummary>>();
  const [suggestions, setSuggestions] = useState("");
  const [value, setValue] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "agent",
      text: "Ready to analyze the Excel workbook. Upload a file or use the existing input.xlsx, then run an instruction.",
    },
  ]);
  const [isBusy, setIsBusy] = useState(false);
  const [fileName, setFileName] = useState("input.xlsx");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch("/api/status")
      .then((response) => response.json())
      .then((data: ApiStatus) => setStatus(data))
      .catch(() => setStatus(null));
  }, []);

  async function analyzeWorkbook() {
    setIsBusy(true);
    try {
      const result = await postJson("/api/analyze");
      setSummary(result.summary);
      setSuggestions(result.suggestions || "");
      setMessages((items) => [
        ...items,
        {
          id: crypto.randomUUID(),
          role: "agent",
          text: "Workbook analyzed.",
          detail: formatSummary(result.summary),
          status: "success",
        },
      ]);
    } catch (error) {
      setMessages((items) => [
        ...items,
        {
          id: crypto.randomUUID(),
          role: "agent",
          text: "Analysis failed.",
          detail: error instanceof Error ? error.message : String(error),
          status: "error",
        },
      ]);
    } finally {
      setIsBusy(false);
    }
  }

  async function uploadFile(file: File) {
    setIsBusy(true);
    setFileName(file.name);
    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        headers: { "X-Filename": file.name },
        body: file,
      });
      const result = (await response.json()) as ApiResult;
      if (!response.ok || result.ok === false) {
        throw new Error(result.error || "Upload failed");
      }
      setSummary(result.summary);
      setSuggestions(result.suggestions || "");
      setMessages((items) => [
        ...items,
        {
          id: crypto.randomUUID(),
          role: "agent",
          text: `${file.name} uploaded and analyzed.`,
          detail: formatSummary(result.summary),
          status: "success",
        },
      ]);
    } catch (error) {
      setMessages((items) => [
        ...items,
        {
          id: crypto.randomUUID(),
          role: "agent",
          text: "Upload failed.",
          detail: error instanceof Error ? error.message : String(error),
          status: "error",
        },
      ]);
    } finally {
      setIsBusy(false);
    }
  }

  async function executeInstruction(instruction = value.trim()) {
    if (!instruction || isBusy) return;
    setValue("");
    setIsBusy(true);
    setMessages((items) => [
      ...items,
      { id: crypto.randomUUID(), role: "user", text: instruction },
    ]);

    try {
      const result = await postJson("/api/execute", { instruction });
      const execStatus = result.execution_result?.status === "error" ? "error" : "success";
      setMessages((items) => [
        ...items,
        {
          id: crypto.randomUUID(),
          role: "agent",
          text: execStatus === "success" ? "Task complete." : "Task failed.",
          detail: formatExecution(result),
          status: execStatus,
        },
      ]);
    } catch (error) {
      setMessages((items) => [
        ...items,
        {
          id: crypto.randomUUID(),
          role: "agent",
          text: "Task failed.",
          detail: error instanceof Error ? error.message : String(error),
          status: "error",
        },
      ]);
    } finally {
      setIsBusy(false);
    }
  }

  const sheetCount = summary ? Object.keys(summary).length : 0;
  const apiOnline = status?.ok === true;

  return (
    <main className="min-h-screen bg-[#f7f8f5] text-neutral-950">
      <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col px-5 py-5">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-200 pb-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-neutral-950 text-white">
              <FileSpreadsheet size={20} />
            </div>
            <div>
              <h1 className="text-xl font-semibold">Data Wrangler</h1>
              <p className="text-sm text-neutral-500">Excel agent connected to P3</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm">
            {apiOnline ? (
              <CheckCircle2 className="text-green-600" size={16} />
            ) : (
              <AlertCircle className="text-red-600" size={16} />
            )}
            <span className={apiOnline ? "text-green-700" : "text-red-700"}>
              {apiOnline ? "API online" : "API offline"}
            </span>
          </div>
        </header>

        <div className="grid flex-1 gap-5 py-5 lg:grid-cols-[340px_minmax(0,1fr)]">
          <aside className="flex flex-col gap-4 border-r border-neutral-200 pr-0 lg:pr-5">
            <section className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2 className="text-sm font-semibold">Workbook</h2>
                  <p className="mt-1 text-xs text-neutral-500">{fileName}</p>
                </div>
                <span className="rounded-md bg-neutral-100 px-2 py-1 text-xs text-neutral-600">
                  {sheetCount || "No"} sheets
                </span>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept=".xlsx,.xls"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void uploadFile(file);
                  event.target.value = "";
                }}
              />

              <div className="mt-4 flex gap-2">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isBusy}
                  className="inline-flex flex-1 items-center justify-center gap-2 rounded-md border border-neutral-200 bg-white px-3 py-2 text-sm font-medium text-neutral-800 hover:bg-neutral-50 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <Paperclip size={16} />
                  Upload
                </button>
                <button
                  onClick={() => void analyzeWorkbook()}
                  disabled={isBusy}
                  className="inline-flex flex-1 items-center justify-center gap-2 rounded-md bg-neutral-950 px-3 py-2 text-sm font-medium text-white hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isBusy ? <Loader2 className="animate-spin" size={16} /> : <RefreshCw size={16} />}
                  Analyze
                </button>
              </div>
            </section>

            <section className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
              <h2 className="text-sm font-semibold">Suggested Tasks</h2>
              <div className="mt-3 flex flex-col gap-2">
                {PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => void executeInstruction(prompt)}
                    disabled={isBusy}
                    className="rounded-md border border-neutral-200 px-3 py-2 text-left text-sm text-neutral-700 hover:border-neutral-300 hover:bg-neutral-50 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </section>

            <section className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
              <h2 className="text-sm font-semibold">AI Suggestions</h2>
              <pre className="mt-3 max-h-56 whitespace-pre-wrap rounded-md bg-neutral-50 p-3 text-xs leading-5 text-neutral-600">
                {suggestions || "Run Analyze to generate suggestions from the workbook."}
              </pre>
            </section>
          </aside>

          <section className="flex min-h-[620px] flex-col rounded-lg border border-neutral-200 bg-white shadow-sm">
            <div className="border-b border-neutral-200 px-4 py-3">
              <h2 className="text-sm font-semibold">Agent Console</h2>
              <p className="text-xs text-neutral-500">
                Type a natural language instruction. Outputs are saved in the P3 output folder.
              </p>
            </div>

            <div className="flex-1 overflow-y-auto p-4">
              <div className="flex flex-col gap-3">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={
                      message.role === "user"
                        ? "ml-auto max-w-[78%] rounded-lg bg-neutral-950 px-4 py-3 text-sm text-white"
                        : "mr-auto max-w-[82%] rounded-lg border border-neutral-200 bg-neutral-50 px-4 py-3 text-sm text-neutral-800"
                    }
                  >
                    <div className="flex items-center gap-2 font-medium">
                      {message.role === "agent" && message.status === "success" ? (
                        <CheckCircle2 className="text-green-600" size={16} />
                      ) : null}
                      {message.role === "agent" && message.status === "error" ? (
                        <AlertCircle className="text-red-600" size={16} />
                      ) : null}
                      <span>{message.text}</span>
                    </div>
                    {message.detail ? (
                      <pre className="mt-2 whitespace-pre-wrap text-xs leading-5 opacity-80">
                        {message.detail}
                      </pre>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>

            <form
              className="border-t border-neutral-200 p-4"
              onSubmit={(event) => {
                event.preventDefault();
                void executeInstruction();
              }}
            >
              <div className="flex gap-2">
                <textarea
                  value={value}
                  onChange={(event) => setValue(event.target.value)}
                  placeholder="Example: merge Sales and Customers on Customer_ID"
                  rows={2}
                  className="min-h-12 flex-1 resize-none rounded-md border border-neutral-200 px-3 py-2 text-sm outline-none focus:border-green-500 focus:ring-4 focus:ring-green-100"
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      void executeInstruction();
                    }
                  }}
                />
                <button
                  type="submit"
                  disabled={isBusy || !value.trim()}
                  className="inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-md bg-green-600 text-white hover:bg-green-700 disabled:cursor-not-allowed disabled:opacity-60"
                  aria-label="Run instruction"
                >
                  {isBusy ? <Loader2 className="animate-spin" size={18} /> : <Send size={18} />}
                </button>
              </div>
            </form>
          </section>
        </div>
      </div>
    </main>
  );
}
