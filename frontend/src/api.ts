import type { ChatMeta, EvalSummary, HealthResponse, Role } from "./types";

export async function fetchHealth(apiBase: string): Promise<HealthResponse> {
  const r = await fetch(`${apiBase}/health`);
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json() as Promise<HealthResponse>;
}

export async function ingestKnowledge(apiBase: string): Promise<{ ingested_chunks: number }> {
  const r = await fetch(`${apiBase}/knowledge/ingest`, { method: "POST" });
  if (!r.ok) throw new Error(`ingest ${r.status}`);
  return r.json() as Promise<{ ingested_chunks: number }>;
}

export async function runEval(apiBase: string): Promise<EvalSummary> {
  const r = await fetch(`${apiBase}/eval/run`, { method: "POST" });
  if (!r.ok) throw new Error(`eval ${r.status}`);
  return r.json() as Promise<EvalSummary>;
}

export type StreamHandlers = {
  onMeta: (meta: ChatMeta) => void;
  onToken: (chunk: string) => void;
};

function parseSseBlock(
  block: string,
  handlers: StreamHandlers,
): { done: boolean } {
  let eventName = "message";
  let data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    else if (line.startsWith("data:")) data = line.slice(5).trim();
  }
  if (!data && eventName === "message") return { done: false };
  if (eventName === "meta") {
    handlers.onMeta(JSON.parse(data) as ChatMeta);
  } else if (eventName === "token") {
    handlers.onToken(data);
  } else if (eventName === "done") {
    return { done: true };
  }
  return { done: false };
}

export async function streamChat(
  apiBase: string,
  body: {
    message: string;
    thread_id: string;
    user_id: string;
    role: Role;
  },
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<string> {
  const r = await fetch(`${apiBase}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });
  if (!r.ok || !r.body) throw new Error(`chat stream ${r.status}`);

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const block of parts) {
      const trimmed = block.trim();
      if (!trimmed) continue;
      const result = parseSseBlock(trimmed, {
        onMeta: handlers.onMeta,
        onToken: (chunk) => {
          answer += chunk;
          handlers.onToken(chunk);
        },
      });
      if (result.done) return answer;
    }
  }
  return answer;
}

export async function chatOnce(
  apiBase: string,
  body: {
    message: string;
    thread_id: string;
    user_id: string;
    role: Role;
  },
): Promise<{ answer: string; meta: ChatMeta }> {
  const r = await fetch(`${apiBase}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, stream: false }),
  });
  if (!r.ok) throw new Error(`chat ${r.status}`);
  const payload = (await r.json()) as ChatMeta & { answer?: string };
  return {
    answer: payload.answer ?? "",
    meta: payload,
  };
}
