export type Role = "user" | "agent" | "admin";

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatMeta = {
  intent?: string | null;
  confidence?: number | null;
  route_path?: string[];
  ticket_id?: string | null;
  needs_human?: boolean;
  trace_id?: string;
  token_usage?: number;
  evidence?: Record<string, unknown>[];
  data_result?: Record<string, unknown>[];
};

export type HealthResponse = {
  status: string;
  model_provider: string;
  knowledge_ready: boolean;
};

export type EvalSummary = {
  report_path: string;
  total: number;
  intent_accuracy: number;
  answer_accuracy: number;
  refuse_rational_rate: number;
  ticket_completion_rate: number;
  avg_recall_at_k: number | null;
};
