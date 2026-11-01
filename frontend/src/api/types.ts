// Shapes of the data exchanged with the backend.
// They mirror the Pydantic schemas in backend/app/schemas/ (knowledge_bases.py, agents.py):
// if a schema changes there, it must change here too.
// (Evolution: generate this file from the backend's OpenAPI description.)

export type HealthResponse = {
  status: string;
  database: string;
};

export type EmbeddingModel = {
  name: string;
  description: string;
  dimension: number;
  max_chunk_size: number;
  is_default: boolean;
};

export type KnowledgeBase = {
  id: number;
  name: string;
  description: string | null;
  embedding_model: string;
  chunk_size: number;
  chunk_overlap: number;
  created_at: string; // ISO date: JSON has no date type
  document_count: number;
};

// What the creation form sends. Optional fields fall back to the model's defaults.
export type KnowledgeBaseCreate = {
  name: string;
  description?: string;
  embedding_model?: string;
  chunk_size?: number;
  chunk_overlap?: number;
};

export type DocumentStatus = "pending" | "processing" | "ready" | "error";

// Named KbDocument, not Document: `Document` is already a built-in browser type
// (the web page itself), and reusing the name would cause confusing errors.
export type KbDocument = {
  id: number;
  knowledge_base_id: number;
  filename: string;
  status: DocumentStatus;
  error_message: string | null;
  created_at: string;
  chunk_count: number;
};

export type SearchHit = {
  chunk_id: number;
  document_id: number;
  knowledge_base_id: number;
  filename: string;
  chunk_index: number;
  content: string;
  score: number; // cosine similarity
};

export type SearchResponse = {
  query: string;
  embedding_model: string;
  results: SearchHit[];
};

// --- Module B: agents and runs (backend/app/schemas/agents.py) -------------------

export type LlmStatus = {
  mode: "api" | "mock";
  default_model: string;
  base_url: string | null;
};

// One group of the model dropdown: a provider and the models it offers (GET /llm/models)
export type ModelOption = {
  id: string; // the name sent to the provider, e.g. "qwen2.5:7b"
  label: string;
  tools: boolean | null; // supports tool calling? null = the provider doesn't say
  free: boolean | null;
};

export type ProviderModels = {
  provider: string; // "ollama", "openrouter", "default"
  label: string;
  default: boolean; // the provider used by agents without one
  error: string | null; // its models couldn't be listed
  models: ModelOption[];
};

export type ToolInfo = {
  name: string;
  description: string;
  automatic: boolean; // the KB search tool: enabled by linking a KB, not by a switch
  available: boolean;
  unavailable_reason: string | null; // e.g. "Needs TAVILY_API_KEY in .env"
};

export type Agent = {
  id: number;
  name: string;
  description: string | null;
  system_prompt: string;
  model: string | null; // null = the platform's default model
  provider: string | null; // null = the default provider
  tools: string[];
  knowledge_base_ids: number[];
  max_iterations: number;
  created_at: string;
};

export type AgentWrite = {
  name: string;
  description?: string | null;
  system_prompt: string;
  model?: string | null;
  provider?: string | null;
  tools: string[];
  knowledge_base_ids: number[];
  max_iterations: number;
};

// Where a piece of an answer came from: a KB chunk or a web page
export type DocumentSource = {
  type?: "document"; // optional: runs saved before Day 5 have no type
  rank: number;
  filename: string;
  chunk_index: number;
  chunk_id: number;
  document_id: number;
  knowledge_base_id: number;
  score: number;
  excerpt: string;
};

export type WebSource = {
  type: "web";
  rank: number;
  title: string;
  url: string;
  score: number | null;
  published_date: string | null;
  excerpt: string;
};

export type Source = DocumentSource | WebSource;

export type RunStep = {
  position: number;
  kind: "llm_call" | "tool_call";
  duration_ms: number;
  content: string | null; // llm_call: text written next to tool calls, or the answer
  prompt_tokens: number;
  completion_tokens: number;
  tool_name: string | null;
  arguments: Record<string, unknown> | null;
  output: string | null;
  is_error: boolean;
  sources: Source[];
};

export type RunStatus = "running" | "completed" | "max_iterations" | "error";

export type RunSummary = {
  id: number;
  agent_id: number;
  conversation_id: number | null; // null: a run from before conversations existed
  user_message: string;
  status: RunStatus;
  answer: string | null;
  error_message: string | null;
  model: string;
  provider: string | null; // null: mock mode, or a run saved before providers existed
  prompt_tokens: number;
  completion_tokens: number;
  duration_ms: number | null;
  created_at: string;
};

export type Run = RunSummary & {
  system_prompt: string;
  tools: string[];
  steps: RunStep[];
};

export type ConversationSummary = {
  id: number;
  agent_id: number;
  title: string; // the first question, shortened
  created_at: string;
  turn_count: number;
  last_activity: string;
};

export type Conversation = {
  id: number;
  agent_id: number;
  title: string;
  created_at: string;
  runs: Run[]; // in order, each with its full trace
};
