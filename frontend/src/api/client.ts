// Every call to the backend goes through this file: components never call fetch
// directly (same idea as routes vs services in the backend).
import type {
  Agent,
  AgentWrite,
  Conversation,
  ConversationSummary,
  EmbeddingModel,
  HealthResponse,
  KbDocument,
  KnowledgeBase,
  KnowledgeBaseCreate,
  LlmStatus,
  ProviderModels,
  Run,
  RunSummary,
  SearchResponse,
  ToolInfo,
} from "./types";

// Written into the JavaScript at build time (see the frontend Dockerfile)
export const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// An error with the HTTP status, so callers can react differently to 404, 422...
export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch {
    // fetch only throws when there is no response at all (server down, CORS...)
    throw new ApiError(0, "Can't reach the API. Check that the backend is running.");
  }

  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
  if (response.status === 204) {
    return undefined as T; // No Content: nothing to parse
  }
  return (await response.json()) as T;
}

// FastAPI sends errors as { "detail": "..." } (our HTTPExceptions), or as
// { "detail": [{ "loc": [...], "msg": "..." }] } when Pydantic validation fails (422).
export async function readErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") {
      return body.detail;
    }
    if (Array.isArray(body.detail)) {
      return body.detail.map(formatValidationError).join(" ");
    }
  } catch {
    // The body wasn't JSON: fall through to the generic message
  }
  return `Request failed (HTTP ${response.status}).`;
}

function formatValidationError(item: { loc?: unknown[]; msg?: string }): string {
  // Our own validators raise ValueError, which Pydantic prefixes with "Value error, "
  const message = (item.msg ?? "Invalid value.").replace(/^Value error, /, "");
  const field = item.loc?.[item.loc.length - 1];
  // loc is ["body", "name"] for a field, just ["body"] for a whole-object rule
  return typeof field === "string" && field !== "body" ? `${field}: ${message}` : message;
}

function jsonBody(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

// --- Endpoints --------------------------------------------------------------------

export const getHealth = () => request<HealthResponse>("/health");

export const listEmbeddingModels = () => request<EmbeddingModel[]>("/embedding-models");

export const listKnowledgeBases = () => request<KnowledgeBase[]>("/knowledge-bases");

export const createKnowledgeBase = (payload: KnowledgeBaseCreate) =>
  request<KnowledgeBase>("/knowledge-bases", jsonBody("POST", payload));

export const deleteKnowledgeBase = (kbId: number) =>
  request<void>(`/knowledge-bases/${kbId}`, { method: "DELETE" });

export const listDocuments = (kbId: number) =>
  request<KbDocument[]>(`/knowledge-bases/${kbId}/documents`);

export function uploadDocument(kbId: number, file: File) {
  // Files are sent as multipart/form-data. No Content-Type header here: the
  // browser sets it itself, with the boundary separating the parts.
  const form = new FormData();
  form.append("file", file);
  return request<KbDocument>(`/knowledge-bases/${kbId}/documents`, {
    method: "POST",
    body: form,
  });
}

export const deleteDocument = (kbId: number, documentId: number) =>
  request<void>(`/knowledge-bases/${kbId}/documents/${documentId}`, { method: "DELETE" });

export const searchKnowledgeBase = (kbId: number, query: string, topK: number) =>
  request<SearchResponse>(
    `/knowledge-bases/${kbId}/search`,
    jsonBody("POST", { query, top_k: topK }),
  );

// --- Module B: agents and runs ----------------------------------------------------

export const getLlmStatus = () => request<LlmStatus>("/llm");
export const listModels = () => request<ProviderModels[]>("/llm/models");

export const listTools = () => request<ToolInfo[]>("/tools");

export const listAgents = () => request<Agent[]>("/agents");

export const createAgent = (payload: AgentWrite) =>
  request<Agent>("/agents", jsonBody("POST", payload));

export const updateAgent = (agentId: number, payload: AgentWrite) =>
  request<Agent>(`/agents/${agentId}`, jsonBody("PUT", payload));

export const deleteAgent = (agentId: number) =>
  request<void>(`/agents/${agentId}`, { method: "DELETE" });

export const listRuns = (agentId: number, limit = 20) =>
  request<RunSummary[]>(`/agents/${agentId}/runs?limit=${limit}`);

export const getRun = (runId: number) => request<Run>(`/runs/${runId}`);

export const listConversations = (agentId: number) =>
  request<ConversationSummary[]>(`/agents/${agentId}/conversations`);

export const getConversation = (conversationId: number) =>
  request<Conversation>(`/conversations/${conversationId}`);

export const deleteConversation = (conversationId: number) =>
  request<void>(`/conversations/${conversationId}`, { method: "DELETE" });

// Starting a streamed run is NOT here: it isn't one request/response but a stream
// read over time. See hooks/useRunStream.ts.
