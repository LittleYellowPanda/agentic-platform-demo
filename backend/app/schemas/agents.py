"""API schemas for agents and their runs."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.tools.catalog import SELECTABLE_TOOLS

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely and precisely."

RunStatus = Literal["running", "completed", "max_iterations", "error"]


class ToolInfoRead(BaseModel):
    name: str
    description: str
    automatic: bool
    available: bool
    unavailable_reason: str | None


class LLMStatusRead(BaseModel):
    mode: Literal["api", "mock"]
    default_model: str
    base_url: str | None


class ModelOptionRead(BaseModel):
    id: str
    label: str
    tools: bool | None  # None: the provider doesn't say
    free: bool | None


class ProviderModelsRead(BaseModel):
    """One group of the model dropdown."""

    provider: str
    label: str
    default: bool  # the provider of LLM_* variables (agents without a provider use it)
    error: str | None  # couldn't list its models (down, wrong URL...)
    models: list[ModelOptionRead]


class AgentWrite(BaseModel):
    """Body of POST /agents and PUT /agents/{id}. Unlike a KB, an agent can be edited:
    nothing stored depends on its configuration (each run keeps its own snapshot)."""

    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT, min_length=1, max_length=20_000
    )
    model: str | None = Field(default=None, max_length=255)  # None = platform default
    provider: str | None = Field(default=None, max_length=50)  # None = default provider
    tools: list[str] = Field(default_factory=list)
    knowledge_base_ids: list[int] = Field(default_factory=list)
    max_iterations: int = Field(default=6, ge=1, le=15)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name cannot be blank.")
        return value

    @field_validator("model")
    @classmethod
    def blank_model_means_default(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("provider")
    @classmethod
    def blank_provider_means_default(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def provider_needs_a_model(self) -> "AgentWrite":
        # LLM_MODEL belongs to the default provider: another provider needs its own model
        if self.provider is not None and self.model is None:
            raise ValueError("Choose a model for this provider.")
        return self

    @field_validator("tools")
    @classmethod
    def tools_exist(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(SELECTABLE_TOOLS))
        if unknown:
            allowed = ", ".join(sorted(SELECTABLE_TOOLS))
            raise ValueError(
                f"Unknown tool(s): {', '.join(unknown)}. Allowed: {allowed}."
            )
        return sorted(set(value))  # no duplicates, stable order

    @field_validator("knowledge_base_ids")
    @classmethod
    def unique_ids(cls, value: list[int]) -> list[int]:
        return sorted(set(value))


class AgentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    system_prompt: str
    model: str | None
    provider: str | None
    tools: list[str]
    knowledge_base_ids: list[int]
    max_iterations: int
    created_at: datetime


class RunCreate(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # None: start a new conversation. Otherwise: continue this one (its history is
    # sent to the model). Must belong to the agent: checked by the route (404).
    conversation_id: int | None = None

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be blank.")
        return value


class RunStepRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    position: int
    kind: Literal["llm_call", "tool_call"]
    duration_ms: int
    content: str | None
    prompt_tokens: int
    completion_tokens: int
    tool_name: str | None
    arguments: dict | None
    output: str | None
    is_error: bool
    sources: list[dict]


class RunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    agent_id: int
    conversation_id: int | None
    user_message: str
    status: RunStatus
    answer: str | None
    error_message: str | None
    model: str
    provider: str | None
    prompt_tokens: int
    completion_tokens: int
    duration_ms: int | None
    created_at: datetime


class RunRead(RunSummary):
    system_prompt: str
    tools: list[str]
    steps: list[RunStepRead]


class ConversationSummary(BaseModel):
    """One line of the conversation list."""

    id: int
    agent_id: int
    title: str
    created_at: datetime
    turn_count: int
    last_activity: datetime


class ConversationRead(BaseModel):
    """A whole thread: its runs in order, each with its full trace."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    agent_id: int
    title: str
    created_at: datetime
    runs: list[RunRead]
