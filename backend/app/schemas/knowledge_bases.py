"""API schemas for knowledge bases: what the API accepts and what it returns.

They are separate from the SQLAlchemy models on purpose: the database schema and
the public API contract evolve independently (e.g. the API never exposes vectors).
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.embeddings import DEFAULT_EMBEDDING_MODEL, get_model_config

MIN_CHUNK_SIZE = 20

DocumentStatus = Literal["pending", "processing", "ready", "error"]


# --- Embedding models -----------------------------------------------------------


class EmbeddingModelRead(BaseModel):
    name: str
    description: str
    dimension: int
    max_chunk_size: int
    is_default: bool


# --- Knowledge bases --------------------------------------------------------------


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # Optional: when omitted, sensible defaults are derived from the model
    chunk_size: int | None = Field(default=None, ge=MIN_CHUNK_SIZE)
    chunk_overlap: int | None = Field(default=None, ge=0)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name cannot be blank.")
        return value

    @field_validator("embedding_model")
    @classmethod
    def model_is_allowed(cls, value: str) -> str:
        get_model_config(value)  # raises ValueError with the list of allowed models
        return value

    @model_validator(mode="after")
    def fill_and_check_chunking(self) -> "KnowledgeBaseCreate":
        config = get_model_config(self.embedding_model)

        if self.chunk_size is None:
            self.chunk_size = config.max_chunk_size
        elif self.chunk_size > config.max_chunk_size:
            raise ValueError(
                f"chunk_size {self.chunk_size} is too large for {config.name}: "
                f"maximum {config.max_chunk_size} tokens."
            )

        if self.chunk_overlap is None:
            self.chunk_overlap = self.chunk_size // 10
        elif self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")

        return self


class KnowledgeBaseRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    created_at: datetime
    document_count: int = 0


# --- Documents ------------------------------------------------------------------


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    knowledge_base_id: int
    filename: str
    status: DocumentStatus
    error_message: str | None
    created_at: datetime
    chunk_count: int = 0


# --- Search ---------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query cannot be blank.")
        return value


class SearchHit(BaseModel):
    chunk_id: int
    document_id: int
    knowledge_base_id: int
    filename: str
    chunk_index: int
    content: str
    score: float  # cosine similarity: 1 = same direction, 0 = unrelated


class SearchResponse(BaseModel):
    query: str
    embedding_model: str
    results: list[SearchHit]
