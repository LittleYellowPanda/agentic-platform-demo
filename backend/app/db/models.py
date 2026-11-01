from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

EMBEDDING_DIM = 384


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(String(255))
    chunk_size: Mapped[int] = mapped_column(default=500)
    chunk_overlap: Mapped[int] = mapped_column(default=50)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    documents: Mapped[list["Document"]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    knowledge_base_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    knowledge_base_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    document: Mapped["Document"] = relationship(back_populates="chunks")
    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


# --- Module B: agents and their runs ---------------------------------------------------

# Many-to-many: an agent can search several KBs, a KB can serve several agents.
# A pure link table, so a plain Table rather than a model class.
agent_knowledge_bases = Table(
    "agent_knowledge_bases",
    Base.metadata,
    Column("agent_id", ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True),
    Column(
        "knowledge_base_id",
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    system_prompt: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(
        String(255)
    )  # None = LLM_MODEL from settings
    # Where the model runs: "ollama", "openrouter"... None = the default provider (LLM_*)
    provider: Mapped[str | None] = mapped_column(String(50))
    # Tool names, e.g. ["calculator", "weather"]. Tools are defined in code, not in the
    # database, so a list of names (validated against the registry) is enough.
    tools: Mapped[list[str]] = mapped_column(JSONB, default=list)
    max_iterations: Mapped[int] = mapped_column(default=6)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    knowledge_bases: Mapped[list["KnowledgeBase"]] = relationship(
        secondary=agent_knowledge_bases
    )
    runs: Mapped[list["AgentRun"]] = relationship(
        back_populates="agent", cascade="all, delete-orphan", passive_deletes=True
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="agent", cascade="all, delete-orphan", passive_deletes=True
    )


class Conversation(Base):
    """A chat thread with one agent: its runs, in order, form the conversation.

    Each new question is sent with the previous questions and final answers of
    the thread (see services/conversation.py), so the agent "remembers".
    """

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_id: Mapped[int] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(255))  # the first question, shortened
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    agent: Mapped["Agent"] = relationship(back_populates="conversations")
    runs: Mapped[list["AgentRun"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="AgentRun.id",  # ids grow with time: the order the questions came
    )


class AgentRun(Base):
    """One question asked to an agent, and everything that happened to answer it."""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_id: Mapped[int] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), index=True
    )
    # Nullable: runs created before conversations existed belong to none
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    user_message: Mapped[str] = mapped_column(Text)
    # running -> completed | max_iterations | error
    status: Mapped[str] = mapped_column(String(20), default="running")
    answer: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    # Snapshot of the configuration used: the agent may be edited later, and an
    # audit trail must show what actually produced this answer.
    model: Mapped[str] = mapped_column(String(255))
    provider: Mapped[str | None] = mapped_column(String(50))  # None: mock, or old runs
    system_prompt: Mapped[str] = mapped_column(Text)
    tools: Mapped[list[str]] = mapped_column(JSONB, default=list)
    prompt_tokens: Mapped[int] = mapped_column(default=0)
    completion_tokens: Mapped[int] = mapped_column(default=0)
    duration_ms: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    agent: Mapped["Agent"] = relationship(back_populates="runs")
    conversation: Mapped["Conversation | None"] = relationship(back_populates="runs")
    steps: Mapped[list["RunStep"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="RunStep.position",
    )


class RunStep(Base):
    """One step of a run, as recorded by the agent loop: an LLM call or a tool call."""

    __tablename__ = "run_steps"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int]
    kind: Mapped[str] = mapped_column(String(20))  # llm_call | tool_call
    duration_ms: Mapped[int]
    content: Mapped[str | None] = mapped_column(Text)  # llm_call: what the model said
    prompt_tokens: Mapped[int] = mapped_column(default=0)
    completion_tokens: Mapped[int] = mapped_column(default=0)
    tool_name: Mapped[str | None] = mapped_column(String(100))
    arguments: Mapped[dict | None] = mapped_column(JSONB)
    output: Mapped[str | None] = mapped_column(Text)
    is_error: Mapped[bool] = mapped_column(Boolean, default=False)
    sources: Mapped[list[dict]] = mapped_column(JSONB, default=list)

    run: Mapped["AgentRun"] = relationship(back_populates="steps")
