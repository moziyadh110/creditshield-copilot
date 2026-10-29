from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# BAAI/bge-small-en-v1.5 output size. Changing the embedding model means a new migration.
EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class Chunk(Base):
    """One section of Reg B text (e.g. 1002.9(a)(2)) or one Appendix C sample form."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    section_id: Mapped[str] = mapped_column(String(64), unique=True)
    title: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RunLog(Base):
    """One notice run: the queryable record in our own database. The MLflow trace
    (trace_id) holds the step-by-step detail."""

    __tablename__ = "run_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    applicant_id: Mapped[int] = mapped_column(BigInteger, index=True)
    decision: Mapped[str] = mapped_column(String(16))
    outcome: Mapped[str] = mapped_column(String(16))  # sent | revised | escalated
    escalation_reason: Mapped[str | None] = mapped_column(Text)
    agent_input: Mapped[dict[str, Any]] = mapped_column(JSON)  # exactly what the LLM saw
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSON)
    draft: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    notice: Mapped[str | None] = mapped_column(Text)
    check_results: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    prompt_name: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[int | None] = mapped_column(Integer)
    model: Mapped[str] = mapped_column(String(128))
    llm_calls: Mapped[int] = mapped_column(Integer)
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[int] = mapped_column(Integer)
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True)
    # Served from the response cache: no LLM call, the notice of an earlier passing run.
    cache_hit: Mapped[bool] = mapped_column(Boolean, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReviewItem(Base):
    """A case a person must handle: escalated, blocked by a check, or a tool failure.
    Nothing in this queue is ever sent automatically."""

    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(primary_key=True)
    applicant_id: Mapped[int] = mapped_column(BigInteger, index=True)
    decision: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(Text)
    check_results: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    draft: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("run_log.id"))
    trace_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), server_default="pending", index=True)
    reviewer_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DailySpend(Base):
    """LLM spend per UTC day, for the DAILY_BUDGET_USD cap. One row per day."""

    __tablename__ = "daily_spend"

    day: Mapped[date] = mapped_column(Date, primary_key=True)
    cost_usd: Mapped[float] = mapped_column(Float, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
