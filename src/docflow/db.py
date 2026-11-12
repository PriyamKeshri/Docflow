"""SQLAlchemy engine, session factory and ORM model."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Engine,
    Float,
    Integer,
    String,
    Text,
    Uuid,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(16), default="api")
    size_bytes: Mapped[int] = mapped_column(Integer)
    mime_type: Mapped[str | None] = mapped_column(String(64))
    storage_path: Mapped[str | None] = mapped_column(String(1024))

    # ingest
    page_count: Mapped[int | None] = mapped_column(Integer)
    ingest_method: Mapped[str | None] = mapped_column(String(8))
    ocr_confidence: Mapped[float | None] = mapped_column(Float)
    raw_text: Mapped[str | None] = mapped_column(Text)

    # extraction + validation
    status: Mapped[str] = mapped_column(String(16), index=True, default="processing")
    doc_type: Mapped[str | None] = mapped_column(String(16), index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    data: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    field_confidence: Mapped[dict[str, float] | None] = mapped_column(JSONType)
    issues: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONType)
    review_reasons: Mapped[list[str] | None] = mapped_column(JSONType)
    notes: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    retryable: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)

    # cost + latency
    provider: Mapped[str | None] = mapped_column(String(16))
    model: Mapped[str | None] = mapped_column(String(64))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    extract_latency_ms: Mapped[int | None] = mapped_column(Integer)
    total_latency_ms: Mapped[int | None] = mapped_column(Integer)

    # human review
    corrected_data: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    reviewed_by: Mapped[str | None] = mapped_column(String(128))
    review_note: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    sheets_exported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    @property
    def effective_data(self) -> dict[str, Any] | None:
        """What downstream systems should use: the reviewer's correction if there is one."""
        return self.corrected_data if self.corrected_data is not None else self.data


def make_engine(url: str) -> Engine:
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create tables if missing. Swap for Alembic migrations once the schema settles."""
    Base.metadata.create_all(engine)


def ping(engine: Engine) -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
