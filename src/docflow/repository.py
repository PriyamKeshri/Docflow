"""Queries over the documents table."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from docflow.db import Document
from docflow.schemas import Status


def get(session: Session, doc_id: uuid.UUID) -> Document | None:
    return session.get(Document, doc_id)


def get_by_sha(session: Session, sha256: str) -> Document | None:
    return session.scalar(select(Document).where(Document.sha256 == sha256))


def list_documents(
    session: Session,
    *,
    status: str | None = None,
    doc_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Document], int]:
    query = select(Document)
    if status:
        query = query.where(Document.status == status)
    if doc_type:
        query = query.where(Document.doc_type == doc_type)
    total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = session.scalars(
        query.order_by(Document.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return list(rows), total


def retry_candidates(session: Session, max_attempts: int) -> list[Document]:
    return list(
        session.scalars(
            select(Document)
            .where(
                Document.status == Status.FAILED,
                Document.retryable.is_(True),
                Document.attempts < max_attempts,
                Document.storage_path.is_not(None),
            )
            .order_by(Document.created_at)
        )
    )


def pending_exports(session: Session) -> list[Document]:
    return list(
        session.scalars(
            select(Document)
            .where(
                Document.status.in_([Status.ACCEPTED, Status.APPROVED]),
                Document.sheets_exported_at.is_(None),
            )
            .order_by(Document.created_at)
        )
    )


def summary(session: Session, since: datetime) -> dict[str, Any]:
    window = Document.processed_at >= since
    by_status = dict(
        session.execute(
            select(Document.status, func.count()).where(window).group_by(Document.status)
        ).all()
    )
    by_type = dict(
        session.execute(
            select(Document.doc_type, func.count()).where(window).group_by(Document.doc_type)
        ).all()
    )
    totals = session.execute(
        select(
            func.count(),
            func.coalesce(func.sum(Document.cost_usd), 0.0),
            func.avg(Document.total_latency_ms),
            func.avg(Document.confidence),
        ).where(window)
    ).one()
    failures = session.scalars(
        select(Document).where(window, Document.status == Status.FAILED).limit(20)
    ).all()
    review = session.scalars(
        select(Document).where(window, Document.status == Status.NEEDS_REVIEW).limit(20)
    ).all()
    return {
        "since": since.isoformat(),
        "documents": totals[0],
        "by_status": {str(k): v for k, v in by_status.items()},
        "by_type": {str(k): v for k, v in by_type.items()},
        "cost_usd": round(float(totals[1]), 6),
        "avg_latency_ms": round(float(totals[2]), 1) if totals[2] is not None else None,
        "avg_confidence": round(float(totals[3]), 4) if totals[3] is not None else None,
        "failures": [{"id": str(d.id), "filename": d.filename, "error": d.error} for d in failures],
        "needs_review": [
            {"id": str(d.id), "filename": d.filename, "reasons": d.review_reasons or []}
            for d in review
        ],
    }
