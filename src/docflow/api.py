"""FastAPI app: upload, query, review, batch and reporting endpoints."""

from __future__ import annotations

import logging
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from docflow import __version__
from docflow import repository as repo
from docflow.config import Settings, get_settings
from docflow.db import Document
from docflow.pipeline import SOURCES, Pipeline, ReviewError, build_pipeline
from docflow.schemas import DocType, Status

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ models


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    source: str
    status: str
    doc_type: str | None
    confidence: float | None
    review_reasons: list[str] = Field(default_factory=list)
    error: str | None
    page_count: int | None
    ingest_method: str | None
    ocr_confidence: float | None
    provider: str | None
    model: str | None
    cost_usd: float | None
    total_latency_ms: int | None
    created_at: datetime
    processed_at: datetime | None
    duplicate: bool = False

    @field_validator("review_reasons", mode="before")
    @classmethod
    def _none_to_list(cls, value: Any) -> Any:
        return value or []

    @field_validator("created_at", "processed_at", mode="after")
    @classmethod
    def _assume_utc(cls, value: datetime | None) -> datetime | None:
        # SQLite drops tzinfo; everything is stored in UTC.
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


class DocumentDetail(DocumentOut):
    sha256: str
    mime_type: str | None
    size_bytes: int
    data: dict[str, Any] | None
    corrected_data: dict[str, Any] | None
    field_confidence: dict[str, float] | None
    issues: list[dict[str, Any]] = Field(default_factory=list)
    notes: str | None
    attempts: int
    input_tokens: int | None
    output_tokens: int | None
    extract_latency_ms: int | None
    reviewed_by: str | None
    review_note: str | None
    reviewed_at: datetime | None

    @field_validator("issues", mode="before")
    @classmethod
    def _issues_list(cls, value: Any) -> Any:
        return value or []


class DocumentPage(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class ReviewRequest(BaseModel):
    action: Literal["approve", "reject"]
    corrected_data: dict[str, Any] | None = Field(
        None, description="Full corrected section (e.g. the invoice object)."
    )
    doc_type: DocType | None = Field(None, description="Override the detected type.")
    reviewer: str | None = Field(None, max_length=128)
    note: str | None = Field(None, max_length=2000)


def _detail(doc: Document, duplicate: bool = False) -> DocumentDetail:
    out = DocumentDetail.model_validate(doc)
    out.duplicate = duplicate
    return out


# ------------------------------------------------------------ dependencies


def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    expected = request.app.state.settings.api_key
    if expected is None:
        return  # auth disabled (local dev); a warning is logged at startup
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key header")


# ----------------------------------------------------------------- routes

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


@router.post("/documents", response_model=DocumentDetail, status_code=201)
def upload_document(
    response: Response,
    request: Request,
    file: UploadFile = File(...),
    source: str = Form("api"),
    doc_type: str | None = Form(None, description="Optional hint: invoice, bank_statement, resume"),
    force: bool = Form(False, description="Re-extract even if this exact file was seen before"),
    pipeline: Pipeline = Depends(get_pipeline),
) -> DocumentDetail:
    settings: Settings = request.app.state.settings
    if source not in SOURCES:
        raise HTTPException(422, f"source must be one of {', '.join(SOURCES)}")
    hint: DocType | None = None
    if doc_type:
        try:
            hint = DocType(doc_type)
        except ValueError as exc:
            raise HTTPException(422, f"unknown doc_type '{doc_type}'") from exc

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"file is larger than {settings.max_upload_mb} MB")
    if not data:
        raise HTTPException(400, "file is empty")

    result = pipeline.process_bytes(
        data, file.filename or "upload", source=source, hint=hint, force=force
    )
    if result.duplicate:
        response.status_code = 200
    return _detail(result.document, duplicate=result.duplicate)


@router.get("/documents", response_model=DocumentPage)
def list_documents(
    status: Status | None = None,
    doc_type: DocType | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    pipeline: Pipeline = Depends(get_pipeline),
) -> DocumentPage:
    with pipeline.session_factory() as session:
        rows, total = repo.list_documents(
            session,
            status=status.value if status else None,
            doc_type=doc_type.value if doc_type else None,
            limit=limit,
            offset=offset,
        )
        items = [DocumentOut.model_validate(row) for row in rows]
    return DocumentPage(items=items, total=total, limit=limit, offset=offset)


@router.get("/documents/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: uuid.UUID, pipeline: Pipeline = Depends(get_pipeline)) -> DocumentDetail:
    with pipeline.session_factory() as session:
        doc = repo.get(session, doc_id)
        if doc is None:
            raise HTTPException(404, "document not found")
        return _detail(doc)


@router.post("/documents/{doc_id}/review", response_model=DocumentDetail)
def review_document(
    doc_id: uuid.UUID, body: ReviewRequest, pipeline: Pipeline = Depends(get_pipeline)
) -> DocumentDetail:
    try:
        doc = pipeline.review(
            doc_id,
            approve=body.action == "approve",
            corrected_data=body.corrected_data,
            doc_type=body.doc_type,
            reviewer=body.reviewer,
            note=body.note,
        )
    except KeyError as exc:
        raise HTTPException(404, "document not found") from exc
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_url=False, include_context=False)) from exc
    except ReviewError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _detail(doc)


@router.post("/documents/{doc_id}/reprocess", response_model=DocumentDetail)
def reprocess_document(
    doc_id: uuid.UUID, pipeline: Pipeline = Depends(get_pipeline)
) -> DocumentDetail:
    try:
        result = pipeline.reprocess(doc_id)
    except KeyError as exc:
        raise HTTPException(404, "document not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(410, str(exc)) from exc
    return _detail(result.document)


@router.post("/batch/run", status_code=202)
def run_batch(
    background: BackgroundTasks,
    response: Response,
    wait: bool = Query(False, description="Run inline and return the summary"),
    pipeline: Pipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    if not pipeline.batch_lock.acquire(blocking=False):
        raise HTTPException(409, "a batch is already running")
    if wait:
        try:
            summary = pipeline.run_batch()
        finally:
            pipeline.batch_lock.release()
        response.status_code = 200
        return {"status": "completed", "summary": asdict(summary)}

    def run() -> None:
        try:
            pipeline.run_batch()
        except Exception:
            log.exception("batch run failed")
        finally:
            pipeline.batch_lock.release()

    background.add_task(run)
    return {"status": "started", "detail": "the summary is posted to the notify webhook"}


@router.get("/reports/summary")
def report_summary(
    hours: int = Query(24, ge=1, le=24 * 31), pipeline: Pipeline = Depends(get_pipeline)
) -> dict[str, Any]:
    return pipeline.summary(hours)


# -------------------------------------------------------------------- app


def create_app(settings: Settings | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.pipeline = pipeline or build_pipeline(settings)
        if settings.api_key is None:
            log.warning("DOCFLOW_API_KEY is not set: the API accepts unauthenticated requests")
        yield

    app = FastAPI(
        title="DocFlow",
        version=__version__,
        description="Automated document intake: ingest, extract, validate, store, notify.",
        lifespan=lifespan,
    )

    @app.get("/health", tags=["ops"])
    def health(request: Request) -> dict[str, Any]:
        current: Pipeline = request.app.state.pipeline
        db_ok = current.ping()
        return {
            "status": "ok" if db_ok else "degraded",
            "database": db_ok,
            "provider": current.extractor.provider,
            "model": current.extractor.model,
            "version": __version__,
        }

    app.include_router(router)
    return app


def __getattr__(name: str) -> Any:
    # ``uvicorn docflow.api:app`` builds the app on first access, not on import,
    # so importing this module (tests, CLI) never reads settings or touches a DB.
    if name == "app":
        return create_app()
    raise AttributeError(name)
