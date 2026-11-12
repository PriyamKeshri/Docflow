"""The pipeline: ingest -> extract -> validate -> store -> export -> notify.

One ``Pipeline`` instance serves the API, the batch job and the folder watcher.
Every original file is kept (content-addressed by SHA-256), which gives us
idempotent uploads and lets the nightly batch retry failures.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from docflow import repository as repo
from docflow.config import Settings
from docflow.db import Document, init_db, make_engine, make_session_factory, ping, utcnow
from docflow.extract import ExtractionError, Extractor, build_extractor
from docflow.ingest import SUPPORTED_EXTENSIONS, IngestError, ingest
from docflow.notify import BatchSummary, Notifier
from docflow.schemas import SECTION_MODELS, DocType, Status
from docflow.sheets import SheetsExporter
from docflow.validation import validate

log = logging.getLogger(__name__)

SOURCES = ("api", "webhook", "email", "folder", "batch")


@dataclass
class ProcessResult:
    document: Document
    duplicate: bool = False


class ReviewError(ValueError):
    pass


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        extractor: Extractor,
        session_factory: sessionmaker[Session],
        *,
        notifier: Notifier | None = None,
        exporter: SheetsExporter | None = None,
        engine: Any | None = None,
    ) -> None:
        self.settings = settings
        self.extractor = extractor
        self.session_factory = session_factory
        self.notifier = notifier
        self.exporter = exporter
        self.engine = engine
        self.batch_lock = threading.Lock()
        settings.ensure_dirs()

    # ------------------------------------------------------------------ core

    def process_bytes(
        self,
        data: bytes,
        filename: str,
        *,
        source: str = "api",
        hint: DocType | None = None,
        force: bool = False,
        notify: bool = True,
    ) -> ProcessResult:
        filename = Path(filename).name[:255] or "upload"
        sha = hashlib.sha256(data).hexdigest()
        with self.session_factory() as session:
            doc = repo.get_by_sha(session, sha)
            if doc is not None and not force and doc.status != Status.FAILED:
                return ProcessResult(doc, duplicate=True)
            if doc is None:
                doc = Document(
                    sha256=sha,
                    filename=filename,
                    source=source,
                    size_bytes=len(data),
                    status=Status.PROCESSING,
                    attempts=0,
                )
                session.add(doc)
                try:
                    session.commit()
                except IntegrityError:  # the same file arrived twice at once
                    session.rollback()
                    existing = repo.get_by_sha(session, sha)
                    if existing is None:
                        raise
                    return ProcessResult(existing, duplicate=True)
            else:
                _reset(doc)
            doc.storage_path = str(self._store(data, sha, filename))
            self._run(doc, data, hint)
            session.commit()

        self._after_processing(doc, notify=notify)
        return ProcessResult(doc)

    def _store(self, data: bytes, sha: str, filename: str) -> Path:
        suffix = Path(filename).suffix.lower()
        path = (
            self.settings.storage_dir / f"{sha}{suffix if suffix in SUPPORTED_EXTENSIONS else ''}"
        )
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return path

    def _run(self, doc: Document, data: bytes, hint: DocType | None) -> None:
        doc.attempts = (doc.attempts or 0) + 1
        start = time.perf_counter()
        try:
            ingested = ingest(data, doc.filename, self.settings)
            doc.mime_type = ingested.mime_type
            doc.page_count = ingested.page_count
            doc.ingest_method = ingested.method
            doc.ocr_confidence = ingested.ocr_confidence
            doc.raw_text = ingested.text

            outcome = self.extractor.extract(ingested.text, hint)
            extraction = outcome.extraction
            report = validate(extraction, self.settings, ocr_confidence=ingested.ocr_confidence)
            section = extraction.section()

            doc.doc_type = extraction.document_type.value
            doc.data = section.model_dump(mode="json") if section is not None else None
            doc.field_confidence = extraction.confidence_map()
            doc.notes = extraction.notes
            doc.issues = [issue.to_dict() for issue in report.issues]
            doc.review_reasons = report.review_reasons
            doc.confidence = report.confidence
            doc.status = Status.NEEDS_REVIEW if report.needs_review else Status.ACCEPTED
            doc.provider = outcome.provider
            doc.model = outcome.model
            doc.input_tokens = outcome.usage.input_tokens
            doc.output_tokens = outcome.usage.output_tokens
            doc.cost_usd = outcome.usage.cost_usd
            doc.extract_latency_ms = outcome.latency_ms
        except IngestError as exc:
            self._fail(doc, f"ingest: {exc}", retryable=exc.retryable)
        except ExtractionError as exc:
            self._fail(doc, f"extraction: {exc}", retryable=True)
            doc.provider = self.extractor.provider
            doc.model = exc.model or self.extractor.model
            doc.input_tokens = exc.usage.input_tokens
            doc.output_tokens = exc.usage.output_tokens
            doc.cost_usd = exc.usage.cost_usd
        except Exception as exc:  # a bug, not a bad document: keep the batch going
            log.exception("unexpected error processing %s", doc.filename)
            self._fail(doc, f"internal error: {type(exc).__name__}: {exc}", retryable=True)
        doc.total_latency_ms = int((time.perf_counter() - start) * 1000)
        doc.processed_at = utcnow()
        log.info(
            "%s -> %s (%s, confidence=%s, %d ms)",
            doc.filename,
            doc.status,
            doc.doc_type,
            doc.confidence,
            doc.total_latency_ms,
        )

    @staticmethod
    def _fail(doc: Document, message: str, *, retryable: bool) -> None:
        doc.status = Status.FAILED
        doc.error = message
        doc.retryable = retryable
        doc.review_reasons = ["processing failed"]

    def _after_processing(self, doc: Document, *, notify: bool) -> None:
        if doc.status in (Status.ACCEPTED, Status.APPROVED):
            self._export(doc)
        if notify and self.notifier is not None:
            self.notifier.document_processed(doc)

    def _export(self, doc: Document) -> bool:
        if self.exporter is None:
            return False
        try:
            self.exporter.export(doc)
        except Exception as exc:  # Sheets is a convenience copy; the DB is the record
            log.warning("Sheets export failed for %s: %s", doc.id, exc)
            return False
        with self.session_factory() as session:
            stored = repo.get(session, doc.id)
            if stored is not None:
                stored.sheets_exported_at = utcnow()
                doc.sheets_exported_at = stored.sheets_exported_at
                session.commit()
        return True

    # --------------------------------------------------------------- review

    def review(
        self,
        doc_id: uuid.UUID,
        *,
        approve: bool,
        corrected_data: dict[str, Any] | None = None,
        doc_type: DocType | None = None,
        reviewer: str | None = None,
        note: str | None = None,
    ) -> Document:
        with self.session_factory() as session:
            doc = repo.get(session, doc_id)
            if doc is None:
                raise KeyError(doc_id)
            if doc.status in (Status.PROCESSING, Status.FAILED):
                raise ReviewError(f"cannot review a document whose status is {doc.status}")
            if doc_type is not None:
                doc.doc_type = doc_type.value
            if corrected_data is not None:
                try:
                    model = SECTION_MODELS[DocType(doc.doc_type)]
                except (KeyError, ValueError) as exc:
                    raise ReviewError("set doc_type before sending corrected_data") from exc
                doc.corrected_data = model.model_validate(corrected_data).model_dump(mode="json")
            doc.status = Status.APPROVED if approve else Status.REJECTED
            doc.reviewed_by = reviewer
            doc.review_note = note
            doc.reviewed_at = utcnow()
            doc.sheets_exported_at = None
            session.commit()
        if approve:
            self._export(doc)
        return doc

    def reprocess(self, doc_id: uuid.UUID) -> ProcessResult:
        with self.session_factory() as session:
            doc = repo.get(session, doc_id)
            if doc is None:
                raise KeyError(doc_id)
            path = Path(doc.storage_path) if doc.storage_path else None
            filename, source = doc.filename, doc.source
        if path is None or not path.exists():
            raise FileNotFoundError(f"original file for {doc_id} is no longer stored")
        return self.process_bytes(path.read_bytes(), filename, source=source, force=True)

    # ---------------------------------------------------------------- batch

    def process_inbox(
        self, *, notify_each: bool = True, source: str = "folder"
    ) -> list[ProcessResult]:
        """Process files dropped into the inbox, then move them to processed/ or failed/."""
        results: list[ProcessResult] = []
        settle_before = time.time() - self.settings.watch_settle_seconds
        for path in sorted(self.settings.inbox_dir.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            try:
                if path.stat().st_mtime > settle_before:
                    continue  # still being written; pick it up next pass
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    log.warning("skipping unsupported file %s", path.name)
                    _move(path, self.settings.failed_dir)
                    continue
                data = path.read_bytes()
            except FileNotFoundError:  # another worker took it
                continue
            result = self.process_bytes(data, path.name, source=source, notify=notify_each)
            target = (
                self.settings.failed_dir
                if result.document.status == Status.FAILED
                else self.settings.processed_dir
            )
            _move(path, target)
            results.append(result)
        return results

    def retry_failed(self) -> list[ProcessResult]:
        with self.session_factory() as session:
            candidates = [
                (doc.storage_path, doc.filename, doc.source)
                for doc in repo.retry_candidates(session, self.settings.max_retry_attempts)
            ]
        results = []
        for storage_path, filename, source in candidates:
            path = Path(storage_path)
            if not path.exists():
                log.warning("cannot retry %s: original file missing", filename)
                continue
            results.append(
                self.process_bytes(
                    path.read_bytes(), filename, source=source, force=True, notify=False
                )
            )
        return results

    def export_pending(self) -> int:
        if self.exporter is None:
            return 0
        with self.session_factory() as session:
            pending = repo.pending_exports(session)
        return sum(1 for doc in pending if self._export(doc))

    def run_batch(self) -> BatchSummary:
        started = datetime.now(UTC)
        start = time.perf_counter()
        new = self.process_inbox(notify_each=False, source="batch")
        retried = self.retry_failed()
        exported = self.export_pending()
        docs = [r.document for r in new + retried if not r.duplicate]
        summary = BatchSummary(
            started_at=started.isoformat(),
            duration_s=round(time.perf_counter() - start, 2),
            inbox_files=len(new),
            retried=len(retried),
            accepted=sum(d.status == Status.ACCEPTED for d in docs),
            needs_review=sum(d.status == Status.NEEDS_REVIEW for d in docs),
            failed=sum(d.status == Status.FAILED for d in docs),
            exported=exported,
            cost_usd=round(sum(d.cost_usd or 0.0 for d in docs), 6),
            failures=[
                {"id": str(d.id), "filename": d.filename, "error": d.error}
                for d in docs
                if d.status == Status.FAILED
            ],
            review=[
                {"id": str(d.id), "filename": d.filename, "reasons": d.review_reasons or []}
                for d in docs
                if d.status == Status.NEEDS_REVIEW
            ],
        )
        if self.notifier is not None:
            self.notifier.batch_completed(summary)
        return summary

    def watch(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        log.info(
            "watching %s every %.0fs", self.settings.inbox_dir, self.settings.watch_interval_seconds
        )
        while not stop.is_set():
            try:
                self.process_inbox(notify_each=True)
            except Exception:
                log.exception("inbox pass failed")
            stop.wait(self.settings.watch_interval_seconds)

    # --------------------------------------------------------------- report

    def summary(self, hours: int = 24) -> dict[str, Any]:
        with self.session_factory() as session:
            return repo.summary(session, datetime.now(UTC) - timedelta(hours=hours))

    def ping(self) -> bool:
        return ping(self.engine) if self.engine is not None else True


def _reset(doc: Document) -> None:
    """Clear the previous run's results before reprocessing."""
    doc.status = Status.PROCESSING
    for attr in (
        "error",
        "doc_type",
        "confidence",
        "data",
        "field_confidence",
        "issues",
        "review_reasons",
        "notes",
        "corrected_data",
        "reviewed_by",
        "review_note",
        "reviewed_at",
        "sheets_exported_at",
    ):
        setattr(doc, attr, None)
    doc.retryable = False


def _move(path: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / path.name
    if target.exists():
        target = directory / f"{path.stem}-{uuid.uuid4().hex[:8]}{path.suffix}"
    with contextlib.suppress(FileNotFoundError):  # another worker moved it first
        shutil.move(str(path), target)


def build_pipeline(settings: Settings) -> Pipeline:
    engine = make_engine(settings.database_url)
    init_db(engine)
    exporter = SheetsExporter(settings) if settings.sheets_enabled else None
    return Pipeline(
        settings,
        build_extractor(settings),
        make_session_factory(engine),
        notifier=Notifier(settings),
        exporter=exporter,
        engine=engine,
    )
