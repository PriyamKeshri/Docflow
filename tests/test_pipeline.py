from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from docflow import repository as repo
from docflow.config import Settings
from docflow.pipeline import ReviewError
from docflow.schemas import Status
from tests.conftest import RecordingExporter, RecordingNotifier, make_pdf


def test_process_accepts_a_clean_invoice(make_pipeline, invoice_pdf, invoice_response) -> None:
    pipeline = make_pipeline([invoice_response])
    result = pipeline.process_bytes(invoice_pdf, "invoice.pdf")
    doc = result.document
    assert not result.duplicate
    assert doc.status == Status.ACCEPTED
    assert doc.doc_type == "invoice"
    assert doc.data["total"] == 270.0
    assert doc.ingest_method == "text"
    assert doc.provider == "fake" and doc.input_tokens > 0
    assert doc.total_latency_ms is not None and doc.processed_at is not None
    assert Path(doc.storage_path).read_bytes() == invoice_pdf
    assert "INV-2026-0042" in doc.raw_text


def test_same_file_twice_is_a_duplicate(make_pipeline, invoice_pdf, invoice_response) -> None:
    pipeline = make_pipeline([invoice_response])
    first = pipeline.process_bytes(invoice_pdf, "invoice.pdf")
    second = pipeline.process_bytes(invoice_pdf, "copy-of-invoice.pdf")
    assert second.duplicate
    assert second.document.id == first.document.id
    assert len(pipeline.extractor.prompts) == 1  # no second model call


def test_force_reprocesses(make_pipeline, invoice_pdf, invoice_response) -> None:
    pipeline = make_pipeline([invoice_response, invoice_response])
    first = pipeline.process_bytes(invoice_pdf, "invoice.pdf")
    again = pipeline.process_bytes(invoice_pdf, "invoice.pdf", force=True)
    assert not again.duplicate
    assert again.document.id == first.document.id
    assert again.document.attempts == 2


def test_validation_errors_go_to_review(make_pipeline, invoice_pdf, invoice_response) -> None:
    invoice_response["invoice"]["total"] = 999.0
    doc = make_pipeline([invoice_response]).process_bytes(invoice_pdf, "invoice.pdf").document
    assert doc.status == Status.NEEDS_REVIEW
    assert any(issue["code"] == "totals_mismatch" for issue in doc.issues)
    assert doc.review_reasons


def test_failed_extraction_is_retried_by_the_batch(
    make_pipeline, invoice_pdf, invoice_response
) -> None:
    broken = {"document_type": "invoice", "invoice": None}
    pipeline = make_pipeline([broken, broken, invoice_response])
    doc = pipeline.process_bytes(invoice_pdf, "invoice.pdf").document
    assert doc.status == Status.FAILED
    assert doc.retryable
    assert doc.error.startswith("extraction:")

    retried = pipeline.retry_failed()
    assert len(retried) == 1
    assert retried[0].document.status == Status.ACCEPTED
    assert retried[0].document.attempts == 2
    assert retried[0].document.error is None


def test_bad_files_fail_without_retry(make_pipeline) -> None:
    pipeline = make_pipeline()
    doc = pipeline.process_bytes(b"\x00\x01binary", "blob.bin").document
    assert doc.status == Status.FAILED
    assert not doc.retryable
    assert pipeline.retry_failed() == []


def test_retries_stop_at_the_limit(make_pipeline, settings: Settings, invoice_pdf) -> None:
    settings.max_retry_attempts = 2
    broken = {"document_type": "invoice", "invoice": None}
    pipeline = make_pipeline(default=broken)
    pipeline.process_bytes(invoice_pdf, "invoice.pdf")
    assert len(pipeline.retry_failed()) == 1
    assert pipeline.retry_failed() == []


def test_inbox_files_are_processed_and_moved(
    make_pipeline, settings: Settings, invoice_pdf, invoice_response
) -> None:
    pipeline = make_pipeline([invoice_response])
    (settings.inbox_dir / "invoice.pdf").write_bytes(invoice_pdf)
    (settings.inbox_dir / "notes.docx").write_bytes(b"PK\x03\x04")
    (settings.inbox_dir / ".DS_Store").write_bytes(b"x")

    results = pipeline.process_inbox()
    assert [r.document.status for r in results] == [Status.ACCEPTED]
    assert (settings.processed_dir / "invoice.pdf").exists()
    assert (settings.failed_dir / "notes.docx").exists()
    assert not (settings.inbox_dir / "invoice.pdf").exists()


def test_inbox_waits_for_files_to_settle(make_pipeline, settings: Settings, invoice_pdf) -> None:
    settings.watch_settle_seconds = 60
    pipeline = make_pipeline()
    path = settings.inbox_dir / "still-copying.pdf"
    path.write_bytes(invoice_pdf)
    assert pipeline.process_inbox() == []
    assert path.exists()
    old = time.time() - 120
    os.utime(path, (old, old))
    assert len(pipeline.process_inbox()) == 1


def test_batch_summary_is_sent(
    make_pipeline, settings: Settings, invoice_response, resume_response
) -> None:
    notifier = RecordingNotifier()
    invoice_response["invoice"]["total"] = 1.0  # -> needs review
    pipeline = make_pipeline([invoice_response, resume_response], notifier=notifier)
    (settings.inbox_dir / "a.pdf").write_bytes(
        make_pdf(["INVOICE A: enough text for the text layer"])
    )
    (settings.inbox_dir / "b.pdf").write_bytes(
        make_pdf(["RESUME B: enough text for the text layer"])
    )
    (settings.inbox_dir / "c.txt").write_bytes(b"")  # empty -> failed

    summary = pipeline.run_batch()
    assert (summary.inbox_files, summary.accepted, summary.needs_review, summary.failed) == (
        3,
        1,
        1,
        1,
    )
    assert summary.failures[0]["filename"] == "c.txt"
    assert notifier.batches == [summary]
    assert notifier.documents == []  # batch runs send one summary, not one message per file


def test_each_document_notifies_outside_batches(
    make_pipeline, invoice_pdf, invoice_response
) -> None:
    notifier = RecordingNotifier()
    make_pipeline([invoice_response], notifier=notifier).process_bytes(invoice_pdf, "a.pdf")
    assert [d.filename for d in notifier.documents] == ["a.pdf"]


def test_accepted_documents_are_exported(make_pipeline, invoice_pdf, invoice_response) -> None:
    exporter = RecordingExporter()
    pipeline = make_pipeline([invoice_response], exporter=exporter)
    doc = pipeline.process_bytes(invoice_pdf, "a.pdf").document
    assert exporter.exported == [doc]
    with pipeline.session_factory() as session:
        assert repo.get(session, doc.id).sheets_exported_at is not None


def test_export_failure_does_not_fail_the_document(
    make_pipeline, invoice_pdf, invoice_response
) -> None:
    exporter = RecordingExporter(fail=True)
    pipeline = make_pipeline([invoice_response], exporter=exporter)
    doc = pipeline.process_bytes(invoice_pdf, "a.pdf").document
    assert doc.status == Status.ACCEPTED
    with pipeline.session_factory() as session:
        assert repo.get(session, doc.id).sheets_exported_at is None
    exporter.fail = False
    assert pipeline.export_pending() == 1


def test_review_with_corrections(make_pipeline, invoice_pdf, invoice_response) -> None:
    invoice_response["invoice"]["total"] = 280.0
    exporter = RecordingExporter()
    pipeline = make_pipeline([invoice_response], exporter=exporter)
    doc = pipeline.process_bytes(invoice_pdf, "a.pdf").document
    assert doc.status == Status.NEEDS_REVIEW and exporter.exported == []

    corrected = dict(invoice_response["invoice"], total="270.00")
    reviewed = pipeline.review(doc.id, approve=True, corrected_data=corrected, reviewer="sam")
    assert reviewed.status == Status.APPROVED
    assert reviewed.corrected_data["total"] == 270.0
    assert reviewed.effective_data["total"] == 270.0
    assert reviewed.data["total"] == 280.0  # the model's answer is kept for audit
    assert exporter.exported[-1].id == doc.id


def test_review_rejects_bad_corrections(make_pipeline, invoice_pdf, invoice_response) -> None:
    pipeline = make_pipeline([invoice_response])
    doc = pipeline.process_bytes(invoice_pdf, "a.pdf").document
    with pytest.raises(ValidationError):
        pipeline.review(doc.id, approve=True, corrected_data={"total": "lots", "colour": "red"})


def test_failed_documents_cannot_be_reviewed(make_pipeline) -> None:
    pipeline = make_pipeline()
    doc = pipeline.process_bytes(b"", "empty.txt").document
    with pytest.raises(ReviewError):
        pipeline.review(doc.id, approve=True)


def test_reprocess_uses_the_stored_original(make_pipeline, invoice_pdf, invoice_response) -> None:
    pipeline = make_pipeline([invoice_response, invoice_response])
    doc = pipeline.process_bytes(invoice_pdf, "a.pdf").document
    again = pipeline.reprocess(doc.id).document
    assert again.id == doc.id and again.attempts == 2


def test_summary_report(make_pipeline, invoice_pdf, invoice_response) -> None:
    pipeline = make_pipeline([invoice_response])
    pipeline.process_bytes(invoice_pdf, "a.pdf")
    pipeline.process_bytes(b"", "empty.txt")
    report = pipeline.summary(hours=1)
    assert report["documents"] == 2
    assert report["by_status"] == {"accepted": 1, "failed": 1}
    assert report["failures"][0]["filename"] == "empty.txt"
