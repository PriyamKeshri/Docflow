from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from docflow.config import Settings
from docflow.db import Document
from docflow.notify import BatchSummary, Notifier, batch_event, document_event
from docflow.sheets import SheetsExporter, build_row


def make_doc(**overrides: Any) -> Document:
    values: dict[str, Any] = {
        "id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
        "filename": "acme-march.pdf",
        "source": "email",
        "status": "needs_review",
        "doc_type": "invoice",
        "confidence": 0.62,
        "review_reasons": ["1 validation error(s): totals_mismatch"],
        "error": None,
        "data": {"invoice_number": "INV-1", "total": 270.0, "line_items": [{}, {}]},
        "corrected_data": None,
        "processed_at": datetime(2026, 3, 2, 9, 30, tzinfo=UTC),
        "cost_usd": 0.0123,
        "total_latency_ms": 2100,
    }
    values.update(overrides)
    return Document(**values)


# ------------------------------------------------------------- notify


def test_document_event_text(settings: Settings) -> None:
    event = document_event(make_doc(), settings)
    assert event["event"] == "document.processed"
    assert event["needs_attention"] and not event["has_failures"]
    assert ":warning: *acme-march.pdf* needs review" in event["text"]
    assert "totals_mismatch" in event["text"]
    assert "/v1/documents/11111111-1111-1111-1111-111111111111|Open in DocFlow>" in event["text"]
    assert ":warning:" not in event["plain_text"]


def test_failed_document_event(settings: Settings) -> None:
    event = document_event(make_doc(status="failed", error="ingest: file is empty"), settings)
    assert event["has_failures"]
    assert "file is empty" in event["text"]


def test_batch_event_lists_problems() -> None:
    summary = BatchSummary(
        started_at="2026-03-02T02:00:00+00:00",
        duration_s=12.5,
        inbox_files=3,
        accepted=1,
        needs_review=1,
        failed=1,
        cost_usd=0.04,
        failures=[{"id": "1", "filename": "bad.pdf", "error": "ingest: unreadable"}],
        review=[{"id": "2", "filename": "odd.pdf", "reasons": ["confidence 0.50 is below 0.80"]}],
    )
    event = batch_event(summary)
    assert event["has_failures"]
    assert "bad.pdf" in event["text"] and "odd.pdf" in event["plain_text"]
    assert event["summary"]["inbox_files"] == 3


def test_empty_batch_says_so() -> None:
    event = batch_event(BatchSummary(started_at="x", duration_s=0.1))
    assert "Nothing new" in event["text"]
    assert not event["has_failures"]


def notifier_with(settings: Settings, handler) -> Notifier:
    settings.notify_webhook_url = "http://n8n.test/webhook/docflow-notify"
    settings.notify_secret = "s3cret"
    return Notifier(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_notifier_posts_with_the_secret(settings: Settings) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    notifier = notifier_with(settings, handler)
    assert notifier.document_processed(make_doc())
    assert seen[0].headers["X-DocFlow-Secret"] == "s3cret"
    assert json.loads(seen[0].content)["document"]["filename"] == "acme-march.pdf"


@pytest.mark.parametrize(
    ("mode", "status", "sent"),
    [
        ("issues", "accepted", False),
        ("issues", "needs_review", True),
        ("issues", "failed", True),
        ("all", "accepted", True),
        ("none", "failed", False),
    ],
)
def test_notify_modes(settings: Settings, mode: str, status: str, sent: bool) -> None:
    notifier = notifier_with(settings, lambda request: httpx.Response(200))
    settings.notify_on = mode
    assert notifier.document_processed(make_doc(status=status)) is sent


def test_notifier_swallows_http_errors(settings: Settings) -> None:
    notifier = notifier_with(settings, lambda request: httpx.Response(502))
    assert notifier.batch_completed(BatchSummary(started_at="x", duration_s=1)) is False


def test_notifier_disabled_without_url(settings: Settings) -> None:
    assert not Notifier(settings).enabled


# ------------------------------------------------------------- sheets


def test_build_row_flattens_the_section() -> None:
    header, row = build_row(make_doc(status="accepted"))
    record = dict(zip(header, row, strict=True))
    assert record["document_id"] == "11111111-1111-1111-1111-111111111111"
    assert record["invoice_number"] == "INV-1"
    assert record["total"] == 270.0
    assert record["vendor_name"] == ""  # None -> empty cell
    assert record["line_items_count"] == 2
    assert record["processed_at"] == "2026-03-02T09:30:00+00:00"


def test_build_row_prefers_reviewer_corrections() -> None:
    doc = make_doc(corrected_data={"invoice_number": "INV-1", "total": 260.0})
    header, row = build_row(doc)
    assert dict(zip(header, row, strict=True))["total"] == 260.0


class FakeWorksheet:
    def __init__(self) -> None:
        self.rows: list[tuple[list[Any], str]] = []

    def append_row(self, row: list[Any], value_input_option: str) -> None:
        self.rows.append((row, value_input_option))


class FakeBook:
    def __init__(self) -> None:
        self.sheets: dict[str, FakeWorksheet] = {}

    def worksheet(self, title: str) -> FakeWorksheet:
        import gspread

        if title not in self.sheets:
            raise gspread.WorksheetNotFound(title)
        return self.sheets[title]

    def add_worksheet(self, title: str, rows: int, cols: int) -> FakeWorksheet:
        self.sheets[title] = FakeWorksheet()
        return self.sheets[title]


def test_exporter_creates_the_sheet_and_appends_raw(settings: Settings) -> None:
    settings.sheets_spreadsheet_id = "sheet-id"
    book = FakeBook()
    client = type("Client", (), {"open_by_key": lambda self, key: book})()
    exporter = SheetsExporter(settings, client=client)
    exporter.export(make_doc(status="accepted"))
    exporter.export(make_doc(status="accepted", filename="second.pdf"))

    rows = book.sheets["invoice"].rows
    assert rows[0][0][0] == "document_id"  # header written once
    assert len(rows) == 3
    assert all(option == "RAW" for _, option in rows)  # no formula injection


def test_exporter_requires_a_spreadsheet_id(settings: Settings) -> None:
    with pytest.raises(ValueError):
        SheetsExporter(settings)
