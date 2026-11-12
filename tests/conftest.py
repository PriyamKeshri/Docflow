from __future__ import annotations

import copy
import io
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from docflow.config import Settings
from docflow.db import Base, make_engine, make_session_factory
from docflow.extract.fake import FakeExtractor
from docflow.pipeline import Pipeline

HAS_TESSERACT = shutil.which("tesseract") is not None
REQUIRE_TESSERACT = os.environ.get("DOCFLOW_REQUIRE_TESSERACT") == "1"


def needs_tesseract(func: Callable[..., Any]) -> Callable[..., Any]:
    """Skip OCR tests locally when Tesseract is missing; CI sets DOCFLOW_REQUIRE_TESSERACT=1."""
    if HAS_TESSERACT or REQUIRE_TESSERACT:
        return func
    return pytest.mark.skip(reason="tesseract binary not installed")(func)


# ---------------------------------------------------------------- settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        database_url=os.environ.get(
            "DOCFLOW_TEST_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
        ),
        data_dir=tmp_path / "data",
        api_key="test-key",
        llm_provider="fake",
        watch_settle_seconds=0,
        notify_webhook_url=None,
        sheets_enabled=False,
    )


@pytest.fixture
def engine(settings: Settings) -> Iterator[Engine]:
    eng = make_engine(settings.database_url)
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


# -------------------------------------------------------- canned responses

INVOICE: dict[str, Any] = {
    "document_type": "invoice",
    "invoice": {
        "invoice_number": "INV-2026-0042",
        "vendor_name": "Brightline Office Supply Co.",
        "vendor_address": "418 Alder Street, Portland, OR 97205",
        "vendor_tax_id": None,
        "customer_name": "Meridian Health Partners",
        "customer_address": "900 Grant Street, Denver, CO 80203",
        "invoice_date": "2026-03-02",
        "due_date": "2026-04-01",
        "currency": "USD",
        "subtotal": 250.0,
        "discount": None,
        "tax": 20.0,
        "shipping": None,
        "total": 270.0,
        "payment_terms": "Net 30",
        "line_items": [
            {"description": "Toner cartridge", "quantity": 2, "unit_price": 100.0, "amount": 200.0},
            {"description": "Copy paper", "quantity": 5, "unit_price": 10.0, "amount": 50.0},
        ],
    },
    "bank_statement": None,
    "resume": None,
    "field_confidence": [
        {"field": "invoice_number", "confidence": 0.99},
        {"field": "vendor_name", "confidence": 0.98},
        {"field": "invoice_date", "confidence": 0.97},
        {"field": "total", "confidence": 0.99},
    ],
    "notes": None,
}

BANK: dict[str, Any] = {
    "document_type": "bank_statement",
    "invoice": None,
    "bank_statement": {
        "bank_name": "Northwind Savings Bank",
        "account_holder": "Maya Thornton",
        "account_number_last4": "4821",
        "statement_period_start": "2026-01-01",
        "statement_period_end": "2026-01-31",
        "currency": "USD",
        "opening_balance": 1000.0,
        "closing_balance": 1350.0,
        "total_credits": 500.0,
        "total_debits": 150.0,
        "transactions": [
            {
                "date": "2026-01-03",
                "description": "Card purchase",
                "amount": -150.0,
                "balance": 850.0,
            },
            {"date": "2026-01-25", "description": "Salary", "amount": 500.0, "balance": 1350.0},
        ],
    },
    "resume": None,
    "field_confidence": [
        {"field": name, "confidence": 0.95}
        for name in (
            "bank_name",
            "account_holder",
            "statement_period_start",
            "statement_period_end",
            "opening_balance",
            "closing_balance",
        )
    ],
    "notes": None,
}

RESUME: dict[str, Any] = {
    "document_type": "resume",
    "invoice": None,
    "bank_statement": None,
    "resume": {
        "full_name": "Priya Raman",
        "email": "priya.raman@example.com",
        "phone": "(555) 0143-9101",
        "location": "Austin, TX",
        "skills": ["Python", "SQL"],
        "experience": [
            {
                "company": "Fernbrook Labs",
                "title": "Data Engineer",
                "location": "Austin, TX",
                "start_date": "2023-02",
                "end_date": "present",
            }
        ],
        "education": [
            {
                "institution": "Lakeside State University",
                "degree": "B.Sc.",
                "field_of_study": "Statistics",
                "graduation_date": "2019",
            }
        ],
    },
    "field_confidence": [{"field": "full_name", "confidence": 0.99}],
    "notes": None,
}


@pytest.fixture
def invoice_response() -> dict[str, Any]:
    return copy.deepcopy(INVOICE)


@pytest.fixture
def bank_response() -> dict[str, Any]:
    return copy.deepcopy(BANK)


@pytest.fixture
def resume_response() -> dict[str, Any]:
    return copy.deepcopy(RESUME)


# ------------------------------------------------------------- pipelines


class RecordingNotifier:
    def __init__(self) -> None:
        self.documents: list[Any] = []
        self.batches: list[Any] = []

    def document_processed(self, doc: Any) -> bool:
        self.documents.append(doc)
        return True

    def batch_completed(self, summary: Any) -> bool:
        self.batches.append(summary)
        return True


class RecordingExporter:
    def __init__(self, fail: bool = False) -> None:
        self.exported: list[Any] = []
        self.fail = fail

    def export(self, doc: Any) -> None:
        if self.fail:
            raise RuntimeError("Sheets API is down")
        self.exported.append(doc)


@pytest.fixture
def make_pipeline(settings: Settings, engine: Engine) -> Callable[..., Pipeline]:
    def factory(responses: list[Any] | None = None, default: Any = None, **kwargs: Any) -> Pipeline:
        extractor = FakeExtractor(settings, responses or [], default=default)
        return Pipeline(settings, extractor, make_session_factory(engine), engine=engine, **kwargs)

    return factory


# ------------------------------------------------------------- documents


def make_pdf(lines: list[str], pages: int = 1) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen.canvas import Canvas

    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=letter, invariant=1)
    for _ in range(pages):
        y = 720
        for line in lines:
            canvas.setFont("Helvetica", 14)
            canvas.drawString(72, y, line)
            y -= 24
        canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def rasterize(pdf: bytes, dpi: int = 200) -> list[Any]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf)
    images = [doc[i].render(scale=dpi / 72).to_pil().convert("L") for i in range(len(doc))]
    doc.close()
    return images


INVOICE_LINES = [
    "INVOICE INV-2026-0042",
    "Brightline Office Supply Co.",
    "Invoice date: March 2, 2026",
    "Toner cartridge 2 x 100.00 = 200.00",
    "Copy paper 5 x 10.00 = 50.00",
    "Subtotal 250.00   Tax 20.00   Total 270.00",
]


@pytest.fixture
def invoice_pdf() -> bytes:
    return make_pdf(INVOICE_LINES)
