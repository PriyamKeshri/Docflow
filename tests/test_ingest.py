from __future__ import annotations

import io

import pytest

from docflow.config import Settings
from docflow.ingest import IngestError, detect_mime, ingest
from tests.conftest import INVOICE_LINES, make_pdf, needs_tesseract, rasterize


def test_text_pdf_uses_the_text_layer(settings: Settings, invoice_pdf: bytes) -> None:
    result = ingest(invoice_pdf, "invoice.pdf", settings)
    assert result.method == "text"
    assert result.ocr_confidence is None
    assert result.page_count == 1
    assert "INV-2026-0042" in result.text
    assert "Total 270.00" in result.text


def test_multi_page_text_is_marked_by_page(settings: Settings) -> None:
    result = ingest(make_pdf(INVOICE_LINES, pages=2), "two.pdf", settings)
    assert result.page_count == 2
    assert "--- Page 2 ---" in result.text


def test_plain_text_file(settings: Settings) -> None:
    result = ingest(b"Invoice 42\nTotal 10.00", "note.txt", settings)
    assert result.method == "text"
    assert result.text.startswith("Invoice 42")


@pytest.mark.parametrize(
    ("data", "name"),
    [
        (b"PK\x03\x04 not really a docx", "letter.docx"),
        (b"\x00\x01\x02", "mystery.bin"),
    ],
)
def test_unsupported_files_are_rejected(settings: Settings, data: bytes, name: str) -> None:
    with pytest.raises(IngestError, match="unsupported"):
        ingest(data, name, settings)


def test_extension_that_lies_is_rejected(settings: Settings) -> None:
    with pytest.raises(IngestError, match="does not look like"):
        detect_mime(b"just text", "scan.png")


def test_magic_bytes_beat_the_extension() -> None:
    assert detect_mime(b"%PDF-1.7 ...", "upload.bin") == "application/pdf"
    assert detect_mime(b"\x89PNG\r\n\x1a\n", "photo.jpg") == "image/png"


def test_empty_and_corrupt_files(settings: Settings) -> None:
    with pytest.raises(IngestError, match="empty"):
        ingest(b"", "empty.pdf", settings)
    with pytest.raises(IngestError, match="unreadable"):
        ingest(b"%PDF-1.4 this is not a real pdf", "broken.pdf", settings)


def test_page_limit(settings: Settings) -> None:
    settings.max_pages = 2
    with pytest.raises(IngestError, match="exceeds"):
        ingest(make_pdf(["page"], pages=3), "long.pdf", settings)


def test_missing_tesseract_is_retryable(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pytesseract

    def boom(*args, **kwargs):
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_data", boom)
    buffer = io.BytesIO()
    rasterize(make_pdf(["hello"]))[0].save(buffer, "PNG")
    with pytest.raises(IngestError) as info:
        ingest(buffer.getvalue(), "scan.png", settings)
    assert info.value.retryable


@needs_tesseract
def test_image_goes_through_ocr(settings: Settings, invoice_pdf: bytes) -> None:
    buffer = io.BytesIO()
    rasterize(invoice_pdf)[0].save(buffer, "PNG")
    result = ingest(buffer.getvalue(), "scan.png", settings)
    assert result.method == "ocr"
    assert result.ocr_confidence is not None and result.ocr_confidence > 0.6
    assert "INV-2026-0042" in result.text


@needs_tesseract
def test_scanned_pdf_pages_fall_back_to_ocr(settings: Settings, invoice_pdf: bytes) -> None:
    image = rasterize(invoice_pdf)[0]
    buffer = io.BytesIO()
    image.save(buffer, "PDF", resolution=200)  # image-only PDF, no text layer
    result = ingest(buffer.getvalue(), "scan.pdf", settings)
    assert result.method == "ocr"
    assert "Brightline" in result.text
