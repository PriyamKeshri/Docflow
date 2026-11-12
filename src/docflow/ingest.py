"""Turn an uploaded file into text.

Text-layer PDFs go through pdfplumber. Pages with (almost) no text layer — scans,
photos saved as PDF — are rasterised and sent to Tesseract, page by page, so a
PDF that mixes typed and scanned pages is handled correctly. Images go straight
to Tesseract. Word-level Tesseract confidence is kept: it feeds the review flag.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image, ImageOps, ImageSequence

from docflow.config import Settings

log = logging.getLogger(__name__)

Method = Literal["text", "ocr"]

SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".txt": "text/plain",
}


class IngestError(Exception):
    """The file could not be turned into text (bad format, encrypted, no OCR...).

    ``retryable`` is False for problems with the file itself (retrying won't help)
    and True for environment problems such as a missing Tesseract binary.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass
class PageText:
    number: int
    text: str
    method: Method
    ocr_confidence: float | None = None


@dataclass
class IngestResult:
    mime_type: str
    pages: list[PageText] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def method(self) -> str:
        methods = {p.method for p in self.pages}
        if methods == {"text"}:
            return "text"
        if methods == {"ocr"}:
            return "ocr"
        return "mixed"

    @property
    def ocr_confidence(self) -> float | None:
        scores = [p.ocr_confidence for p in self.pages if p.ocr_confidence is not None]
        return round(sum(scores) / len(scores), 4) if scores else None

    @property
    def text(self) -> str:
        if len(self.pages) == 1:
            return self.pages[0].text
        return "\n\n".join(f"--- Page {p.number} ---\n{p.text}" for p in self.pages)


def detect_mime(data: bytes, filename: str) -> str:
    """Sniff magic bytes first; fall back to the extension."""
    head = data[:8]
    if head.startswith(b"%PDF"):
        return "application/pdf"
    if head.startswith(b"\x89PNG"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime = SUPPORTED_EXTENSIONS.get(suffix)
    if mime == "text/plain":
        try:
            data.decode("utf-8")
            return mime
        except UnicodeDecodeError:
            pass
    if mime and mime != "text/plain":
        # Extension claims a binary format but the bytes don't match it.
        raise IngestError(f"{filename}: content does not look like {mime}")
    raise IngestError(f"{filename}: unsupported file type (supported: PDF, PNG, JPEG, TIFF, TXT)")


def ingest(data: bytes, filename: str, settings: Settings) -> IngestResult:
    if not data:
        raise IngestError(f"{filename}: file is empty")
    mime = detect_mime(data, filename)
    if mime == "application/pdf":
        result = _ingest_pdf(data, filename, settings)
    elif mime == "text/plain":
        result = IngestResult(mime, [PageText(1, data.decode("utf-8").strip(), "text")])
    else:
        result = _ingest_image(data, filename, mime, settings)
    if not any(p.text.strip() for p in result.pages):
        raise IngestError(f"{filename}: no readable text found")
    return result


def _ingest_pdf(data: bytes, filename: str, settings: Settings) -> IngestResult:
    import pdfplumber
    from pdfminer.pdfdocument import PDFPasswordIncorrect
    from pdfplumber.utils.exceptions import PdfminerException

    result = IngestResult("application/pdf")
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > settings.max_pages:
                raise IngestError(
                    f"{filename}: {len(pdf.pages)} pages exceeds the limit of {settings.max_pages}"
                )
            for number, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if len(text) >= settings.min_text_chars_per_page:
                    result.pages.append(PageText(number, text, "text"))
                    continue
                log.info("%s page %d has no text layer; running OCR", filename, number)
                image = page.to_image(resolution=settings.ocr_dpi).original
                ocr_text, confidence = ocr_image(image, settings)
                result.pages.append(PageText(number, ocr_text, "ocr", confidence))
    except IngestError:
        raise
    except (PDFPasswordIncorrect, PdfminerException) as exc:
        raise IngestError(f"{filename}: unreadable or encrypted PDF ({exc})") from exc
    return result


def _ingest_image(data: bytes, filename: str, mime: str, settings: Settings) -> IngestResult:
    try:
        image = Image.open(io.BytesIO(data))
        frames = [frame.copy() for frame in ImageSequence.Iterator(image)]
    except Exception as exc:  # PIL raises a zoo of exception types
        raise IngestError(f"{filename}: could not open image ({exc})") from exc
    if len(frames) > settings.max_pages:
        raise IngestError(f"{filename}: {len(frames)} frames exceeds the limit")
    result = IngestResult(mime)
    for number, frame in enumerate(frames, start=1):
        text, confidence = ocr_image(frame, settings)
        result.pages.append(PageText(number, text, "ocr", confidence))
    return result


def ocr_image(image: Image.Image, settings: Settings) -> tuple[str, float]:
    """OCR one image; returns (text, mean word confidence 0..1).

    Uses a single ``image_to_data`` pass and rebuilds lines from Tesseract's
    block/paragraph/line numbering, so we get text and confidences without
    running OCR twice.
    """
    import pytesseract

    prepared = ImageOps.autocontrast(ImageOps.grayscale(ImageOps.exif_transpose(image)))
    try:
        data = pytesseract.image_to_data(
            prepared,
            lang=settings.ocr_lang,
            config="--oem 1 --psm 3",
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise IngestError(
            "Tesseract is not installed (brew install tesseract / apt install tesseract-ocr)",
            retryable=True,
        ) from exc

    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        conf = float(data["conf"][i])
        if conf >= 0:
            confidences.append(conf)

    out: list[str] = []
    previous_block = None
    for (block, _par, _line), words in lines.items():
        if previous_block is not None and block != previous_block:
            out.append("")
        out.append(" ".join(words))
        previous_block = block
    mean_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.0
    return "\n".join(out), round(mean_conf, 4)
