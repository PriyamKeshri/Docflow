"""Append accepted / approved documents to Google Sheets (one worksheet per type)."""

from __future__ import annotations

import logging
import typing
from typing import Any

from docflow.config import Settings
from docflow.db import Document
from docflow.schemas import SECTION_MODELS, DocType

log = logging.getLogger(__name__)

META_COLUMNS = ["document_id", "filename", "status", "confidence", "processed_at"]


def _columns(doc_type: DocType) -> tuple[list[str], list[str]]:
    model = SECTION_MODELS[doc_type]
    scalars, lists = [], []
    for name, info in model.model_fields.items():
        (lists if typing.get_origin(info.annotation) is list else scalars).append(name)
    return scalars, lists


def build_row(doc: Document) -> tuple[list[str], list[Any]]:
    """Header and row for a document. Lists (line items, transactions...) become counts."""
    doc_type = DocType(doc.doc_type)
    scalars, lists = _columns(doc_type)
    header = META_COLUMNS + scalars + [f"{name}_count" for name in lists]
    data = doc.effective_data or {}
    row: list[Any] = [
        str(doc.id),
        doc.filename,
        doc.status,
        doc.confidence,
        doc.processed_at.isoformat() if doc.processed_at else None,
    ]
    row += [data.get(name) for name in scalars]
    row += [len(data.get(name) or []) for name in lists]
    return header, ["" if value is None else value for value in row]


class SheetsExporter:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if not settings.sheets_spreadsheet_id:
            raise ValueError("DOCFLOW_SHEETS_SPREADSHEET_ID is required when Sheets export is on")
        self.settings = settings
        self._client = client
        self._book: Any | None = None

    def _spreadsheet(self) -> Any:
        if self._book is None:
            if self._client is None:
                import gspread

                self._client = gspread.service_account(
                    filename=str(self.settings.google_service_account_file)
                )
            self._book = self._client.open_by_key(self.settings.sheets_spreadsheet_id)
        return self._book

    def _worksheet(self, title: str, header: list[str]) -> Any:
        import gspread

        book = self._spreadsheet()
        try:
            return book.worksheet(title)
        except gspread.WorksheetNotFound:
            sheet = book.add_worksheet(title=title, rows=1000, cols=len(header))
            sheet.append_row(header, value_input_option="RAW")
            return sheet

    def export(self, doc: Document) -> None:
        if doc.doc_type not in {t.value for t in SECTION_MODELS}:
            return
        header, row = build_row(doc)
        # RAW, not USER_ENTERED: a vendor called "=IMPORTXML(...)" must stay text.
        self._worksheet(doc.doc_type, header).append_row(row, value_input_option="RAW")
        log.info("exported %s to Google Sheets", doc.id)
