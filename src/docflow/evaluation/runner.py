"""Run the pipeline over a labelled dataset and collect per-document results.

Dataset layout::

    eval/dataset/
      docs/inv-001.pdf ...
      labels/inv-001.json   {"file": "docs/inv-001.pdf", "document_type": "invoice",
                             "fields": {...}, "meta": {"verified": true, ...}}

The runner uses ingest + extract + validate directly (no database), so it
measures the same code path the API runs.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docflow.config import Settings
from docflow.evaluation.scoring import DocScore, score_document
from docflow.extract import ExtractionError, Extractor
from docflow.ingest import IngestError, ingest
from docflow.schemas import DocType
from docflow.validation import validate

log = logging.getLogger(__name__)


@dataclass
class Label:
    id: str
    path: Path
    document_type: DocType
    fields: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocResult:
    label: Label
    score: DocScore | None = None
    predicted_type: str | None = None
    ingest_method: str | None = None
    ocr_confidence: float | None = None
    ingest_ms: int = 0
    extract_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    needs_review: bool = True
    review_reasons: list[str] = field(default_factory=list)
    error: str | None = None
    prediction: dict[str, Any] | None = None

    @property
    def total_ms(self) -> int:
        return self.ingest_ms + self.extract_ms


def load_labels(dataset_dir: Path, *, verified_only: bool = False) -> list[Label]:
    labels = []
    for path in sorted((dataset_dir / "labels").glob("*.json")):
        raw = json.loads(path.read_text())
        meta = raw.get("meta", {})
        if verified_only and not meta.get("verified"):
            continue
        labels.append(
            Label(
                id=path.stem,
                path=dataset_dir / raw["file"],
                document_type=DocType(raw["document_type"]),
                fields=raw.get("fields") or {},
                meta=meta,
            )
        )
    return labels


def evaluate_one(label: Label, extractor: Extractor, settings: Settings) -> DocResult:
    result = DocResult(label=label)
    start = time.perf_counter()
    try:
        ingested = ingest(label.path.read_bytes(), label.path.name, settings)
    except (IngestError, OSError) as exc:
        result.error = f"ingest: {exc}"
        result.score = score_document(label.document_type, label.fields, None, None)
        return result
    result.ingest_ms = int((time.perf_counter() - start) * 1000)
    result.ingest_method = ingested.method
    result.ocr_confidence = ingested.ocr_confidence

    start = time.perf_counter()
    try:
        outcome = extractor.extract(ingested.text)
    except ExtractionError as exc:
        result.extract_ms = int((time.perf_counter() - start) * 1000)
        result.error = f"extraction: {exc}"
        result.input_tokens, result.output_tokens = exc.usage.input_tokens, exc.usage.output_tokens
        result.cost_usd = exc.usage.cost_usd
        result.score = score_document(label.document_type, label.fields, None, None)
        return result
    result.extract_ms = int((time.perf_counter() - start) * 1000)

    extraction = outcome.extraction
    section = extraction.section()
    report = validate(extraction, settings, ocr_confidence=ingested.ocr_confidence)
    result.predicted_type = extraction.document_type.value
    result.prediction = section.model_dump(mode="json") if section is not None else None
    result.input_tokens = outcome.usage.input_tokens
    result.output_tokens = outcome.usage.output_tokens
    result.cost_usd = outcome.usage.cost_usd
    result.needs_review = report.needs_review
    result.review_reasons = report.review_reasons
    result.score = score_document(
        label.document_type, label.fields, extraction.document_type, result.prediction
    )
    return result


def run(
    labels: list[Label], extractor: Extractor, settings: Settings, *, workers: int = 4
) -> list[DocResult]:
    def task(label: Label) -> DocResult:
        result = evaluate_one(label, extractor, settings)
        status = "ok" if result.score and result.score.all_correct else "diff"
        log.info("%-12s %-4s %6d ms  %s", label.id, status, result.total_ms, result.error or "")
        return result

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(task, labels))
