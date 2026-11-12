"""Events DocFlow sends to the n8n notify webhook.

Each event carries ready-to-post ``text`` (Slack mrkdwn) and ``plain_text``
(email), so the n8n workflow only routes; all formatting is tested here.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from docflow.config import Settings
from docflow.db import Document
from docflow.schemas import Status

log = logging.getLogger(__name__)

_EMOJI = {
    Status.ACCEPTED: ":white_check_mark:",
    Status.APPROVED: ":white_check_mark:",
    Status.NEEDS_REVIEW: ":warning:",
    Status.FAILED: ":x:",
    Status.REJECTED: ":no_entry:",
}
_LABEL = {
    Status.ACCEPTED: "accepted",
    Status.APPROVED: "approved",
    Status.NEEDS_REVIEW: "needs review",
    Status.FAILED: "failed",
    Status.REJECTED: "rejected",
}


@dataclass
class BatchSummary:
    started_at: str
    duration_s: float
    inbox_files: int = 0
    retried: int = 0
    accepted: int = 0
    needs_review: int = 0
    failed: int = 0
    exported: int = 0
    cost_usd: float = 0.0
    failures: list[dict[str, Any]] = field(default_factory=list)
    review: list[dict[str, Any]] = field(default_factory=list)


def _doc_url(settings: Settings, doc: Document) -> str:
    return f"{settings.public_base_url.rstrip('/')}/v1/documents/{doc.id}"


def document_event(doc: Document, settings: Settings) -> dict[str, Any]:
    status = Status(doc.status)
    url = _doc_url(settings, doc)
    confidence = f"{doc.confidence:.2f}" if doc.confidence is not None else "n/a"
    details = f"type {doc.doc_type or 'unknown'} · confidence {confidence} · via {doc.source}"
    lines = [
        f"{_EMOJI.get(status, '')} *{doc.filename}* {_LABEL.get(status, status.value)}",
        details,
    ]
    plain = [f"{doc.filename}: {_LABEL.get(status, status.value)}", details]
    if doc.review_reasons:
        why = "Why: " + "; ".join(doc.review_reasons)
        lines.append(why)
        plain.append(why)
    if doc.error:
        lines.append(f"Error: `{doc.error[:300]}`")
        plain.append(f"Error: {doc.error[:300]}")
    lines.append(f"<{url}|Open in DocFlow>")
    plain.append(url)
    return {
        "event": "document.processed",
        "has_failures": status == Status.FAILED,
        "needs_attention": status in (Status.FAILED, Status.NEEDS_REVIEW),
        "subject": f"[DocFlow] {doc.filename} {_LABEL.get(status, status.value)}",
        "text": "\n".join(lines),
        "plain_text": "\n".join(plain),
        "document": {
            "id": str(doc.id),
            "filename": doc.filename,
            "source": doc.source,
            "status": doc.status,
            "doc_type": doc.doc_type,
            "confidence": doc.confidence,
            "review_reasons": doc.review_reasons or [],
            "error": doc.error,
            "cost_usd": doc.cost_usd,
            "latency_ms": doc.total_latency_ms,
            "url": url,
        },
    }


def batch_event(summary: BatchSummary) -> dict[str, Any]:
    handled = summary.inbox_files + summary.retried
    head = (
        f"*DocFlow batch* {summary.inbox_files} new and {summary.retried} retried "
        f"in {summary.duration_s:.0f}s"
    )
    counts = (
        f":white_check_mark: {summary.accepted} accepted · :warning: {summary.needs_review} "
        f"need review · :x: {summary.failed} failed · ${summary.cost_usd:.4f}"
    )
    plain_counts = (
        f"{summary.accepted} accepted, {summary.needs_review} need review, "
        f"{summary.failed} failed, cost ${summary.cost_usd:.4f}"
    )
    lines, plain = [head, counts], [head.replace("*", ""), plain_counts]
    if handled == 0:
        lines.append("Nothing new in the inbox.")
        plain.append("Nothing new in the inbox.")
    for item in summary.failures[:10]:
        lines.append(f"• :x: {item['filename']}: {str(item.get('error'))[:200]}")
        plain.append(f"- FAILED {item['filename']}: {str(item.get('error'))[:200]}")
    for item in summary.review[:10]:
        reasons = "; ".join(item.get("reasons") or [])
        lines.append(f"• :warning: {item['filename']}: {reasons}")
        plain.append(f"- REVIEW {item['filename']}: {reasons}")
    return {
        "event": "batch.completed",
        "has_failures": summary.failed > 0,
        "needs_attention": summary.failed > 0 or summary.needs_review > 0,
        "subject": f"[DocFlow] batch: {summary.failed} failed, {summary.needs_review} need review",
        "text": "\n".join(lines),
        "plain_text": "\n".join(plain),
        "summary": asdict(summary),
    }


class Notifier:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.Client(timeout=10.0)

    @property
    def enabled(self) -> bool:
        return bool(self.settings.notify_webhook_url) and self.settings.notify_on != "none"

    def should_notify(self, status: str) -> bool:
        if not self.enabled:
            return False
        if self.settings.notify_on == "all":
            return True
        return status in (Status.FAILED, Status.NEEDS_REVIEW)

    def send(self, payload: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        headers = {}
        if self.settings.notify_secret:
            headers["X-DocFlow-Secret"] = self.settings.notify_secret
        try:
            response = self.client.post(
                self.settings.notify_webhook_url, json=payload, headers=headers
            )
            response.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            # Never fail a document because Slack/n8n is down; it's in the DB either way.
            log.warning("notification failed (%s): %s", payload.get("event"), exc)
            return False

    def document_processed(self, doc: Document) -> bool:
        if not self.should_notify(doc.status):
            return False
        return self.send(document_event(doc, self.settings))

    def batch_completed(self, summary: BatchSummary) -> bool:
        return self.send(batch_event(summary))
