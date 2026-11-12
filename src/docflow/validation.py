"""Business rules for extracted documents, plus the human-review decision.

Errors force review. Warnings are shown to the reviewer but don't force it on
their own. A document also goes to review when its confidence (the lowest model
confidence across required fields, scaled down for poor OCR) is below the
threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from typing import Any

from docflow.config import Settings
from docflow.schemas import (
    REQUIRED_FIELDS,
    BankStatement,
    DocType,
    Extraction,
    Invoice,
    Resume,
)

# Tesseract's mean word confidence on a clean 300-dpi scan is about 0.9; below
# that we scale the document's confidence down proportionally.
OCR_REFERENCE_CONFIDENCE = 0.90
LOW_OCR_CONFIDENCE = 0.75
UNREPORTED_FIELD_CONFIDENCE = 0.6
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class Issue:
    code: str
    message: str
    severity: Severity
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "field": self.field,
        }


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    confidence: float = 0.0
    needs_review: bool = True
    review_reasons: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]


class _Rules:
    def __init__(self, settings: Settings, today: date) -> None:
        self.tol = settings.amount_tolerance
        self.today = today
        self.issues: list[Issue] = []

    def error(self, code: str, message: str, field_name: str | None = None) -> None:
        self.issues.append(Issue(code, message, Severity.ERROR, field_name))

    def warn(self, code: str, message: str, field_name: str | None = None) -> None:
        self.issues.append(Issue(code, message, Severity.WARNING, field_name))

    def close(self, a: float, b: float, tolerance: float | None = None) -> bool:
        return abs(a - b) <= (self.tol if tolerance is None else tolerance) + 1e-9

    # -- shared -----------------------------------------------------------

    def required(self, section: Any, doc_type: DocType) -> None:
        for name in REQUIRED_FIELDS[doc_type]:
            if getattr(section, name) in (None, "", []):
                self.error("missing_required", f"{name} is missing", name)

    def iso_date(self, value: str | None, name: str, *, allow_future: bool = False) -> date | None:
        if value is None:
            return None
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            self.error("invalid_date", f"{name} '{value}' is not a YYYY-MM-DD date", name)
            return None
        if not allow_future and parsed > self.today + timedelta(days=1):
            self.warn("future_date", f"{name} {value} is in the future", name)
        if parsed < self.today - timedelta(days=365 * 15):
            self.warn("old_date", f"{name} {value} is more than 15 years old", name)
        return parsed

    def currency(self, value: str | None) -> None:
        if value is not None and not re.fullmatch(r"[A-Z]{3}", value):
            self.warn("invalid_currency", f"currency '{value}' is not an ISO 4217 code", "currency")

    # -- invoice ----------------------------------------------------------

    def invoice(self, inv: Invoice) -> None:
        self.required(inv, DocType.INVOICE)
        issued = self.iso_date(inv.invoice_date, "invoice_date")
        due = self.iso_date(inv.due_date, "due_date", allow_future=True)
        if issued and due and due < issued:
            self.warn(
                "due_before_issue", f"due_date {due} is before invoice_date {issued}", "due_date"
            )
        self.currency(inv.currency)
        for name in ("subtotal", "tax", "discount", "shipping", "total"):
            value = getattr(inv, name)
            if value is not None and value < 0:
                self.warn("negative_amount", f"{name} is negative ({value:.2f})", name)

        items_sum: float | None = None
        if inv.line_items:
            for i, item in enumerate(inv.line_items):
                if None not in (item.quantity, item.unit_price, item.amount) and not self.close(
                    item.quantity * item.unit_price,
                    item.amount,
                    max(self.tol, 0.005 * abs(item.amount)),
                ):
                    self.warn(
                        "line_item_math",
                        f"line {i + 1}: {item.quantity:g} x {item.unit_price:.2f} "
                        f"!= {item.amount:.2f}",
                        f"line_items[{i}]",
                    )
            amounts = [item.amount for item in inv.line_items]
            if None in amounts:
                self.warn("line_item_missing_amount", "a line item has no amount", "line_items")
            else:
                items_sum = round(sum(amounts), 2)
                if inv.subtotal is not None and not self.close(
                    items_sum, inv.subtotal, self.tol * len(amounts)
                ):
                    self.error(
                        "line_items_mismatch",
                        f"line items sum to {items_sum:.2f} but subtotal is {inv.subtotal:.2f}",
                        "subtotal",
                    )

        base = inv.subtotal if inv.subtotal is not None else items_sum
        if base is not None and inv.total is not None:
            discount = abs(inv.discount or 0.0)
            tax = inv.tax or 0.0
            shipping = inv.shipping or 0.0
            expected = round(base - discount + tax + shipping, 2)
            if not self.close(expected, inv.total, self.tol * 2):
                if tax and self.close(base - discount + shipping, inv.total, self.tol * 2):
                    self.warn(
                        "tax_inclusive",
                        "total equals subtotal without tax; prices may be tax-inclusive",
                        "tax",
                    )
                else:
                    self.error(
                        "totals_mismatch",
                        f"subtotal {base:.2f} - discount {discount:.2f} + tax {tax:.2f} + "
                        f"shipping {shipping:.2f} = {expected:.2f}, but total is {inv.total:.2f}",
                        "total",
                    )

    # -- bank statement ---------------------------------------------------

    def bank_statement(self, bs: BankStatement) -> None:
        self.required(bs, DocType.BANK_STATEMENT)
        start = self.iso_date(bs.statement_period_start, "statement_period_start")
        end = self.iso_date(bs.statement_period_end, "statement_period_end")
        if start and end and start > end:
            self.error("period_inverted", f"statement period starts {start} after it ends {end}")
        self.currency(bs.currency)

        txs = bs.transactions
        if not txs:
            self.warn("no_transactions", "no transactions were extracted", "transactions")
            if (
                bs.opening_balance is not None
                and bs.closing_balance is not None
                and not self.close(bs.opening_balance, bs.closing_balance)
            ):
                self.error(
                    "balance_mismatch",
                    "balances changed but no transactions were extracted",
                    "transactions",
                )
            return

        amounts = [t.amount for t in txs]
        if None in amounts:
            self.error(
                "transaction_missing_amount",
                f"{amounts.count(None)} transaction(s) have no amount",
                "transactions",
            )
            return

        if bs.opening_balance is not None and bs.closing_balance is not None:
            expected = round(bs.opening_balance + sum(amounts), 2)
            if not self.close(expected, bs.closing_balance):
                self.error(
                    "balance_mismatch",
                    f"opening {bs.opening_balance:.2f} + transactions {sum(amounts):.2f} = "
                    f"{expected:.2f}, but closing balance is {bs.closing_balance:.2f}",
                    "closing_balance",
                )

        credits = round(sum(a for a in amounts if a > 0), 2)
        debits = round(-sum(a for a in amounts if a < 0), 2)
        if bs.total_credits is not None and not self.close(credits, abs(bs.total_credits)):
            self.warn(
                "credits_mismatch",
                f"credits sum to {credits:.2f} but total_credits is {bs.total_credits:.2f}",
                "total_credits",
            )
        if bs.total_debits is not None and not self.close(debits, abs(bs.total_debits)):
            self.warn(
                "debits_mismatch",
                f"debits sum to {debits:.2f} but total_debits is {bs.total_debits:.2f}",
                "total_debits",
            )

        running = bs.opening_balance
        for i, tx in enumerate(txs):
            printed = tx.balance
            if (
                running is not None
                and printed is not None
                and not self.close(running + tx.amount, printed)
            ):
                self.warn(
                    "running_balance_mismatch",
                    f"transaction {i + 1}: {running:.2f} + {tx.amount:.2f} != "
                    f"printed balance {printed:.2f}",
                    f"transactions[{i}]",
                )
                break
            running = (
                tx.balance
                if tx.balance is not None
                else (running + tx.amount if running is not None else None)
            )

        for i, tx in enumerate(txs):
            if tx.date is None:
                continue
            try:
                when = date.fromisoformat(tx.date)
            except ValueError:
                self.warn(
                    "invalid_date",
                    f"transaction {i + 1} date '{tx.date}' is invalid",
                    f"transactions[{i}]",
                )
                break
            if start and end and not (start - timedelta(days=3) <= when <= end + timedelta(days=3)):
                self.warn(
                    "transaction_outside_period",
                    f"transaction {i + 1} ({when}) is outside the statement period",
                    f"transactions[{i}]",
                )
                break

    # -- resume -----------------------------------------------------------

    def resume(self, cv: Resume) -> None:
        self.required(cv, DocType.RESUME)
        if cv.email and not EMAIL_RE.fullmatch(cv.email):
            self.warn("invalid_email", f"email '{cv.email}' looks invalid", "email")
        if cv.phone:
            digits = re.sub(r"\D", "", cv.phone)
            if not 7 <= len(digits) <= 15:
                self.warn("invalid_phone", f"phone '{cv.phone}' looks invalid", "phone")
        if not cv.email and not cv.phone:
            self.warn("no_contact", "no email or phone number found")
        if not cv.experience and not cv.education:
            self.warn("empty_history", "no experience or education found")
        for i, job in enumerate(cv.experience):
            started = self.partial_date(job.start_date, f"experience[{i}].start_date")
            ended = self.partial_date(job.end_date, f"experience[{i}].end_date")
            if started and ended and ended < started:
                self.warn(
                    "experience_dates_inverted",
                    f"{job.company or 'job ' + str(i + 1)}: ends before it starts",
                    f"experience[{i}]",
                )

    def partial_date(self, value: str | None, name: str) -> date | None:
        if value is None:
            return None
        if value == "present":
            return self.today
        match = re.fullmatch(r"(\d{4})(?:-(\d{2}))?", value)
        if not match or not 1 <= int(match.group(2) or 1) <= 12:
            self.warn("invalid_date", f"{name} '{value}' is not YYYY-MM, YYYY or present", name)
            return None
        return date(int(match.group(1)), int(match.group(2) or 1), 1)


def validate(
    extraction: Extraction,
    settings: Settings,
    *,
    ocr_confidence: float | None = None,
    today: date | None = None,
) -> ValidationReport:
    rules = _Rules(settings, today or date.today())
    doc_type = extraction.document_type
    section = extraction.section()

    if doc_type == DocType.OTHER:
        rules.error(
            "unsupported_document_type",
            "document was not recognised as an invoice, bank statement or resume",
        )
    elif section is None:
        rules.error("missing_section", f"no {doc_type.value} fields were extracted")
    elif doc_type == DocType.INVOICE:
        rules.invoice(section)  # type: ignore[arg-type]
    elif doc_type == DocType.BANK_STATEMENT:
        rules.bank_statement(section)  # type: ignore[arg-type]
    elif doc_type == DocType.RESUME:
        rules.resume(section)  # type: ignore[arg-type]

    confidence = _confidence(extraction, ocr_confidence)
    threshold = settings.review_confidence_threshold
    for name, score in sorted(extraction.confidence_map().items()):
        if score < threshold:
            rules.warn("low_confidence", f"model confidence for {name} is {score:.2f}", name)

    reasons: list[str] = []
    errors = [i for i in rules.issues if i.severity == Severity.ERROR]
    if errors:
        codes = ", ".join(dict.fromkeys(i.code for i in errors))
        reasons.append(f"{len(errors)} validation error(s): {codes}")
    if confidence < threshold:
        reasons.append(f"confidence {confidence:.2f} is below {threshold:.2f}")
    if ocr_confidence is not None and ocr_confidence < LOW_OCR_CONFIDENCE:
        reasons.append(f"low OCR quality ({ocr_confidence:.2f})")

    return ValidationReport(
        issues=rules.issues,
        confidence=confidence,
        needs_review=bool(reasons),
        review_reasons=reasons,
    )


def _confidence(extraction: Extraction, ocr_confidence: float | None) -> float:
    """Lowest model confidence across the required fields that were found."""
    section = extraction.section()
    if extraction.document_type == DocType.OTHER or section is None:
        return 0.0
    reported = extraction.confidence_map()
    scores = [
        reported.get(name, UNREPORTED_FIELD_CONFIDENCE)
        for name in REQUIRED_FIELDS[extraction.document_type]
        if getattr(section, name) not in (None, "", [])
    ]
    score = min(scores) if scores else 0.0
    if ocr_confidence is not None:
        score *= min(1.0, ocr_confidence / OCR_REFERENCE_CONFIDENCE)
    return round(score, 4)
