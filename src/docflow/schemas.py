"""Pydantic schemas for extracted documents.

These models are the contract with the LLM: their JSON Schema is sent as the
structured-output format, and every response is validated against them.
Types stay deliberately loose (strings for dates, floats for money) so a single
odd value becomes a validation *issue* for a human, not a hard parse failure
that throws the whole document away. Semantic checks live in ``validation.py``.
"""

from __future__ import annotations

import copy
import re
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator


class DocType(StrEnum):
    INVOICE = "invoice"
    BANK_STATEMENT = "bank_statement"
    RESUME = "resume"
    OTHER = "other"


class Status(StrEnum):
    PROCESSING = "processing"
    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"
    APPROVED = "approved"
    REJECTED = "rejected"


# --------------------------------------------------------------------------
# Normalizers (also used by the eval scorer, so labels and predictions match)
# --------------------------------------------------------------------------

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d-%b-%Y",
    "%d.%m.%Y",
)
_MONTH_FORMATS = ("%Y-%m", "%m/%Y", "%B %Y", "%b %Y", "%Y/%m")
_PRESENT = {"present", "current", "now", "ongoing", "to date", "today"}


def normalize_date(value: Any) -> str | None:
    """Return an ISO ``YYYY-MM-DD`` string, or the raw string if it can't be parsed."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return text  # left as-is; validation flags it as an invalid date


def normalize_partial_date(value: Any) -> str | None:
    """Resume dates: ``YYYY-MM``, ``YYYY`` or ``present``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in _PRESENT:
        return "present"
    if re.fullmatch(r"\d{4}", text) or re.fullmatch(r"\d{4}-\d{2}", text):
        return text
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text[:7]
    for fmt in _MONTH_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m")
        except ValueError:
            continue
    return text


def parse_amount(value: Any) -> float | None:
    """Parse ``1,234.50``, ``$1 234,50``, ``(12.00)`` or ``12.00 DR`` into a float."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("expected a number, got a boolean")
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    negative = (
        (text.startswith("(") and text.endswith(")"))
        or text.startswith("-")
        or text.endswith("-")
        or text.upper().endswith(("DR", "CR-"))
    )
    # Drop currency words first so the dot in "Rs. 99" isn't read as a decimal point.
    digits = re.sub(r"[^0-9.,]", "", re.sub(r"[A-Za-z]+\.?", "", text))
    if not digits:
        raise ValueError(f"not a number: {value!r}")
    if "," in digits and "." in digits:
        if digits.rfind(",") > digits.rfind("."):  # 1.234,56
            digits = digits.replace(".", "").replace(",", ".")
        else:  # 1,234.56
            digits = digits.replace(",", "")
    elif "," in digits:
        head, _, tail = digits.rpartition(",")
        digits = (
            f"{head.replace(',', '')}.{tail}" if len(tail) in (1, 2) else digits.replace(",", "")
        )
    try:
        number = float(digits)
    except ValueError as exc:
        raise ValueError(f"not a number: {value!r}") from exc
    return -number if negative else number


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _currency(value: Any) -> Any:
    value = _blank_to_none(value)
    return value.upper() if isinstance(value, str) else value


OptStr = Annotated[str | None, BeforeValidator(_blank_to_none)]
Number = Annotated[float | None, BeforeValidator(parse_amount)]
IsoDate = Annotated[str | None, BeforeValidator(normalize_date)]
PartialDate = Annotated[str | None, BeforeValidator(normalize_partial_date)]
Currency = Annotated[str | None, BeforeValidator(_currency)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Invoice
# --------------------------------------------------------------------------


class LineItem(_Strict):
    description: OptStr = Field(None, description="Item or service, as printed.")
    quantity: Number = None
    unit_price: Number = None
    amount: Number = Field(None, description="Line total as printed.")


class Invoice(_Strict):
    invoice_number: OptStr = None
    vendor_name: OptStr = Field(None, description="The business issuing the invoice.")
    vendor_address: OptStr = Field(None, description="Single line, comma separated.")
    vendor_tax_id: OptStr = Field(None, description="VAT / GST / EIN number if printed.")
    customer_name: OptStr = Field(None, description="The party being billed.")
    customer_address: OptStr = Field(None, description="Single line, comma separated.")
    invoice_date: IsoDate = Field(None, description="YYYY-MM-DD")
    due_date: IsoDate = Field(None, description="YYYY-MM-DD")
    currency: Currency = Field(None, description="ISO 4217 code, e.g. USD, EUR, INR.")
    subtotal: Number = None
    discount: Number = Field(None, description="Total discount as a positive number.")
    tax: Number = Field(None, description="Total tax / VAT / GST amount.")
    shipping: Number = None
    total: Number = Field(None, description="Grand total / amount due.")
    payment_terms: OptStr = Field(None, description="e.g. 'Net 30'.")
    line_items: list[LineItem] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Bank statement
# --------------------------------------------------------------------------


class Transaction(_Strict):
    date: IsoDate = Field(None, description="YYYY-MM-DD")
    description: OptStr = None
    amount: Number = Field(
        None, description="Signed: credits/deposits positive, debits/withdrawals negative."
    )
    balance: Number = Field(None, description="Running balance after this line, if printed.")


class BankStatement(_Strict):
    bank_name: OptStr = None
    account_holder: OptStr = None
    account_number_last4: OptStr = Field(
        None, description="Only the last 4 digits of the account number."
    )
    statement_period_start: IsoDate = Field(None, description="YYYY-MM-DD")
    statement_period_end: IsoDate = Field(None, description="YYYY-MM-DD")
    currency: Currency = Field(None, description="ISO 4217 code.")
    opening_balance: Number = None
    closing_balance: Number = None
    total_credits: Number = Field(None, description="Sum of credits, positive number.")
    total_debits: Number = Field(None, description="Sum of debits, positive number.")
    transactions: list[Transaction] = Field(default_factory=list)

    @field_validator("account_number_last4", mode="after")
    @classmethod
    def _mask_account_number(cls, value: str | None) -> str | None:
        """Never store a full account number, even if the model returns one."""
        if value is None:
            return None
        digits = re.sub(r"\D", "", value)
        return digits[-4:] if digits else None


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------


class Experience(_Strict):
    company: OptStr = None
    title: OptStr = None
    location: OptStr = None
    start_date: PartialDate = Field(None, description="YYYY-MM, or YYYY if no month.")
    end_date: PartialDate = Field(None, description="YYYY-MM, YYYY, or 'present'.")


class Education(_Strict):
    institution: OptStr = None
    degree: OptStr = Field(None, description="e.g. 'B.Sc.', 'MBA', 'Ph.D.'")
    field_of_study: OptStr = None
    graduation_date: PartialDate = Field(None, description="YYYY-MM or YYYY.")


class Resume(_Strict):
    full_name: OptStr = None
    email: OptStr = None
    phone: OptStr = None
    location: OptStr = None
    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list, description="Most recent first.")
    education: list[Education] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Envelope returned by the LLM
# --------------------------------------------------------------------------


class FieldConfidence(_Strict):
    field: str = Field(description="Top-level field name of the filled section, e.g. 'total'.")
    confidence: float = Field(description="0.0 (guess) to 1.0 (clearly printed).")

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, value: Any) -> float:
        return min(1.0, max(0.0, float(value)))


class Extraction(_Strict):
    document_type: DocType
    invoice: Invoice | None = Field(None, description="Only when document_type is invoice.")
    bank_statement: BankStatement | None = Field(
        None, description="Only when document_type is bank_statement."
    )
    resume: Resume | None = Field(None, description="Only when document_type is resume.")
    field_confidence: list[FieldConfidence] = Field(default_factory=list)
    notes: OptStr = Field(None, description="Ambiguities a reviewer should know about.")

    @model_validator(mode="after")
    def _keep_only_matching_section(self) -> Extraction:
        for doc_type, attr in SECTION_ATTR.items():
            if doc_type != self.document_type and getattr(self, attr) is not None:
                setattr(self, attr, None)
        return self

    def section(self) -> BaseModel | None:
        attr = SECTION_ATTR.get(self.document_type)
        return getattr(self, attr) if attr else None

    def confidence_map(self) -> dict[str, float]:
        return {fc.field.split(".")[-1]: fc.confidence for fc in self.field_confidence}


SECTION_ATTR: dict[DocType, str] = {
    DocType.INVOICE: "invoice",
    DocType.BANK_STATEMENT: "bank_statement",
    DocType.RESUME: "resume",
}

SECTION_MODELS: dict[DocType, type[BaseModel]] = {
    DocType.INVOICE: Invoice,
    DocType.BANK_STATEMENT: BankStatement,
    DocType.RESUME: Resume,
}

REQUIRED_FIELDS: dict[DocType, tuple[str, ...]] = {
    DocType.INVOICE: ("invoice_number", "vendor_name", "invoice_date", "total"),
    DocType.BANK_STATEMENT: (
        "bank_name",
        "account_holder",
        "statement_period_start",
        "statement_period_end",
        "opening_balance",
        "closing_balance",
    ),
    DocType.RESUME: ("full_name",),
}


# --------------------------------------------------------------------------
# JSON Schema for structured outputs
# --------------------------------------------------------------------------

# Keywords structured-output APIs reject or don't need. Constraints that matter
# are re-checked client-side by Pydantic and validation.py.
_DROP_KEYWORDS = {
    "title",
    "default",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
}


def llm_json_schema(model: type[BaseModel] = Extraction) -> dict[str, Any]:
    """Pydantic's schema, made portable for Claude and Gemini structured outputs.

    ``$ref``s are inlined, unsupported keywords dropped, and every object gets
    ``additionalProperties: false`` with all properties required (nullable
    fields stay ``anyOf [..., null]``), so the model must consider each field.
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def clean(node: Any) -> Any:
        if isinstance(node, list):
            return [clean(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = clean(copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]]))
            siblings = {k: v for k, v in node.items() if k != "$ref" and k not in _DROP_KEYWORDS}
            return {**target, **siblings}
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _DROP_KEYWORDS:
                continue
            if key == "properties":
                out[key] = {name: clean(sub) for name, sub in value.items()}
            else:
                out[key] = clean(value)
        if out.get("type") == "object":
            out["additionalProperties"] = False
            out["required"] = list(out.get("properties", {}))
        return out

    return clean(raw)
