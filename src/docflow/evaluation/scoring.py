"""Field-level comparison of an extraction against a hand-made label.

Matching rules (also documented in the README):
- ids, codes, emails: exact after trimming and case-folding
- text (names, addresses): equal after normalising case, whitespace and
  punctuation, or a similarity ratio >= 0.92 (absorbs a stray OCR character)
- money: within 0.01; plain numbers within 0.1 %
- dates: identical ISO strings after normalisation
- phone numbers: last 10 digits
- both null counts as correct; one null counts as wrong
- lists of objects are matched item-by-item (an item is right only if every
  sub-field is right) and scored with precision/recall/F1; the list *field* is
  right only when every item matches and nothing is extra or missing
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from docflow.schemas import DocType, normalize_date, normalize_partial_date

TEXT_SIMILARITY = 0.92


def _norm_text(value: Any) -> str:
    text = str(value).casefold()
    text = re.sub(r"[.,;:'\"()\[\]]", " ", text)
    return " ".join(text.split())


def match_text(a: Any, b: Any) -> bool:
    x, y = _norm_text(a), _norm_text(b)
    return x == y or SequenceMatcher(None, x, y).ratio() >= TEXT_SIMILARITY


def match_id(a: Any, b: Any) -> bool:
    return "".join(str(a).split()).casefold() == "".join(str(b).split()).casefold()


_TAX_ID_LABEL = re.compile(
    r"^(ein|tin|vat( no\.?| reg\.? no\.?)?|gstin|gst|ust-?idnr\.?|abn)[:\s]*", re.I
)


def match_tax_id(a: Any, b: Any) -> bool:
    """Tax ids, ignoring a leading "VAT"/"EIN"/"GSTIN" label the model may or may not copy."""
    return match_id(_TAX_ID_LABEL.sub("", str(a).strip()), _TAX_ID_LABEL.sub("", str(b).strip()))


def match_money(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) <= 0.01 + 1e-9
    except (TypeError, ValueError):
        return False


def match_number(a: Any, b: Any) -> bool:
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(x - y) <= max(1e-6, 0.001 * abs(y))


def match_date(a: Any, b: Any) -> bool:
    return normalize_date(a) == normalize_date(b)


def match_partial_date(a: Any, b: Any) -> bool:
    return normalize_partial_date(a) == normalize_partial_date(b)


def match_phone(a: Any, b: Any) -> bool:
    x, y = re.sub(r"\D", "", str(a)), re.sub(r"\D", "", str(b))
    return bool(x) and x[-10:] == y[-10:]


Matcher = Callable[[Any, Any], bool]

# Scalar fields scored per document type.
SCALARS: dict[DocType, dict[str, Matcher]] = {
    DocType.INVOICE: {
        "invoice_number": match_id,
        "vendor_name": match_text,
        "vendor_address": match_text,
        "vendor_tax_id": match_tax_id,
        "customer_name": match_text,
        "customer_address": match_text,
        "invoice_date": match_date,
        "due_date": match_date,
        "currency": match_id,
        "subtotal": match_money,
        "discount": match_money,
        "tax": match_money,
        "shipping": match_money,
        "total": match_money,
        "payment_terms": match_text,
    },
    DocType.BANK_STATEMENT: {
        "bank_name": match_text,
        "account_holder": match_text,
        "account_number_last4": match_id,
        "statement_period_start": match_date,
        "statement_period_end": match_date,
        "currency": match_id,
        "opening_balance": match_money,
        "closing_balance": match_money,
        "total_credits": match_money,
        "total_debits": match_money,
    },
    DocType.RESUME: {
        "full_name": match_text,
        "email": match_id,
        "phone": match_phone,
        "location": match_text,
    },
}

# List-of-object fields and how to compare their items.
LISTS: dict[DocType, dict[str, dict[str, Matcher]]] = {
    DocType.INVOICE: {
        "line_items": {
            "description": match_text,
            "quantity": match_number,
            "unit_price": match_money,
            "amount": match_money,
        }
    },
    DocType.BANK_STATEMENT: {
        "transactions": {
            "date": match_date,
            "description": match_text,
            "amount": match_money,
            "balance": match_money,
        }
    },
    DocType.RESUME: {
        "experience": {
            "company": match_text,
            "title": match_text,
            "start_date": match_partial_date,
            "end_date": match_partial_date,
        },
        "education": {
            "institution": match_text,
            "degree": match_text,
            "field_of_study": match_text,
            "graduation_date": match_partial_date,
        },
    },
}

# Lists of plain strings, scored as sets.
STRING_SETS: dict[DocType, tuple[str, ...]] = {DocType.RESUME: ("skills",)}


def _match(matcher: Matcher, predicted: Any, expected: Any) -> bool:
    if predicted is None or expected is None:
        return predicted is None and expected is None
    return matcher(predicted, expected)


@dataclass
class ListScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def exact(self) -> bool:
        return self.fp == 0 and self.fn == 0


@dataclass
class DocScore:
    doc_type_correct: bool
    fields: dict[str, bool] = field(default_factory=dict)
    lists: dict[str, ListScore] = field(default_factory=dict)

    @property
    def all_correct(self) -> bool:
        return self.doc_type_correct and all(self.fields.values())


def _score_items(
    predicted: list[dict[str, Any]], expected: list[dict[str, Any]], spec: dict[str, Matcher]
) -> ListScore:
    """Greedy one-to-one matching: each label item takes the best unused prediction."""
    remaining = list(range(len(predicted)))
    score = ListScore()
    for item in expected:
        best, best_hits = None, -1
        for idx in remaining:
            hits = sum(_match(m, predicted[idx].get(k), item.get(k)) for k, m in spec.items())
            if hits > best_hits:
                best, best_hits = idx, hits
        if best is not None and best_hits == len(spec):
            score.tp += 1
            remaining.remove(best)
        else:
            score.fn += 1
    score.fp = len(remaining)
    return score


def _score_strings(predicted: list[str], expected: list[str]) -> ListScore:
    pred = {_norm_text(s) for s in predicted or []}
    gold = {_norm_text(s) for s in expected or []}
    return ListScore(tp=len(pred & gold), fp=len(pred - gold), fn=len(gold - pred))


def score_document(
    label_type: DocType,
    expected: dict[str, Any],
    predicted_type: DocType | None,
    predicted: dict[str, Any] | None,
) -> DocScore:
    """Score one document. A wrong document type makes every field wrong."""
    predicted = predicted if predicted_type == label_type and predicted else {}
    result = DocScore(doc_type_correct=predicted_type == label_type)

    for name, matcher in SCALARS.get(label_type, {}).items():
        result.fields[name] = result.doc_type_correct and _match(
            matcher, predicted.get(name), expected.get(name)
        )
    for name, spec in LISTS.get(label_type, {}).items():
        ls = _score_items(predicted.get(name) or [], expected.get(name) or [], spec)
        result.lists[name] = ls
        result.fields[name] = result.doc_type_correct and ls.exact
    for name in STRING_SETS.get(label_type, ()):
        ls = _score_strings(predicted.get(name) or [], expected.get(name) or [])
        result.lists[name] = ls
        result.fields[name] = result.doc_type_correct and ls.exact
    return result
