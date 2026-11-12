from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from docflow.schemas import (
    BankStatement,
    DocType,
    Extraction,
    llm_json_schema,
    normalize_date,
    normalize_partial_date,
    parse_amount,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-03-02", "2026-03-02"),
        ("March 2, 2026", "2026-03-02"),
        ("2 Mar 2026", "2026-03-02"),
        ("02.03.2026", "2026-03-02"),
        ("2026/03/02", "2026-03-02"),
        ("", None),
        (None, None),
        ("sometime soon", "sometime soon"),  # left for validation to flag
    ],
)
def test_normalize_date(raw: Any, expected: str | None) -> None:
    assert normalize_date(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Present", "present"),
        ("current", "present"),
        ("2021", "2021"),
        ("2021-04", "2021-04"),
        ("2021-04-17", "2021-04"),
        ("Apr 2021", "2021-04"),
        ("04/2021", "2021-04"),
        ("April 2021", "2021-04"),
    ],
)
def test_normalize_partial_date(raw: str, expected: str) -> None:
    assert normalize_partial_date(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (12, 12.0),
        ("1,234.50", 1234.5),
        ("$1,234.50", 1234.5),
        ("1.234,56", 1234.56),
        ("1.234,56 €", 1234.56),
        ("12,5", 12.5),
        ("1,234", 1234.0),
        ("(12.00)", -12.0),
        ("12.00 DR", -12.0),
        ("-7.25", -7.25),
        ("Rs. 99", 99.0),
        ("", None),
        (None, None),
    ],
)
def test_parse_amount(raw: Any, expected: float | None) -> None:
    assert parse_amount(raw) == expected


def test_parse_amount_rejects_text() -> None:
    with pytest.raises(ValueError):
        parse_amount("n/a")


def test_account_number_is_masked_to_last_four() -> None:
    assert (
        BankStatement(account_number_last4="DE89 3704 0044 0532 0130 00").account_number_last4
        == "3000"
    )
    assert BankStatement(account_number_last4="****4821").account_number_last4 == "4821"


def test_extraction_drops_sections_for_other_types(invoice_response: dict) -> None:
    invoice_response["resume"] = {"full_name": "Stray Name"}
    extraction = Extraction.model_validate(invoice_response)
    assert extraction.resume is None
    assert extraction.section() is extraction.invoice


def test_extraction_normalises_values(invoice_response: dict) -> None:
    invoice_response["invoice"]["total"] = "$270.00"
    invoice_response["invoice"]["invoice_date"] = "March 2, 2026"
    invoice_response["invoice"]["currency"] = " usd "
    invoice_response["field_confidence"].append({"field": "invoice.total", "confidence": 1.7})
    extraction = Extraction.model_validate(invoice_response)
    assert extraction.invoice.total == 270.0
    assert extraction.invoice.invoice_date == "2026-03-02"
    assert extraction.invoice.currency == "USD"
    assert extraction.confidence_map()["total"] == 1.0  # clamped, prefix stripped


def test_extraction_rejects_unknown_fields(invoice_response: dict) -> None:
    invoice_response["invoice"]["surprise"] = 1
    with pytest.raises(ValidationError):
        Extraction.model_validate(invoice_response)


def test_extraction_rejects_unknown_document_type(invoice_response: dict) -> None:
    invoice_response["document_type"] = "receipt"
    with pytest.raises(ValidationError):
        Extraction.model_validate(invoice_response)


def _walk(node: Any, visit) -> None:
    if isinstance(node, dict):
        visit(node)
        for key, value in node.items():
            if key == "properties":
                for sub in value.values():
                    _walk(sub, visit)
            else:
                _walk(value, visit)
    elif isinstance(node, list):
        for item in node:
            _walk(item, visit)


def test_llm_schema_is_portable() -> None:
    schema = llm_json_schema()
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text

    objects: list[dict] = []

    def visit(node: dict) -> None:
        assert "default" not in node and "minimum" not in node
        if node.get("type") == "object":
            objects.append(node)

    _walk(schema, visit)
    assert objects
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert obj["required"] == list(obj["properties"])

    assert schema["properties"]["document_type"]["enum"] == [t.value for t in DocType]
    # A *field* called "title" must survive even though the "title" keyword is dropped.
    resume = schema["properties"]["resume"]["anyOf"][0]
    experience = resume["properties"]["experience"]["items"]
    assert "title" in experience["properties"]
