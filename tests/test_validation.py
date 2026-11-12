from __future__ import annotations

from datetime import date

from docflow.config import Settings
from docflow.schemas import Extraction
from docflow.validation import Severity, validate

TODAY = date(2026, 11, 12)


def run(settings: Settings, response: dict, **kwargs):
    return validate(Extraction.model_validate(response), settings, today=TODAY, **kwargs)


def codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_clean_invoice_is_accepted(settings: Settings, invoice_response: dict) -> None:
    report = run(settings, invoice_response)
    assert report.errors == []
    assert not report.needs_review
    assert report.confidence == 0.97


def test_totals_mismatch_forces_review(settings: Settings, invoice_response: dict) -> None:
    invoice_response["invoice"]["total"] = 280.0
    report = run(settings, invoice_response)
    assert "totals_mismatch" in codes(report)
    assert report.needs_review
    assert "totals_mismatch" in report.review_reasons[0]


def test_line_items_must_add_up_to_subtotal(settings: Settings, invoice_response: dict) -> None:
    invoice_response["invoice"]["line_items"][1]["amount"] = 55.0
    report = run(settings, invoice_response)
    assert {"line_items_mismatch", "line_item_math"} <= codes(report)


def test_discount_and_shipping_are_part_of_the_total(
    settings: Settings, invoice_response: dict
) -> None:
    invoice_response["invoice"].update(discount=-25.0, shipping=10.0, total=255.0)
    report = run(settings, invoice_response)
    assert "totals_mismatch" not in codes(report)


def test_tax_inclusive_total_is_a_warning(settings: Settings, invoice_response: dict) -> None:
    invoice_response["invoice"]["total"] = 250.0
    report = run(settings, invoice_response)
    assert "tax_inclusive" in codes(report)
    assert "totals_mismatch" not in codes(report)


def test_missing_required_and_bad_dates(settings: Settings, invoice_response: dict) -> None:
    invoice_response["invoice"].update(invoice_number=None, invoice_date="31/31/2026")
    invoice_response["invoice"]["due_date"] = "2025-01-01"
    report = run(settings, invoice_response)
    found = codes(report)
    assert {"missing_required", "invalid_date"} <= found
    assert report.needs_review


def test_due_date_before_issue_date(settings: Settings, invoice_response: dict) -> None:
    invoice_response["invoice"]["due_date"] = "2026-02-01"
    report = run(settings, invoice_response)
    issue = next(i for i in report.issues if i.code == "due_before_issue")
    assert issue.severity == Severity.WARNING


def test_future_invoice_date_is_flagged(settings: Settings, invoice_response: dict) -> None:
    invoice_response["invoice"]["invoice_date"] = "2027-06-01"
    invoice_response["invoice"]["due_date"] = "2027-07-01"
    assert "future_date" in codes(run(settings, invoice_response))


def test_low_model_confidence_forces_review(settings: Settings, invoice_response: dict) -> None:
    invoice_response["field_confidence"][3]["confidence"] = 0.4
    report = run(settings, invoice_response)
    assert report.confidence == 0.4
    assert report.needs_review
    assert "low_confidence" in codes(report)


def test_poor_ocr_lowers_confidence(settings: Settings, invoice_response: dict) -> None:
    clean = run(settings, invoice_response, ocr_confidence=0.95)
    noisy = run(settings, invoice_response, ocr_confidence=0.6)
    assert clean.confidence == 0.97
    assert noisy.confidence < 0.7
    assert noisy.needs_review
    assert any("OCR" in reason for reason in noisy.review_reasons)


def test_unreported_confidence_uses_a_cautious_default(
    settings: Settings, invoice_response: dict
) -> None:
    invoice_response["field_confidence"] = []
    report = run(settings, invoice_response)
    assert report.confidence == 0.6
    assert report.needs_review


def test_clean_bank_statement(settings: Settings, bank_response: dict) -> None:
    report = run(settings, bank_response)
    assert report.issues == []
    assert not report.needs_review


def test_bank_balance_mismatch(settings: Settings, bank_response: dict) -> None:
    bank_response["bank_statement"]["closing_balance"] = 1400.0
    report = run(settings, bank_response)
    assert "balance_mismatch" in codes(report)
    assert report.needs_review


def test_bank_sign_error_is_caught(settings: Settings, bank_response: dict) -> None:
    # A debit read as a credit: balance, running balance and totals all disagree.
    bank_response["bank_statement"]["transactions"][0]["amount"] = 150.0
    found = codes(run(settings, bank_response))
    assert {"balance_mismatch", "running_balance_mismatch", "debits_mismatch"} <= found


def test_bank_period_and_transaction_dates(settings: Settings, bank_response: dict) -> None:
    statement = bank_response["bank_statement"]
    statement["statement_period_start"], statement["statement_period_end"] = (
        "2026-02-01",
        "2026-01-01",
    )
    statement["transactions"][1]["date"] = "2026-03-15"
    found = codes(run(settings, bank_response))
    assert {"period_inverted", "transaction_outside_period"} <= found


def test_bank_transaction_without_amount(settings: Settings, bank_response: dict) -> None:
    bank_response["bank_statement"]["transactions"][0]["amount"] = None
    assert "transaction_missing_amount" in codes(run(settings, bank_response))


def test_clean_resume(settings: Settings, resume_response: dict) -> None:
    report = run(settings, resume_response)
    assert report.errors == []
    assert not report.needs_review


def test_resume_checks(settings: Settings, resume_response: dict) -> None:
    cv = resume_response["resume"]
    cv.update(email="not-an-email", phone="12")
    cv["experience"][0].update(start_date="2024-05", end_date="2022-01")
    found = codes(run(settings, resume_response))
    assert {"invalid_email", "invalid_phone", "experience_dates_inverted"} <= found


def test_resume_without_name_needs_review(settings: Settings, resume_response: dict) -> None:
    resume_response["resume"]["full_name"] = None
    report = run(settings, resume_response)
    assert "missing_required" in codes(report)
    assert report.needs_review


def test_other_documents_always_go_to_review(settings: Settings) -> None:
    report = run(settings, {"document_type": "other"})
    assert "unsupported_document_type" in codes(report)
    assert report.confidence == 0.0
    assert report.needs_review
