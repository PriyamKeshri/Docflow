from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import pytest

from docflow.cli import main
from docflow.config import Settings
from docflow.evaluation import report, runner
from docflow.evaluation.scoring import (
    match_phone,
    match_tax_id,
    match_text,
    score_document,
)
from docflow.schemas import REQUIRED_FIELDS, SECTION_ATTR, SECTION_MODELS, DocType, Extraction
from docflow.validation import validate
from tests.conftest import INVOICE, RESUME

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval" / "dataset"


# ------------------------------------------------------------- scoring


def test_perfect_prediction_scores_100() -> None:
    fields = INVOICE["invoice"]
    score = score_document(DocType.INVOICE, fields, DocType.INVOICE, copy.deepcopy(fields))
    assert score.all_correct
    assert score.lists["line_items"].tp == 2


def test_wrong_document_type_makes_every_field_wrong() -> None:
    fields = INVOICE["invoice"]
    score = score_document(DocType.INVOICE, fields, DocType.RESUME, {"full_name": "x"})
    assert not score.doc_type_correct
    assert not any(score.fields.values())


def test_field_level_differences() -> None:
    expected = INVOICE["invoice"]
    predicted = copy.deepcopy(expected)
    predicted["total"] = 270.004  # within a cent: right
    predicted["vendor_name"] = "Brightline Office Supply Co"  # punctuation only: right
    predicted["invoice_date"] = "2026-02-03"  # day/month swapped: wrong
    predicted["discount"] = 0.0  # label says not printed (None): wrong
    predicted["line_items"] = list(reversed(predicted["line_items"]))  # order doesn't matter
    score = score_document(DocType.INVOICE, expected, DocType.INVOICE, predicted)
    assert score.fields["total"] and score.fields["vendor_name"] and score.fields["line_items"]
    assert not score.fields["invoice_date"] and not score.fields["discount"]


def test_list_items_extra_and_missing() -> None:
    expected = RESUME["resume"]
    predicted = copy.deepcopy(expected)
    predicted["skills"] = ["python", "Go"]  # case-insensitive hit, one extra, one missing
    predicted["experience"].append({"company": "Ghost Inc", "title": "CEO"})
    score = score_document(DocType.RESUME, expected, DocType.RESUME, predicted)
    skills = score.lists["skills"]
    assert (skills.tp, skills.fp, skills.fn) == (1, 1, 1)
    assert score.lists["experience"].fp == 1
    assert not score.fields["experience"]


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("EIN 47-3815562", "47-3815562", True),
        ("GB 284 6153 07", "VAT GB284615307", True),
        ("DE298765432", "DE298765433", False),
    ],
)
def test_tax_id_matching(a: str, b: str, same: bool) -> None:
    assert match_tax_id(a, b) is same


def test_phone_and_text_matching() -> None:
    assert match_phone("+1 (555) 014-3910", "555.014.3910")
    assert match_text("12 Harbor Way, Suite 4", "12 Harbor Way Suite 4")
    assert not match_text("Acme Ltd", "Apex Ltd")


# -------------------------------------------------------------- report


def test_summarize_and_render(tmp_path: Path) -> None:
    label = runner.Label(
        "inv-x", tmp_path / "x.pdf", DocType.INVOICE, INVOICE["invoice"], {"scanned": False}
    )
    good = runner.DocResult(
        label=label,
        predicted_type="invoice",
        ingest_method="text",
        ingest_ms=50,
        extract_ms=1500,
        cost_usd=0.01,
        needs_review=False,
        score=score_document(DocType.INVOICE, label.fields, DocType.INVOICE, label.fields),
    )
    bad_fields = dict(label.fields, total=1.0)
    bad = runner.DocResult(
        label=label,
        predicted_type="invoice",
        ingest_method="ocr",
        ingest_ms=900,
        extract_ms=2500,
        cost_usd=0.02,
        needs_review=True,
        score=score_document(DocType.INVOICE, label.fields, DocType.INVOICE, bad_fields),
    )
    metrics = report.summarize(
        [good, bad], provider="claude", model="claude-opus-5-5", effort="medium"
    )
    assert metrics["docs_fully_correct"] == 50.0
    assert metrics["review"]["flagged_when_wrong"] == 100.0
    assert metrics["review"]["flagged_when_right"] == 0.0
    assert metrics["cost_usd"]["per_doc_mean"] == pytest.approx(0.015)

    markdown = report.render_markdown(metrics)
    assert "Field-level accuracy" in markdown and "invoice.total" in markdown

    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.README_START}\nold\n{report.README_END}\noutro\n")
    assert report.update_readme(readme, markdown)
    text = readme.read_text()
    assert "old" not in text and "Field-level accuracy" in text and text.endswith("outro\n")


# ------------------------------------------------------------- dataset


def test_dataset_labels_are_complete_and_valid() -> None:
    labels = runner.load_labels(DATASET)
    assert 30 <= len(labels) <= 50
    assert {label.document_type for label in labels} == {
        DocType.INVOICE,
        DocType.BANK_STATEMENT,
        DocType.RESUME,
    }
    settings = Settings(_env_file=None)
    for label in labels:
        assert label.path.exists(), label.path
        SECTION_MODELS[label.document_type].model_validate(label.fields)
        # A perfect extraction of a clean document must pass validation;
        # the documents with a deliberate error on them must be flagged.
        extraction = Extraction.model_validate(
            {
                "document_type": label.document_type.value,
                SECTION_ATTR[label.document_type]: label.fields,
                "field_confidence": [
                    {"field": f, "confidence": 1.0} for f in REQUIRED_FIELDS[label.document_type]
                ],
            }
        )
        flagged = validate(extraction, settings, today=date(2026, 11, 12)).needs_review
        assert flagged == bool(label.meta.get("anomaly")), label.id


def test_eval_command_runs_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "results"
    code = main(
        [
            "eval",
            "--provider",
            "fake",
            "--dataset",
            str(DATASET),
            "--type",
            "invoice",
            "--limit",
            "2",
            "--workers",
            "1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    metrics = json.loads((out / "latest.json").read_text())
    assert metrics["documents"] == 2
    assert metrics["doc_type_accuracy"] == 0.0  # the fake provider answers "other"
    assert "Field-level accuracy" in capsys.readouterr().out
