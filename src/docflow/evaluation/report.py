"""Aggregate eval results into metrics, a Markdown report, and the README section."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docflow.evaluation.runner import DocResult
from docflow.evaluation.scoring import ListScore

README_START = "<!-- EVAL:START -->"
README_END = "<!-- EVAL:END -->"


def _pct(numerator: float, denominator: float) -> float | None:
    return round(100 * numerator / denominator, 1) if denominator else None


def _percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _f1(ls: ListScore) -> float | None:
    if ls.tp + ls.fp + ls.fn == 0:
        return None
    precision = ls.tp / (ls.tp + ls.fp) if ls.tp + ls.fp else 0.0
    recall = ls.tp / (ls.tp + ls.fn) if ls.tp + ls.fn else 0.0
    return (
        round(100 * 2 * precision * recall / (precision + recall), 1) if precision + recall else 0.0
    )


def summarize(
    results: list[DocResult], *, provider: str, model: str, effort: str | None
) -> dict[str, Any]:
    n = len(results)
    field_hits = field_total = 0
    per_field: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_type: dict[str, dict[str, Any]] = defaultdict(lambda: {"docs": 0, "hits": 0, "total": 0})
    per_method: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    list_scores: dict[str, ListScore] = defaultdict(ListScore)

    for r in results:
        doc_type = r.label.document_type.value
        per_type[doc_type]["docs"] += 1
        method = r.ingest_method or "error"
        for name, ok in r.score.fields.items() if r.score else []:
            key = f"{doc_type}.{name}"
            per_field[key][0] += int(ok)
            per_field[key][1] += 1
            per_type[doc_type]["hits"] += int(ok)
            per_type[doc_type]["total"] += 1
            per_method[method][0] += int(ok)
            per_method[method][1] += 1
            field_hits += int(ok)
            field_total += 1
        for name, ls in r.score.lists.items() if r.score else []:
            agg = list_scores[f"{doc_type}.{name}"]
            agg.tp, agg.fp, agg.fn = agg.tp + ls.tp, agg.fp + ls.fp, agg.fn + ls.fn

    latencies = [r.total_ms for r in results if r.error is None]
    extract_latencies = [r.extract_ms for r in results if r.error is None]
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    wrong = [r for r in results if not (r.score and r.score.all_correct)]
    right = [r for r in results if r.score and r.score.all_correct]

    return {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": provider,
        "model": model,
        "effort": effort,
        "documents": n,
        "errors": sum(r.error is not None for r in results),
        "doc_type_accuracy": _pct(
            sum(bool(r.score and r.score.doc_type_correct) for r in results), n
        ),
        "field_accuracy": _pct(field_hits, field_total),
        "fields_scored": field_total,
        "docs_fully_correct": _pct(len(right), n),
        "by_type": {
            t: {"docs": v["docs"], "field_accuracy": _pct(v["hits"], v["total"])}
            for t, v in sorted(per_type.items())
        },
        "by_ingest_method": {
            m: {"fields": v[1], "field_accuracy": _pct(v[0], v[1])}
            for m, v in sorted(per_method.items())
        },
        "fields": {
            k: {"accuracy": _pct(v[0], v[1]), "n": v[1]} for k, v in sorted(per_field.items())
        },
        "list_item_f1": {k: _f1(v) for k, v in sorted(list_scores.items())},
        "latency_ms": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "mean_extract": round(statistics.mean(extract_latencies))
            if extract_latencies
            else None,
        },
        "cost_usd": {
            "per_doc_mean": round(statistics.mean(costs), 5) if costs else None,
            "total": round(sum(costs), 4) if costs else None,
        },
        "tokens_per_doc": {
            "input": round(statistics.mean(r.input_tokens for r in results)) if n else None,
            "output": round(statistics.mean(r.output_tokens for r in results)) if n else None,
        },
        "review": {
            "flag_rate": _pct(sum(r.needs_review for r in results), n),
            # Of the documents with at least one wrong field, how many did we flag?
            "flagged_when_wrong": _pct(sum(r.needs_review for r in wrong), len(wrong)),
            # Of the fully correct documents, how many did we flag anyway?
            "flagged_when_right": _pct(sum(r.needs_review for r in right), len(right)),
        },
        "documents_detail": [
            {
                "id": r.label.id,
                "type": r.label.document_type.value,
                "scanned": bool(r.label.meta.get("scanned")),
                "ingest": r.ingest_method,
                "ocr_confidence": r.ocr_confidence,
                "all_correct": bool(r.score and r.score.all_correct),
                "wrong_fields": [
                    k for k, ok in (r.score.fields.items() if r.score else []) if not ok
                ],
                "needs_review": r.needs_review,
                "review_reasons": r.review_reasons,
                "latency_ms": r.total_ms,
                "cost_usd": r.cost_usd,
                "error": r.error,
            }
            for r in results
        ],
    }


def _fmt(value: Any, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value}{suffix}"


def render_markdown(metrics: dict[str, Any], *, dataset_note: str = "") -> str:
    lat, cost, review = metrics["latency_ms"], metrics["cost_usd"], metrics["review"]
    per_doc = cost["per_doc_mean"]
    lines = [
        f"Run {metrics['run_at']} · `{metrics['provider']}` / `{metrics['model']}`"
        + (f" (effort `{metrics['effort']}`)" if metrics.get("effort") else "")
        + f" · {metrics['documents']} documents{dataset_note}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Field-level accuracy | **{_fmt(metrics['field_accuracy'], '%')}** "
        f"({metrics['fields_scored']} fields) |",
        f"| Document type accuracy | {_fmt(metrics['doc_type_accuracy'], '%')} |",
        f"| Documents with every field right | {_fmt(metrics['docs_fully_correct'], '%')} |",
        f"| Latency p50 / p95 (ingest + extract) | {_fmt(lat['p50'], ' ms')} / "
        f"{_fmt(lat['p95'], ' ms')} |",
        f"| Cost per document (mean) | {'n/a' if per_doc is None else f'${per_doc:.4f}'} |",
        f"| Tokens per document (in / out) | {_fmt(metrics['tokens_per_doc']['input'])} / "
        f"{_fmt(metrics['tokens_per_doc']['output'])} |",
        f"| Flagged for review | {_fmt(review['flag_rate'], '%')} |",
        f"| Flagged when a field was wrong | {_fmt(review['flagged_when_wrong'], '%')} |",
        f"| Flagged when everything was right | {_fmt(review['flagged_when_right'], '%')} |",
        f"| Pipeline errors | {metrics['errors']} |",
        "",
        "| Document type | Docs | Field accuracy |",
        "| --- | --- | --- |",
    ]
    for doc_type, v in metrics["by_type"].items():
        lines.append(f"| {doc_type} | {v['docs']} | {_fmt(v['field_accuracy'], '%')} |")
    lines += ["", "| Ingest path | Fields | Field accuracy |", "| --- | --- | --- |"]
    for method, v in metrics["by_ingest_method"].items():
        lines.append(f"| {method} | {v['fields']} | {_fmt(v['field_accuracy'], '%')} |")
    weakest = sorted(
        ((k, v) for k, v in metrics["fields"].items() if v["accuracy"] is not None),
        key=lambda kv: kv[1]["accuracy"],
    )[:8]
    if weakest:
        lines += ["", "Weakest fields:", ""]
        lines += [f"- `{k}`: {v['accuracy']}% (n={v['n']})" for k, v in weakest]
    if metrics["list_item_f1"]:
        lines += [
            "",
            "List items, F1: "
            + ", ".join(f"`{k}` {_fmt(v, '%')}" for k, v in metrics["list_item_f1"].items()),
        ]
    return "\n".join(lines) + "\n"


def write_outputs(metrics: dict[str, Any], markdown: str, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "latest.json", out_dir / "latest.md"
    json_path.write_text(json.dumps(metrics, indent=2) + "\n")
    md_path.write_text(markdown)
    runs = out_dir / "runs"
    runs.mkdir(exist_ok=True)
    stamp = metrics["run_at"].replace(":", "").replace("-", "")
    (runs / f"{stamp}-{metrics['model']}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return json_path, md_path


def update_readme(readme: Path, markdown: str) -> bool:
    text = readme.read_text()
    if README_START not in text or README_END not in text:
        return False
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    readme.write_text(f"{head}{README_START}\n{markdown}{README_END}{tail}")
    return True
