"""Command-line entry point: ``docflow <command>``."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from docflow.config import Settings


def _settings(**overrides: Any) -> Settings:
    return Settings(**{k: v for k, v in overrides.items() if v is not None})


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "docflow.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
    return 0


def cmd_init_db(args: argparse.Namespace) -> int:
    from docflow.db import init_db, make_engine

    init_db(make_engine(_settings().database_url))
    print("tables are ready")
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    from docflow.pipeline import build_pipeline
    from docflow.schemas import DocType

    pipeline = build_pipeline(_settings())
    hint = DocType(args.type) if args.type else None
    failed = False
    for name in args.files:
        path = Path(name)
        result = pipeline.process_bytes(
            path.read_bytes(), path.name, source="batch", hint=hint, force=args.force
        )
        doc = result.document
        failed |= doc.status == "failed"
        record = {
            "file": str(path),
            "id": str(doc.id),
            "status": doc.status,
            "doc_type": doc.doc_type,
            "confidence": doc.confidence,
            "review_reasons": doc.review_reasons,
            "error": doc.error,
            "duplicate": result.duplicate,
            "cost_usd": doc.cost_usd,
        }
        if args.show_data:
            record["data"] = doc.effective_data
        print(json.dumps(record, default=str, indent=2 if args.show_data else None))
    return 1 if failed else 0


def cmd_batch(args: argparse.Namespace) -> int:
    from docflow.pipeline import build_pipeline

    summary = build_pipeline(_settings()).run_batch()
    print(json.dumps(asdict(summary), indent=2))
    return 1 if args.fail_on_error and summary.failed else 0


def cmd_watch(args: argparse.Namespace) -> int:
    from docflow.pipeline import build_pipeline

    with contextlib.suppress(KeyboardInterrupt):
        build_pipeline(_settings()).watch()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from docflow.evaluation import report, runner
    from docflow.extract import build_extractor

    overrides: dict[str, Any] = {"llm_provider": args.provider}
    if args.model:
        overrides["gemini_model" if args.provider == "gemini" else "claude_model"] = args.model
    if args.effort:
        overrides["claude_effort"] = args.effort
    settings = _settings(**overrides)

    labels = runner.load_labels(Path(args.dataset), verified_only=args.verified_only)
    if args.type:
        labels = [label for label in labels if label.document_type.value == args.type]
    if args.limit:
        labels = labels[: args.limit]
    if not labels:
        print(f"no labels found under {args.dataset}/labels", file=sys.stderr)
        return 2

    extractor = build_extractor(settings)
    results = runner.run(labels, extractor, settings, workers=args.workers)
    metrics = report.summarize(
        results,
        provider=settings.llm_provider,
        model=extractor.model,
        effort=settings.claude_effort if settings.llm_provider == "claude" else None,
    )
    sources = Counter(label.meta.get("source", "hand") for label in labels)
    note = " (" + ", ".join(f"{count} {source}" for source, count in sorted(sources.items())) + ")"
    markdown = report.render_markdown(metrics, dataset_note=note)
    json_path, md_path = report.write_outputs(metrics, markdown, Path(args.out))
    print(markdown)
    print(f"wrote {json_path} and {md_path}", file=sys.stderr)
    if args.update_readme:
        if report.update_readme(Path(args.update_readme), markdown):
            print(f"updated the results section of {args.update_readme}", file=sys.stderr)
        else:
            print(f"{args.update_readme} has no EVAL markers; not updated", file=sys.stderr)
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    """Pre-fill labels with the model so a person corrects them instead of typing them."""
    from docflow.extract import build_extractor
    from docflow.ingest import ingest

    settings = _settings()
    extractor = build_extractor(settings)
    dataset = Path(args.dataset)
    (dataset / "docs").mkdir(parents=True, exist_ok=True)
    (dataset / "labels").mkdir(parents=True, exist_ok=True)
    for name in args.files:
        source = Path(name)
        target = dataset / "docs" / source.name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        label_path = dataset / "labels" / f"{target.stem}.json"
        if label_path.exists() and not args.overwrite:
            print(f"skip {label_path} (exists; pass --overwrite)")
            continue
        ingested = ingest(target.read_bytes(), target.name, settings)
        outcome = extractor.extract(ingested.text)
        section = outcome.extraction.section()
        label = {
            "file": f"docs/{target.name}",
            "document_type": outcome.extraction.document_type.value,
            "fields": section.model_dump(mode="json") if section is not None else {},
            "meta": {
                "source": "hand",
                "verified": False,
                "drafted_by": outcome.model,
                "scanned": ingested.method != "text",
            },
        }
        label_path.write_text(json.dumps(label, indent=2, ensure_ascii=False) + "\n")
        print(
            f"draft: {label_path}  -> check every field against the document, then set "
            "meta.verified to true"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docflow", description="DocFlow document intake")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("init-db", help="create database tables")
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("process", help="process files now and print the results")
    p.add_argument("files", nargs="+")
    p.add_argument("--type", choices=["invoice", "bank_statement", "resume"])
    p.add_argument("--force", action="store_true", help="re-extract files seen before")
    p.add_argument("--show-data", action="store_true", help="print the extracted fields")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("batch", help="process the inbox, retry failures, export to Sheets")
    p.add_argument("--fail-on-error", action="store_true", help="exit 1 if any document failed")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("watch", help="watch the inbox folder and process new files")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("eval", help="measure accuracy, latency and cost on a labelled set")
    p.add_argument("--dataset", default="eval/dataset")
    p.add_argument("--provider", choices=["claude", "gemini", "fake"], default="claude")
    p.add_argument("--model", help="override the model for this run")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--type", choices=["invoice", "bank_statement", "resume"])
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--verified-only", action="store_true", help="skip unverified draft labels")
    p.add_argument("--out", default="eval/results")
    p.add_argument("--update-readme", metavar="PATH", help="rewrite the results section of PATH")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("label", help="draft labels for new documents, for you to correct")
    p.add_argument("files", nargs="+")
    p.add_argument("--dataset", default="eval/dataset")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_label)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    for noisy in ("pdfminer", "httpx", "httpcore", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
