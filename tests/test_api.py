from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docflow.api import create_app
from docflow.config import Settings
from docflow.pipeline import Pipeline

AUTH = {"X-API-Key": "test-key"}


@pytest.fixture
def client_for(settings: Settings, make_pipeline) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def factory(responses: list[Any] | None = None) -> TestClient:
        pipeline: Pipeline = make_pipeline(responses)
        client = TestClient(create_app(settings, pipeline))
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


def upload(client: TestClient, data: bytes, name: str = "invoice.pdf", **form: str):
    return client.post("/v1/documents", files={"file": (name, data)}, data=form, headers=AUTH)


def test_health_needs_no_key(client_for) -> None:
    body = client_for().get("/health").json()
    assert body["status"] == "ok" and body["database"] is True and body["provider"] == "fake"


def test_v1_routes_need_the_api_key(client_for) -> None:
    client = client_for()
    assert client.get("/v1/documents").status_code == 401
    assert client.get("/v1/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/documents", headers=AUTH).status_code == 200


def test_upload_then_duplicate(client_for, invoice_pdf, invoice_response) -> None:
    client = client_for([invoice_response])
    first = upload(client, invoice_pdf, source="webhook", doc_type="invoice")
    assert first.status_code == 201
    body = first.json()
    assert body["status"] == "accepted"
    assert body["source"] == "webhook"
    assert body["data"]["invoice_number"] == "INV-2026-0042"
    assert body["created_at"].endswith("Z") or "+00:00" in body["created_at"]

    second = upload(client, invoice_pdf)
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["id"] == body["id"]


@pytest.mark.parametrize(
    ("form", "status"),
    [({"source": "carrier-pigeon"}, 422), ({"doc_type": "receipt"}, 422)],
)
def test_upload_rejects_bad_form_values(client_for, invoice_pdf, form, status) -> None:
    assert upload(client_for(), invoice_pdf, **form).status_code == status


def test_upload_limits(client_for, settings: Settings) -> None:
    client = client_for()
    assert upload(client, b"").status_code == 400
    settings.max_upload_mb = 1
    assert upload(client, b"%PDF" + b"0" * (1024 * 1024 + 1)).status_code == 413


def test_list_filter_and_get(client_for, invoice_pdf, invoice_response) -> None:
    client = client_for([invoice_response])
    doc_id = upload(client, invoice_pdf).json()["id"]
    upload(client, b"", "empty.txt")  # 400, not stored

    page = client.get("/v1/documents", params={"status": "accepted"}, headers=AUTH).json()
    assert page["total"] == 1 and page["items"][0]["id"] == doc_id
    assert (
        client.get("/v1/documents", params={"status": "failed"}, headers=AUTH).json()["total"] == 0
    )

    detail = client.get(f"/v1/documents/{doc_id}", headers=AUTH).json()
    assert detail["field_confidence"]["total"] == 0.99
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/v1/documents/{missing}", headers=AUTH).status_code == 404


def test_review_flow(client_for, invoice_pdf, invoice_response) -> None:
    invoice_response["invoice"]["total"] = 280.0
    client = client_for([invoice_response])
    doc = upload(client, invoice_pdf).json()
    assert doc["status"] == "needs_review"

    bad = client.post(
        f"/v1/documents/{doc['id']}/review",
        json={"action": "approve", "corrected_data": {"total": "lots"}},
        headers=AUTH,
    )
    assert bad.status_code == 422

    good = client.post(
        f"/v1/documents/{doc['id']}/review",
        json={
            "action": "approve",
            "corrected_data": dict(invoice_response["invoice"], total=270.0),
            "reviewer": "sam",
            "note": "total typo on the invoice",
        },
        headers=AUTH,
    )
    assert good.status_code == 200
    assert good.json()["status"] == "approved"
    assert good.json()["corrected_data"]["total"] == 270.0


def test_cannot_review_a_failed_document(client_for) -> None:
    client = client_for()
    doc = upload(client, b"\x00\x01", "blob.bin").json()
    assert doc["status"] == "failed"
    response = client.post(
        f"/v1/documents/{doc['id']}/review", json={"action": "approve"}, headers=AUTH
    )
    assert response.status_code == 409


def test_reprocess_endpoint(client_for, invoice_pdf, invoice_response) -> None:
    client = client_for([invoice_response, invoice_response])
    doc = upload(client, invoice_pdf).json()
    again = client.post(f"/v1/documents/{doc['id']}/reprocess", headers=AUTH)
    assert again.status_code == 200 and again.json()["attempts"] == 2


def test_batch_run_and_lock(client_for, settings: Settings, invoice_pdf, invoice_response) -> None:
    client = client_for([invoice_response])
    (settings.inbox_dir / "a.pdf").write_bytes(invoice_pdf)
    response = client.post("/v1/batch/run", params={"wait": "true"}, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["summary"]["accepted"] == 1

    pipeline: Pipeline = client.app.state.pipeline
    pipeline.batch_lock.acquire()
    try:
        assert client.post("/v1/batch/run", headers=AUTH).status_code == 409
    finally:
        pipeline.batch_lock.release()

    started = client.post("/v1/batch/run", headers=AUTH)
    assert started.status_code == 202
    assert not pipeline.batch_lock.locked()  # the background run finished and released it


def test_summary_endpoint(client_for, invoice_pdf, invoice_response) -> None:
    client = client_for([invoice_response])
    upload(client, invoice_pdf)
    report = client.get("/v1/reports/summary", params={"hours": 2}, headers=AUTH).json()
    assert report["documents"] == 1 and report["by_type"] == {"invoice": 1}
