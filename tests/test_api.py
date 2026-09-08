"""apps/api tests: HTTP-level, against a real ASGI TestClient (no running
server needed), pointed at an isolated temp SQLite DB so this suite never
touches data/store.db. Ingest itself is mocked (extract_blocks/
extract_facts/adjudicator.generate) -- real live behavior (upload, SSE
streaming, page-image rendering, all route responses) was verified
separately with a real running server against real PDFs."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from apps.api import deps
from apps.api.main import app
from src.fkl.extract.extractor import ExtractionResult
from src.fkl.reconcile.adjudicator import ExplanationResponse
from src.fkl.store.models import (
    Confidence,
    Entity,
    Evidence,
    Fact,
    Measure,
    Provenance,
    Relation,
    RelationType,
    ReasonCode,
    Value,
)
from src.fkl.store.repo import Repo


def make_fact(subject: str, measure: str, value_raw: str, qualifiers: dict | None = None, doc_id: str = "doc1") -> Fact:
    return Fact(
        id=str(uuid.uuid4()),
        doc_id=doc_id,
        block_id=f"{doc_id}:p1:b0",
        subject=Entity(canonical_id="x", surface_form=subject),
        measure=Measure(canonical_id="m", surface_form=measure),
        fact_type="numeric",
        value=Value(raw=value_raw),
        qualifiers=qualifiers or {},
        evidence=Evidence(page_no=1, char_start=0, char_end=len(value_raw), bbox=[(1.0, 2.0, 3.0, 4.0)], quote=value_raw),
        confidence=Confidence(extraction=0.9),
        provenance=Provenance(model="test", prompt_version="v1", extracted_at=datetime.now(UTC), parser="test"),
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    # deps.DB_PATH is a module attribute read at call time by BOTH
    # Depends(get_repo) routes and routes/documents.py's background
    # ingest thread (see deps.py's docstring) -- monkeypatching it here
    # is what makes the upload test below actually isolated, not just
    # the request-scoped routes.
    monkeypatch.setattr(deps, "DB_PATH", tmp_path / "test_store.db")
    with TestClient(app) as c:
        yield c


def _seeded_repo(tmp_path) -> Repo:
    return Repo(tmp_path / "test_store.db")


def test_empty_endpoints_return_empty_lists(client):
    assert client.get("/documents").json() == []
    assert client.get("/facts").json() == []
    assert client.get("/relations").json() == []
    assert client.get("/review").json() == []


def test_get_fact_404_for_unknown_id(client):
    resp = client.get("/facts/nonexistent")
    assert resp.status_code == 404


def test_get_relation_404_for_unknown_id(client):
    resp = client.get("/relations/nonexistent")
    assert resp.status_code == 404


def test_facts_and_relations_endpoints_against_seeded_data(client, tmp_path):
    repo = _seeded_repo(tmp_path)
    repo.add_document("doc1", "x.pdf", {"doc_id": "doc1", "publisher": "Delhivery Limited"})
    fact_a = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24"})
    fact_b = make_fact("Delhivery Limited", "Revenue from Operations", "200", {"period_label": "FY24"})
    repo.add_fact(fact_a)
    repo.add_fact(fact_b)
    relation = Relation(
        id=str(uuid.uuid4()), fact_a_id=fact_a.id, fact_b_id=fact_b.id,
        type=RelationType.CONTRADICTS, reason_code=ReasonCode.VALUES_DIVERGE,
        explanation="test explanation", confidence=0.9, adjudicator="rule",
    )
    repo.add_relation(relation)
    repo.close()

    docs = client.get("/documents").json()
    assert len(docs) == 1 and docs[0]["doc_id"] == "doc1"

    facts = client.get("/facts").json()
    assert len(facts) == 2

    facts_filtered = client.get("/facts", params={"subject": "delhivery"}).json()
    assert len(facts_filtered) == 2

    fact_resp = client.get(f"/facts/{fact_a.id}")
    assert fact_resp.status_code == 200
    assert fact_resp.json()["value"]["raw"] == "100"

    relations = client.get("/relations").json()
    assert len(relations) == 1

    relations_filtered = client.get("/relations", params={"type": "CORROBORATES"}).json()
    assert relations_filtered == []

    detail = client.get(f"/relations/{relation.id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["relation"]["type"] == "CONTRADICTS"
    # repo.py normalizes relation storage to sorted(fact_a_id, fact_b_id)
    # (documented in repo.py), so which one comes back as "fact_a" isn't
    # guaranteed to match the original reconcile() call order -- check the
    # pair as a set, and check each fact's own content is intact.
    assert {body["fact_a"]["id"], body["fact_b"]["id"]} == {fact_a.id, fact_b.id}
    quotes = {body["fact_a"]["evidence"]["quote"], body["fact_b"]["evidence"]["quote"]}
    assert quotes == {"100", "200"}


def test_review_endpoint_against_seeded_quarantine(client, tmp_path):
    repo = _seeded_repo(tmp_path)
    repo.add_document("doc1", "x.pdf", {"doc_id": "doc1"})
    bad_fact = make_fact("Delhivery Limited", "revenue", "garbled")
    repo.add_quarantine("doc1", bad_fact, "partial_ratio 40.0 below threshold 92", 40.0)
    repo.close()

    entries = client.get("/review").json()
    assert len(entries) == 1
    assert entries[0]["reason"] == "partial_ratio 40.0 below threshold 92"
    assert entries[0]["fact"]["evidence"]["quote"] == "garbled"


def test_page_image_404_for_unknown_document(client):
    resp = client.get("/documents/nonexistent/page/1.png")
    assert resp.status_code == 404


def test_upload_document_streams_sse_and_persists(client, tmp_path):
    # doc_id must match what the mocked derive_doc_context below returns —
    # add_fact enforces a real FK constraint against documents.doc_id.
    fact = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24"}, doc_id="uploaded")
    extraction = ExtractionResult(facts=[fact])

    with (
        patch("src.fkl.pipeline.extract_blocks", return_value=[]),
        patch("src.fkl.pipeline.extract_facts", return_value=extraction),
        patch("src.fkl.pipeline.derive_doc_context", return_value={"doc_id": "uploaded", "publisher": "Delhivery Limited"}),
    ):
        resp = client.post(
            "/documents",
            files={"file": ("uploaded.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "data:" in body
    assert '"stage": "started"' in body
    assert '"stage": "done"' in body

    repo = _seeded_repo(tmp_path)
    assert repo.has_document("uploaded")
    assert len(repo.facts_by_doc("uploaded")) == 1
    repo.close()


def test_upload_rejects_non_pdf(client):
    resp = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_quota_exhaustion_surfaces_once_as_a_clear_message(client):
    """Regression: ingest_document() already notifies an "error" stage
    itself before re-raising -- the route's own except used to ALSO push a
    second, identical error event for the same failure, so a user watching
    the upload log would see "Error: ..." twice for one quota-exhaustion
    event. Must appear exactly once, and as the exception's own clear
    message -- not a stack trace."""
    from src.fkl.extract.providers import AllKeysExhaustedError

    with patch(
        "src.fkl.pipeline.extract_blocks",
        side_effect=AllKeysExhaustedError("all 5 API key(s) exhausted their daily quota"),
    ):
        resp = client.post(
            "/documents",
            files={"file": ("quota_test.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )

    assert resp.status_code == 200
    body = resp.text
    assert body.count('"stage": "error"') == 1
    assert "all 5 API key(s) exhausted their daily quota" in body
    assert "Traceback" not in body
