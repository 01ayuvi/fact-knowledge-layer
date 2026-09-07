"""pipeline.py tests: orchestration only, mocked at the LLM boundary
(extract_blocks, derive_doc_context, extract_facts, and
reconcile/adjudicator.generate) so this suite is fast and deterministic.
Real live end-to-end behavior was verified separately against actual PDFs
and real Groq/Gemini calls, including the incremental-skip path and
cross-document reconciliation firing correctly."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from src.fkl.extract.extractor import ExtractionResult
from src.fkl.pipeline import ingest_document
from src.fkl.reconcile.adjudicator import ExplanationResponse
from src.fkl.store.models import (
    Confidence,
    Entity,
    Evidence,
    Fact,
    Measure,
    Provenance,
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
        evidence=Evidence(page_no=1, char_start=0, char_end=len(value_raw), bbox=[(0, 0, 1, 1)], quote=value_raw),
        confidence=Confidence(extraction=0.9),
        provenance=Provenance(model="test", prompt_version="v1", extracted_at=datetime.now(UTC), parser="test"),
    )


@pytest.fixture
def repo():
    r = Repo(":memory:")
    yield r
    r.close()


@pytest.fixture(autouse=True)
def mock_llm_layer():
    """Never let write_explanation/adjudicate hit a real provider in this
    suite -- return the input relation's own explanation-shaped stand-in."""
    with patch(
        "src.fkl.reconcile.adjudicator.generate",
        return_value=(ExplanationResponse(explanation="mocked explanation"), "test-model"),
    ):
        yield


def test_derive_doc_context_called_only_when_not_supplied(repo):
    fact = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24"})
    extraction = ExtractionResult(facts=[fact])

    with (
        patch("src.fkl.pipeline.extract_blocks", return_value=[]),
        patch("src.fkl.pipeline.extract_facts", return_value=extraction),
        patch("src.fkl.pipeline.derive_doc_context") as mock_derive,
    ):
        ingest_document("x.pdf", doc_context={"doc_id": "doc1", "publisher": "Delhivery Limited"}, repo=repo)
        assert mock_derive.call_count == 0

        ingest_document("y.pdf", doc_context=None, repo=repo)
        assert mock_derive.call_count == 1


def test_ingest_persists_facts_and_returns_counts(repo):
    facts = [
        make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24"}),
        make_fact("Delhivery Limited", "PAT", "50", {"period_label": "FY24"}),
    ]
    extraction = ExtractionResult(facts=facts)

    with (
        patch("src.fkl.pipeline.extract_blocks", return_value=[]),
        patch("src.fkl.pipeline.extract_facts", return_value=extraction),
    ):
        result = ingest_document("x.pdf", doc_context={"doc_id": "doc1"}, repo=repo)

    assert result.doc_id == "doc1"
    assert result.facts == 2
    assert result.quarantined == 0
    assert len(repo.facts_by_doc("doc1")) == 2


def test_ingest_persists_quarantined_facts(repo):
    from src.fkl.extract.grounding_gate import QuarantineEntry

    bad_fact = make_fact("Delhivery Limited", "revenue", "garbled")
    extraction = ExtractionResult(facts=[], quarantined=[QuarantineEntry(fact=bad_fact, reason="low score", score=40.0)])

    with (
        patch("src.fkl.pipeline.extract_blocks", return_value=[]),
        patch("src.fkl.pipeline.extract_facts", return_value=extraction),
    ):
        result = ingest_document("x.pdf", doc_context={"doc_id": "doc1"}, repo=repo)

    assert result.quarantined == 1
    assert repo.get_fact(bad_fact.id) is None
    assert len(repo.quarantined_by_doc("doc1")) == 1


def test_incremental_ingest_skips_already_ingested_document(repo):
    fact = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24"})
    extraction = ExtractionResult(facts=[fact])

    with (
        patch("src.fkl.pipeline.extract_blocks", return_value=[]) as mock_blocks,
        patch("src.fkl.pipeline.extract_facts", return_value=extraction) as mock_extract,
    ):
        ingest_document("x.pdf", doc_context={"doc_id": "doc1"}, repo=repo)
        assert mock_blocks.call_count == 1 and mock_extract.call_count == 1

        result2 = ingest_document("x.pdf", doc_context={"doc_id": "doc1"}, repo=repo)
        # must NOT re-extract -- this is the point of incremental ingest
        assert mock_blocks.call_count == 1
        assert mock_extract.call_count == 1
        assert result2.facts == 1


def test_reingest_does_not_touch_existing_facts_or_relations(repo):
    """The literal requirement: adding a document must not touch existing
    facts or relations."""
    fact_a = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24", "consolidation": "consolidated"}, doc_id="doc1")
    with (
        patch("src.fkl.pipeline.extract_blocks", return_value=[]),
        patch("src.fkl.pipeline.extract_facts", return_value=ExtractionResult(facts=[fact_a])),
    ):
        ingest_document("a.pdf", doc_context={"doc_id": "doc1"}, repo=repo)

    stored_before = repo.get_fact(fact_a.id)

    fact_b = make_fact("Delhivery Limited", "Revenue from Operations", "101", {"period_label": "FY24", "consolidation": "standalone"}, doc_id="doc2")
    with (
        patch("src.fkl.pipeline.extract_blocks", return_value=[]),
        patch("src.fkl.pipeline.extract_facts", return_value=ExtractionResult(facts=[fact_b])),
    ):
        result2 = ingest_document("b.pdf", doc_context={"doc_id": "doc2"}, repo=repo)

    stored_after = repo.get_fact(fact_a.id)
    assert stored_before == stored_after  # doc1's fact is byte-identical, untouched
    assert result2.relations_by_type["RECONCILED_BY_CONTEXT"] == 1  # SCOPE_MISMATCH, consolidation differs


def test_only_new_candidate_pairs_get_reconciled(repo):
    """A second call for a document already ingested must not re-run
    reconciliation for pairs already related in a prior ingest."""
    fact_a = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24"}, doc_id="doc1")
    fact_b = make_fact("Delhivery Limited", "Revenue from Operations", "101", {"period_label": "FY24"}, doc_id="doc2")

    with (
        patch("src.fkl.pipeline.extract_blocks", return_value=[]),
        patch("src.fkl.pipeline.extract_facts", return_value=ExtractionResult(facts=[fact_a])),
    ):
        ingest_document("a.pdf", doc_context={"doc_id": "doc1"}, repo=repo)

    with (
        patch("src.fkl.pipeline.extract_blocks", return_value=[]),
        patch("src.fkl.pipeline.extract_facts", return_value=ExtractionResult(facts=[fact_b])),
        patch("src.fkl.reconcile.adjudicator.generate") as mock_generate,
    ):
        mock_generate.return_value = (ExplanationResponse(explanation="x"), "test-model")
        ingest_document("b.pdf", doc_context={"doc_id": "doc2"}, repo=repo)
        first_call_count = mock_generate.call_count
        assert first_call_count >= 1

        # Re-ingesting doc2 again: already ingested, must short-circuit
        # entirely (see test_incremental_ingest_skips_already_ingested_document)
        # -- no new reconciliation, no new LLM calls.
        ingest_document("b.pdf", doc_context={"doc_id": "doc2"}, repo=repo)
        assert mock_generate.call_count == first_call_count


def test_explanation_writing_runs_concurrently_bounded_at_four(repo):
    """Real proof of parallelism, not just that results come back right:
    5 mutually-matching facts -> 10 independent pairs. A slow (0.3s)
    mocked call, run fully sequentially, would take ~3s; run with real
    concurrency capped at 4, it should take roughly ceil(10/4)*0.3 =~ 0.9s.
    Also tracks the actual concurrent-call high-water mark directly, to
    confirm it never exceeds MAX_EXPLANATION_WORKERS."""
    import threading
    import time

    facts = [
        make_fact("Delhivery Limited", "Revenue from Operations", str(100 + i), {"period_label": f"FY2{i}"})
        for i in range(5)
    ]
    extraction = ExtractionResult(facts=facts)

    call_delay = 0.3
    in_flight = {"count": 0, "max": 0}
    lock = threading.Lock()

    def slow_generate(*args, **kwargs):
        with lock:
            in_flight["count"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["count"])
        time.sleep(call_delay)
        with lock:
            in_flight["count"] -= 1
        return (ExplanationResponse(explanation="x"), "test-model")

    with (
        patch("src.fkl.pipeline.extract_blocks", return_value=[]),
        patch("src.fkl.pipeline.extract_facts", return_value=extraction),
        patch("src.fkl.reconcile.adjudicator.generate", side_effect=slow_generate),
    ):
        t0 = time.time()
        result = ingest_document("x.pdf", doc_context={"doc_id": "doc1"}, repo=repo)
        elapsed = time.time() - t0

    n_pairs = 5 * 4 // 2  # C(5,2)
    assert sum(result.relations_by_type.values()) == n_pairs
    assert len(repo.relations_by_doc("doc1")) == n_pairs
    assert in_flight["max"] <= 4, f"exceeded max concurrency: {in_flight['max']}"
    assert in_flight["max"] > 1, "no real concurrency observed — looks sequential"
    sequential_estimate = n_pairs * call_delay
    assert elapsed < sequential_estimate * 0.6, (
        f"elapsed {elapsed:.2f}s not meaningfully faster than sequential {sequential_estimate:.2f}s"
    )


def test_on_progress_reports_expected_stages(repo):
    fact_a = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24"})
    fact_b = make_fact("Delhivery Limited", "Revenue from Operations", "101", {"period_label": "FY24"}, doc_id="doc2")

    events: list[tuple[str, dict]] = []

    with (
        patch("src.fkl.pipeline.extract_blocks", return_value=[]),
        patch("src.fkl.pipeline.extract_facts", return_value=ExtractionResult(facts=[fact_a])),
    ):
        ingest_document("a.pdf", doc_context={"doc_id": "doc1"}, repo=repo, on_progress=lambda s, d: events.append((s, d)))

    stages = [s for s, _ in events]
    assert stages[0] == "started"
    assert "blocks_extracted" in stages
    assert "doc_context" in stages
    assert "extraction_complete" in stages
    assert "facts_persisted" in stages
    assert "candidates_found" in stages
    assert stages[-1] == "done"
    assert "deriving_doc_context" not in stages  # doc_context was supplied, not derived

    done_event = next(d for s, d in events if s == "done")
    assert done_event["result"].facts == 1

    # Second document: candidate found against doc1's fact -> a "relation" event fires.
    events.clear()
    with (
        patch("src.fkl.pipeline.extract_blocks", return_value=[]),
        patch("src.fkl.pipeline.extract_facts", return_value=ExtractionResult(facts=[fact_b])),
        patch("src.fkl.reconcile.adjudicator.generate", return_value=(ExplanationResponse(explanation="x"), "test-model")),
    ):
        ingest_document("b.pdf", doc_context={"doc_id": "doc2"}, repo=repo, on_progress=lambda s, d: events.append((s, d)))
    relation_events = [d for s, d in events if s == "relation"]
    assert len(relation_events) == 1
    # fact_a="100", fact_b="101": same exact claim key (subject/measure/period
    # all match, no other qualifiers), ~0.99% apart -- beyond the 0.1%
    # rounding tolerance, so this is a genuine CONTRADICTS/VALUES_DIVERGE.
    assert relation_events[0] == {"done": 1, "total": 1, "type": "CONTRADICTS", "reason_code": "VALUES_DIVERGE"}


def test_on_progress_reports_already_ingested_short_circuit(repo):
    fact = make_fact("Delhivery Limited", "Revenue from Operations", "100")
    with (
        patch("src.fkl.pipeline.extract_blocks", return_value=[]),
        patch("src.fkl.pipeline.extract_facts", return_value=ExtractionResult(facts=[fact])),
    ):
        ingest_document("a.pdf", doc_context={"doc_id": "doc1"}, repo=repo)

    events: list[str] = []
    ingest_document("a.pdf", doc_context={"doc_id": "doc1"}, repo=repo, on_progress=lambda s, d: events.append(s))
    assert events == ["started", "already_ingested", "done"]
