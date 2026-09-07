"""store/repo.py tests: round-trip fidelity, retrieval-bounded candidate
lookup, relation dedup, and the incremental-ingest invariant (adding a
document must not touch existing facts or relations)."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

import pytest

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
    ValidityInterval,
)
from src.fkl.store.repo import Repo


def make_fact(subject, measure, value_raw, qualifiers=None, doc_id="doc1", **kwargs) -> Fact:
    return Fact(
        id=str(uuid.uuid4()),
        doc_id=doc_id,
        block_id=f"{doc_id}:p1:b0",
        subject=Entity(canonical_id="x", surface_form=subject),
        measure=Measure(canonical_id="m", surface_form=measure),
        fact_type=kwargs.get("fact_type", "numeric"),
        value=Value(raw=value_raw),
        qualifiers=qualifiers or {},
        validity_interval=ValidityInterval(start=kwargs.get("validity_start"), end=kwargs.get("validity_end")),
        evidence=Evidence(page_no=1, char_start=0, char_end=len(value_raw), bbox=[(0, 0, 1, 1)], quote=value_raw),
        confidence=Confidence(extraction=0.9),
        provenance=Provenance(model="test", prompt_version="v1", extracted_at=datetime.now(UTC), parser="test"),
    )


@pytest.fixture
def repo():
    r = Repo(":memory:")
    r.add_document("doc1", "x.pdf", {"doc_id": "doc1"})
    yield r
    r.close()


def test_fact_round_trip(repo):
    fact = make_fact(
        "Delhivery Limited", "Revenue from Operations", "100",
        {"period_label": "FY24"}, validity_start=date(2023, 1, 1), validity_end=date(2024, 1, 1),
    )
    repo.add_fact(fact)
    got = repo.get_fact(fact.id)
    assert got is not None
    assert got.subject.surface_form == "Delhivery Limited"
    assert got.measure.surface_form == "Revenue from Operations"
    assert got.value.raw == "100"
    assert got.qualifiers == {"period_label": "FY24"}
    assert got.validity_interval.start == date(2023, 1, 1)
    assert got.validity_interval.end == date(2024, 1, 1)
    assert got.evidence.bbox == [(0.0, 0.0, 1.0, 1.0)]


def test_document_round_trip(repo):
    repo.add_document("doc_meta", "x.pdf", {"doc_id": "doc_meta", "publisher": "Delhivery Limited"})
    assert repo.has_document("doc_meta") is True
    assert repo.has_document("doc_unseen") is False
    assert repo.get_document_context("doc_meta") == {"doc_id": "doc_meta", "publisher": "Delhivery Limited"}


def test_candidates_for_fact_exact_before_loose(repo):
    exact_match = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24", "consolidation": "consolidated"})
    loose_match = make_fact("Delhivery Limited", "Revenue from Operations", "101", {"period_label": "FY24", "consolidation": "standalone"})
    unrelated = make_fact("Falcon Autotech", "ownership stake", "39%")

    repo.add_fact(exact_match)
    repo.add_fact(loose_match)
    repo.add_fact(unrelated)

    new_fact = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24", "consolidation": "consolidated"})
    repo.add_fact(new_fact)

    candidates = repo.candidates_for_fact(new_fact)
    levels_by_id = {f.id: level for f, level in candidates}
    assert levels_by_id[exact_match.id] == "exact"
    assert levels_by_id[loose_match.id] == "loose"
    assert unrelated.id not in levels_by_id
    assert new_fact.id not in levels_by_id  # never a candidate against itself


def test_relation_dedup_is_order_independent(repo):
    fact_a = make_fact("Delhivery Limited", "Revenue from Operations", "100")
    fact_b = make_fact("Delhivery Limited", "Revenue from Operations", "101")
    repo.add_fact(fact_a)
    repo.add_fact(fact_b)

    assert repo.relation_exists(fact_a.id, fact_b.id) is False
    relation = Relation(
        id=str(uuid.uuid4()), fact_a_id=fact_b.id, fact_b_id=fact_a.id,
        type=RelationType.CORROBORATES, reason_code=ReasonCode.VALUES_MATCH,
        explanation="x", confidence=0.9, adjudicator="rule",
    )
    repo.add_relation(relation)
    assert repo.relation_exists(fact_a.id, fact_b.id) is True
    assert repo.relation_exists(fact_b.id, fact_a.id) is True


def test_duplicate_relation_insert_is_ignored_not_duplicated(repo):
    fact_a = make_fact("Delhivery Limited", "Revenue from Operations", "100")
    fact_b = make_fact("Delhivery Limited", "Revenue from Operations", "101")
    repo.add_fact(fact_a)
    repo.add_fact(fact_b)

    first = Relation(
        id=str(uuid.uuid4()), fact_a_id=fact_a.id, fact_b_id=fact_b.id,
        type=RelationType.CORROBORATES, reason_code=ReasonCode.VALUES_MATCH,
        explanation="first", confidence=0.9, adjudicator="rule",
    )
    second = Relation(
        id=str(uuid.uuid4()), fact_a_id=fact_a.id, fact_b_id=fact_b.id,
        type=RelationType.CONTRADICTS, reason_code=ReasonCode.VALUES_DIVERGE,
        explanation="second, should be ignored", confidence=0.5, adjudicator="rule",
    )
    repo.add_relation(first)
    repo.add_relation(second)

    relations = repo.relations_by_doc("doc1")
    assert len(relations) == 1
    assert relations[0].explanation == "first"


def test_reingesting_same_fact_does_not_duplicate_or_error(repo):
    """Incremental ingest: adding a document must not touch existing
    facts."""
    fact = make_fact("Delhivery Limited", "Revenue from Operations", "100")
    repo.add_fact(fact)
    repo.add_fact(fact)  # re-add, same id -- must be a safe no-op
    assert len(repo.facts_by_doc("doc1")) == 1


def test_quarantine_round_trip(repo):
    fact = make_fact("Delhivery Limited", "revenue", "not a real number")
    repo.add_quarantine("doc1", fact, "partial_ratio 40.0 below threshold 92", 40.0)
    entries = repo.quarantined_by_doc("doc1")
    assert len(entries) == 1
    assert entries[0]["fact"].id == fact.id
    assert entries[0]["reason"] == "partial_ratio 40.0 below threshold 92"
    assert entries[0]["score"] == 40.0
    # quarantined facts must never appear in the facts table
    assert repo.get_fact(fact.id) is None


def test_facts_from_different_docs_still_share_claim_key(repo):
    """The whole point of persistence: candidate-finding must work ACROSS
    documents, not just within one."""
    fact_doc1 = make_fact("Delhivery Limited", "Revenue from Operations", "8142", {"period_label": "FY24", "consolidation": "consolidated"}, doc_id="doc1")
    fact_doc2 = make_fact("Delhivery Limited", "Revenue from Operations", "81415.38", {"period_label": "FY24", "consolidation": "consolidated"}, doc_id="doc2")
    repo.add_document("doc1", "a.pdf", {"doc_id": "doc1"})
    repo.add_document("doc2", "b.pdf", {"doc_id": "doc2"})
    repo.add_fact(fact_doc1)
    repo.add_fact(fact_doc2)

    candidates = repo.candidates_for_fact(fact_doc2)
    assert len(candidates) == 1
    assert candidates[0][0].id == fact_doc1.id
    assert candidates[0][1] == "exact"
