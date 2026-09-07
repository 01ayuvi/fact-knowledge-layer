"""link/candidates.py tests: exact matches before loose matches, no
duplicate pairs across the two, and group-size truncation instead of an
unbounded O(n^2) blowup."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from src.fkl.link.candidates import find_candidates
from src.fkl.store.models import Confidence, Entity, Evidence, Fact, Measure, Provenance, Value


def make_fact(subject: str, measure: str, value_raw: str, qualifiers: dict | None = None) -> Fact:
    return Fact(
        id=str(uuid.uuid4()),
        doc_id="doc",
        block_id="doc:p1:b0",
        subject=Entity(canonical_id="x", surface_form=subject),
        measure=Measure(canonical_id="m", surface_form=measure),
        fact_type="numeric",
        value=Value(raw=value_raw),
        qualifiers=qualifiers or {},
        evidence=Evidence(page_no=1, char_start=0, char_end=1, bbox=[(0, 0, 1, 1)], quote=value_raw),
        confidence=Confidence(extraction=0.9),
        provenance=Provenance(model="test", prompt_version="v1", extracted_at=datetime.now(UTC), parser="test"),
    )


def test_exact_match_pair_found():
    a = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24", "consolidation": "consolidated"})
    b = make_fact("Delhivery Limited", "Revenue from Operations", "101", {"period_label": "FY24", "consolidation": "consolidated"})
    pairs = find_candidates([a, b])
    assert len(pairs) == 1
    assert pairs[0].match_level == "exact"


def test_loose_match_pair_found_when_qualifiers_differ():
    a = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24", "consolidation": "standalone"})
    b = make_fact("Delhivery Limited", "Revenue from Operations", "101", {"period_label": "FY24", "consolidation": "consolidated"})
    pairs = find_candidates([a, b])
    assert len(pairs) == 1
    assert pairs[0].match_level == "loose"


def test_unrelated_facts_produce_no_pairs():
    a = make_fact("Delhivery Limited", "Revenue from Operations", "100")
    b = make_fact("Falcon Autotech", "ownership stake", "39%")
    assert find_candidates([a, b]) == []


def test_no_duplicate_pair_across_exact_and_loose():
    a = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24"})
    b = make_fact("Delhivery Limited", "Revenue from Operations", "101", {"period_label": "FY24"})
    pairs = find_candidates([a, b])
    pair_id_sets = [frozenset((p.fact_a.id, p.fact_b.id)) for p in pairs]
    assert len(pair_id_sets) == len(set(pair_id_sets))


def test_group_size_is_bounded_not_quadratic_over_full_corpus():
    """20 facts sharing one claim key should still only ever compare
    within that group -- retrieval-bounded, not a full-corpus O(n^2)
    scan."""
    facts = [
        make_fact("Delhivery Limited", "Revenue from Operations", str(i), {"period_label": "FY24"})
        for i in range(20)
    ]
    pairs = find_candidates(facts, max_group_size=5)
    # C(5,2) = 10 pairs max after truncation to 5, not C(20,2) = 190
    assert len(pairs) == 10


def test_single_fact_produces_no_pairs():
    a = make_fact("Delhivery Limited", "Revenue from Operations", "100")
    assert find_candidates([a]) == []
