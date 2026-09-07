"""link/claim_key.py tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from src.fkl.link.claim_key import claim_key, claim_key_components, loose_key
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


def test_claim_key_stable_for_identical_facts():
    a = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24"})
    b = make_fact("Delhivery Ltd", "Revenue from Operations", "200", {"period_label": "FY24"})
    # same claim (entity alias resolves), even though the stated value differs
    assert claim_key(a) == claim_key(b)


def test_claim_key_differs_on_consolidation():
    a = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24", "consolidation": "standalone"})
    b = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24", "consolidation": "consolidated"})
    assert claim_key(a) != claim_key(b)
    assert loose_key(a) == loose_key(b)  # still the same subject+measure


def test_claim_key_matches_across_equivalent_period_labels_critical():
    """The exact claim key must be identical for the dossier's Q1
    FY2025/26 / 2025Q2 pair -- that's what makes them land in the same
    exact-match candidate group in link/candidates.py."""
    a = make_fact("India", "real GDP growth", "7.8%", {"period_label": "Q1 FY2025/26"})
    b = make_fact("India", "real GDP growth", "7.8%", {"period_label": "2025Q2"})
    assert claim_key(a) == claim_key(b)


def test_loose_key_ignores_period_but_not_measure():
    a = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24"})
    b = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY25"})
    c = make_fact("Delhivery Limited", "PAT", "100", {"period_label": "FY24"})
    assert loose_key(a) == loose_key(b)
    assert loose_key(a) != loose_key(c)


def test_components_expose_resolved_period():
    a = make_fact("Delhivery Limited", "Revenue from Operations", "100", {"period_label": "FY24"})
    c = claim_key_components(a)
    assert c.subject_id == "delhivery_limited"
    assert c.period_start is not None and c.period_end is not None
