"""reconcile/rules.py tests: table-driven, pair in -> (RelationType,
ReasonCode) out. The four CRITICAL cases are the dossier's own headline
cases (docs/CASE_DOSSIER.md) -- everything else is synthetic coverage for
branches the dossier doesn't happen to exercise."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

import pytest

from src.fkl.reconcile.rules import reconcile
from src.fkl.store.models import (
    Confidence,
    Entity,
    Evidence,
    Fact,
    Measure,
    Provenance,
    RelationType,
    ReasonCode,
    Value,
    ValidityInterval,
)


def make_fact(
    *,
    subject: str = "Delhivery Limited",
    measure: str = "Revenue from Operations",
    fact_type: str = "numeric",
    value_raw: str,
    qualifiers: dict | None = None,
    validity_start: date | None = None,
    validity_end: date | None = None,
) -> Fact:
    return Fact(
        id=str(uuid.uuid4()),
        doc_id="doc",
        block_id="doc:p1:b0",
        subject=Entity(canonical_id="x", surface_form=subject),
        measure=Measure(canonical_id="m", surface_form=measure),
        fact_type=fact_type,
        value=Value(raw=value_raw),
        qualifiers=qualifiers or {},
        validity_interval=ValidityInterval(start=validity_start, end=validity_end),
        evidence=Evidence(page_no=1, char_start=0, char_end=len(value_raw), bbox=[(0, 0, 1, 1)], quote=value_raw),
        confidence=Confidence(extraction=0.9),
        provenance=Provenance(model="test", prompt_version="v1", extracted_at=datetime.now(UTC), parser="test"),
    )


CASES = [
    pytest.param(
        # docs/CASE_DOSSIER.md §1: "Annual Report FY24 p22: Revenue from
        # Operations ... 81,415.38 (₹ in Million, Consolidated, FY ended
        # March 31, 2024)" vs "Q4 FY24 earnings deck p6: ₹8,142 Cr FY24
        # revenue from services". "81,415.38 mn / 10 = 8,141.54 Cr vs
        # stated 8,142 Cr -> delta 0.006%."
        dict(value_raw="₹81,415.38 million", qualifiers={"period_label": "FY24", "consolidation": "consolidated"}),
        dict(value_raw="₹8,142 Cr", qualifiers={"period_label": "FY24", "consolidation": "consolidated"}),
        RelationType.CORROBORATES,
        ReasonCode.ROUNDING_ARTIFACT,
        id="dossier_case1_rounding_mn_vs_cr",
    ),
    pytest.param(
        # docs/CASE_DOSSIER.md §1: "Second instance, same measure, prior
        # year: AR FY23 consolidated = 72,253.01 mn = 7,225.3 Cr; deck
        # p14 table = 7,225 Cr. Exact."
        dict(value_raw="72,253.01 mn", qualifiers={"period_label": "FY23", "consolidation": "consolidated"}),
        dict(value_raw="7,225.3 Cr", qualifiers={"period_label": "FY23", "consolidation": "consolidated"}),
        RelationType.CORROBORATES,
        ReasonCode.SCALE_NORMALIZED,
        id="dossier_case1_exact_prior_year",
    ),
    pytest.param(
        # docs/CASE_DOSSIER.md §3(a): "Particulars / Standalone FY24
        # 74,540.82 / Consolidated FY24 81,415.38 ... Differs only by the
        # consolidation qualifier."
        dict(value_raw="74,540.82", qualifiers={"period_label": "FY24", "consolidation": "standalone"}),
        dict(value_raw="81,415.38", qualifiers={"period_label": "FY24", "consolidation": "consolidated"}),
        RelationType.RECONCILED_BY_CONTEXT,
        ReasonCode.SCOPE_MISMATCH,
        id="dossier_case3a_scope_mismatch",
    ),
    pytest.param(
        # docs/CASE_DOSSIER.md §3(d): IMF p3 "the first quarter of
        # FY2025/26" (7.8%) vs IMF p10 "2025Q2" (7.8%) -- "Q1 of Indian
        # FY2025/26 (Apr-Jun 2025) IS calendar 2025Q2." Once the period
        # normalizer resolves both to the same interval, this is a plain
        # corroboration, not a mismatch code.
        dict(subject="India", measure="real GDP growth", value_raw="7.8%", qualifiers={"period_label": "Q1 FY2025/26"}),
        dict(subject="India", measure="real GDP growth", value_raw="7.8%", qualifiers={"period_label": "2025Q2"}),
        RelationType.CORROBORATES,
        ReasonCode.VALUES_MATCH,
        id="dossier_case3d_fy_quarter_equals_cy_quarter",
    ),
    # -- synthetic coverage for branches the four dossier cases don't hit --
    pytest.param(
        dict(subject="India", measure="real GDP growth", value_raw="6.5%", qualifiers={"period_label": "FY24"}),
        dict(subject="India", measure="real GDP growth", value_raw="6.5%", qualifiers={"period_label": "FY25"}),
        RelationType.RECONCILED_BY_CONTEXT,
        ReasonCode.PERIOD_DISJOINT,
        id="period_disjoint",
    ),
    pytest.param(
        dict(subject="India", measure="real GDP growth", value_raw="6.5%", qualifiers={"period_label": "FY24", "geography": "India"}),
        dict(subject="India", measure="real GDP growth", value_raw="6.5%", qualifiers={"period_label": "FY24", "geography": "global"}),
        RelationType.RECONCILED_BY_CONTEXT,
        ReasonCode.GEOGRAPHY_MISMATCH,
        id="geography_mismatch",
    ),
    pytest.param(
        # docs/CASE_DOSSIER.md §3(b): Economic Survey "first advance
        # estimates ... 6.4 per cent in FY25" vs RBI/IMF actual "6.5 per
        # cent" for the same year.
        dict(subject="India", measure="real GDP growth", value_raw="6.4%", qualifiers={"period_label": "FY25", "modality": "first advance estimate"}),
        dict(subject="India", measure="real GDP growth", value_raw="6.5%", qualifiers={"period_label": "FY25", "modality": "actual"}),
        RelationType.RECONCILED_BY_CONTEXT,
        ReasonCode.VINTAGE_RESTATEMENT,
        id="vintage_restatement",
    ),
    pytest.param(
        dict(value_raw="$100 million", qualifiers={"period_label": "FY24", "consolidation": "consolidated"}),
        dict(value_raw="₹100 million", qualifiers={"period_label": "FY24", "consolidation": "consolidated"}),
        RelationType.RECONCILED_BY_CONTEXT,
        ReasonCode.UNIT_MISMATCH,
        id="currency_mismatch",
    ),
    pytest.param(
        dict(value_raw="₹100 million", qualifiers={"period_label": "FY24", "consolidation": "consolidated"}),
        dict(value_raw="₹150 million", qualifiers={"period_label": "FY24", "consolidation": "consolidated"}),
        RelationType.CONTRADICTS,
        ReasonCode.VALUES_DIVERGE,
        id="genuine_contradiction_numeric",
    ),
    pytest.param(
        dict(subject="Delhivery Limited", measure="director status", fact_type="categorical",
             value_raw="Non-Executive Independent Director",
             qualifiers={"period_label": "FY24", "consolidation": "consolidated"}),
        dict(subject="Delhivery Limited", measure="director status", fact_type="categorical",
             value_raw="Resigned",
             qualifiers={"period_label": "FY24", "consolidation": "consolidated"}),
        RelationType.CONTRADICTS,
        ReasonCode.VALUES_DIVERGE,
        id="genuine_contradiction_categorical_overlapping_intervals",
    ),
    pytest.param(
        dict(subject="Delhivery Limited", measure="Revenue from Operations", value_raw="100"),
        dict(subject="Falcon Autotech", measure="Revenue from Operations", value_raw="100"),
        RelationType.RECONCILED_BY_CONTEXT,
        ReasonCode.UNRESOLVED,
        id="different_subject_not_comparable",
    ),
    pytest.param(
        dict(value_raw="100", qualifiers={"period_label": "sometime unparseable"}),
        dict(value_raw="100", qualifiers={"period_label": "FY24"}),
        RelationType.RECONCILED_BY_CONTEXT,
        ReasonCode.UNRESOLVED,
        id="unparseable_period_routed_to_review",
    ),
]


@pytest.mark.parametrize("fact_a_kwargs,fact_b_kwargs,expected_type,expected_reason", CASES)
def test_reconcile_table(fact_a_kwargs, fact_b_kwargs, expected_type, expected_reason):
    fact_a = make_fact(**fact_a_kwargs)
    fact_b = make_fact(**fact_b_kwargs)
    relation = reconcile(fact_a, fact_b)
    assert relation.type == expected_type
    assert relation.reason_code == expected_reason
    assert relation.adjudicator == "rule"
    assert relation.fact_a_id == fact_a.id
    assert relation.fact_b_id == fact_b.id


def test_dossier_case2_morparia_supersedes_critical():
    """docs/CASE_DOSSIER.md §2: 'Prospectus 2022, p88: Kalpana Jaisingh
    Morparia is a Non-Executive Independent Director...' (present tense)
    vs 'Annual Report FY24, p91: Ms. Kalpana Jaisingh Morparia ...
    (resigned w.e.f. February 11, 2023)'. 'The fact has a validity
    interval [prospectus_date, 2023-02-11) and the later document closes
    it... the correct verdict is not CONTRADICTS' -- both assertions are
    true over non-overlapping validity intervals, so this must resolve to
    SUPERSEDES / TEMPORAL_STATE_CHANGE, not CONTRADICTS."""
    prospectus_fact = make_fact(
        subject="Kalpana Jaisingh Morparia",
        measure="director status",
        fact_type="categorical",
        value_raw="Non-Executive Independent Director",
        validity_start=date(2022, 1, 1),
        validity_end=date(2023, 2, 11),
    )
    ar_fact = make_fact(
        subject="Ms. Kalpana Jaisingh Morparia",
        measure="director status",
        fact_type="categorical",
        value_raw="Non Executive - Independent Director (resigned)",
        validity_start=date(2023, 2, 11),
        validity_end=None,
    )
    relation = reconcile(prospectus_fact, ar_fact)
    assert relation.type == RelationType.SUPERSEDES
    assert relation.reason_code == ReasonCode.TEMPORAL_STATE_CHANGE
    assert relation.type != RelationType.CONTRADICTS  # the whole point of the case


def test_relation_delta_populated_for_rounding_artifact():
    fact_a = make_fact(value_raw="₹81,415.38 million", qualifiers={"period_label": "FY24", "consolidation": "consolidated"})
    fact_b = make_fact(value_raw="₹8,142 Cr", qualifiers={"period_label": "FY24", "consolidation": "consolidated"})
    relation = reconcile(fact_a, fact_b)
    assert relation.delta is not None
    assert relation.delta.relative == pytest.approx(0.0000567, abs=0.00002)
    assert relation.delta.after_normalization is True


def test_reconcile_is_symmetric_on_verdict_type():
    """Order shouldn't flip CORROBORATES into CONTRADICTS or vice versa."""
    fact_a = make_fact(value_raw="₹100 million", qualifiers={"period_label": "FY24", "consolidation": "consolidated"})
    fact_b = make_fact(value_raw="₹100 million", qualifiers={"period_label": "FY24", "consolidation": "consolidated"})
    forward = reconcile(fact_a, fact_b)
    backward = reconcile(fact_b, fact_a)
    assert forward.type == backward.type == RelationType.CORROBORATES
    assert forward.reason_code == backward.reason_code == ReasonCode.VALUES_MATCH
