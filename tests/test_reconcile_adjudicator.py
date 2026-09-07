"""reconcile/adjudicator.py tests. All mocked at the provider boundary
(src.fkl.extract.providers.generate) -- no real API calls, and no reliance
on Groq/Gemini quota being available."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from unittest.mock import patch

import pytest

from src.fkl.reconcile.adjudicator import (
    AdjudicationResponse,
    ExplanationResponse,
    adjudicate,
    is_adjudication_candidate,
    write_explanation,
)
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


def make_fact(
    *,
    subject: str = "Delhivery Limited",
    measure: str = "director status",
    fact_type: str = "categorical",
    value_raw: str,
    quote: str | None = None,
    validity_start: date | None = None,
    validity_end: date | None = None,
) -> Fact:
    quote = quote if quote is not None else value_raw
    return Fact(
        id=str(uuid.uuid4()),
        doc_id="doc",
        block_id="doc:p1:b0",
        subject=Entity(canonical_id="x", surface_form=subject),
        measure=Measure(canonical_id="m", surface_form=measure),
        fact_type=fact_type,
        value=Value(raw=value_raw),
        validity_interval=ValidityInterval(start=validity_start, end=validity_end),
        evidence=Evidence(page_no=1, char_start=0, char_end=len(quote), bbox=[(0, 0, 1, 1)], quote=quote),
        confidence=Confidence(extraction=0.9),
        provenance=Provenance(model="test", prompt_version="v1", extracted_at=datetime.now(UTC), parser="test"),
    )


def make_relation(
    fact_a: Fact, fact_b: Fact, rel_type: RelationType, reason: ReasonCode, explanation: str = ""
) -> Relation:
    return Relation(
        id=str(uuid.uuid4()),
        fact_a_id=fact_a.id,
        fact_b_id=fact_b.id,
        type=rel_type,
        reason_code=reason,
        explanation=explanation,
        confidence=0.9,
        adjudicator="rule",
    )


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Never touch the real data/cache/adjudicate/ directory from tests."""
    import src.fkl.reconcile.adjudicator as adjudicator_module

    monkeypatch.setattr(adjudicator_module, "CACHE_DIR", tmp_path / "adjudicate")


def test_write_explanation_cannot_change_verdict():
    """The core guarantee: whatever the model returns, type/reason_code on
    the output Relation are exactly what rules.py already decided."""
    fact_a = make_fact(value_raw="active", quote="active director")
    fact_b = make_fact(value_raw="resigned", quote="resigned director")
    decided = make_relation(fact_a, fact_b, RelationType.SUPERSEDES, ReasonCode.TEMPORAL_STATE_CHANGE, "placeholder")

    mock_response = ExplanationResponse(explanation="Fact A precedes fact B; the state changed.")
    with patch("src.fkl.reconcile.adjudicator.generate", return_value=(mock_response, "test-model")):
        result = write_explanation(decided, fact_a, fact_b)

    assert result.type == RelationType.SUPERSEDES
    assert result.reason_code == ReasonCode.TEMPORAL_STATE_CHANGE
    assert result.confidence == decided.confidence
    assert result.adjudicator == "rule"
    assert result.explanation == "Fact A precedes fact B; the state changed."


def test_explanation_response_schema_has_no_verdict_fields():
    """Structural enforcement, not just instruction: there is no field in
    this schema a model could even populate to attempt a verdict change."""
    fields = ExplanationResponse.model_fields
    assert set(fields) == {"explanation"}


def test_write_explanation_uses_cache_on_second_call():
    fact_a = make_fact(value_raw="active")
    fact_b = make_fact(value_raw="resigned")
    decided = make_relation(fact_a, fact_b, RelationType.SUPERSEDES, ReasonCode.TEMPORAL_STATE_CHANGE)

    mock_response = ExplanationResponse(explanation="Grounded explanation.")
    with patch("src.fkl.reconcile.adjudicator.generate", return_value=(mock_response, "test-model")) as mock_gen:
        write_explanation(decided, fact_a, fact_b)
        write_explanation(decided, fact_a, fact_b)

    assert mock_gen.call_count == 1


def test_adjudicate_returns_llm_adjudicator():
    fact_a = make_fact(value_raw="Chief Business Officer", measure="role")
    fact_b = make_fact(value_raw="resigned from the office of Executive Director", measure="role")
    unresolved = make_relation(
        fact_a, fact_b, RelationType.RECONCILED_BY_CONTEXT, ReasonCode.UNRESOLVED,
        "Facts describe different subjects or measures — not a comparable pair.",
    )
    mock_response = AdjudicationResponse(
        type=RelationType.RECONCILED_BY_CONTEXT,
        reason_code=ReasonCode.SCOPE_MISMATCH,
        explanation="One role is a subset of the other's responsibilities.",
        confidence=0.7,
    )
    with patch("src.fkl.reconcile.adjudicator.generate", return_value=(mock_response, "test-model")):
        result = adjudicate(fact_a, fact_b, unresolved)

    assert result.adjudicator == "llm"
    assert result.type == RelationType.RECONCILED_BY_CONTEXT
    assert result.reason_code == ReasonCode.SCOPE_MISMATCH
    assert result.confidence == 0.7
    assert result.fact_a_id == fact_a.id
    assert result.fact_b_id == fact_b.id


def test_adjudicate_rejects_out_of_vocabulary_verdict():
    """reason_code is typed as the real ReasonCode enum -- a model response
    naming something outside the closed vocabulary fails validation rather
    than silently becoming a new, unvetted category."""
    with pytest.raises(Exception):
        AdjudicationResponse(
            type=RelationType.CONTRADICTS,
            reason_code="TOTALLY_MADE_UP_CODE",
            explanation="x",
            confidence=0.5,
        )


def test_adjudicate_uses_cache_on_second_call():
    fact_a = make_fact(value_raw="active")
    fact_b = make_fact(value_raw="resigned")
    unresolved = make_relation(fact_a, fact_b, RelationType.RECONCILED_BY_CONTEXT, ReasonCode.UNRESOLVED)
    mock_response = AdjudicationResponse(
        type=RelationType.CONTRADICTS, reason_code=ReasonCode.VALUES_DIVERGE, explanation="x", confidence=0.6,
    )
    with patch("src.fkl.reconcile.adjudicator.generate", return_value=(mock_response, "test-model")) as mock_gen:
        adjudicate(fact_a, fact_b, unresolved)
        adjudicate(fact_a, fact_b, unresolved)

    assert mock_gen.call_count == 1


def test_is_adjudication_candidate_true_for_unresolved_non_numeric():
    fact_a = make_fact(value_raw="active", fact_type="categorical")
    fact_b = make_fact(value_raw="resigned", fact_type="categorical")
    unresolved = make_relation(fact_a, fact_b, RelationType.RECONCILED_BY_CONTEXT, ReasonCode.UNRESOLVED)
    assert is_adjudication_candidate(fact_a, fact_b, unresolved) is True


def test_is_adjudication_candidate_false_for_numeric_facts():
    fact_a = make_fact(value_raw="100", fact_type="numeric")
    fact_b = make_fact(value_raw="200", fact_type="numeric")
    unresolved = make_relation(fact_a, fact_b, RelationType.RECONCILED_BY_CONTEXT, ReasonCode.UNRESOLVED)
    assert is_adjudication_candidate(fact_a, fact_b, unresolved) is False


def test_is_adjudication_candidate_false_for_already_resolved_relation():
    fact_a = make_fact(value_raw="active", fact_type="categorical")
    fact_b = make_fact(value_raw="resigned", fact_type="categorical")
    resolved = make_relation(fact_a, fact_b, RelationType.CORROBORATES, ReasonCode.VALUES_MATCH)
    assert is_adjudication_candidate(fact_a, fact_b, resolved) is False


def test_pair_hash_order_independent():
    """Cache should hit regardless of which fact is passed as A vs B."""
    fact_a = make_fact(value_raw="active")
    fact_b = make_fact(value_raw="resigned")
    decided = make_relation(fact_a, fact_b, RelationType.SUPERSEDES, ReasonCode.TEMPORAL_STATE_CHANGE)
    decided_swapped = make_relation(fact_b, fact_a, RelationType.SUPERSEDES, ReasonCode.TEMPORAL_STATE_CHANGE)

    mock_response = ExplanationResponse(explanation="x")
    with patch("src.fkl.reconcile.adjudicator.generate", return_value=(mock_response, "test-model")) as mock_gen:
        write_explanation(decided, fact_a, fact_b)
        write_explanation(decided_swapped, fact_b, fact_a)

    assert mock_gen.call_count == 1
