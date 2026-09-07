"""LLM-backed adjudication, layered on top of reconcile/rules.py's
deterministic verdicts. Uses the same Groq-first/Gemini-fallback provider
layer as extract/extractor.py (src/fkl/extract/providers.py) — no
duplicated retry/rotation logic.

Two responsibilities:

1. write_explanation() — the verdict and reason_code are already decided
   (by rules.py). The model only writes the sentence explaining them,
   grounded in both facts' evidence quotes. This is enforced structurally,
   not just by instruction: ExplanationResponse has no type/reason_code
   field at all, so there is no field for the model's output to populate
   even if it tried to change the verdict — the returned Relation copies
   type/reason_code/confidence/etc. from the INPUT relation unchanged, and
   only explanation comes from the model.

2. adjudicate() — for a pair rules.py returned as UNRESOLVED where both
   facts are non-numeric (categorical/relational/assertion — the cases
   deterministic rules have no principled way to compare), ask the model
   for a real verdict from the same closed vocabulary rules.py uses
   (RelationType, ReasonCode — reusing those enums as the Pydantic field
   types means an out-of-vocabulary answer is a validation error, not a
   silently-accepted new category). adjudicator="llm" on the result;
   write_explanation()'s output keeps whatever adjudicator the input
   Relation already had ("rule" for a rules.py verdict).

No LLM calls happen in reconcile/rules.py itself — this module is what
"the explanation-writing adjudicator comes next" (referenced in that
module's docstring) turned into.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from src.fkl.extract.providers import KeyRotator, generate
from src.fkl.store.models import Fact, Relation, RelationType, ReasonCode

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/adjudicate")
PROMPT_VERSION = "adjudicate-v1"

NON_NUMERIC_FACT_TYPES = {"categorical", "relational", "assertion"}


class ExplanationResponse(BaseModel):
    """Deliberately has no type/reason_code field — see module docstring.
    The model cannot change the verdict because there is nowhere in this
    schema for a changed verdict to go."""

    explanation: str


class AdjudicationResponse(BaseModel):
    """Reuses RelationType/ReasonCode as the field types themselves, not
    free strings re-parsed afterward — a response outside that closed
    vocabulary fails Pydantic validation rather than silently being
    accepted as some new, unvetted category."""

    type: RelationType
    reason_code: ReasonCode
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


def _fact_summary(fact: Fact) -> str:
    return (
        f"subject: {fact.subject.surface_form}\n"
        f"measure: {fact.measure.surface_form}\n"
        f"value: {fact.value.raw}\n"
        f"fact_type: {fact.fact_type}\n"
        f"qualifiers: {json.dumps(fact.qualifiers, default=str)}\n"
        f"validity_interval: [{fact.validity_interval.start}, {fact.validity_interval.end})\n"
        f'evidence_quote: "{fact.evidence.quote}"'
    )


def _pair_hash(fact_a: Fact, fact_b: Fact, *extra: object) -> str:
    a_id, b_id = sorted((fact_a.id, fact_b.id))
    parts = [a_id, b_id, PROMPT_VERSION, *[str(e) for e in extra]]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _load_cache(key: str, response_model: type[BaseModel]) -> BaseModel | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    return response_model.model_validate_json(path.read_text(encoding="utf-8"))


def _save_cache(key: str, response: BaseModel) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(response.model_dump_json(indent=2), encoding="utf-8")


def _build_explanation_prompt(relation: Relation, fact_a: Fact, fact_b: Fact) -> str:
    return (
        "You are writing the explanation for a fact-reconciliation verdict that has "
        "ALREADY been decided by a deterministic rule — you are not deciding it, and "
        "there is no field in your response for a verdict, only for the explanation "
        "sentence. Do not state or imply a different verdict than the one given.\n\n"
        f"DECIDED VERDICT (fixed, not yours to change): {relation.type.value} / "
        f"{relation.reason_code.value}\n\n"
        f"FACT A:\n{_fact_summary(fact_a)}\n\n"
        f"FACT B:\n{_fact_summary(fact_b)}\n\n"
        "Write a ONE- or TWO-sentence explanation of why this verdict holds, grounded "
        "explicitly in both evidence_quote values above — reference what each quote "
        "actually says. Do not introduce facts not present in the two summaries."
    )


def write_explanation(
    relation: Relation, fact_a: Fact, fact_b: Fact, rotator: KeyRotator | None = None
) -> Relation:
    """Returns a new Relation: same type/reason_code/confidence/adjudicator
    as the input, explanation replaced by the model's grounded sentence."""
    cache_key = _pair_hash(fact_a, fact_b, "explain", relation.type.value, relation.reason_code.value)
    cached = _load_cache(cache_key, ExplanationResponse)
    if cached is None:
        if rotator is None:
            rotator = KeyRotator.from_env()
        prompt = _build_explanation_prompt(relation, fact_a, fact_b)
        response, _model_used = generate(prompt, ExplanationResponse, rotator)
        _save_cache(cache_key, response)
        cached = response

    return relation.model_copy(update={"explanation": cached.explanation})


def _build_adjudication_prompt(fact_a: Fact, fact_b: Fact, unresolved: Relation) -> str:
    type_values = ", ".join(t.value for t in RelationType)
    reason_values = ", ".join(r.value for r in ReasonCode)
    return (
        "A deterministic rule engine compared two facts and could not confidently "
        f"classify their relationship (verdict: RECONCILED_BY_CONTEXT / UNRESOLVED — "
        f'its own note: "{unresolved.explanation}"). Both facts are non-numeric, where '
        "semantic judgement genuinely matters and a rule can't safely guess. Decide the "
        "relationship yourself.\n\n"
        f"FACT A:\n{_fact_summary(fact_a)}\n\n"
        f"FACT B:\n{_fact_summary(fact_b)}\n\n"
        f"type must be exactly one of: {type_values}\n"
        f"reason_code must be exactly one of: {reason_values}\n"
        "reason_code must never just restate type (e.g. type=CONTRADICTS with "
        "reason_code=VALUES_DIVERGE is correct; reason_code=CONTRADICTS is not a valid "
        "value at all). Ground your explanation in both evidence_quote values. If you "
        "genuinely cannot decide, use type=RECONCILED_BY_CONTEXT, "
        "reason_code=UNRESOLVED rather than guessing."
    )


def adjudicate(
    fact_a: Fact, fact_b: Fact, unresolved: Relation, rotator: KeyRotator | None = None
) -> Relation:
    """Only call this for a Relation rules.py returned with type ==
    RECONCILED_BY_CONTEXT and reason_code == UNRESOLVED, where both facts'
    fact_type is in NON_NUMERIC_FACT_TYPES — callers should check
    is_adjudication_candidate() first; this function trusts the caller and
    does not re-check, so it stays a pure "given this pair, ask the model"
    function usable in tests without needing a real UNRESOLVED Relation.
    """
    cache_key = _pair_hash(fact_a, fact_b, "adjudicate")
    cached = _load_cache(cache_key, AdjudicationResponse)
    if cached is None:
        if rotator is None:
            rotator = KeyRotator.from_env()
        prompt = _build_adjudication_prompt(fact_a, fact_b, unresolved)
        response, _model_used = generate(prompt, AdjudicationResponse, rotator)
        _save_cache(cache_key, response)
        cached = response

    return Relation(
        id=unresolved.id,
        fact_a_id=fact_a.id,
        fact_b_id=fact_b.id,
        type=cached.type,
        reason_code=cached.reason_code,
        explanation=cached.explanation,
        delta=None,
        confidence=cached.confidence,
        adjudicator="llm",
    )


def is_adjudication_candidate(fact_a: Fact, fact_b: Fact, relation: Relation) -> bool:
    """True when relation is exactly the UNRESOLVED verdict rules.py
    produces AND both facts are non-numeric — the precondition adjudicate()
    assumes but does not itself enforce."""
    return (
        relation.type == RelationType.RECONCILED_BY_CONTEXT
        and relation.reason_code == ReasonCode.UNRESOLVED
        and fact_a.fact_type in NON_NUMERIC_FACT_TYPES
        and fact_b.fact_type in NON_NUMERIC_FACT_TYPES
    )
