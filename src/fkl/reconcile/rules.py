"""The typed decision procedure: given two Facts, decide a Relation
(verdict + reason_code) using deterministic logic only. No LLM calls --
the explanation strings here are template-generated, not model-written;
"the LLM writes the sentence; the code writes the verdict" (plan §2 D3)
becomes true once an adjudicator wraps this module to rewrite explanation
in natural language for contested/semantic pairs. This module is the
verdict, always.

Decision order (see reconcile() for the actual branches):
  1. same subject+measure at all? if not, not a comparable pair -> UNRESOLVED.
  2. non-overlapping validity intervals + differing value -> SUPERSEDES /
     TEMPORAL_STATE_CHANGE (checked before qualifier/value comparison,
     since a state change changes what "matching" even means here).
  3. consolidation differs -> RECONCILED_BY_CONTEXT / SCOPE_MISMATCH.
  4. geography differs -> RECONCILED_BY_CONTEXT / GEOGRAPHY_MISMATCH.
  5. resolved periods disjoint -> RECONCILED_BY_CONTEXT / PERIOD_DISJOINT;
     resolved periods overlap but aren't equal -> .../PERIOD_BASIS_MISMATCH.
     (Periods that resolve to the SAME interval, e.g. "Q1 FY2025/26" and
     "2025Q2" via normalize/periods.py, are NOT a mismatch -- they fall
     through to value comparison as docs/CASE_DOSSIER.md §3(d) intends:
     the correct outcome for that pair is CORROBORATES, not a mismatch
     code, because the period normalizer resolved them to be identical.)
  6. modality differs and one side reads as an estimate/advance figure
     -> RECONCILED_BY_CONTEXT / VINTAGE_RESTATEMENT.
  7. currencies determinable and differ -> RECONCILED_BY_CONTEXT /
     UNIT_MISMATCH.
  8. values compared (numeric via normalize/units.py base_value, else raw
     string equality):
       - equal within EXACT_RELATIVE_TOLERANCE, same scale/currency
         representation -> CORROBORATES / VALUES_MATCH
       - equal within EXACT_RELATIVE_TOLERANCE, different scale/currency
         representation -> CORROBORATES / SCALE_NORMALIZED
       - equal within ROUNDING_RELATIVE_TOLERANCE -> CORROBORATES /
         ROUNDING_ARTIFACT
       - otherwise -> CONTRADICTS / VALUES_DIVERGE
  9. anything that couldn't be confidently resolved along the way (an
     unparseable period or value on either side) -> UNRESOLVED, routed to
     review rather than guessed at.

RelationType has no "not comparable" / "unknown" member (see
src/fkl/store/models.py) -- UNRESOLVED as a reason_code always pairs with
RECONCILED_BY_CONTEXT as the type: the closest fit among the five for "we
looked at the context and couldn't confidently resolve it," never a
confident CORROBORATES/CONTRADICTS/SUPERSEDES claim.

Compound cases (docs/CASE_DOSSIER.md §3(e): "SCOPE_MISMATCH +
PERIOD_DISJOINT") don't fit a single reason_code field -- the first
explaining dimension found (fixed priority order above) becomes the
reason_code, and the runner-up is named in the explanation text instead of
silently dropped.
"""

from __future__ import annotations

import uuid
from dataclasses import replace as dc_replace
from datetime import date

from src.fkl.link.claim_key import claim_key_components, normalize_qualifier
from src.fkl.normalize.measures import measures_aliased_not_identical
from src.fkl.normalize.units import SCALE_MULTIPLIERS, NormalizedValue, normalize_value
from src.fkl.store.models import (
    Delta,
    Fact,
    Relation,
    RelationType,
    ReasonCode,
    ValidityInterval,
)

EXACT_RELATIVE_TOLERANCE = 1e-6
# 0.1% -- looser than every rounding gap actually observed in the corpus
# (docs/CASE_DOSSIER.md §1: Delhivery FY24 revenue mn-vs-Cr delta is
# ~0.006%; §5(b): the FY23 7,224-vs-7,225 Cr split is ~0.01%) but tight
# enough that a real disagreement (the dossier's own Case 2/genuine-
# contradiction candidates run well into double-digit percent) still
# lands as CONTRADICTS, not a false ROUNDING_ARTIFACT.
ROUNDING_RELATIVE_TOLERANCE = 0.001

_ESTIMATE_MODALITY_MARKERS = ("estimate", "advance", "provisional", "projected", "forecast")

_RULE_CONFIDENCE: dict[ReasonCode, float] = {
    ReasonCode.VALUES_MATCH: 1.0,
    ReasonCode.SCALE_NORMALIZED: 0.97,
    ReasonCode.ROUNDING_ARTIFACT: 0.9,
    ReasonCode.SCOPE_MISMATCH: 0.9,
    ReasonCode.GEOGRAPHY_MISMATCH: 0.85,
    ReasonCode.PERIOD_DISJOINT: 0.9,
    ReasonCode.PERIOD_BASIS_MISMATCH: 0.75,
    ReasonCode.VINTAGE_RESTATEMENT: 0.85,
    ReasonCode.UNIT_MISMATCH: 0.85,
    ReasonCode.TEMPORAL_STATE_CHANGE: 0.9,
    ReasonCode.VALUES_DIVERGE: 0.9,
    ReasonCode.UNRESOLVED: 0.2,
}


def _relation(
    fact_a: Fact,
    fact_b: Fact,
    rel_type: RelationType,
    reason: ReasonCode,
    explanation: str,
    delta: Delta | None = None,
) -> Relation:
    return Relation(
        id=str(uuid.uuid4()),
        fact_a_id=fact_a.id,
        fact_b_id=fact_b.id,
        type=rel_type,
        reason_code=reason,
        explanation=explanation,
        delta=delta,
        confidence=_RULE_CONFIDENCE.get(reason, 0.7),
        adjudicator="rule",
    )


def _bounds(vi: ValidityInterval) -> tuple[date, date]:
    return vi.start or date.min, vi.end or date.max


def _intervals_overlap(a: ValidityInterval, b: ValidityInterval) -> bool:
    a0, a1 = _bounds(a)
    b0, b1 = _bounds(b)
    return a0 < b1 and b0 < a1


def _period_relation(
    a: tuple[date | None, date | None], b: tuple[date | None, date | None]
) -> str | None:
    a_start, a_end = a
    b_start, b_end = b
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return None
    if a_start == b_start and a_end == b_end:
        return "equal"
    return "overlap" if (a_start <= b_end and b_start <= a_end) else "disjoint"


def _is_estimate_modality(modality: str | None) -> bool:
    if modality is None:
        return False
    return any(marker in modality for marker in _ESTIMATE_MODALITY_MARKERS)


def _values_equal_raw(fact_a: Fact, fact_b: Fact) -> bool:
    return fact_a.value.raw.strip().lower() == fact_b.value.raw.strip().lower()


def _effective_value(fact: Fact) -> NormalizedValue | None:
    """normalize/units.py parses scale/currency embedded IN the raw text
    ("8,142 Cr"). A fact's scale/currency sometimes instead lives in a
    separate structured field (Value.scale/Value.currency) with no suffix
    in the raw text itself -- common for table cells, where the scale is
    stated once in the table caption, not repeated per cell. This fills
    that gap in rather than under-normalizing such facts."""
    parsed = normalize_value(fact.value.raw)
    if parsed is None:
        return None
    scale = parsed.scale
    multiplier = parsed.scale_multiplier
    if scale is None and fact.value.scale:
        scale = fact.value.scale.strip().lower()
        multiplier = SCALE_MULTIPLIERS.get(scale, 1.0)
    currency = parsed.currency
    if currency is None and fact.value.currency:
        currency = fact.value.currency.strip().upper()
    return dc_replace(
        parsed,
        scale=scale,
        scale_multiplier=multiplier,
        currency=currency,
        base_value=parsed.number * multiplier,
    )


def reconcile(fact_a: Fact, fact_b: Fact) -> Relation:
    """Public entry point: decides the verdict via _reconcile_decide(), then
    checks whether the pair was only linked through measure ALIASING
    (normalize/measures.py's canonicalize_measure, consulted inside
    claim_key_components — an alias match means "worth comparing," not
    "the same measure"). A resulting CORROBORATES gets its reason_code
    forced to SCALE_NORMALIZED plus a caveat recording which two surface
    forms were linked and that their definitions may differ — see
    docs/CASE_DOSSIER.md §1. CONTRADICTS/other verdicts on an aliased pair
    are left as rules.py already decided them; only a CORROBORATES needs
    the caveat, since that's the case that could otherwise be mistaken for
    "these two measures mean the same thing"."""
    relation = _reconcile_decide(fact_a, fact_b)
    if relation.type == RelationType.CORROBORATES and measures_aliased_not_identical(
        fact_a.measure.surface_form, fact_b.measure.surface_form
    ):
        caveat = (
            f"Linked via measure aliasing ({fact_a.measure.surface_form!r} ~ "
            f"{fact_b.measure.surface_form!r}) — not identical wording. The definitions "
            f"may differ (e.g. one may exclude a component the other includes); this "
            f"agreement is contingent, not guaranteed to hold in general."
        )
        return relation.model_copy(update={"reason_code": ReasonCode.SCALE_NORMALIZED, "caveat": caveat})
    return relation


def _reconcile_decide(fact_a: Fact, fact_b: Fact) -> Relation:
    comp_a = claim_key_components(fact_a)
    comp_b = claim_key_components(fact_b)

    if comp_a.subject_id != comp_b.subject_id or comp_a.measure_key != comp_b.measure_key:
        return _relation(
            fact_a, fact_b, RelationType.RECONCILED_BY_CONTEXT, ReasonCode.UNRESOLVED,
            "Facts describe different subjects or measures — not a comparable pair.",
        )

    # -- 2. temporal supersession --
    if not _intervals_overlap(fact_a.validity_interval, fact_b.validity_interval):
        if not _values_equal_raw(fact_a, fact_b):
            return _relation(
                fact_a, fact_b, RelationType.SUPERSEDES, ReasonCode.TEMPORAL_STATE_CHANGE,
                f"Non-overlapping validity intervals "
                f"([{fact_a.validity_interval.start}, {fact_a.validity_interval.end}) vs "
                f"[{fact_b.validity_interval.start}, {fact_b.validity_interval.end})) with a "
                f"changed value — the later fact supersedes the earlier one rather than "
                f"contradicting it.",
            )

    # -- 3. consolidation --
    # Only a CONFIRMED mismatch (both sides state a basis, and they
    # differ) counts here — one side simply not stating a consolidation
    # basis at all (common for an investor deck vs a statutory filing,
    # e.g. docs/CASE_DOSSIER.md §1's Q4 deck, which never uses the word
    # "consolidated") is silence, not evidence of disagreement. Treating
    # it as a mismatch would block every deck-vs-filing comparison at this
    # step before value comparison (and the caveated-CORROBORATES path for
    # aliased measures, see reconcile()) ever gets a chance to run.
    if comp_a.consolidation and comp_b.consolidation and comp_a.consolidation != comp_b.consolidation:
        return _relation(
            fact_a, fact_b, RelationType.RECONCILED_BY_CONTEXT, ReasonCode.SCOPE_MISMATCH,
            f"Consolidation differs: {comp_a.consolidation!r} vs {comp_b.consolidation!r} — "
            f"apparent disagreement explained by scope, not a real contradiction.",
        )

    # -- 4. geography --
    if comp_a.geography != comp_b.geography:
        return _relation(
            fact_a, fact_b, RelationType.RECONCILED_BY_CONTEXT, ReasonCode.GEOGRAPHY_MISMATCH,
            f"Geography differs: {comp_a.geography!r} vs {comp_b.geography!r}.",
        )

    # -- 5. period --
    period_a = (comp_a.period_start, comp_a.period_end)
    period_b = (comp_b.period_start, comp_b.period_end)
    period_relation = _period_relation(period_a, period_b)
    if period_relation == "disjoint":
        return _relation(
            fact_a, fact_b, RelationType.RECONCILED_BY_CONTEXT, ReasonCode.PERIOD_DISJOINT,
            f"Periods do not overlap: {period_a} vs {period_b}.",
        )
    if period_relation == "overlap":
        return _relation(
            fact_a, fact_b, RelationType.RECONCILED_BY_CONTEXT, ReasonCode.PERIOD_BASIS_MISMATCH,
            f"Periods overlap but are not identical after normalization: {period_a} vs "
            f"{period_b} — likely a fiscal-vs-calendar or vintage basis difference.",
        )

    # -- 6. modality / vintage --
    modality_a = normalize_qualifier(fact_a.qualifiers.get("modality"))
    modality_b = normalize_qualifier(fact_b.qualifiers.get("modality"))
    if modality_a != modality_b and (_is_estimate_modality(modality_a) or _is_estimate_modality(modality_b)):
        return _relation(
            fact_a, fact_b, RelationType.RECONCILED_BY_CONTEXT, ReasonCode.VINTAGE_RESTATEMENT,
            f"Modality differs ({modality_a!r} vs {modality_b!r}) and at least one side is an "
            f"estimate/advance figure — a later vintage restating an earlier one, not a "
            f"contradiction.",
        )

    # period_relation is "equal" or None (unresolved) at this point.
    if period_relation is None:
        return _relation(
            fact_a, fact_b, RelationType.RECONCILED_BY_CONTEXT, ReasonCode.UNRESOLVED,
            "Period could not be resolved on at least one side — routed to review.",
        )

    # -- 8. values --
    if fact_a.fact_type != "numeric" or fact_b.fact_type != "numeric":
        if _values_equal_raw(fact_a, fact_b):
            return _relation(
                fact_a, fact_b, RelationType.CORROBORATES, ReasonCode.VALUES_MATCH,
                "Non-numeric values are equal after normalization.",
            )
        return _relation(
            fact_a, fact_b, RelationType.CONTRADICTS, ReasonCode.VALUES_DIVERGE,
            f"Non-numeric values disagree: {fact_a.value.raw!r} vs {fact_b.value.raw!r}, "
            f"same subject/measure/period/qualifiers.",
        )

    va = _effective_value(fact_a)
    vb = _effective_value(fact_b)
    if va is None or vb is None:
        return _relation(
            fact_a, fact_b, RelationType.RECONCILED_BY_CONTEXT, ReasonCode.UNRESOLVED,
            "Value could not be parsed on at least one side — routed to review.",
        )

    if va.currency and vb.currency and va.currency != vb.currency:
        return _relation(
            fact_a, fact_b, RelationType.RECONCILED_BY_CONTEXT, ReasonCode.UNIT_MISMATCH,
            f"Currencies differ ({va.currency} vs {vb.currency}) — not comparable without FX "
            f"conversion, out of scope here.",
        )

    denom = max(abs(va.base_value), abs(vb.base_value), 1e-9)
    relative_delta = abs(va.base_value - vb.base_value) / denom
    delta = Delta(
        absolute=va.base_value - vb.base_value,
        relative=relative_delta,
        after_normalization=(va.scale != vb.scale) or (va.currency != vb.currency),
    )

    if relative_delta <= EXACT_RELATIVE_TOLERANCE:
        if delta.after_normalization:
            return _relation(
                fact_a, fact_b, RelationType.CORROBORATES, ReasonCode.SCALE_NORMALIZED,
                f"{va.base_value:,.2f} == {vb.base_value:,.2f} after scale normalization "
                f"({va.scale or 'none'} vs {vb.scale or 'none'}).",
                delta=delta,
            )
        return _relation(
            fact_a, fact_b, RelationType.CORROBORATES, ReasonCode.VALUES_MATCH,
            f"{va.base_value:,.2f} == {vb.base_value:,.2f}, same scale/currency.",
            delta=delta,
        )

    if relative_delta <= ROUNDING_RELATIVE_TOLERANCE:
        return _relation(
            fact_a, fact_b, RelationType.CORROBORATES, ReasonCode.ROUNDING_ARTIFACT,
            f"{va.base_value:,.2f} vs {vb.base_value:,.2f} — {relative_delta:.4%} apart, "
            f"within the defended rounding tolerance of {ROUNDING_RELATIVE_TOLERANCE:.2%}.",
            delta=delta,
        )

    return _relation(
        fact_a, fact_b, RelationType.CONTRADICTS, ReasonCode.VALUES_DIVERGE,
        f"{va.base_value:,.2f} vs {vb.base_value:,.2f} — {relative_delta:.2%} apart, beyond "
        f"the {ROUNDING_RELATIVE_TOLERANCE:.2%} rounding tolerance, same subject/measure/"
        f"period/qualifiers.",
        delta=delta,
    )
