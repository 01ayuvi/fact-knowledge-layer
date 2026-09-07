"""Fact / Relation schema, per docs/SUPERJOIN_BUILD_PLAN.md §4.

Pydantic v2 models only — no persistence or query logic here.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

FactType = Literal["numeric", "temporal", "categorical", "relational", "assertion"]
Adjudicator = Literal["rule", "llm", "rule+llm"]


class RelationType(str, Enum):
    CORROBORATES = "CORROBORATES"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDES = "SUPERSEDES"
    RECONCILED_BY_CONTEXT = "RECONCILED_BY_CONTEXT"
    REFINES = "REFINES"


class ReasonCode(str, Enum):
    """Closed vocabulary from the plan's §2 D3 table, minus CORROBORATES and
    CONTRADICTS (those are RelationTypes — the verdict — not reasons; a
    reason_code must never just restate the relation's own type), plus
    additions: VALUES_MATCH / SCALE_NORMALIZED (why a CORROBORATES relation
    holds — raw match vs. match only after scale normalization),
    VALUES_DIVERGE (why a CONTRADICTS relation holds when no explanatory
    dimension — scale, unit, period, scope, geography, modality, vintage —
    accounts for the difference), TEMPORAL_STATE_CHANGE (Morparia-style
    validity-interval supersession, from docs/CASE_DOSSIER.md), and
    ROUNDING_ARTIFACT (sub-tolerance numeric drift, e.g. the FY23 7,224 vs
    7,225 Cr split, also from the dossier)."""

    VALUES_MATCH = "VALUES_MATCH"
    SCALE_NORMALIZED = "SCALE_NORMALIZED"
    SCALE_MISMATCH = "SCALE_MISMATCH"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    PERIOD_DISJOINT = "PERIOD_DISJOINT"
    PERIOD_BASIS_MISMATCH = "PERIOD_BASIS_MISMATCH"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    GEOGRAPHY_MISMATCH = "GEOGRAPHY_MISMATCH"
    MODALITY_MISMATCH = "MODALITY_MISMATCH"
    VINTAGE_RESTATEMENT = "VINTAGE_RESTATEMENT"
    VALUES_DIVERGE = "VALUES_DIVERGE"
    UNRESOLVED = "UNRESOLVED"
    TEMPORAL_STATE_CHANGE = "TEMPORAL_STATE_CHANGE"
    ROUNDING_ARTIFACT = "ROUNDING_ARTIFACT"


class Entity(BaseModel):
    canonical_id: str
    surface_form: str
    type: str | None = None


class Measure(BaseModel):
    canonical_id: str
    surface_form: str


class Value(BaseModel):
    raw: str
    number: float | None = None
    unit: str | None = None
    currency: str | None = None
    scale: str | None = None
    direction: str | None = None


class ValidityInterval(BaseModel):
    """Half-open interval [start, end) over which the fact holds. Either end
    left open (None) means unbounded in that direction."""

    start: date | None = None
    end: date | None = None


class Evidence(BaseModel):
    page_no: int
    char_start: int
    char_end: int
    bbox: list[tuple[float, float, float, float]] = Field(default_factory=list)
    quote: str
    section_path: list[str] = Field(default_factory=list)
    table_ref: str | None = None


class Confidence(BaseModel):
    extraction: float
    normalization: float | None = None


class Provenance(BaseModel):
    model: str
    prompt_version: str
    extracted_at: datetime
    parser: str


class Fact(BaseModel):
    id: str
    doc_id: str
    block_id: str
    subject: Entity
    measure: Measure
    fact_type: FactType
    value: Value
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    validity_interval: ValidityInterval = Field(default_factory=ValidityInterval)
    evidence: Evidence
    confidence: Confidence
    provenance: Provenance

    def claim_key(self) -> str:
        """hash(subject.canonical_id, measure.canonical_id, period_norm,
        consolidation, geography) — see plan §4. period_norm is derived from
        the qualifiers bag (period_start/period_end/period_basis) since
        qualifiers is an open dict, not a fixed field."""
        period_norm = (
            self.qualifiers.get("period_start"),
            self.qualifiers.get("period_end"),
            self.qualifiers.get("period_basis"),
        )
        parts = (
            self.subject.canonical_id,
            self.measure.canonical_id,
            repr(period_norm),
            str(self.qualifiers.get("consolidation")),
            str(self.qualifiers.get("geography")),
        )
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


class Delta(BaseModel):
    absolute: float | None = None
    relative: float | None = None
    after_normalization: bool = False


class Relation(BaseModel):
    id: str
    fact_a_id: str
    fact_b_id: str
    type: RelationType
    reason_code: ReasonCode
    explanation: str
    delta: Delta | None = None
    confidence: float
    adjudicator: Adjudicator
