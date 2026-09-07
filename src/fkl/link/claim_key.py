"""Build claim keys from Facts using the real normalizers (periods.py,
units.py, entities.py) -- NOT Fact.claim_key() on the Pydantic model
itself (src/fkl/store/models.py), which just hashes whatever canonical_id
extraction happened to assign (currently a crude slugify — see
extractor.py's _slugify). This module re-derives canonical subject and a
resolved period interval from the Fact's raw surface forms and qualifiers,
which is what "canonical" is supposed to mean for linking purposes.

exact claim key: subject + measure + resolved period interval +
consolidation + geography -- facts sharing this key describe the literal
same claim and should be directly value-compared.

loose key: subject + measure only -- facts sharing this (but not the exact
key) are CANDIDATES whose qualifiers need to be reconciled (see
reconcile/rules.py), not assumed equal.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import date

from src.fkl.normalize.entities import canonicalize
from src.fkl.normalize.periods import parse_period
from src.fkl.store.models import Fact


def normalize_measure(surface_form: str) -> str:
    """No measures.py normalizer exists yet (out of scope here) -- light
    string normalization only (NFKC, lowercase, collapse whitespace).
    Deliberately does NOT alias "revenue from operations" to "revenue from
    services": those genuinely differ (docs/CASE_DOSSIER.md §1's explicit
    trap -- they only coincide because FY24 traded-goods revenue was ~0),
    and silently merging them here would hide that instead of surfacing it
    as the caveated CORROBORATES the dossier calls for."""
    text = unicodedata.normalize("NFKC", surface_form).strip().lower()
    return re.sub(r"\s+", " ", text)


def normalize_qualifier(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def fact_period(fact: Fact) -> tuple[date | None, date | None]:
    """Resolves the fact's period to a concrete (start, end) interval.
    Prefers already-resolved period_start/period_end qualifiers (set
    deterministically for table facts by
    extractor.py's _deterministic_qualifiers_from_header); falls back to
    parsing period_label through normalize/periods.py. Returns (None,
    None) if neither is present or parseable."""
    q = fact.qualifiers
    start_raw, end_raw = q.get("period_start"), q.get("period_end")
    if start_raw and end_raw:
        try:
            return date.fromisoformat(str(start_raw)), date.fromisoformat(str(end_raw))
        except ValueError:
            pass
    label = q.get("period_label")
    if label:
        period = parse_period(str(label))
        if period is not None:
            return period.start, period.end
    return None, None


@dataclass(frozen=True)
class ClaimKeyComponents:
    subject_id: str
    measure_key: str
    period_start: date | None
    period_end: date | None
    consolidation: str | None
    geography: str | None


def claim_key_components(fact: Fact) -> ClaimKeyComponents:
    period_start, period_end = fact_period(fact)
    return ClaimKeyComponents(
        subject_id=canonicalize(fact.subject.surface_form),
        measure_key=normalize_measure(fact.measure.surface_form),
        period_start=period_start,
        period_end=period_end,
        consolidation=normalize_qualifier(fact.qualifiers.get("consolidation")),
        geography=normalize_qualifier(fact.qualifiers.get("geography")),
    )


def _hash(*parts: object) -> str:
    joined = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def claim_key(fact: Fact) -> str:
    c = claim_key_components(fact)
    return _hash(c.subject_id, c.measure_key, c.period_start, c.period_end, c.consolidation, c.geography)


def loose_key(fact: Fact) -> str:
    c = claim_key_components(fact)
    return _hash(c.subject_id, c.measure_key)
