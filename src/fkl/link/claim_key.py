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
from src.fkl.normalize.measures import canonicalize_measure
from src.fkl.normalize.periods import parse_period
from src.fkl.store.models import Fact


def normalize_qualifier(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def fact_period(fact: Fact) -> tuple[date | None, date | None]:
    """Resolves the fact's period to a concrete (start, end) interval, most
    precise available signal first:

    1. a `date` qualifier -- a single calendar date, strictly more precise
       than any range. Folded in as a same-day (date, date) interval so
       two facts that share a fiscal-year period_label but sit on
       different specific dates (e.g. a monthly ESOP-exercise table: 12
       rows, one per month, all carrying period_label "FY2023-24" but each
       with its own `date`) no longer collapse onto the SAME claim key and
       get diffed against each other as if they were repeated measurements
       of one thing -- see docs/LIMITATIONS.md's 105/107 false-CONTRADICTS
       incident. Distinct dates now correctly resolve as non-overlapping
       periods (PERIOD_DISJOINT in reconcile/rules.py), not a value
       disagreement.
    2. already-resolved period_start/period_end qualifiers (set
       deterministically for table facts by extractor.py's
       _deterministic_qualifiers_from_header) -- a resolved range, more
       precise than a label still needing parsing.
    3. period_label, parsed through normalize/periods.py.

    Returns (None, None) if nothing above is present or parseable."""
    q = fact.qualifiers
    date_raw = q.get("date")
    if date_raw:
        try:
            d = date.fromisoformat(str(date_raw))
            return d, d
        except ValueError:
            pass
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
        # canonicalize_measure folds a KNOWN alias group (e.g. "Revenue for
        # services" / "Revenue from customers" -> "Revenue from Operations",
        # see normalize/measures.py) onto one measure_key so aliased facts
        # become CANDIDATES for reconciliation — but aliasing is not the
        # same as identity: reconcile/rules.py checks
        # measures_aliased_not_identical() separately and must caveat any
        # resulting CORROBORATES, never treat the match as unqualified.
        measure_key=canonicalize_measure(fact.measure.surface_form),
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
