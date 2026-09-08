"""Canonicalize measure surface forms that are semantically RELATED but not
identical, for cross-document linking -- mirrors normalize/entities.py's
alias-table pattern, with one deliberate difference: an alias match here
never means "these are the same measure," only "these are worth comparing."

docs/CASE_DOSSIER.md §1 is the reason this exists: "Revenue from Operations"
(the annual report's statutory line item) and "Revenue for services" /
"Revenue from customers" (the Q4 earnings deck's framing) agree in value for
FY24 only because traded-goods revenue was ~0 that year -- the deck's own
footnote says revenue from services EXCLUDES revenue from traded goods, a
genuinely narrower definition. link/claim_key.py's normalize_measure()
deliberately does NOT fold these together for exact-match purposes; this
module is the explicit, caveated bridge reconcile/rules.py uses instead --
see measures_aliased_not_identical(), consumed there to force a caveat
rather than a silent CORROBORATES whenever an alias (not literal identity)
is what linked the pair.
"""

from __future__ import annotations

import re
import unicodedata

# canonical_id -> every surface form known to refer to the same underlying
# concept, even where the definitions aren't strictly identical (see
# module docstring). Grows by adding entries, not by retraining anything.
MEASURE_ALIASES: dict[str, list[str]] = {
    "revenue_from_operations": [
        "Revenue from Operations",
        "Revenue for services",
        "Revenue for services (A)",
        "Revenue from customers",
        "Revenue from customers (A+B)",
    ],
}


def _normalize_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"\s+", " ", text)


_ALIAS_LOOKUP: dict[str, str] = {
    _normalize_key(alias): canonical_id
    for canonical_id, aliases in MEASURE_ALIASES.items()
    for alias in aliases
}


def canonicalize_measure(surface_form: str) -> str:
    """Returns a canonical id for surface_form: the alias group's id if
    it's a known alias, else a slugified version of the normalized surface
    form itself -- same fallback contract as normalize/entities.py's
    canonicalize(), so an unaliased measure still gets a stable id rather
    than no id at all."""
    key = _normalize_key(surface_form)
    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]
    slug = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return slug or "unknown_measure"


def measures_aliased_not_identical(measure_a: str, measure_b: str) -> bool:
    """True when measure_a and measure_b are linked ONLY through the alias
    table -- same alias group, but not the same normalized text. False if
    they're literally identical (nothing to caveat) or unrelated (they
    won't share a measure_key at all, via canonicalize_measure's fallback,
    so reconcile/rules.py would never see them as a candidate pair in the
    first place). Callers use this to decide whether a resulting
    CORROBORATES needs a caveat: the values may agree, but the definitions
    underneath them might not."""
    key_a, key_b = _normalize_key(measure_a), _normalize_key(measure_b)
    if key_a == key_b:
        return False
    canon_a, canon_b = _ALIAS_LOOKUP.get(key_a), _ALIAS_LOOKUP.get(key_b)
    return canon_a is not None and canon_a == canon_b
