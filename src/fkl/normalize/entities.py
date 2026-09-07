"""Canonicalize entity surface forms to a stable id. Pure Python, no LLM
calls, no embeddings -- an explicit alias table plus light string
normalization, per docs/SUPERJOIN_BUILD_PLAN.md's measure/entity registry
brownie point: this grows by adding alias entries, not by retraining
anything.

"the Company" / "your Company" are ambiguous on their own -- they only
resolve to a specific entity given a document's subject context (the
filing's own company). canonicalize() takes that as doc_subject.
"""

from __future__ import annotations

import re
import unicodedata

# canonical_id -> every surface form known to refer to it (lowercased,
# whitespace-normalized forms are matched against this after normalization;
# see _normalize_key).
ALIASES: dict[str, list[str]] = {
    "delhivery_limited": [
        "Delhivery Limited",
        "Delhivery Ltd",
        "Delhivery Ltd.",
        "Delhivery",
    ],
    "kalpana_jaisingh_morparia": [
        "Kalpana Jaisingh Morparia",
        "Ms. Kalpana Jaisingh Morparia",
        "Ms Kalpana Jaisingh Morparia",
    ],
    "munish_ravinder_varma": [
        "Munish Ravinder Varma",
        "Mr. Munish Ravinder Varma",
    ],
    "agus_tandiono": [
        "Agus Tandiono",
        "Mr. Agus Tandiono",
    ],
}

# Generic referring phrases that mean "whichever company this document is
# about" -- not a name, so they can't get their own alias entry above.
_SELF_REFERENCE_PHRASES = {"the company", "your company", "our company"}

_HONORIFIC_RE = re.compile(r"^(mr|mrs|ms|dr|messrs)\.?\s+", re.IGNORECASE)
_SUFFIX_RE = re.compile(r"\b(limited|ltd)\.?\s*$", re.IGNORECASE)


def _normalize_key(text: str) -> str:
    """NFKC-normalize, lowercase, strip honorifics/corporate suffixes,
    collapse whitespace -- so "Delhivery Ltd." and "delhivery limited"
    (and "Ms. Kalpana Jaisingh Morparia" vs "Kalpana Jaisingh Morparia")
    land on the same key before alias lookup."""
    text = unicodedata.normalize("NFKC", text).strip()
    text = _HONORIFIC_RE.sub("", text)
    text = _SUFFIX_RE.sub("", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


_ALIAS_LOOKUP: dict[str, str] = {
    _normalize_key(alias): canonical_id
    for canonical_id, aliases in ALIASES.items()
    for alias in aliases
}


def canonicalize(surface_form: str, doc_subject: str | None = None) -> str:
    """Returns a canonical_id for surface_form. Falls back to a slugified
    version of the (normalized) surface form itself when it isn't in the
    alias table -- so an unknown entity still gets a stable, deterministic
    id rather than no id at all; add it to ALIASES once a second surface
    form for it shows up.

    doc_subject: the canonical_id (or a surface form -- either works) of
    "the Company" for the document being processed, used to resolve
    self-referential phrases like "the Company" / "your Company".
    """
    key = _normalize_key(surface_form)

    if key in _SELF_REFERENCE_PHRASES:
        if doc_subject is None:
            return "unknown_self_reference"
        if doc_subject in ALIASES:
            return doc_subject
        return canonicalize(doc_subject)

    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]

    slug = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return slug or "unknown"
