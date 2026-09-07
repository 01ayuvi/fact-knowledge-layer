"""Normalize a raw numeric-value string to a canonical base value. Pure
Python, no LLM calls.

"Canonical" here means: absolute units (no crore/million/etc multiplier
folded in), with the raw scale/currency preserved alongside so two figures
expressed at different scales (₹81,415.38 million vs ₹8,142 Cr) become
directly comparable after normalization -- see docs/CASE_DOSSIER.md §1,
which is also why the test for that pair asserts approximate, not exact,
equality: the source documents themselves round to different precision
("81,415.38 mn / 10 = 8,141.54 Cr vs stated 8,142 Cr -> delta 0.006%"), and
pretending otherwise would misrepresent real data as a bug.

"percent" is handled as its own scale, not a currency amount: a value like
"1.6%" normalizes to base_value == number (1.6), with currency=None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SCALE_MULTIPLIERS: dict[str, float] = {
    "thousand": 1_000,
    "k": 1_000,
    "lakh": 100_000,
    "million": 1_000_000,
    "mn": 1_000_000,
    "crore": 10_000_000,
    "cr": 10_000_000,
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
}

_CURRENCY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"₹"), "INR"),
    (re.compile(r"\bRs\.?\b", re.IGNORECASE), "INR"),
    (re.compile(r"\bINR\b", re.IGNORECASE), "INR"),
    (re.compile(r"\$"), "USD"),
    (re.compile(r"\bUSD\b", re.IGNORECASE), "USD"),
    (re.compile(r"€"), "EUR"),
    (re.compile(r"\bEUR\b", re.IGNORECASE), "EUR"),
    (re.compile(r"£"), "GBP"),
    (re.compile(r"\bGBP\b", re.IGNORECASE), "GBP"),
]

_CANONICAL_SCALE_NAME = {
    "thousand": "thousand",
    "k": "thousand",
    "lakh": "lakh",
    "million": "million",
    "mn": "million",
    "crore": "crore",
    "cr": "crore",
    "billion": "billion",
    "bn": "billion",
}

_NUMBER_RE = re.compile(r"\(?-?[\d,]+(?:\.\d+)?\)?")
# Not \b on the left: a digit immediately followed by a letter ("10K",
# "81,415.38mn") is NOT a regex word boundary, since digits and letters are
# both \w -- so \b would silently fail to match this extremely common
# compact notation. (?<![a-zA-Z]) / (?![a-zA-Z]) allow a digit neighbor
# while still rejecting a match inside an unrelated word.
_SCALE_WORD_RE = re.compile(
    r"(?<![a-zA-Z])(" + "|".join(sorted(SCALE_MULTIPLIERS, key=len, reverse=True)) + r")(?![a-zA-Z])",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"%|\bper\s*cent\b|\bpercent\b", re.IGNORECASE)


@dataclass(frozen=True)
class NormalizedValue:
    raw: str
    number: float
    scale: str | None
    scale_multiplier: float
    currency: str | None
    base_value: float


def normalize_value(raw: str) -> NormalizedValue | None:
    """Returns None if no numeric token can be found in raw."""
    text = raw.strip()

    m = _NUMBER_RE.search(text)
    if not m:
        return None
    token = m.group(0)
    negative = token.startswith("(") and token.endswith(")")
    cleaned = token.strip("()").replace(",", "")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if negative:
        number = -number

    if _PERCENT_RE.search(text):
        return NormalizedValue(
            raw=raw, number=number, scale="percent", scale_multiplier=1.0,
            currency=None, base_value=number,
        )

    currency: str | None = None
    for pattern, code in _CURRENCY_PATTERNS:
        if pattern.search(text):
            currency = code
            break

    scale: str | None = None
    multiplier = 1.0
    if sm := _SCALE_WORD_RE.search(text):
        matched = sm.group(1).lower()
        scale = _CANONICAL_SCALE_NAME[matched]
        multiplier = SCALE_MULTIPLIERS[matched]

    return NormalizedValue(
        raw=raw,
        number=number,
        scale=scale,
        scale_multiplier=multiplier,
        currency=currency,
        base_value=number * multiplier,
    )
