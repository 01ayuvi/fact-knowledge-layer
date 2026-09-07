"""Fuzzy grounding verification: does a Fact's evidence_quote actually
appear in its source block's text?

extractor.py used to answer this with a strict str.find() (_locate_quote) --
an exact match or nothing, so a single unicode dash, ligature, or extra
space between the LLM's quote and the PDF's real text would sink an
otherwise-correct fact with no record of why. This gate normalizes both
sides (unicode NFKC, dash variants -> "-", whitespace runs -> a single
space) before fuzzy-matching with rapidfuzz's partial_ratio, and maps the
match's position back to the ORIGINAL (unnormalized) text so char_start/
char_end stay exact -- not approximate, and not off because normalization
changed the string's length (a ligature expands one character into two).

On pass: char_start/char_end on the fact's Evidence are corrected to the
real match position. bbox is carried through unchanged -- that geometry
came from wherever the fact was built (pdf.py's block bbox, or tables.py's
per-cell bbox), not from this gate, which only ever sees flattened text and
has no span-level geometry of its own to recompute it from.

On fail: the fact is quarantined with a reason and a score, not dropped --
see GroundingGate.quarantined. "0 facts in this store are ungrounded, by
construction" (docs/SUPERJOIN_BUILD_PLAN.md D1) means every fact that
reaches a caller's result either passed this gate or is sitting in the
quarantine list with an explanation, never silently discarded.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from src.fkl.store.models import Fact

DEFAULT_THRESHOLD = 92.0

# Hyphen, non-breaking hyphen, figure dash, en dash, em dash, horizontal
# bar, minus sign, soft hyphen -- all fold to a plain "-".
_DASH_CODEPOINTS = {
    0x2010,
    0x2011,
    0x2012,
    0x2013,
    0x2014,
    0x2015,
    0x2212,
    0x00AD,
}


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """NFKC-normalize (folds ligatures: "ﬁ" -> "fi"), canonicalize dash
    variants to "-", and collapse whitespace runs to a single space -- while
    tracking, per output character, which original index it came from,
    since none of those transforms are guaranteed length-preserving."""
    norm_chars: list[str] = []
    pos_map: list[int] = []
    prev_was_space = False
    for i, ch in enumerate(text):
        piece = "-" if ord(ch) in _DASH_CODEPOINTS else unicodedata.normalize("NFKC", ch)
        for out_ch in piece:
            if out_ch.isspace():
                if prev_was_space:
                    continue
                out_ch = " "
                prev_was_space = True
            else:
                prev_was_space = False
            norm_chars.append(out_ch)
            pos_map.append(i)
    return "".join(norm_chars), pos_map


@dataclass
class QuarantineEntry:
    fact: Fact
    reason: str
    score: float


@dataclass
class GroundingGate:
    """Stateful across a run: construct one, call verify() per candidate
    fact, then read verified_count / quarantined_count / quarantined for a
    summary."""

    threshold: float = DEFAULT_THRESHOLD
    verified_count: int = 0
    quarantined: list[QuarantineEntry] = field(default_factory=list)

    @property
    def quarantined_count(self) -> int:
        return len(self.quarantined)

    def verify(self, fact: Fact, block_text: str) -> bool:
        """Mutates fact.evidence.char_start/char_end in place and returns
        True on pass. On fail, appends a QuarantineEntry to self.quarantined
        (fact left untouched) and returns False -- the caller must not add
        this fact to its result."""
        quote = fact.evidence.quote
        if not quote.strip():
            self._quarantine(fact, "empty evidence_quote", 0.0)
            return False

        quote_norm, _ = _normalize_with_map(quote)
        block_norm, block_map = _normalize_with_map(block_text)
        if not quote_norm or not block_norm:
            self._quarantine(fact, "quote or block empty after normalization", 0.0)
            return False

        alignment = fuzz.partial_ratio_alignment(quote_norm, block_norm)
        if alignment.score < self.threshold:
            self._quarantine(
                fact,
                f"partial_ratio {alignment.score:.1f} below threshold {self.threshold:.0f}",
                alignment.score,
            )
            return False

        fact.evidence.char_start = block_map[alignment.dest_start]
        fact.evidence.char_end = block_map[alignment.dest_end - 1] + 1
        self.verified_count += 1
        return True

    def quarantine(self, fact: Fact, reason: str, score: float = 0.0) -> None:
        """For a caller-side failure that never reaches verify() at all —
        e.g. no source block text could be found for this fact — so it
        still ends up recorded here rather than silently dropped."""
        self._quarantine(fact, reason, score)

    def _quarantine(self, fact: Fact, reason: str, score: float) -> None:
        self.quarantined.append(QuarantineEntry(fact=fact, reason=reason, score=score))
