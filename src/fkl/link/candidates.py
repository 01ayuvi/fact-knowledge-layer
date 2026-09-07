"""Retrieval-bounded candidate-pair generation: exact claim-key matches
first, then loose-key matches. No embeddings -- string/key matching only
(see docs/SUPERJOIN_BUILD_PLAN.md: "candidate linking is retrieval-bounded,
not O(n^2)").

"fact store" here is an in-memory list -- src/fkl/store/repo.py (a real
persistent store) doesn't exist yet. The indexing approach carries over
unchanged once it does: swap building exact_index/loose_index in Python
for a query that returns facts sharing a claim_key/loose_key, computed by
the store. What must NOT change is the shape of the result: bounded
same-key groups, never a full pairwise scan of the corpus.
"""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from src.fkl.link.claim_key import claim_key, loose_key
from src.fkl.store.models import Fact

logger = logging.getLogger(__name__)

# A same-key group larger than this is truncated rather than fully
# expanded -- a single very common (subject, measure) pair should not be
# allowed to blow up into O(k^2) pairs unbounded by corpus size.
DEFAULT_MAX_GROUP_SIZE = 50


@dataclass(frozen=True)
class CandidatePair:
    fact_a: Fact
    fact_b: Fact
    match_level: Literal["exact", "loose"]


def _pairs_from_groups(
    index: dict[str, list[Fact]],
    match_level: Literal["exact", "loose"],
    max_group_size: int,
    already_paired: set[frozenset[str]],
) -> list[CandidatePair]:
    pairs: list[CandidatePair] = []
    for key, group in index.items():
        if len(group) < 2:
            continue
        if len(group) > max_group_size:
            logger.warning(
                "%s-match group %r has %d facts, exceeding max_group_size=%d; truncating",
                match_level, key, len(group), max_group_size,
            )
            group = group[:max_group_size]
        for a, b in itertools.combinations(group, 2):
            pair_ids = frozenset((a.id, b.id))
            if pair_ids in already_paired:
                continue
            pairs.append(CandidatePair(a, b, match_level))
            already_paired.add(pair_ids)
    return pairs


def find_candidates(
    facts: list[Fact], max_group_size: int = DEFAULT_MAX_GROUP_SIZE
) -> list[CandidatePair]:
    """facts stands in for "a fact store" per the module docstring. Builds
    two indexes in O(n) (n = len(facts)), then only pairwise-compares
    within a shared key's group -- never facts that share no key at all."""
    exact_index: dict[str, list[Fact]] = defaultdict(list)
    loose_index: dict[str, list[Fact]] = defaultdict(list)
    for f in facts:
        exact_index[claim_key(f)].append(f)
        loose_index[loose_key(f)].append(f)

    seen: set[frozenset[str]] = set()
    exact_pairs = _pairs_from_groups(exact_index, "exact", max_group_size, seen)
    loose_pairs = _pairs_from_groups(loose_index, "loose", max_group_size, seen)
    return exact_pairs + loose_pairs
