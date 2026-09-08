"""normalize/measures.py tests."""

from __future__ import annotations

from src.fkl.normalize.measures import canonicalize_measure, measures_aliased_not_identical


def test_known_aliases_share_canonical_id():
    assert canonicalize_measure("Revenue from Operations") == canonicalize_measure("Revenue for services (A)")
    assert canonicalize_measure("Revenue from Operations") == canonicalize_measure("Revenue from customers (A+B)")


def test_unrelated_measure_gets_stable_fallback_slug():
    assert canonicalize_measure("PAT") == canonicalize_measure("PAT")
    assert canonicalize_measure("PAT") != canonicalize_measure("Revenue from Operations")


def test_aliased_not_identical_true_only_for_alias_group_with_differing_wording():
    assert measures_aliased_not_identical("Revenue from Operations", "Revenue for services (A)") is True
    # identical wording (even differently-cased/spaced) is NOT an alias link -- nothing to caveat
    assert measures_aliased_not_identical("Revenue from Operations", "revenue   from operations") is False
    # unrelated measures never alias-link at all
    assert measures_aliased_not_identical("Revenue from Operations", "PAT") is False
