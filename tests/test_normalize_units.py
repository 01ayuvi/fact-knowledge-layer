"""units.py tests, grounded in docs/CASE_DOSSIER.md §1's Delhivery revenue
figures -- both the near-match pair (real rounding, ~0.006% apart) and the
exact-match pair (the dossier explicitly calls "Exact.")."""

import pytest

from src.fkl.normalize.units import normalize_value


def test_million_scale():
    v = normalize_value("₹81,415.38 million")
    assert v.number == pytest.approx(81415.38)
    assert v.scale == "million"
    assert v.currency == "INR"
    assert v.base_value == pytest.approx(81_415_380_000)


def test_crore_scale():
    v = normalize_value("₹8,142 Cr")
    assert v.number == pytest.approx(8142)
    assert v.scale == "crore"
    assert v.currency == "INR"
    assert v.base_value == pytest.approx(81_420_000_000)


def test_million_and_crore_normalize_to_near_equal_base_critical():
    """docs/CASE_DOSSIER.md §1: 'Annual Report FY24 p22: Revenue from
    Operations ... 81,415.38' (₹ in Million, Consolidated, FY24) vs 'Q4
    FY24 earnings deck p6: ₹8,142 Cr FY24 revenue from services'.
    '81,415.38 mn / 10 = 8,141.54 Cr vs stated 8,142 Cr -> delta 0.006%.'
    Requiring bit-exact equality here would misrepresent real rounding in
    the source documents as a bug -- the dossier itself frames the correct
    tolerance as an engineering decision to defend, not zero."""
    ar = normalize_value("₹81,415.38 million")
    deck = normalize_value("₹8,142 Cr")
    relative_delta = abs(ar.base_value - deck.base_value) / deck.base_value
    assert relative_delta < 0.001  # dossier's own observed delta is ~0.0057%
    assert relative_delta > 0  # and it should be a real, non-zero rounding gap


def test_million_and_crore_exact_match_prior_year():
    """docs/CASE_DOSSIER.md §1: 'Second instance, same measure, prior
    year: AR FY23 consolidated = 72,253.01 mn = 7,225.3 Cr; deck p14
    table = 7,225 Cr. Exact.'"""
    ar = normalize_value("72,253.01 mn")
    deck = normalize_value("7,225.3 Cr")
    assert ar.base_value == pytest.approx(deck.base_value, rel=1e-4)


def test_lakh_thousand_billion_k_bn_mn_cr():
    assert normalize_value("5 lakh").base_value == pytest.approx(500_000)
    assert normalize_value("3 thousand").base_value == pytest.approx(3_000)
    assert normalize_value("2 billion").base_value == pytest.approx(2_000_000_000)
    assert normalize_value("10K").base_value == pytest.approx(10_000)
    assert normalize_value("1.5bn").base_value == pytest.approx(1_500_000_000)
    assert normalize_value("81,415.38mn").base_value == pytest.approx(81_415_380_000)
    assert normalize_value("8,142Cr").base_value == pytest.approx(81_420_000_000)


def test_percent_symbol():
    """docs/CASE_DOSSIER.md §3(c): 'GDP at constant prices grew 6.5% in
    2024-25.'"""
    v = normalize_value("6.5%")
    assert v.scale == "percent"
    assert v.currency is None
    assert v.base_value == pytest.approx(6.5)


def test_percent_word_form():
    """docs/CASE_DOSSIER.md §1: RBI AR p8, 'moderated to 6.5 per cent in
    2024-25.'"""
    v = normalize_value("6.5 per cent")
    assert v.scale == "percent"
    assert v.base_value == pytest.approx(6.5)


def test_negative_parenthesized_value():
    v = normalize_value("(1,679.68)")
    assert v.number == pytest.approx(-1679.68)
    assert v.base_value == pytest.approx(-1679.68)


def test_currency_word_form_and_footnote_figure():
    """docs/CASE_DOSSIER.md §5(c) footnote: 'Revenue from Cross Border
    Services in FY22 included freight revenue of Rs 46 Cr from ...
    traded goods.'"""
    v = normalize_value("Rs 46 Cr")
    assert v.currency == "INR"
    assert v.base_value == pytest.approx(460_000_000)


def test_bare_number_no_scale_no_currency():
    v = normalize_value("1,600")
    assert v.number == pytest.approx(1600)
    assert v.scale is None
    assert v.currency is None
    assert v.base_value == pytest.approx(1600)


def test_raw_preserved():
    v = normalize_value("₹8,142 Cr")
    assert v.raw == "₹8,142 Cr"


def test_no_number_returns_none():
    assert normalize_value("no digits here") is None
