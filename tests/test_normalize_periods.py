"""periods.py tests. The two equivalence tests marked CRITICAL are the
exact scenarios docs/CASE_DOSSIER.md calls out as the cases that prove the
period normalizer is real, not string pattern-matching."""

from datetime import date

from src.fkl.normalize.periods import parse_period


def test_fy24():
    p = parse_period("FY24")
    assert (p.start, p.end, p.basis) == (date(2023, 4, 1), date(2024, 3, 31), "FY_IN")


def test_fy2023_24_matches_fy24():
    a = parse_period("FY2023-24")
    b = parse_period("FY24")
    assert (a.start, a.end, a.basis) == (b.start, b.end, b.basis)


def test_fy2024_25():
    p = parse_period("FY2024/25")
    assert (p.start, p.end, p.basis) == (date(2024, 4, 1), date(2025, 3, 31), "FY_IN")


def test_bare_2024_25_matches_fy2024_25_critical():
    """docs/CASE_DOSSIER.md §1: RBI AR p8 says 'moderated to 6.5 per cent
    in 2024-25'; IMF Article IV p10 says 'grew by 6.5 percent in
    FY2024/25'. 'Same value, different period label (2024-25 vs
    FY2024/25) ... Proving these are the same period is what your
    periods.py normalizer is for.'"""
    rbi = parse_period("2024-25")
    imf = parse_period("FY2024/25")
    assert (rbi.start, rbi.end) == (imf.start, imf.end)
    assert rbi.basis == imf.basis == "FY_IN"


def test_q4_fy24():
    p = parse_period("Q4 FY24")
    assert (p.start, p.end, p.basis) == (date(2024, 1, 1), date(2024, 3, 31), "QUARTER")


def test_q1_fy2025_26_matches_calendar_2025q2_critical():
    """docs/CASE_DOSSIER.md §3(d), the exact required demo case: IMF p3
    says 'real GDP expanded by 7.8 percent in the first quarter of
    FY2025/26'; IMF p10 says 'real GDP growth of 7.8 percent in 2025Q2'.
    'Q1 of Indian FY2025/26 (Apr-Jun 2025) IS calendar 2025Q2. A system
    that resolves this proves its period normalizer genuinely understands
    the Apr-Mar fiscal convention.'"""
    fy_quarter = parse_period("Q1 FY2025/26")
    cy_quarter = parse_period("2025Q2")
    assert (fy_quarter.start, fy_quarter.end) == (cy_quarter.start, cy_quarter.end)
    assert fy_quarter.start == date(2025, 4, 1)
    assert fy_quarter.end == date(2025, 6, 30)


def test_cy2024():
    p = parse_period("CY2024")
    assert (p.start, p.end, p.basis) == (date(2024, 1, 1), date(2024, 12, 31), "CY")


def test_nine_months_ended_december_31_2021():
    p = parse_period("nine months ended December 31, 2021")
    assert (p.start, p.end) == (date(2021, 4, 1), date(2021, 12, 31))
    assert p.basis == "FY_IN"  # starts April 1 -- fiscal-year-aligned


def test_as_of_march_31_2024():
    p = parse_period("as of March 31, 2024")
    assert p.start == p.end == date(2024, 3, 31)


def test_unparseable_returns_none():
    assert parse_period("sometime next quarter maybe") is None


def test_full_sentence_rbi_quote():
    """parse_period must work on the real sentence, not just an isolated
    token -- docs/CASE_DOSSIER.md §1, RBI AR p8, verbatim."""
    p = parse_period(
        "real gross domestic product (GDP) growth moderated to 6.5 per cent in 2024-25"
    )
    assert (p.start, p.end, p.basis) == (date(2024, 4, 1), date(2025, 3, 31), "FY_IN")


def test_full_sentence_imf_quote():
    """docs/CASE_DOSSIER.md §1, IMF Article IV p10, verbatim."""
    p = parse_period("India's real GDP grew by 6.5 percent in FY2024/25")
    assert (p.start, p.end, p.basis) == (date(2024, 4, 1), date(2025, 3, 31), "FY_IN")


def test_full_sentence_imf_fy_quarter_quote():
    """docs/CASE_DOSSIER.md §3(d), IMF p3, verbatim."""
    p = parse_period("real GDP expanded by 7.8 percent in the first quarter of FY2025/26")
    assert (p.start, p.end) == (date(2025, 4, 1), date(2025, 6, 30))


def test_full_sentence_imf_cy_quarter_quote():
    """docs/CASE_DOSSIER.md §3(d), IMF p10, verbatim."""
    p = parse_period("real GDP growth of 7.8 percent in 2025Q2")
    assert (p.start, p.end) == (date(2025, 4, 1), date(2025, 6, 30))
