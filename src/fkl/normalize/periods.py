"""Parse period expressions into (start_date, end_date, basis). Pure Python,
no LLM calls.

basis is one of:
  FY_IN    Indian fiscal year, April 1 - March 31
  CY       calendar year, January 1 - December 31
  QUARTER  a single quarter, either FY-anchored (Apr start) or CY-anchored
           (Jan start) -- both resolve through the same basis label since
           what matters for reconciliation is that the resulting interval
           is correct, not which label produced it. See PERIOD_BASIS_MISMATCH
           in docs/SUPERJOIN_BUILD_PLAN.md D3.

The critical case this module exists for (docs/CASE_DOSSIER.md §3d): Q1 of
Indian FY2025/26 and calendar 2025Q2 are different LABELS for the same
interval (Apr 1 - Jun 30, 2025) -- parse_period must resolve both to
identical (start, end).
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

Basis = Literal["FY_IN", "CY", "QUARTER"]

_MONTH_NAMES = {
    name.lower(): i
    for i, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}

_ORDINAL_QUARTERS = {"first": 1, "second": 2, "third": 3, "fourth": 4}

_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


@dataclass(frozen=True)
class Period:
    start: date
    end: date
    basis: Basis
    raw: str


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _fy_in_range(fy_start_year: int) -> tuple[date, date]:
    """FY_IN "FY24" means fiscal year ending March 2024, i.e. the year
    starting April 2023 -- fy_start_year is that April-start year (2023)."""
    return date(fy_start_year, 4, 1), _month_end(fy_start_year + 1, 3)


def _cy_range(year: int) -> tuple[date, date]:
    return date(year, 1, 1), date(year, 12, 31)


def _fy_in_quarter_range(fy_start_year: int, quarter: int) -> tuple[date, date]:
    """Indian FY quarters: Q1=Apr-Jun, Q2=Jul-Sep, Q3=Oct-Dec (all in
    fy_start_year), Q4=Jan-Mar (in fy_start_year + 1)."""
    if quarter == 4:
        return date(fy_start_year + 1, 1, 1), _month_end(fy_start_year + 1, 3)
    start_month = 4 + (quarter - 1) * 3
    return date(fy_start_year, start_month, 1), _month_end(fy_start_year, start_month + 2)


def _cy_quarter_range(year: int, quarter: int) -> tuple[date, date]:
    start_month = 1 + (quarter - 1) * 3
    return date(year, start_month, 1), _month_end(year, start_month + 2)


def _two_digit_year(suffix: str) -> int:
    """"24" -> 2024. This whole domain is post-2000, so no century pivot
    logic is needed."""
    return 2000 + int(suffix)


def _parse_date(text: str) -> date | None:
    """"March 31, 2024" / "December 31, 2021" style dates only -- the
    formats this module's callers (period expressions) actually produce."""
    m = re.search(
        r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})", text
    )
    if not m:
        return None
    month = _MONTH_NAMES.get(m.group("month").lower())
    if month is None:
        return None
    try:
        return date(int(m.group("year")), month, int(m.group("day")))
    except ValueError:
        return None


def _months_before(end: date, n_months: int) -> date:
    """The 1st of the month that is n_months before end's month, inclusive
    of end's own month -- "nine months ended Dec 31" starts in April
    (Apr..Dec is 9 months)."""
    zero_based = (end.year * 12 + (end.month - 1)) - (n_months - 1)
    year, month = divmod(zero_based, 12)
    return date(year, month + 1, 1)


def _basis_for_range(start: date, end: date) -> Basis:
    if start.month == 4 and start.day == 1:
        return "FY_IN"
    return "CY"


# --- format-specific parsers, tried in order (most specific first) ---

_ORDINAL_QUARTER_OF_FY_RE = re.compile(
    r"\b(?P<word>first|second|third|fourth)\s+quarter\s+of\s+FY\s*(?P<y1>\d{4})\s*[/-]\s*(?P<y2>\d{2,4})\b",
    re.IGNORECASE,
)
_ORDINAL_QUARTER_OF_FY_SHORT_RE = re.compile(
    r"\b(?P<word>first|second|third|fourth)\s+quarter\s+of\s+FY\s*(?P<y>\d{2})\b",
    re.IGNORECASE,
)
_QUARTER_FY_SLASH_RE = re.compile(r"\bQ(?P<q>[1-4])\s*FY\s*(?P<y1>\d{4})\s*/\s*(?P<y2>\d{2})\b", re.IGNORECASE)
_QUARTER_FY_DASH_RE = re.compile(r"\bQ(?P<q>[1-4])\s*FY\s*(?P<y1>\d{4})\s*-\s*(?P<y2>\d{2})\b", re.IGNORECASE)
_QUARTER_FY_SHORT_RE = re.compile(r"\bQ(?P<q>[1-4])\s*FY\s*(?P<y>\d{2})\b", re.IGNORECASE)
_YEAR_Q_RE = re.compile(r"\b(?P<y>\d{4})\s*Q(?P<q>[1-4])\b", re.IGNORECASE)
_Q_YEAR_RE = re.compile(r"\bQ(?P<q>[1-4])\s+(?P<y>\d{4})\b(?!\s*/)", re.IGNORECASE)

_FY_SLASH_RE = re.compile(r"\bFY\s*(?P<y1>\d{4})\s*/\s*(?P<y2>\d{2})\b", re.IGNORECASE)
_FY_DASH_LONG_RE = re.compile(r"\bFY\s*(?P<y1>\d{4})\s*-\s*(?P<y2>\d{2})\b", re.IGNORECASE)
_FY_SHORT_RE = re.compile(r"\bFY\s*(?P<y>\d{2})\b", re.IGNORECASE)
_BARE_YEAR_DASH_RE = re.compile(r"(?<!\d)(?P<y1>\d{4})\s*-\s*(?P<y2>\d{2})(?!\d)")

_CY_RE = re.compile(r"\bCY\s*(?P<y>\d{4})\b", re.IGNORECASE)

_MONTHS_ENDED_RE = re.compile(
    r"(?P<n>\d+|" + "|".join(_WORD_NUMBERS) + r")\s+months?\s+ended\s+(?P<date>.+)",
    re.IGNORECASE,
)
_AS_OF_RE = re.compile(r"\bas\s+of\s+(?P<date>.+)", re.IGNORECASE)


def parse_period(expr: str) -> Period | None:
    """Returns None if expr doesn't match any known period format."""
    text = expr.strip()

    if m := _ORDINAL_QUARTER_OF_FY_RE.search(text):
        fy_start = int(m.group("y1"))
        quarter = _ORDINAL_QUARTERS[m.group("word").lower()]
        start, end = _fy_in_quarter_range(fy_start, quarter)
        return Period(start, end, "QUARTER", expr)

    if m := _ORDINAL_QUARTER_OF_FY_SHORT_RE.search(text):
        fy_start = _two_digit_year(m.group("y")) - 1
        quarter = _ORDINAL_QUARTERS[m.group("word").lower()]
        start, end = _fy_in_quarter_range(fy_start, quarter)
        return Period(start, end, "QUARTER", expr)

    if m := _QUARTER_FY_SLASH_RE.search(text):
        fy_start = int(m.group("y1"))
        start, end = _fy_in_quarter_range(fy_start, int(m.group("q")))
        return Period(start, end, "QUARTER", expr)

    if m := _QUARTER_FY_DASH_RE.search(text):
        fy_start = int(m.group("y1"))
        start, end = _fy_in_quarter_range(fy_start, int(m.group("q")))
        return Period(start, end, "QUARTER", expr)

    if m := _QUARTER_FY_SHORT_RE.search(text):
        fy_start = _two_digit_year(m.group("y")) - 1
        start, end = _fy_in_quarter_range(fy_start, int(m.group("q")))
        return Period(start, end, "QUARTER", expr)

    if m := _YEAR_Q_RE.search(text):
        start, end = _cy_quarter_range(int(m.group("y")), int(m.group("q")))
        return Period(start, end, "QUARTER", expr)

    if m := _Q_YEAR_RE.search(text):
        start, end = _cy_quarter_range(int(m.group("y")), int(m.group("q")))
        return Period(start, end, "QUARTER", expr)

    if m := _FY_SLASH_RE.search(text):
        start, end = _fy_in_range(int(m.group("y1")))
        return Period(start, end, "FY_IN", expr)

    if m := _FY_DASH_LONG_RE.search(text):
        start, end = _fy_in_range(int(m.group("y1")))
        return Period(start, end, "FY_IN", expr)

    if m := _FY_SHORT_RE.search(text):
        start, end = _fy_in_range(_two_digit_year(m.group("y")) - 1)
        return Period(start, end, "FY_IN", expr)

    if m := _CY_RE.search(text):
        start, end = _cy_range(int(m.group("y")))
        return Period(start, end, "CY", expr)

    if m := _MONTHS_ENDED_RE.search(text):
        n_raw = m.group("n").lower()
        n_months = int(n_raw) if n_raw.isdigit() else _WORD_NUMBERS[n_raw]
        end = _parse_date(m.group("date"))
        if end is None:
            return None
        start = _months_before(end, n_months)
        return Period(start, end, _basis_for_range(start, end), expr)

    if m := _AS_OF_RE.search(text):
        at = _parse_date(m.group("date"))
        if at is None:
            return None
        return Period(at, at, "FY_IN", expr)

    # Bare "2024-25" (no "FY" prefix) -- this corpus's macro documents use
    # this to mean the Indian fiscal year (docs/CASE_DOSSIER.md §1: RBI's
    # "2024-25" and IMF's "FY2024/25" are the same period).
    if m := _BARE_YEAR_DASH_RE.search(text):
        start, end = _fy_in_range(int(m.group("y1")))
        return Period(start, end, "FY_IN", expr)

    # Bare date, no "as of" prefix -- treat as a point in time.
    if (at := _parse_date(text)) is not None:
        return Period(at, at, "FY_IN", expr)

    return None
