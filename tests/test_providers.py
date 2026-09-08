"""providers.py unit tests for the pure parsing helpers -- no real network
calls, no API keys needed."""

from __future__ import annotations

import httpx

from src.fkl.extract.providers import _is_daily_quota_error_groq, _parse_retry_after_seconds


def make_429(message: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, json={"error": {"message": message}}, request=request)
    return httpx.HTTPStatusError(message, request=request, response=response)


def test_retry_after_milliseconds_not_misread_as_minutes():
    """Regression: (\\d+)m alone greedily matches the "m" in "432ms",
    turning a sub-second cooldown into a 7+ hour one (432 minutes) -- seen
    live on a real Groq 429 message during a Delhivery ingest."""
    exc = make_429(
        "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): "
        "Limit 200000, Used 197300, Requested 2701. Please try again in 432ms."
    )
    assert _parse_retry_after_seconds(exc) == 0.432


def test_retry_after_minutes_and_seconds():
    exc = make_429("... on tokens per day (TPD): ... Please try again in 4m45.552s.")
    assert _parse_retry_after_seconds(exc) == 4 * 60 + 45.552


def test_retry_after_hours_minutes_seconds():
    exc = make_429("... on requests per day (RPD): ... Please try again in 23h59m59s.")
    assert _parse_retry_after_seconds(exc) == 23 * 3600 + 59 * 60 + 59


def test_retry_after_unparseable_falls_back_to_default():
    exc = make_429("... on tokens per day (TPD): ... no retry hint here.")
    from src.fkl.extract.providers import DEFAULT_DAILY_COOLDOWN_SECONDS

    assert _parse_retry_after_seconds(exc) == DEFAULT_DAILY_COOLDOWN_SECONDS


def test_daily_quota_signal_requires_explicit_unit():
    assert _is_daily_quota_error_groq(make_429("... on tokens per day (TPD): ...")) is True
    assert _is_daily_quota_error_groq(make_429("... on requests per minute (RPM): ...")) is False
    assert _is_daily_quota_error_groq(make_429("... on tokens per minute (TPM): ...")) is False
