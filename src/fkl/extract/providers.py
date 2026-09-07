"""Shared LLM provider layer: Groq-first, Gemini-fallback, structured
output either way. Extracted out of extractor.py so reconcile/adjudicator.py
can reuse the exact same rotation/retry/fallback logic against a third
response schema, rather than reimplementing it or reaching into another
module's private names.

Providers: Groq (OpenAI-compatible endpoint, JSON mode) is the default.
Gemini is the fallback when Groq fails, and the sole provider for anything
needing native image input (e.g. slide/vision extraction), which Groq's
endpoint here is too text-only for.

Both providers rotate across multiple API keys the same way: a 429 whose
quota violation is a daily cap rotates immediately to the next key, with no
wait (since a daily cap won't clear by waiting); a 429 that's a per-minute
rate limit, or a 5xx, keeps the same key and backs off exponentially.
- Gemini: reads GOOGLE_API_KEYS (comma-separated) from .env, falling back
  to a single GOOGLE_API_KEY for backward compatibility (see KeyRotator,
  is_retryable, is_daily_quota_error).
- Groq: reads GROQ_API_KEYS (comma-separated) from .env, falling back to a
  single GROQ_API_KEY for backward compatibility (see GroqKeyRotator,
  _is_retryable_groq, _is_daily_quota_error_groq). Groq is the primary
  provider here, so it needs rotation more than Gemini's fallback path
  does — it burns through its own rate-limit budget far faster in
  practice.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import TypeVar

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_ResponseT = TypeVar("_ResponseT", bound=BaseModel)

# Gemini: fallback provider, and (once implemented) the sole provider for
# slide/vision extraction, which needs native image input Groq's text-only
# endpoint can't do.
MODEL_NAME = os.environ.get("GEMINI_EXTRACT_MODEL", "gemini-flash-latest")

# Groq: default provider — OpenAI-compatible endpoint, JSON mode (schema
# described in-prompt, not enforced structurally the way Gemini's
# response_schema is). openai/gpt-oss-120b chosen from Groq's live
# /v1/models listing: a general-purpose production model with "json_mode"
# in supported_features (unlike the guard/audio/TTS/compound specialty
# models also listed there).
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.environ.get("GROQ_EXTRACT_MODEL", "openai/gpt-oss-120b")


def is_daily_quota_error(exc: BaseException) -> bool:
    """True for a 429 whose QuotaFailure violation names a per-DAY quota
    (quotaId like "GenerateRequestsPerDayPerProjectPerModel-FreeTier", seen
    live: 'limit: 20, model: gemini-3.8-flash'). False for a per-minute rate
    limit or any other error — those keep the existing same-key backoff."""
    if not (isinstance(exc, errors.APIError) and exc.code == 429):
        return False
    details = exc.details if isinstance(exc.details, dict) else {}
    error_obj = details.get("error", details)
    for item in error_obj.get("details", []):
        if str(item.get("@type", "")).endswith("QuotaFailure"):
            for violation in item.get("violations", []):
                if "day" in str(violation.get("quotaId", "")).lower():
                    return True
    return "perday" in str(exc).lower().replace(" ", "").replace("-", "")


def is_retryable(exc: BaseException) -> bool:
    """Per-minute 429s and any ServerError (5xx — e.g. 503 'high demand',
    seen live from gemini-flash-latest during testing) get exponential
    backoff on the SAME key. A daily-quota 429 is explicitly excluded here —
    KeyRotator.generate handles that by switching keys immediately, with no
    wait. Other APIErrors (4xx like a bad request) are not transient and
    should fail immediately rather than waste retry budget."""
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.APIError) and exc.code == 429:
        return not is_daily_quota_error(exc)
    return False


@retry(
    retry=retry_if_exception(is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def call_gemini(client: genai.Client, prompt: str, response_model: type[_ResponseT]) -> _ResponseT:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_model,
        ),
    )
    return response_model.model_validate_json(response.text)


class AllKeysExhaustedError(RuntimeError):
    pass


class KeyRotator:
    """Round-robins across GOOGLE_API_KEYS. A key that hits a daily-quota
    429 (see is_daily_quota_error) is marked exhausted for the rest of the
    session and generate() immediately retries the same prompt on the next
    non-exhausted key — no backoff wait, since a daily cap won't clear by
    waiting. Per-minute 429s and 5xx errors are NOT rotation triggers; they
    stay on the current key and get call_gemini's exponential backoff, and
    propagate to the caller unchanged if that backoff is exhausted.

    Safe to share across threads (e.g. a parallel explanation-writing pool):
    _lock guards state (_current/_exhausted/_clients) but is released
    before the actual network call, so concurrent generate() calls don't
    serialize on each other — only the brief key-selection step does."""

    def __init__(self, api_keys: list[str]):
        if not api_keys:
            raise ValueError("KeyRotator requires at least one API key")
        self._api_keys = api_keys
        self._clients: dict[int, genai.Client] = {}
        self._exhausted: set[int] = set()
        self._current = 0
        self._lock = threading.RLock()

    @classmethod
    def from_env(cls) -> KeyRotator:
        load_dotenv()
        raw = os.environ.get("GOOGLE_API_KEYS", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            single = os.environ.get("GOOGLE_API_KEY", "").strip()
            keys = [single] if single else []
        if not keys:
            raise RuntimeError(
                "no API key found: set GOOGLE_API_KEYS (comma-separated) or "
                "GOOGLE_API_KEY in .env"
            )
        return cls(keys)

    def __len__(self) -> int:
        return len(self._api_keys)

    def current_index(self) -> int:
        with self._lock:
            return self._current

    def _client_for(self, index: int) -> genai.Client:
        # caller already holds _lock
        if index not in self._clients:
            self._clients[index] = genai.Client(api_key=self._api_keys[index])
        return self._clients[index]

    def _advance(self) -> None:
        # caller already holds _lock
        total = len(self._api_keys)
        for offset in range(1, total + 1):
            candidate = (self._current + offset) % total
            if candidate not in self._exhausted:
                self._current = candidate
                return

    def generate(self, prompt: str, response_model: type[_ResponseT]) -> _ResponseT:
        total = len(self._api_keys)
        while True:
            with self._lock:
                if len(self._exhausted) >= total:
                    raise AllKeysExhaustedError(f"all {total} API key(s) exhausted their daily quota")
                if self._current in self._exhausted:
                    self._advance()
                    continue
                current_index = self._current
                client = self._client_for(current_index)
            # Lock released here — the network call itself runs unlocked,
            # so other threads can pick their own key and proceed
            # concurrently instead of queuing behind this one call.
            try:
                return call_gemini(client, prompt, response_model)
            except errors.APIError as exc:
                if exc.code == 429 and is_daily_quota_error(exc):
                    with self._lock:
                        self._exhausted.add(current_index)
                        logger.warning(
                            "API key index %d exhausted its daily quota (%d/%d keys exhausted so far)",
                            current_index,
                            len(self._exhausted),
                            total,
                        )
                        self._advance()
                    continue
                raise


def _is_daily_quota_error_groq(exc: httpx.HTTPStatusError) -> bool:
    """True for a 429 whose message names a per-DAY (RPD) limit rather than
    a transient per-minute/per-second rate limit — mirrors
    is_daily_quota_error's role for Gemini. Groq's rate-limit error text
    looks like 'Rate limit reached for model `X` ... on requests per day
    (RPD): Limit 14400, Used 14400, Requested 1. Please try again in
    23h59m59s.' for a hard daily cap, vs '... requests per minute (RPM) ...'
    or '... tokens per minute (TPM) ...' for the transient case that should
    just back off on the same key instead of rotating."""
    try:
        message = str(exc.response.json().get("error", {}).get("message", ""))
    except Exception:
        message = exc.response.text if exc.response is not None else ""
    text = message.lower()
    return "per day" in text or "(rpd)" in text


def _is_retryable_groq(exc: BaseException) -> bool:
    """5xx and a per-minute/per-second 429 get exponential backoff, same
    policy as Gemini's is_retryable. A daily-quota 429 is explicitly
    excluded here — GroqKeyRotator.generate handles that by switching keys
    immediately, with no wait, same split as Gemini's is_retryable /
    KeyRotator. A network-level failure (timeout, connection error) is also
    retried; a non-retryable HTTP error (e.g. 400/401) is not — it will just
    fail the same way again, so fail fast into the Gemini fallback."""
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 429:
            return not _is_daily_quota_error_groq(exc)
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


@retry(
    retry=retry_if_exception(_is_retryable_groq),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _groq_chat_completion(prompt: str, api_key: str) -> str:
    """max_completion_tokens matters here more than it would for a plain
    chat reply: with no explicit value set, a real live call against a
    3-block batch came back with finish_reason="length" — the response was
    truncated mid-JSON, which then fails Groq's own json_object validation
    as a 400 "json_validate_failed" (not a transient error, so it was never
    retried; it just silently produced zero facts for that batch). 16000 is
    comfortably under gpt-oss-120b's 65536 completion-token ceiling and
    well above what any single extraction batch (capped at
    BATCH_CHAR_BUDGET input characters) should ever need to fully emit."""
    response = httpx.post(
        f"{GROQ_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_completion_tokens": 16000,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


class GroqKeyRotator:
    """Round-robins across GROQ_API_KEYS, mirroring KeyRotator's
    exhaustion-and-rotate logic (see its docstring) but adapted to Groq's
    httpx-based errors via _is_daily_quota_error_groq. Groq is the primary
    provider here (Gemini is only the fallback), so it burns through its
    own rate-limit budget far faster in practice — rotation matters more
    for Groq than it does for Gemini.

    Safe to share across threads, same locking discipline as KeyRotator:
    _lock guards key-selection state only, released before the network
    call itself."""

    def __init__(self, api_keys: list[str]):
        if not api_keys:
            raise ValueError("GroqKeyRotator requires at least one API key")
        self._api_keys = api_keys
        self._exhausted: set[int] = set()
        self._current = 0
        self._lock = threading.RLock()

    @classmethod
    def from_env(cls) -> GroqKeyRotator:
        load_dotenv()
        raw = os.environ.get("GROQ_API_KEYS", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            # Backward compatibility: a single GROQ_API_KEY still works,
            # just with no rotation (a 1-key rotator).
            single = os.environ.get("GROQ_API_KEY", "").strip()
            keys = [single] if single else []
        if not keys:
            raise RuntimeError(
                "no Groq API key found: set GROQ_API_KEYS (comma-separated) or "
                "GROQ_API_KEY in .env"
            )
        return cls(keys)

    def __len__(self) -> int:
        return len(self._api_keys)

    def current_index(self) -> int:
        with self._lock:
            return self._current

    def _advance(self) -> None:
        # caller already holds _lock
        total = len(self._api_keys)
        for offset in range(1, total + 1):
            candidate = (self._current + offset) % total
            if candidate not in self._exhausted:
                self._current = candidate
                return

    def generate(self, prompt: str) -> str:
        total = len(self._api_keys)
        while True:
            with self._lock:
                if len(self._exhausted) >= total:
                    raise AllKeysExhaustedError(f"all {total} Groq API key(s) exhausted their daily quota")
                if self._current in self._exhausted:
                    self._advance()
                    continue
                current_index = self._current
                api_key = self._api_keys[current_index]
            # Lock released here, same rationale as KeyRotator.generate.
            try:
                return _groq_chat_completion(prompt, api_key)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and _is_daily_quota_error_groq(exc):
                    with self._lock:
                        self._exhausted.add(current_index)
                        logger.warning(
                            "Groq key index %d exhausted its daily quota (%d/%d keys exhausted so far)",
                            current_index,
                            len(self._exhausted),
                            total,
                        )
                        self._advance()
                    continue
                raise


_groq_rotator: GroqKeyRotator | None = None
_groq_rotator_lock = threading.Lock()


def _get_groq_rotator() -> GroqKeyRotator:
    """Lazy process-wide singleton so callers of generate() don't need to
    thread a Groq rotator through the same call chains that already pass
    around a Gemini KeyRotator (extractor.py, pipeline.py,
    reconcile/adjudicator.py) — those signatures stay unchanged."""
    global _groq_rotator
    with _groq_rotator_lock:
        if _groq_rotator is None:
            _groq_rotator = GroqKeyRotator.from_env()
        return _groq_rotator


def _append_json_schema_instructions(prompt: str, response_model: type[BaseModel]) -> str:
    """Groq's JSON mode (unlike Gemini's response_schema) only guarantees
    syntactically valid JSON, not conformance to a particular shape — the
    shape has to be described in the prompt itself, then validated on our
    side (see call_groq's model_validate_json, which raises ValidationError
    on a non-conforming response)."""
    schema = json.dumps(response_model.model_json_schema())
    return (
        f"{prompt}\n\n"
        "Respond with ONLY a single JSON object (no prose, no markdown code "
        f"fences) that conforms exactly to this JSON Schema:\n{schema}"
    )


def call_groq(prompt: str, response_model: type[_ResponseT]) -> _ResponseT:
    full_prompt = _append_json_schema_instructions(prompt, response_model)
    content = _get_groq_rotator().generate(full_prompt)
    return response_model.model_validate_json(content)


class PayloadTooLargeError(RuntimeError):
    """Groq returned 413 Payload Too Large. This is a batch-sizing problem,
    not a provider problem — falling back to Gemini would just resend the
    SAME oversized prompt to a different endpoint. The caller (extractor.py)
    is expected to catch this specifically and shrink the batch, not treat
    it like any other Groq failure."""


def generate(
    prompt: str,
    response_model: type[_ResponseT],
    rotator: KeyRotator,
) -> tuple[_ResponseT, str]:
    """Groq (JSON mode, openai/gpt-oss-120b) is the default provider. Any
    failure that survives Groq's own retry budget — a non-retryable HTTP
    error, retries exhausted, or a response that doesn't validate against
    the schema — falls back to the existing Gemini key-rotation path
    rather than losing the call. The one exception is 413 (see
    PayloadTooLargeError), which is never retried here and never falls
    through to Gemini.

    Returns (response, model_used) — the caller needs to know which
    provider actually served the request for Fact.provenance.model (or a
    Relation's provenance) to be honest about it, not just always claim
    whichever model is the default. Groq itself rotates across
    GROQ_API_KEYS internally (see GroqKeyRotator) before any of this falls
    back to Gemini at all — Groq is the primary provider, so it needs its
    own key rotation more than Gemini's fallback path does.
    """
    try:
        return call_groq(prompt, response_model), GROQ_MODEL
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 413:
            raise PayloadTooLargeError(str(exc)) from exc
        logger.warning("Groq call failed (%s), falling back to Gemini", exc)
        return rotator.generate(prompt, response_model), MODEL_NAME
    except AllKeysExhaustedError as exc:
        logger.warning("All Groq keys exhausted (%s), falling back to Gemini", exc)
        return rotator.generate(prompt, response_model), MODEL_NAME
    except (httpx.HTTPError, ValidationError) as exc:
        logger.warning("Groq call failed (%s), falling back to Gemini", exc)
        return rotator.generate(prompt, response_model), MODEL_NAME
