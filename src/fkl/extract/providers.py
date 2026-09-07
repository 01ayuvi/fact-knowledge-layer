"""Shared LLM provider layer: Groq-first, Gemini-fallback, structured
output either way. Extracted out of extractor.py so reconcile/adjudicator.py
can reuse the exact same rotation/retry/fallback logic against a third
response schema, rather than reimplementing it or reaching into another
module's private names.

Providers: Groq (OpenAI-compatible endpoint, JSON mode, GROQ_API_KEY) is the
default. Gemini is the fallback when Groq fails, and the sole provider for
anything needing native image input (e.g. slide/vision extraction), which
Groq's endpoint here is too text-only for.

Gemini API keys: reads GOOGLE_API_KEYS (comma-separated) from .env, falling
back to a single GOOGLE_API_KEY for backward compatibility. A 429 whose
quota violation is a daily cap rotates immediately to the next key (see
KeyRotator); a 429 that's a per-minute rate limit, or a 5xx, keeps the same
key and backs off exponentially (see is_retryable).
"""

from __future__ import annotations

import json
import logging
import os
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
    propagate to the caller unchanged if that backoff is exhausted."""

    def __init__(self, api_keys: list[str]):
        if not api_keys:
            raise ValueError("KeyRotator requires at least one API key")
        self._api_keys = api_keys
        self._clients: dict[int, genai.Client] = {}
        self._exhausted: set[int] = set()
        self._current = 0

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
        return self._current

    def _client_for(self, index: int) -> genai.Client:
        if index not in self._clients:
            self._clients[index] = genai.Client(api_key=self._api_keys[index])
        return self._clients[index]

    def _advance(self) -> None:
        total = len(self._api_keys)
        for offset in range(1, total + 1):
            candidate = (self._current + offset) % total
            if candidate not in self._exhausted:
                self._current = candidate
                return

    def generate(self, prompt: str, response_model: type[_ResponseT]) -> _ResponseT:
        total = len(self._api_keys)
        while len(self._exhausted) < total:
            if self._current in self._exhausted:
                self._advance()
                continue
            client = self._client_for(self._current)
            try:
                return call_gemini(client, prompt, response_model)
            except errors.APIError as exc:
                if exc.code == 429 and is_daily_quota_error(exc):
                    self._exhausted.add(self._current)
                    logger.warning(
                        "API key index %d exhausted its daily quota (%d/%d keys exhausted so far)",
                        self._current,
                        len(self._exhausted),
                        total,
                    )
                    self._advance()
                    continue
                raise
        raise AllKeysExhaustedError(f"all {total} API key(s) exhausted their daily quota")


def _is_retryable_groq(exc: BaseException) -> bool:
    """429 and 5xx get exponential backoff, same policy as Gemini's
    is_retryable. A network-level failure (timeout, connection error) is
    also retried; a non-retryable HTTP error (e.g. 400/401) is not — it will
    just fail the same way again, so fail fast into the Gemini fallback."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


@retry(
    retry=retry_if_exception(_is_retryable_groq),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _groq_chat_completion(prompt: str) -> str:
    api_key = os.environ["GROQ_API_KEY"]
    response = httpx.post(
        f"{GROQ_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


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
    content = _groq_chat_completion(_append_json_schema_instructions(prompt, response_model))
    return response_model.model_validate_json(content)


def generate(
    prompt: str,
    response_model: type[_ResponseT],
    rotator: KeyRotator,
) -> tuple[_ResponseT, str]:
    """Groq (JSON mode, openai/gpt-oss-120b) is the default provider. Any
    failure that survives Groq's own retry budget — a non-retryable HTTP
    error, retries exhausted, or a response that doesn't validate against
    the schema — falls back to the existing Gemini key-rotation path
    rather than losing the call.

    Returns (response, model_used) — the caller needs to know which
    provider actually served the request for Fact.provenance.model (or a
    Relation's provenance) to be honest about it, not just always claim
    whichever model is the default.
    """
    try:
        return call_groq(prompt, response_model), GROQ_MODEL
    except (httpx.HTTPError, ValidationError) as exc:
        logger.warning("Groq call failed (%s), falling back to Gemini", exc)
        return rotator.generate(prompt, response_model), MODEL_NAME
