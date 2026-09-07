"""Fact extraction from prose Blocks via the Gemini API, structured output.

Slide and table blocks are not handled yet — callers should filter to
page_type == "prose" upstream, and this module also filters defensively.
The quote-location check here is a cheap exact-match pre-filter, not the
real grounding gate: a dedicated grounding_gate.py will later re-verify with
fuzzy matching and a proper quarantine queue (see docs/SUPERJOIN_BUILD_PLAN.md
D1). Here, a fact whose evidence_quote cannot be found verbatim in its block
is simply dropped, since the Evidence model requires real char offsets.

Providers: Groq (OpenAI-compatible endpoint, JSON mode, GROQ_API_KEY) is the
default for prose and table extraction (see _generate). Gemini is the
fallback when Groq fails, and will be the sole provider for slide/vision
extraction once that's implemented — Groq's endpoint here is text-only.

Gemini API keys: reads GOOGLE_API_KEYS (comma-separated) from .env, falling
back to a single GOOGLE_API_KEY for backward compatibility. A 429 whose
quota violation is a daily cap rotates immediately to the next key (see
KeyRotator); a 429 that's a per-minute rate limit, or a 5xx, keeps the same
key and backs off exponentially (see _is_retryable).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.fkl.ingest.pdf import Block
from src.fkl.ingest.tables import ReconstructedTable, TableCell, reconstruct_table
from src.fkl.store.models import Confidence, Entity, Evidence, Fact, Measure, Provenance, Value

logger = logging.getLogger(__name__)

_ResponseT = TypeVar("_ResponseT")

# Gemini: fallback provider, and (once implemented) the sole provider for
# slide/vision extraction, which needs native image input Groq's text-only
# endpoint can't do.
MODEL_NAME = os.environ.get("GEMINI_EXTRACT_MODEL", "gemini-flash-latest")

# Groq: default provider for prose/table extraction — OpenAI-compatible
# endpoint, JSON mode (schema described in-prompt, not enforced structurally
# the way Gemini's response_schema is). openai/gpt-oss-120b chosen from
# Groq's live /v1/models listing: a general-purpose production model with
# "json_mode" in supported_features (unlike the guard/audio/TTS/compound
# specialty models also listed there).
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.environ.get("GROQ_EXTRACT_MODEL", "openai/gpt-oss-120b")

BATCH_CHAR_BUDGET = 6000
HEADING_MAX_CHARS = 80
CACHE_DIR = Path("data/cache/extract")
TABLE_CACHE_DIR = Path("data/cache/extract_tables")
# Bumped past the Gemini-only versions: the cache key must reflect that a
# batch was attempted under the Groq-default/Gemini-fallback scheme, so an
# old Gemini-only cache entry can never be silently served post-switch.
PROMPT_VERSION = "extract-v4"
TABLE_PROMPT_VERSION = "extract-table-v3"

CONSOLIDATION_RE = re.compile(r"\b(standalone|consolidated)\b", re.IGNORECASE)
DATE_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
FY_LABEL_RE = re.compile(r"FY\s?(\d{2})\b")


class QualifierKV(BaseModel):
    key: str
    value: str


class ExtractedFact(BaseModel):
    block_index: int
    subject_surface_form: str
    subject_type: str | None = None
    measure_surface_form: str
    fact_type: Literal["numeric", "temporal", "categorical", "relational", "assertion"]
    value_raw: str
    value_number: float | None = None
    value_unit: str | None = None
    value_currency: str | None = None
    value_scale: str | None = None
    value_direction: str | None = None
    qualifiers: list[QualifierKV] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quote: str


class ExtractionResponse(BaseModel):
    facts: list[ExtractedFact]


class TableFactSemantics(BaseModel):
    """The LLM's job for a table cell is narrower than for prose: the value,
    its bbox, and its column header are already known exactly from PyMuPDF
    geometry (see tables.py) — no point asking the model to re-derive or
    re-type them. It only supplies the judgement-requiring parts: who/what
    the row is about, the clean measure name, and any qualifiers beyond what
    deterministic column-header parsing already covers (see
    _deterministic_qualifiers_from_header)."""

    cell_index: int
    subject_surface_form: str
    subject_type: str | None = None
    measure_surface_form: str
    fact_type: Literal["numeric", "temporal", "categorical", "relational", "assertion"] = "numeric"
    qualifiers: list[QualifierKV] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class TableExtractionResponse(BaseModel):
    facts: list[TableFactSemantics]


@dataclass
class BatchFailure:
    """A batch that exhausted retries. Not a crash — recorded so the caller
    (and eventually the D5 review queue) can see what was skipped and retry
    or investigate later; the rest of the document still gets processed."""

    batch_index: int
    page_range: tuple[int, int]
    block_count: int
    error: str


@dataclass
class ExtractionResult:
    facts: list[Fact] = field(default_factory=list)
    failed_batches: list[BatchFailure] = field(default_factory=list)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "unknown"


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and len(stripped) <= HEADING_MAX_CHARS and not stripped.endswith(
        (".", ",", ";")
    )


def _tag_sections(blocks: list[Block]) -> list[tuple[Block, str | None]]:
    """Best-effort section heading per block: the nearest preceding
    short, unpunctuated block. No layout/heading classifier exists yet
    (that's the "Layout" stage in the plan's pipeline), so this is a
    provisional heuristic, not a real heading detector."""
    tagged: list[tuple[Block, str | None]] = []
    current: str | None = None
    for block in blocks:
        if _looks_like_heading(block.text):
            current = block.text.strip()
        tagged.append((block, current))
    return tagged


def _make_batches(
    tagged: list[tuple[Block, str | None]],
) -> list[list[tuple[Block, str | None]]]:
    batches: list[list[tuple[Block, str | None]]] = []
    current: list[tuple[Block, str | None]] = []
    current_chars = 0
    for item in tagged:
        block_len = len(item[0].text)
        if current and current_chars + block_len > BATCH_CHAR_BUDGET:
            batches.append(current)
            current, current_chars = [], 0
        current.append(item)
        current_chars += block_len
    if current:
        batches.append(current)
    return batches


def _build_prompt(doc_context: dict[str, Any], batch: list[tuple[Block, str | None]]) -> str:
    lines = [
        "You are a fact-extraction engine for financial and macroeconomic PDF documents.",
        "Extract every discrete, checkable fact from the BLOCK TEXT sections below.",
        "",
        "DOCUMENT CONTEXT (defaults every fact in this document inherits; override a",
        "default ONLY if the block text explicitly states a different value for that",
        "specific fact — otherwise leave the qualifier out and the default applies):",
        json.dumps(doc_context, indent=2, default=str),
        "",
        "BEFORE extracting, apply this COMPARABILITY FILTER: only extract a fact that",
        "could plausibly be stated again, about the same entity, in a DIFFERENT",
        "document — financial measures, operational metrics, entity attributes,",
        "governance facts, dated events. Skip one-off descriptive trivia about a single",
        "facility (e.g. its docking station count, vehicle transit interval, square",
        "footage) UNLESS that facility itself is the named subject of the fact — a",
        "measure describing a specific named facility, attributed to that facility, is",
        "fine; the same number floating with no clear owner is not.",
        "",
        "For each fact return:",
        "- block_index: which BLOCK below it came from",
        "- subject_surface_form: REQUIRED. The entity the measure is actually about —",
        "  the company, a named subsidiary, a named facility, or a named business",
        "  segment. Prefer a specific named entity from the block text; fall back to",
        "  the DOCUMENT CONTEXT publisher only when the block is genuinely describing",
        "  that publisher as a whole (e.g. unattributed company-wide metrics like",
        "  \"our\"/\"we\"). If no subject can be determined even from DOCUMENT CONTEXT,",
        "  do NOT include the fact — an unattributed number is not a usable fact.",
        "- subject_type (e.g. organization, person, country, facility, segment)",
        "- measure_surface_form: REQUIRED, and must be the COMPLETE phrase as written",
        "  or as it would naturally be referred to — never a truncated fragment.",
        "  \"revenues from part truckload\", not \"revenues from part\".",
        "  (e.g. \"revenue from operations\", \"real GDP growth\")",
        "- fact_type: numeric | temporal | categorical | relational | assertion",
        "- value_raw (exact text of the value as written), plus value_number, value_unit,",
        "  value_currency, value_scale, value_direction where applicable",
        "- qualifiers: a list of {key, value} pairs for anything the DOCUMENT CONTEXT",
        "  does not already cover — e.g. period_start, period_end, period_label,",
        "  period_basis, consolidation, scope, geography, segment, modality,",
        "  source_stated_as, period_resolved. Include a key ONLY if the block text",
        "  supports it. RESOLVE RELATIVE TIME EXPRESSIONS against DOCUMENT CONTEXT's",
        "  reporting_period before emitting period_label/period_start/period_end — do",
        "  not pass the literal relative phrase through. E.g. in an FY24 document,",
        "  \"a year ago\" -> period_label \"FY23\"; \"as of March 2024\" -> period_end",
        "  \"2024-03-31\". If you cannot resolve a relative or vague period expression",
        "  to an absolute period from the available context, still emit your best",
        "  period_label AND add a qualifier {key: \"period_resolved\", value: \"false\"}",
        "  to flag it as unresolved; omit period_resolved entirely when a period is",
        "  either resolved or not applicable to the fact.",
        "- confidence: your own confidence in this extraction, 0.0-1.0",
        "- evidence_quote: REQUIRED, and must be copied VERBATIM — character-for-",
        "  character, same spelling, punctuation, spacing, capitalization — from the",
        "  block text below. Never paraphrase, correct, summarize, or reformat it. It",
        "  must be a contiguous substring a plain string search would find in the",
        "  block text. If you cannot produce such a quote, do not include the fact.",
        "  Note: the evidence_quote itself stays verbatim even when period_label is",
        "  resolved to an absolute value in the qualifier — do not rewrite the quote.",
        "",
    ]
    for i, (block, heading) in enumerate(batch):
        lines.append(f"--- BLOCK {i} (page {block.page_no}, section: {heading or 'unknown'}) ---")
        lines.append(block.text)
        lines.append("")
    return "\n".join(lines)


def _batch_cache_key(doc_context: dict[str, Any], batch: list[tuple[Block, str | None]]) -> str:
    payload = {
        "model": f"{GROQ_MODEL}|{MODEL_NAME}",
        "prompt_version": PROMPT_VERSION,
        "doc_context": doc_context,
        "blocks": [
            {"page_no": b.page_no, "block_no": b.block_no, "text": b.text, "heading": h}
            for b, h in batch
        ],
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_cache(key: str) -> tuple[ExtractionResponse, str] | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExtractionResponse.model_validate(payload["response"]), payload["model_used"]


def _save_cache(key: str, response: ExtractionResponse, model_used: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    payload = {"model_used": model_used, "response": response.model_dump(mode="json")}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_daily_quota_error(exc: BaseException) -> bool:
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


def _is_retryable(exc: BaseException) -> bool:
    """Per-minute 429s and any ServerError (5xx — e.g. 503 'high demand',
    seen live from gemini-flash-latest during testing) get exponential
    backoff on the SAME key. A daily-quota 429 is explicitly excluded here —
    KeyRotator.generate handles that by switching keys immediately, with no
    wait. Other APIErrors (4xx like a bad request) are not transient and
    should fail immediately rather than waste retry budget."""
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.APIError) and exc.code == 429:
        return not _is_daily_quota_error(exc)
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _call_gemini(client: genai.Client, prompt: str) -> ExtractionResponse:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExtractionResponse,
        ),
    )
    return ExtractionResponse.model_validate_json(response.text)


class AllKeysExhaustedError(RuntimeError):
    pass


class KeyRotator:
    """Round-robins across GOOGLE_API_KEYS. A key that hits a daily-quota
    429 (see _is_daily_quota_error) is marked exhausted for the rest of the
    session and generate() immediately retries the same prompt on the next
    non-exhausted key — no backoff wait, since a daily cap won't clear by
    waiting. Per-minute 429s and 5xx errors are NOT rotation triggers; they
    stay on the current key and get _call_gemini's exponential backoff, and
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

    def generate(
        self,
        prompt: str,
        call_fn: Callable[[genai.Client, str], _ResponseT] = _call_gemini,
    ) -> _ResponseT:
        total = len(self._api_keys)
        while len(self._exhausted) < total:
            if self._current in self._exhausted:
                self._advance()
                continue
            client = self._client_for(self._current)
            try:
                return call_fn(client, prompt)
            except errors.APIError as exc:
                if exc.code == 429 and _is_daily_quota_error(exc):
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
        raise AllKeysExhaustedError(
            f"all {total} API key(s) exhausted their daily quota"
        )


def _is_retryable_groq(exc: BaseException) -> bool:
    """429 and 5xx get exponential backoff, same policy as Gemini's
    _is_retryable. A network-level failure (timeout, connection error) is
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
    side (see _call_groq's model_validate_json, which raises ValidationError
    on a non-conforming response)."""
    schema = json.dumps(response_model.model_json_schema())
    return (
        f"{prompt}\n\n"
        "Respond with ONLY a single JSON object (no prose, no markdown code "
        f"fences) that conforms exactly to this JSON Schema:\n{schema}"
    )


def _call_groq(prompt: str, response_model: type[_ResponseT]) -> _ResponseT:
    content = _groq_chat_completion(_append_json_schema_instructions(prompt, response_model))
    return response_model.model_validate_json(content)


def _generate(
    prompt: str,
    response_model: type[_ResponseT],
    rotator: KeyRotator,
    gemini_call_fn: Callable[[genai.Client, str], _ResponseT],
) -> tuple[_ResponseT, str]:
    """Groq (JSON mode, openai/gpt-oss-120b) is the default provider for
    prose/table extraction. Any failure that survives Groq's own retry
    budget — a non-retryable HTTP error, retries exhausted, or a response
    that doesn't validate against the schema — falls back to the existing
    Gemini key-rotation path rather than losing the batch.

    Returns (response, model_used) — the caller needs to know which
    provider actually served the request for Fact.provenance.model to be
    honest about it, not just always claim whichever model is the default.
    """
    try:
        return _call_groq(prompt, response_model), GROQ_MODEL
    except (httpx.HTTPError, ValidationError) as exc:
        logger.warning("Groq call failed (%s), falling back to Gemini", exc)
        return rotator.generate(prompt, call_fn=gemini_call_fn), MODEL_NAME


def _locate_quote(block_text: str, quote: str) -> tuple[int, int] | None:
    start = block_text.find(quote)
    if start == -1:
        return None
    return start, start + len(quote)


def _to_fact(
    extracted: ExtractedFact,
    block: Block,
    heading: str | None,
    doc_id: str,
    model_used: str,
) -> Fact | None:
    if not extracted.subject_surface_form.strip():
        logger.warning(
            "dropping fact with no subject: %r (page %d block %d)",
            extracted.measure_surface_form,
            block.page_no,
            block.block_no,
        )
        return None

    span = _locate_quote(block.text, extracted.evidence_quote)
    if span is None:
        logger.warning(
            "dropping ungrounded fact: quote not found verbatim in page %d block %d",
            block.page_no,
            block.block_no,
        )
        return None
    char_start, char_end = span

    qualifiers: dict[str, Any] = {kv.key: kv.value for kv in extracted.qualifiers}
    if "period_resolved" in qualifiers:
        qualifiers["period_resolved"] = str(qualifiers["period_resolved"]).strip().lower() != "false"

    return Fact(
        id=str(uuid.uuid4()),
        doc_id=doc_id,
        block_id=f"{doc_id}:p{block.page_no}:b{block.block_no}",
        subject=Entity(
            canonical_id=_slugify(extracted.subject_surface_form),
            surface_form=extracted.subject_surface_form,
            type=extracted.subject_type,
        ),
        measure=Measure(
            canonical_id=_slugify(extracted.measure_surface_form),
            surface_form=extracted.measure_surface_form,
        ),
        fact_type=extracted.fact_type,
        value=Value(
            raw=extracted.value_raw,
            number=extracted.value_number,
            unit=extracted.value_unit,
            currency=extracted.value_currency,
            scale=extracted.value_scale,
            direction=extracted.value_direction,
        ),
        qualifiers=qualifiers,
        evidence=Evidence(
            page_no=block.page_no,
            char_start=char_start,
            char_end=char_end,
            bbox=[block.bbox],
            quote=extracted.evidence_quote,
            section_path=[heading] if heading else [],
        ),
        confidence=Confidence(extraction=extracted.confidence),
        provenance=Provenance(
            model=model_used,
            prompt_version=PROMPT_VERSION,
            extracted_at=datetime.now(UTC),
            parser="pymupdf",
        ),
    )


def _deterministic_qualifiers_from_header(header_text: str, doc_context: dict[str, Any]) -> dict[str, str]:
    """Rule-first normalization (see plan D3: "deterministic normalization
    first, LLM only for judgement calls") of a reconstructed column header.
    Always attaches the raw header text; adds consolidation/period_label
    when the header text makes them unambiguous via simple pattern matching.
    """
    qualifiers: dict[str, str] = {"column_header": header_text}

    consolidation_match = CONSOLIDATION_RE.search(header_text)
    if consolidation_match:
        qualifiers["consolidation"] = consolidation_match.group(1).lower()

    year_match = DATE_YEAR_RE.search(header_text)
    fy_match = FY_LABEL_RE.search(str(doc_context.get("reporting_period", "")))
    if year_match and fy_match:
        year = int(year_match.group(0))
        fy_end_year = 2000 + int(fy_match.group(1))
        if year == fy_end_year:
            qualifiers["period_label"] = str(doc_context["reporting_period"])
        elif year == fy_end_year - 1:
            qualifiers["period_label"] = f"FY{(fy_end_year - 1) % 100:02d}"

    return qualifiers


@dataclass
class _TableCellRecord:
    page_no: int
    source_block_no: int
    row_label_text: str
    header_text: str
    value_cell: TableCell


def _table_cell_records(table: ReconstructedTable) -> list[_TableCellRecord]:
    records: list[_TableCellRecord] = []
    for row in table.rows:
        label_text = row.label.text if row.label else ""
        for value_cell in row.values:
            records.append(
                _TableCellRecord(
                    page_no=table.page_no,
                    source_block_no=value_cell.source_block_no,
                    row_label_text=label_text,
                    header_text=table.column_header_for(value_cell),
                    value_cell=value_cell,
                )
            )
    return records


def _make_table_batches(records: list[_TableCellRecord]) -> list[list[_TableCellRecord]]:
    batches: list[list[_TableCellRecord]] = []
    current: list[_TableCellRecord] = []
    current_chars = 0
    for record in records:
        record_len = len(record.row_label_text) + len(record.header_text) + len(record.value_cell.text)
        if current and current_chars + record_len > BATCH_CHAR_BUDGET:
            batches.append(current)
            current, current_chars = [], 0
        current.append(record)
        current_chars += record_len
    if current:
        batches.append(current)
    return batches


def _build_table_prompt(
    doc_context: dict[str, Any], caption: str, batch: list[_TableCellRecord]
) -> str:
    lines = [
        "You are a fact-extraction engine reading ONE reconstructed table from a",
        "financial or macroeconomic PDF. Row/column structure and the exact cell",
        "values below were reconstructed deterministically from the PDF's layout —",
        "trust them as given; do not re-read or reinterpret the numbers.",
        "",
        "DOCUMENT CONTEXT (defaults every fact in this document inherits; override a",
        "default ONLY if the row/column context explicitly states otherwise):",
        json.dumps(doc_context, indent=2, default=str),
        "",
        f"TABLE CAPTION: {caption or '(none found)'}",
        "",
        "For each numbered cell below, return:",
        "- cell_index: matching the number in brackets",
        "- subject_surface_form: REQUIRED. The entity this measure is about — the",
        "  company, a named subsidiary, or a named segment. For an ordinary company",
        "  financial-statement row, this is normally the DOCUMENT CONTEXT publisher.",
        "  If no subject can be determined, do NOT include a fact for that cell.",
        "- subject_type (e.g. organization, person, country, segment)",
        "- measure_surface_form: the row's label, cleaned up only if it carries",
        "  stray footnote markers — otherwise use it verbatim, in full.",
        "- fact_type: numeric | temporal | categorical | relational | assertion",
        "  (virtually always numeric for a table cell)",
        "- qualifiers: a list of {key, value} pairs for anything the column header",
        "  implies beyond consolidation/period, which are already handled",
        "  deterministically — e.g. geography, scope, segment, modality,",
        "  source_stated_as. Do not restate consolidation or period_label yourself.",
        "- confidence: your own confidence in this extraction, 0.0-1.0",
        "",
        "Cells:",
    ]
    for i, record in enumerate(batch):
        lines.append(
            f'[{i}] Row "{record.row_label_text}", Column "{record.header_text}": '
            f"{record.value_cell.text}"
        )
    return "\n".join(lines)


def _table_batch_cache_key(doc_context: dict[str, Any], caption: str, batch: list[_TableCellRecord]) -> str:
    payload = {
        "model": f"{GROQ_MODEL}|{MODEL_NAME}",
        "prompt_version": TABLE_PROMPT_VERSION,
        "doc_context": doc_context,
        "caption": caption,
        "cells": [
            {
                "row_label": r.row_label_text,
                "header": r.header_text,
                "value": r.value_cell.text,
                "page_no": r.page_no,
                "block_no": r.source_block_no,
            }
            for r in batch
        ],
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_table_cache(key: str) -> tuple[TableExtractionResponse, str] | None:
    path = TABLE_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TableExtractionResponse.model_validate(payload["response"]), payload["model_used"]


def _save_table_cache(key: str, response: TableExtractionResponse, model_used: str) -> None:
    TABLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_CACHE_DIR / f"{key}.json"
    payload = {"model_used": model_used, "response": response.model_dump(mode="json")}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _call_gemini_table(client: genai.Client, prompt: str) -> TableExtractionResponse:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TableExtractionResponse,
        ),
    )
    return TableExtractionResponse.model_validate_json(response.text)


def _to_table_fact(
    semantics: TableFactSemantics,
    record: _TableCellRecord,
    doc_context: dict[str, Any],
    blocks_by_page_and_no: dict[tuple[int, int], Block],
    doc_id: str,
    model_used: str,
) -> Fact | None:
    if not semantics.subject_surface_form.strip():
        logger.warning(
            "dropping table fact with no subject: %r (page %d block %d)",
            semantics.measure_surface_form,
            record.page_no,
            record.source_block_no,
        )
        return None

    source_block = blocks_by_page_and_no.get((record.page_no, record.source_block_no))
    if source_block is None:
        logger.warning(
            "dropping table fact: no source Block for page %d block %d",
            record.page_no,
            record.source_block_no,
        )
        return None

    span = _locate_quote(source_block.text, record.value_cell.text)
    if span is None:
        logger.warning(
            "dropping ungrounded table fact: %r not found verbatim in page %d block %d",
            record.value_cell.text,
            record.page_no,
            record.source_block_no,
        )
        return None
    char_start, char_end = span

    qualifiers: dict[str, Any] = {kv.key: kv.value for kv in semantics.qualifiers}
    # Deterministic column-header parsing is authoritative over LLM guesses
    # for the keys it computes (consolidation, period_label) — see D3.
    qualifiers.update(_deterministic_qualifiers_from_header(record.header_text, doc_context))

    raw_text = record.value_cell.text
    number: float | None = None
    try:
        number = float(raw_text.replace(",", "").replace("(", "-").replace(")", "").rstrip("%"))
    except ValueError:
        pass

    return Fact(
        id=str(uuid.uuid4()),
        doc_id=doc_id,
        block_id=f"{doc_id}:p{record.page_no}:b{record.source_block_no}",
        subject=Entity(
            canonical_id=_slugify(semantics.subject_surface_form),
            surface_form=semantics.subject_surface_form,
            type=semantics.subject_type,
        ),
        measure=Measure(
            canonical_id=_slugify(semantics.measure_surface_form),
            surface_form=semantics.measure_surface_form,
        ),
        fact_type=semantics.fact_type,
        value=Value(
            raw=raw_text,
            number=number,
            unit="%" if raw_text.strip().endswith("%") else None,
            currency=doc_context.get("currency"),
            scale=doc_context.get("scale"),
        ),
        qualifiers=qualifiers,
        evidence=Evidence(
            page_no=record.page_no,
            char_start=char_start,
            char_end=char_end,
            bbox=[record.value_cell.bbox],
            quote=record.value_cell.text,
            section_path=[],
            table_ref=f"p{record.page_no}",
        ),
        confidence=Confidence(extraction=semantics.confidence),
        provenance=Provenance(
            model=model_used,
            prompt_version=TABLE_PROMPT_VERSION,
            extracted_at=datetime.now(UTC),
            parser="pymupdf+tables",
        ),
    )


def _extract_table_facts(
    blocks: list[Block],
    doc_context: dict[str, Any],
    pdf_path: str,
    rotator: KeyRotator,
    result: ExtractionResult,
) -> None:
    doc_id = doc_context.get("doc_id", "unknown")
    table_page_nos = sorted({b.page_no for b in blocks if b.page_type == "table"})
    blocks_by_page_and_no = {(b.page_no, b.block_no): b for b in blocks}

    for page_no in table_page_nos:
        table = reconstruct_table(pdf_path, page_no)
        if table is None or not table.rows:
            logger.info("page %d: page_type=table but no reconstructable rows found", page_no)
            continue

        records = _table_cell_records(table)
        batches = _make_table_batches(records)

        for batch_index, batch in enumerate(batches):
            cache_key = _table_batch_cache_key(doc_context, table.caption, batch)
            cached = _load_table_cache(cache_key)
            if cached is not None:
                response, model_used = cached
            else:
                logger.info(
                    "table batch %d (page %d, %d cells): trying Groq (%s), Gemini key index %d/%d as fallback",
                    batch_index,
                    page_no,
                    len(batch),
                    GROQ_MODEL,
                    rotator.current_index(),
                    len(rotator) - 1,
                )
                try:
                    prompt = _build_table_prompt(doc_context, table.caption, batch)
                    response, model_used = _generate(prompt, TableExtractionResponse, rotator, _call_gemini_table)
                except (errors.APIError, AllKeysExhaustedError) as exc:
                    logger.error(
                        "table batch %d (page %d, %d cells) failed after retries, skipping: %s",
                        batch_index,
                        page_no,
                        len(batch),
                        exc,
                    )
                    result.failed_batches.append(
                        BatchFailure(
                            batch_index=batch_index,
                            page_range=(page_no, page_no),
                            block_count=len(batch),
                            error=str(exc),
                        )
                    )
                    continue
                _save_table_cache(cache_key, response, model_used)

            for semantics in response.facts:
                if not (0 <= semantics.cell_index < len(batch)):
                    logger.warning("dropping table fact with out-of-range cell_index %d", semantics.cell_index)
                    continue
                record = batch[semantics.cell_index]
                fact = _to_table_fact(semantics, record, doc_context, blocks_by_page_and_no, doc_id, model_used)
                if fact is not None:
                    result.facts.append(fact)


def extract_facts(
    blocks: list[Block],
    doc_context: dict[str, Any],
    pdf_path: str,
    rotator: KeyRotator | None = None,
) -> ExtractionResult:
    """Extract Facts from a document's prose and table Blocks. Slide blocks
    (page_type == "slide") are still skipped — not implemented yet.

    pdf_path is required for table pages: page_type == "table" only tells
    the router a page LOOKS tabular, it doesn't carry row/column structure
    (Block.text is a flattened string) — see src/fkl/ingest/tables.py, which
    re-opens the PDF to rebuild that structure.

    doc_context must include "doc_id"; the rest (publisher, currency, scale,
    consolidation, geography, period, ...) are inherited by every fact as
    defaults unless a block/cell overrides them explicitly.

    A batch that still fails after retry — including exhausting every
    rotated API key (see KeyRotator, AllKeysExhaustedError) — is logged and
    recorded in the result's failed_batches, not raised — one bad batch must
    not abort extraction for the rest of the document.
    """
    doc_id = doc_context.get("doc_id", "unknown")

    prose_blocks = [b for b in blocks if b.page_type == "prose"]
    table_blocks = [b for b in blocks if b.page_type == "table"]
    skipped = len(blocks) - len(prose_blocks) - len(table_blocks)
    if skipped:
        logger.info("skipping %d slide block(s); slide extraction not implemented yet", skipped)

    load_dotenv()  # GROQ_API_KEY, needed even if an external rotator is passed in
    if rotator is None:
        rotator = KeyRotator.from_env()

    tagged = _tag_sections(prose_blocks)
    batches = _make_batches(tagged)

    result = ExtractionResult()
    for batch_index, batch in enumerate(batches):
        cache_key = _batch_cache_key(doc_context, batch)
        cached = _load_cache(cache_key)
        if cached is not None:
            response, model_used = cached
        else:
            page_nos = [block.page_no for block, _ in batch]
            logger.info(
                "batch %d (pages %d-%d, %d blocks): trying Groq (%s), Gemini key index %d/%d as fallback",
                batch_index,
                min(page_nos),
                max(page_nos),
                len(batch),
                GROQ_MODEL,
                rotator.current_index(),
                len(rotator) - 1,
            )
            try:
                prompt = _build_prompt(doc_context, batch)
                response, model_used = _generate(prompt, ExtractionResponse, rotator, _call_gemini)
            except (errors.APIError, AllKeysExhaustedError) as exc:
                logger.error(
                    "batch %d (pages %d-%d, %d blocks) failed after retries, skipping: %s",
                    batch_index,
                    min(page_nos),
                    max(page_nos),
                    len(batch),
                    exc,
                )
                result.failed_batches.append(
                    BatchFailure(
                        batch_index=batch_index,
                        page_range=(min(page_nos), max(page_nos)),
                        block_count=len(batch),
                        error=str(exc),
                    )
                )
                continue
            _save_cache(cache_key, response, model_used)

        for extracted in response.facts:
            if not (0 <= extracted.block_index < len(batch)):
                logger.warning("dropping fact with out-of-range block_index %d", extracted.block_index)
                continue
            block, heading = batch[extracted.block_index]
            fact = _to_fact(extracted, block, heading, doc_id, model_used)
            if fact is not None:
                result.facts.append(fact)

    if table_blocks:
        _extract_table_facts(blocks, doc_context, pdf_path, rotator, result)

    return result
