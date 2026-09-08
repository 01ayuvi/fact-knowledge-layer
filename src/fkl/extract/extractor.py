"""Fact extraction from prose Blocks via the Gemini API, structured output.

Slide and table blocks are not handled yet — callers should filter to
page_type == "prose" upstream, and this module also filters defensively.
The quote-location check here is a cheap exact-match pre-filter, not the
real grounding gate: a dedicated grounding_gate.py will later re-verify with
fuzzy matching and a proper quarantine queue (see docs/SUPERJOIN_BUILD_PLAN.md
D1). Here, a fact whose evidence_quote cannot be found verbatim in its block
is simply dropped, since the Evidence model requires real char offsets.

Providers: Groq-first, Gemini-fallback — see providers.py for the shared
implementation (also reused by reconcile/adjudicator.py). MODEL_NAME/
GROQ_MODEL/KeyRotator/AllKeysExhaustedError are all re-exported from there.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal

from dotenv import load_dotenv
from google.genai import errors
from pydantic import BaseModel, Field

from src.fkl.extract import providers
from src.fkl.extract.grounding_gate import GroundingGate, QuarantineEntry
from src.fkl.extract.providers import GROQ_MODEL, MODEL_NAME, AllKeysExhaustedError, KeyRotator
from src.fkl.ingest.pdf import Block
from src.fkl.ingest.tables import ReconstructedTable, TableCell, reconstruct_table
from src.fkl.store.models import Confidence, Entity, Evidence, Fact, Measure, Provenance, Value

logger = logging.getLogger(__name__)

# BATCH_CHAR_BUDGET bounds the CONTENT portion of a batch (block/cell text
# plus its own per-item template overhead) -- it is NOT the full prompt
# size. _build_prompt/_build_table_prompt also add ~5000 chars of fixed
# instructions plus, for Groq, ~2000 chars of appended JSON Schema (see
# providers._append_json_schema_instructions) on EVERY call, regardless of
# batch size. A real batch on the RBI annual report (52-130 blocks, each
# comfortably under the old text-only budget) produced actual prompts of
# 15,000-23,000 characters and got rejected with 413 Payload Too Large --
# the per-block "--- BLOCK N (page P, section: S) ---" header line was
# never counted at all. PER_BLOCK_OVERHEAD_CHARS/PER_TABLE_CELL_OVERHEAD_CHARS
# below fix the accounting; MAX_BLOCKS_PER_BATCH/MAX_TABLE_CELLS_PER_BATCH
# are a hard backstop independent of character counting, since many
# short blocks can still rack up overhead the char budget alone might
# under-price.
BATCH_CHAR_BUDGET = 3500
PER_BLOCK_OVERHEAD_CHARS = 120
PER_TABLE_CELL_OVERHEAD_CHARS = 30
MAX_BLOCKS_PER_BATCH = 20
MAX_TABLE_CELLS_PER_BATCH = 20
HEADING_MAX_CHARS = 80
CACHE_DIR = Path("data/cache/extract")
TABLE_CACHE_DIR = Path("data/cache/extract_tables")
# Bumped past the Gemini-only versions: the cache key must reflect that a
# batch was attempted under the Groq-default/Gemini-fallback scheme, so an
# old Gemini-only cache entry can never be silently served post-switch.
# Bumped again for the corrected batching (smaller, differently-shaped
# batches invalidate any cache keyed on the old batch boundaries).
PROMPT_VERSION = "extract-v5"
TABLE_PROMPT_VERSION = "extract-table-v4"

CONSOLIDATION_RE = re.compile(r"\b(standalone|consolidated)\b", re.IGNORECASE)
DATE_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
FY_LABEL_RE = re.compile(r"FY\s?(\d{2})\b")
# A column header that IS just a bare fiscal-year label ("FY24", not
# "Q4 FY24" -- the ^...$ anchors exclude the quarter-prefixed case, a
# narrower period this doesn't attempt to resolve) needs no cross-
# referencing against doc_context.reporting_period at all: the label
# itself already says what fiscal year the column covers.
BARE_FY_LABEL_RE = re.compile(r"^FY\s?\d{2}$", re.IGNORECASE)


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
    quarantined: list[QuarantineEntry] = field(default_factory=list)


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
        item_len = len(item[0].text) + PER_BLOCK_OVERHEAD_CHARS
        over_budget = current and current_chars + item_len > BATCH_CHAR_BUDGET
        over_count = current and len(current) >= MAX_BLOCKS_PER_BATCH
        if over_budget or over_count:
            batches.append(current)
            current, current_chars = [], 0
        current.append(item)
        current_chars += item_len
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


def _deterministic_fact_id(fact: Fact) -> str:
    """Fact identity must be a function of content, not a random UUID —
    otherwise re-extracting the SAME batch (e.g. a cache-hit replay on a
    resumed ingest, where the LLM call is skipped but _to_fact/_to_table_fact
    still runs) mints a fresh id every time, and repo.add_fact's INSERT OR
    IGNORE (keyed on id) silently persists a full duplicate instead of
    recognizing "already have this" — see the 298-duplicate incident from a
    resumed Delhivery ingest (docs/LIMITATIONS.md). A sha256 over the fields
    that define "the same claim" makes replay genuinely idempotent, cache
    hit or not.

    Call this AFTER GroundingGate.verify()/quarantine() has had its chance
    to correct evidence.char_start/char_end — fuzzy grounding is itself
    deterministic for identical (doc, block, quote) input, so hashing
    whatever offsets are on the fact at that point (corrected on a pass,
    still the (0, len(quote)) placeholder on a quarantine) is exactly as
    reproducible across a cache-hit replay as any other field here."""
    payload = "|".join(
        [
            fact.doc_id,
            fact.block_id,
            fact.subject.canonical_id,
            fact.measure.canonical_id,
            fact.value.raw,
            str(fact.evidence.char_start),
            str(fact.evidence.char_end),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_fact(
    extracted: ExtractedFact,
    block: Block,
    heading: str | None,
    doc_id: str,
    model_used: str,
) -> Fact | None:
    """Builds a candidate Fact with provisional evidence offsets (0,
    len(quote)) — NOT grounding-verified yet. The caller must run this
    through GroundingGate.verify(), which corrects char_start/char_end on
    pass or quarantines the fact on fail; nothing here decides groundedness
    anymore (see grounding_gate.py). id is left as a placeholder — the
    caller must set it via _deterministic_fact_id() AFTER verify()/
    quarantine() has run, once evidence offsets are in their final state."""
    if not extracted.subject_surface_form.strip():
        logger.warning(
            "dropping fact with no subject: %r (page %d block %d)",
            extracted.measure_surface_form,
            block.page_no,
            block.block_no,
        )
        return None

    qualifiers: dict[str, Any] = {kv.key: kv.value for kv in extracted.qualifiers}
    if "period_resolved" in qualifiers:
        qualifiers["period_resolved"] = str(qualifiers["period_resolved"]).strip().lower() != "false"

    return Fact(
        id="",
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
            char_start=0,
            char_end=len(extracted.evidence_quote),
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

    stripped_header = header_text.strip()
    if BARE_FY_LABEL_RE.match(stripped_header):
        # The header already IS the period label -- e.g. an investor
        # deck's table column literally headed "FY24"/"FY23", no full
        # calendar year anywhere to cross-reference. DATE_YEAR_RE below
        # only matches a 4-digit year ("2024"), so this style would
        # otherwise never resolve to a period at all (see the FY24
        # revenue-corroboration case, docs/CASE_DOSSIER.md §1).
        qualifiers["period_label"] = re.sub(r"\s+", "", stripped_header.upper())
    else:
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
        record_len = (
            len(record.row_label_text)
            + len(record.header_text)
            + len(record.value_cell.text)
            + PER_TABLE_CELL_OVERHEAD_CHARS
        )
        over_budget = current and current_chars + record_len > BATCH_CHAR_BUDGET
        over_count = current and len(current) >= MAX_TABLE_CELLS_PER_BATCH
        if over_budget or over_count:
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




def _to_table_fact(
    semantics: TableFactSemantics,
    record: _TableCellRecord,
    doc_context: dict[str, Any],
    doc_id: str,
    model_used: str,
) -> Fact | None:
    """Builds a candidate Fact with provisional evidence offsets — NOT
    grounding-verified yet, same contract as _to_fact, including the id
    placeholder (see _to_fact's docstring). The caller runs it through
    GroundingGate.verify() against the source block's text."""
    if not semantics.subject_surface_form.strip():
        logger.warning(
            "dropping table fact with no subject: %r (page %d block %d)",
            semantics.measure_surface_form,
            record.page_no,
            record.source_block_no,
        )
        return None

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
        id="",
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
            char_start=0,
            char_end=len(record.value_cell.text),
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
    gate: GroundingGate,
    on_batch: Callable[[list[Fact], list[QuarantineEntry]], None] | None = None,
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

        # Same 413-splitting queue as extract_facts()'s prose loop — see
        # that loop's comment for why this isn't a plain enumerate().
        queue: deque[list[_TableCellRecord]] = deque(batches)
        batch_index = 0
        while queue:
            batch = queue.popleft()
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
                    response, model_used = providers.generate(prompt, TableExtractionResponse, rotator)
                except providers.PayloadTooLargeError as exc:
                    if len(batch) <= 1:
                        logger.error(
                            "table batch %d (page %d): a single cell still triggers 413, giving up: %s",
                            batch_index, page_no, exc,
                        )
                        result.failed_batches.append(
                            BatchFailure(
                                batch_index=batch_index,
                                page_range=(page_no, page_no),
                                block_count=len(batch),
                                error=str(exc),
                            )
                        )
                        batch_index += 1
                        continue
                    mid = len(batch) // 2
                    logger.warning(
                        "table batch %d (page %d, %d cells) got 413 Payload Too Large — "
                        "splitting into %d + %d cell(s) and retrying",
                        batch_index, page_no, len(batch), mid, len(batch) - mid,
                    )
                    queue.appendleft(batch[mid:])
                    queue.appendleft(batch[:mid])
                    continue
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
                    batch_index += 1
                    continue
                _save_table_cache(cache_key, response, model_used)

            batch_facts: list[Fact] = []
            quarantined_before = len(gate.quarantined)
            for semantics in response.facts:
                if not (0 <= semantics.cell_index < len(batch)):
                    logger.warning("dropping table fact with out-of-range cell_index %d", semantics.cell_index)
                    continue
                record = batch[semantics.cell_index]
                fact = _to_table_fact(semantics, record, doc_context, doc_id, model_used)
                if fact is None:
                    continue
                source_block = blocks_by_page_and_no.get((record.page_no, record.source_block_no))
                if source_block is None:
                    gate.quarantine(
                        fact, f"no source Block for page {record.page_no} block {record.source_block_no}"
                    )
                    fact.id = _deterministic_fact_id(fact)
                    continue
                passed = gate.verify(fact, source_block.text)
                fact.id = _deterministic_fact_id(fact)
                if passed:
                    result.facts.append(fact)
                    batch_facts.append(fact)
            # Persist this batch's work immediately rather than waiting for
            # the whole document to finish — a crash or interrupt on a
            # later batch must not lose everything extracted so far (see
            # extract_facts()'s on_batch for the same reasoning).
            if on_batch is not None:
                on_batch(batch_facts, gate.quarantined[quarantined_before:])
            batch_index += 1


def extract_facts(
    blocks: list[Block],
    doc_context: dict[str, Any],
    pdf_path: str,
    rotator: KeyRotator | None = None,
    on_batch: Callable[[list[Fact], list[QuarantineEntry]], None] | None = None,
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

    Every candidate fact — prose or table — passes through a GroundingGate
    before it can reach result.facts (see grounding_gate.py): a fact whose
    evidence_quote doesn't fuzzy-match its source block is quarantined into
    result.quarantined with a reason, never silently dropped.

    on_batch(new_facts, new_quarantined), if given, is called once per
    completed batch (prose or table) with only the facts/quarantine entries
    that batch just produced — NOT the running total. This is the hook
    pipeline.py uses to persist to the repo incrementally: a document with
    hundreds of batches must not lose everything extracted so far just
    because a later batch crashes the process or gets interrupted. Whatever
    on_batch already persisted stays persisted; only result.facts/
    result.quarantined (the full in-memory totals, still returned as
    normal) would be lost on a crash.
    """
    doc_id = doc_context.get("doc_id", "unknown")
    gate = GroundingGate()

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
    # A deque, not enumerate(batches): a 413 (see providers.PayloadTooLargeError)
    # splits the current batch in half and re-queues both halves in place,
    # rather than recording a permanent failure or falling through to
    # Gemini with the same oversized prompt.
    queue: deque[list[tuple[Block, str | None]]] = deque(batches)
    batch_index = 0
    while queue:
        batch = queue.popleft()
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
                response, model_used = providers.generate(prompt, ExtractionResponse, rotator)
            except providers.PayloadTooLargeError as exc:
                if len(batch) <= 1:
                    logger.error(
                        "batch %d (pages %d-%d): a single block still triggers 413, giving up: %s",
                        batch_index, min(page_nos), max(page_nos), exc,
                    )
                    result.failed_batches.append(
                        BatchFailure(
                            batch_index=batch_index,
                            page_range=(min(page_nos), max(page_nos)),
                            block_count=len(batch),
                            error=str(exc),
                        )
                    )
                    batch_index += 1
                    continue
                mid = len(batch) // 2
                logger.warning(
                    "batch %d (pages %d-%d, %d blocks) got 413 Payload Too Large — "
                    "splitting into %d + %d block(s) and retrying",
                    batch_index, min(page_nos), max(page_nos), len(batch), mid, len(batch) - mid,
                )
                queue.appendleft(batch[mid:])
                queue.appendleft(batch[:mid])
                continue
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
                batch_index += 1
                continue
            _save_cache(cache_key, response, model_used)

        batch_facts: list[Fact] = []
        quarantined_before = len(gate.quarantined)
        for extracted in response.facts:
            if not (0 <= extracted.block_index < len(batch)):
                logger.warning("dropping fact with out-of-range block_index %d", extracted.block_index)
                continue
            block, heading = batch[extracted.block_index]
            fact = _to_fact(extracted, block, heading, doc_id, model_used)
            if fact is None:
                continue
            passed = gate.verify(fact, block.text)
            fact.id = _deterministic_fact_id(fact)
            if passed:
                result.facts.append(fact)
                batch_facts.append(fact)
        if on_batch is not None:
            on_batch(batch_facts, gate.quarantined[quarantined_before:])
        batch_index += 1

    if table_blocks:
        _extract_table_facts(blocks, doc_context, pdf_path, rotator, result, gate, on_batch=on_batch)

    result.quarantined = gate.quarantined
    return result
