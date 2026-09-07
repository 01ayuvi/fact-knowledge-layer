"""ingest_document(): the whole chain, one call.

    ingest blocks -> derive doc context (1 LLM call, if not supplied) ->
    extract facts, prose + table, grounding-gate already applied inside
    extract_facts() -> persist -> find candidates against everything
    already in the store -> reconcile -> write explanations / adjudicate
    -> persist relations.

"normalize" isn't a separate step here with its own persisted output — the
normalizers (normalize/periods.py, units.py, entities.py) already run
lazily wherever they're actually needed: inside link/claim_key.py when a
Fact is persisted (claim_key/loose_key), and inside reconcile/rules.py
when two Facts' values are compared. There's nothing a separate upfront
pass would add beyond what those two call sites already guarantee.

doc_context derivation is the part the brief explicitly tests with unseen
PDFs: derive_doc_context() sends a text sample from the document itself
(not any hardcoded per-document logic) through one structured LLM call and
asks for publisher, document type, publication date, reporting period,
currency, scale, consolidation, and geography.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pymupdf
from pydantic import BaseModel

from src.fkl.extract.extractor import extract_facts
from src.fkl.extract.providers import KeyRotator, generate
from src.fkl.ingest.pdf import extract_blocks
from src.fkl.reconcile import adjudicator
from src.fkl.reconcile.rules import reconcile
from src.fkl.store.models import Fact, Relation, RelationType
from src.fkl.store.repo import Repo

logger = logging.getLogger(__name__)

DOC_CONTEXT_SAMPLE_CHARS = 4000

# Currency/scale conventions ("₹ in Million") are a financial-statements-
# section convention, not a cover-page one — they routinely live well past
# any opening-pages sample. A generic regex scan across the whole document
# (cheap, deterministic, no per-document logic) finds a few snippets to
# splice into the SAME single LLM call, rather than growing the sample
# large enough to blindly reach the financial statements every time.
_SCALE_HINT_RE = re.compile(
    r"[₹$€£]\s*(?:in\s+)?(million|crore|lakh|thousand|billion)", re.IGNORECASE
)
MAX_SCALE_HINT_SNIPPETS = 3
SCALE_HINT_CONTEXT_CHARS = 60

# Explanation-writing/adjudication is one LLM call per candidate pair and
# they're independent of each other, so they parallelize cleanly — see
# _reconcile_pair. 4 concurrent workers bounds how hard we hit the
# provider; KeyRotator is thread-safe (see providers.py) and every worker
# still gets the SAME retry/backoff/rotation behavior per call, so this
# caps concurrency without bypassing rate-limit handling.
MAX_EXPLANATION_WORKERS = 4


class DocContextResponse(BaseModel):
    publisher: str
    document_type: str
    publication_date: str | None = None
    reporting_period: str | None = None
    currency: str | None = None
    scale: str | None = None
    consolidation: str | None = None
    geography: str | None = None


def _doc_id_for(pdf_path: str) -> str:
    """Deterministic from the file path alone, so re-ingesting the same
    file is idempotent (see Repo.has_document / add_fact's INSERT OR
    IGNORE). Not a content hash — two different paths to byte-identical
    files would get different doc_ids, which is an accepted simplification
    here, not a claim that it can't happen."""
    return Path(pdf_path).stem


def _sample_text(pdf_path: str, max_chars: int = DOC_CONTEXT_SAMPLE_CHARS) -> str:
    """The first max_chars of the document's own text — no per-document
    logic, so this generalizes to an unseen PDF exactly as well as to any
    PDF already seen during development."""
    doc = pymupdf.open(pdf_path)
    try:
        chunks: list[str] = []
        total = 0
        for page in doc:
            text = page.get_text("text")
            chunks.append(text)
            total += len(text)
            if total >= max_chars:
                break
        return "".join(chunks)[:max_chars]
    finally:
        doc.close()


def _scale_hint_snippets(pdf_path: str) -> list[str]:
    doc = pymupdf.open(pdf_path)
    try:
        snippets: list[str] = []
        for page in doc:
            text = page.get_text("text")
            for m in _SCALE_HINT_RE.finditer(text):
                start = max(0, m.start() - SCALE_HINT_CONTEXT_CHARS)
                end = min(len(text), m.end() + SCALE_HINT_CONTEXT_CHARS)
                snippets.append(" ".join(text[start:end].split()))
                if len(snippets) >= MAX_SCALE_HINT_SNIPPETS:
                    return snippets
        return snippets
    finally:
        doc.close()


def _build_doc_context_prompt(sample: str, scale_hints: list[str]) -> str:
    return (
        "You are reading the opening pages of a financial or macroeconomic PDF "
        "document. Based ONLY on the text below, determine:\n"
        "- publisher: the organization that issued this document\n"
        "- document_type: e.g. annual_report, prospectus, earnings_presentation, "
        "economic_survey, article_iv_report\n"
        "- publication_date: ISO 8601 date if stated or clearly inferable, else null\n"
        "- reporting_period: the primary period this document reports on, e.g. "
        '"FY24", "FY2024/25", "CY2024" — null if not evident from this excerpt\n'
        "- currency: ISO-ish code, e.g. INR, USD — null if not evident\n"
        "- scale: the default numeric scale used for monetary figures, e.g. "
        '"million", "crore", "lakh", "billion" — null if not evident\n'
        "- consolidation: \"consolidated\" or \"standalone\" if the document states "
        "a default basis — null otherwise\n"
        "- geography: the primary country/region this document covers — null if "
        "not evident\n\n"
        "Use null (not a guess) for anything not actually supported by this text.\n\n"
        f"--- DOCUMENT TEXT (first {len(sample)} characters) ---\n{sample}"
        + (
            "\n\n--- ADDITIONAL SNIPPETS FOUND ELSEWHERE IN THE DOCUMENT (currency/scale "
            "conventions are often stated in a financial-statements section, not the "
            "opening pages — use these ONLY for currency/scale/consolidation) ---\n"
            + "\n".join(f"- …{s}…" for s in scale_hints)
            if scale_hints
            else ""
        )
    )


def derive_doc_context(pdf_path: str, rotator: KeyRotator | None = None) -> dict[str, Any]:
    """One LLM call, no hardcoded per-document logic — see module
    docstring. Returns a plain dict in the same shape extract_facts()
    expects as doc_context, with doc_id set and null fields dropped."""
    if rotator is None:
        rotator = KeyRotator.from_env()
    sample = _sample_text(pdf_path)
    scale_hints = _scale_hint_snippets(pdf_path)
    prompt = _build_doc_context_prompt(sample, scale_hints)
    response, model_used = generate(prompt, DocContextResponse, rotator)
    logger.info("doc context for %s derived via %s", pdf_path, model_used)

    context = {k: v for k, v in response.model_dump().items() if v is not None}
    context["doc_id"] = _doc_id_for(pdf_path)
    return context


@dataclass
class IngestResult:
    doc_id: str
    facts: int = 0
    quarantined: int = 0
    relations_by_type: dict[str, int] = field(default_factory=dict)


def _reconcile_pair(fact: Fact, other_fact: Fact, rotator: KeyRotator) -> Relation:
    """Runs in a worker thread — reconcile() is pure/deterministic, and
    write_explanation()/adjudicate() only touch the shared KeyRotator
    (thread-safe, see providers.py) and the network. No repo access here:
    sqlite3 connections aren't thread-safe, so persistence happens back on
    the main thread once this returns (see ingest_document)."""
    relation = reconcile(fact, other_fact)
    if adjudicator.is_adjudication_candidate(fact, other_fact, relation):
        return adjudicator.adjudicate(fact, other_fact, relation, rotator=rotator)
    return adjudicator.write_explanation(relation, fact, other_fact, rotator=rotator)


def ingest_document(
    pdf_path: str,
    doc_context: dict[str, Any] | None = None,
    repo: Repo | None = None,
    rotator: KeyRotator | None = None,
    on_progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> IngestResult:
    """on_progress(stage, data), called synchronously at each checkpoint
    below — a no-op by default. This is the hook apps/api's SSE endpoint
    wires up to stream progress; ingest_document itself has no notion of
    HTTP or SSE, it just reports where it is."""
    notify = on_progress or (lambda stage, data: None)
    owns_repo = repo is None
    if repo is None:
        repo = Repo()
    if rotator is None:
        rotator = KeyRotator.from_env()

    try:
        doc_id = (doc_context or {}).get("doc_id") or _doc_id_for(pdf_path)
        notify("started", {"pdf_path": pdf_path, "doc_id": doc_id})

        if repo.has_document(doc_id):
            # Already ingested: skip extraction and doc-context derivation
            # entirely, not just rely on add_fact's idempotent INSERT OR
            # IGNORE — re-running the LLM calls for a document already
            # fully processed would be wasted cost, not just wasted time.
            logger.info("doc_id %s already ingested, skipping re-extraction", doc_id)
            relations_by_type = {t.value: 0 for t in RelationType}
            for relation in repo.relations_by_doc(doc_id):
                relations_by_type[relation.type.value] += 1
            result = IngestResult(
                doc_id=doc_id,
                facts=len(repo.facts_by_doc(doc_id)),
                quarantined=len(repo.quarantined_by_doc(doc_id)),
                relations_by_type=relations_by_type,
            )
            notify("already_ingested", {"doc_id": doc_id, "result": result})
            notify("done", {"result": result})
            return result

        blocks = extract_blocks(pdf_path)
        notify("blocks_extracted", {"count": len(blocks)})

        if doc_context is None:
            notify("deriving_doc_context", {})
            doc_context = derive_doc_context(pdf_path, rotator=rotator)
        doc_context["doc_id"] = doc_id
        notify("doc_context", {"doc_context": doc_context})

        repo.add_document(doc_id, pdf_path, doc_context)

        # extract_facts() already applies the grounding gate internally
        # (see extract/extractor.py, extract/grounding_gate.py) — a fact
        # never reaches result.facts unverified.
        notify("extracting", {})
        extraction = extract_facts(blocks, doc_context, pdf_path, rotator=rotator)
        notify(
            "extraction_complete",
            {
                "facts": len(extraction.facts),
                "quarantined": len(extraction.quarantined),
                "failed_batches": len(extraction.failed_batches),
            },
        )

        for fact in extraction.facts:
            repo.add_fact(fact)
        for entry in extraction.quarantined:
            repo.add_quarantine(doc_id, entry.fact, entry.reason, entry.score)
        notify("facts_persisted", {"count": len(extraction.facts), "quarantined": len(extraction.quarantined)})

        # Candidate lookup is repo reads only — stays on the main thread
        # (sqlite3 connections aren't thread-safe). Dedupe pairs found from
        # both directions (fact A's search finds B, and later B's search
        # finds A, since by then A is already persisted) before any work
        # is submitted, not just before persisting — otherwise the same
        # pair gets explained twice by two different workers.
        pending_pairs: list[tuple[Fact, Fact]] = []
        seen_pairs: set[frozenset[str]] = set()
        for fact in extraction.facts:
            for other_fact, _match_level in repo.candidates_for_fact(fact):
                if repo.relation_exists(fact.id, other_fact.id):
                    continue  # already reconciled in a previous ingest — skip, don't re-spend an LLM call
                pair_ids = frozenset((fact.id, other_fact.id))
                if pair_ids in seen_pairs:
                    continue
                seen_pairs.add(pair_ids)
                pending_pairs.append((fact, other_fact))
        notify("candidates_found", {"count": len(pending_pairs)})

        relations_by_type: dict[str, int] = {t.value: 0 for t in RelationType}
        if pending_pairs:
            total_pairs = len(pending_pairs)
            done_pairs = 0
            with ThreadPoolExecutor(max_workers=MAX_EXPLANATION_WORKERS) as pool:
                futures = [pool.submit(_reconcile_pair, a, b, rotator) for a, b in pending_pairs]
                for future in as_completed(futures):
                    relation = future.result()
                    repo.add_relation(relation)  # persistence stays on the main thread
                    relations_by_type[relation.type.value] += 1
                    done_pairs += 1
                    notify(
                        "relation",
                        {
                            "done": done_pairs,
                            "total": total_pairs,
                            "type": relation.type.value,
                            "reason_code": relation.reason_code.value,
                        },
                    )

        result = IngestResult(
            doc_id=doc_id,
            facts=len(extraction.facts),
            quarantined=len(extraction.quarantined),
            relations_by_type=relations_by_type,
        )
        notify("done", {"result": result})
        return result
    except Exception as exc:
        notify("error", {"message": str(exc)})
        raise
    finally:
        if owns_repo:
            repo.close()
