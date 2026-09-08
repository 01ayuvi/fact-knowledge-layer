"""extract_facts() tests: the LLM call is mocked at the provider boundary
(src.fkl.extract.providers.generate) so the batch-cache and fact-identity
logic run for real -- no real API calls, no reliance on env-configured
API keys."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.fkl.extract.extractor import ExtractedFact, ExtractionResponse, extract_facts
from src.fkl.ingest.pdf import Block
from src.fkl.store.repo import Repo

DOC_CONTEXT = {"doc_id": "doc1", "publisher": "Delhivery Limited", "currency": "INR", "scale": "crore"}


class FakeRotator:
    """Stands in for a real KeyRotator -- only used for the logging calls
    extract_facts() makes on a cache MISS (current_index/__len__); the
    actual LLM call is mocked at the providers.generate boundary, so
    nothing here ever needs to behave like a real rotator."""

    def current_index(self) -> int:
        return 0

    def __len__(self) -> int:
        return 1


def make_block() -> Block:
    return Block(
        page_no=1,
        block_no=0,
        text="Revenue from operations was Rs. 100 crore in FY24.",
        bbox=(0.0, 0.0, 100.0, 20.0),
        page_type="prose",
    )


def mock_response() -> ExtractionResponse:
    return ExtractionResponse(
        facts=[
            ExtractedFact(
                block_index=0,
                subject_surface_form="Delhivery Limited",
                subject_type="organization",
                measure_surface_form="Revenue from operations",
                fact_type="numeric",
                value_raw="Rs. 100 crore",
                value_number=100.0,
                value_currency="INR",
                value_scale="crore",
                confidence=0.95,
                evidence_quote="Rs. 100 crore",
            )
        ]
    )


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Never touch the real data/cache/extract/ directory from tests --
    and this test specifically NEEDS the real on-disk cache mechanics (not
    just a mock), since the bug being regression-tested here is about
    cache-hit replay specifically."""
    import src.fkl.extract.extractor as extractor_module

    monkeypatch.setattr(extractor_module, "CACHE_DIR", tmp_path / "extract")
    monkeypatch.setattr(extractor_module, "TABLE_CACHE_DIR", tmp_path / "extract_tables")


def test_fact_id_is_deterministic_across_a_cache_hit_replay():
    """The regression this guards against: resuming an ingest re-runs
    extract_facts() over the same blocks. Batches already on disk replay
    from cache (no new LLM call) -- but _to_fact() used to mint a fresh
    uuid4() every time regardless, so a "free" cache-hit replay silently
    persisted a full duplicate copy of every previously-extracted fact
    (see docs/LIMITATIONS.md, the 298-duplicate incident). Fact identity
    must be a deterministic function of content so the second run
    reproduces the exact same fact.id, and Repo.add_fact's INSERT OR
    IGNORE actually dedupes instead of silently double-inserting."""
    blocks = [make_block()]

    with patch("src.fkl.extract.providers.generate", return_value=(mock_response(), "test-model")) as mock_generate:
        result1 = extract_facts(blocks, DOC_CONTEXT, "x.pdf", rotator=FakeRotator())
        assert mock_generate.call_count == 1  # first call: cache miss, real (mocked) LLM call

        result2 = extract_facts(blocks, DOC_CONTEXT, "x.pdf", rotator=FakeRotator())
        assert mock_generate.call_count == 1  # second call: cache hit, no new LLM call

    assert len(result1.facts) == 1
    assert len(result2.facts) == 1
    assert result1.facts[0].id == result2.facts[0].id
    assert result1.facts[0].id != ""


def test_ingesting_same_document_twice_yields_identical_fact_counts():
    """The literal requirement: persisting a cache-hit replay must not
    double the fact count in the store."""
    blocks = [make_block()]
    repo = Repo(":memory:")
    try:
        repo.add_document("doc1", "x.pdf", DOC_CONTEXT)
        with patch("src.fkl.extract.providers.generate", return_value=(mock_response(), "test-model")):
            result1 = extract_facts(blocks, DOC_CONTEXT, "x.pdf", rotator=FakeRotator())
            for fact in result1.facts:
                repo.add_fact(fact)
            count_after_first = len(repo.facts_by_doc("doc1"))

            result2 = extract_facts(blocks, DOC_CONTEXT, "x.pdf", rotator=FakeRotator())
            for fact in result2.facts:
                repo.add_fact(fact)
            count_after_second = len(repo.facts_by_doc("doc1"))

        assert count_after_first == 1
        assert count_after_second == count_after_first
    finally:
        repo.close()
