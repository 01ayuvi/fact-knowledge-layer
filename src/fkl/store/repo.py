"""SQLite persistence for documents, facts, relations, and quarantined
facts. Qualifiers/evidence/provenance/delta are stored as JSON columns;
everything a query actually needs to filter or join on (subject, measure,
value, validity, and the claim_key/loose_key from link/claim_key.py) gets
its own indexed column.

Retrieval-bounded candidate lookup: facts.claim_key and facts.loose_key are
both indexed, so candidates_for_fact() is an indexed WHERE lookup, never a
full-table scan — the same guarantee link/candidates.py's in-memory version
gives, just backed by SQLite instead of a Python dict.

Incremental ingest: a Fact row, once inserted, is never updated (no
add_fact ever does an UPDATE) — a re-ingested document just fails to insert
new rows for facts whose id already exists (INSERT OR IGNORE), and a
relation is keyed UNIQUE on (fact_a_id, fact_b_id) so the same pair is
never reconciled and stored twice. pipeline.py is what actually skips
re-reconciling an already-related pair (via relation_exists(), checked
before spending an LLM call on it) — this module's job is only to make
that check and that insert both correct and cheap.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.fkl.link.claim_key import claim_key as compute_claim_key
from src.fkl.link.claim_key import loose_key as compute_loose_key
from src.fkl.store.models import (
    Confidence,
    Delta,
    Entity,
    Evidence,
    Fact,
    Measure,
    Provenance,
    Relation,
    RelationType,
    ReasonCode,
    Value,
    ValidityInterval,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    pdf_path TEXT NOT NULL,
    doc_context TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    block_id TEXT NOT NULL,
    subject_canonical_id TEXT NOT NULL,
    subject_surface_form TEXT NOT NULL,
    subject_type TEXT,
    measure_canonical_id TEXT NOT NULL,
    measure_surface_form TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    value_raw TEXT NOT NULL,
    value_number REAL,
    value_unit TEXT,
    value_currency TEXT,
    value_scale TEXT,
    value_direction TEXT,
    qualifiers TEXT NOT NULL,
    validity_start TEXT,
    validity_end TEXT,
    evidence TEXT NOT NULL,
    confidence_extraction REAL NOT NULL,
    confidence_normalization REAL,
    provenance TEXT NOT NULL,
    claim_key TEXT NOT NULL,
    loose_key TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);
CREATE INDEX IF NOT EXISTS idx_facts_claim_key ON facts(claim_key);
CREATE INDEX IF NOT EXISTS idx_facts_loose_key ON facts(loose_key);
CREATE INDEX IF NOT EXISTS idx_facts_doc_id ON facts(doc_id);

CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    fact_a_id TEXT NOT NULL,
    fact_b_id TEXT NOT NULL,
    type TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    explanation TEXT NOT NULL,
    delta TEXT,
    confidence REAL NOT NULL,
    adjudicator TEXT NOT NULL,
    UNIQUE(fact_a_id, fact_b_id)
);
CREATE INDEX IF NOT EXISTS idx_relations_fact_a ON relations(fact_a_id);
CREATE INDEX IF NOT EXISTS idx_relations_fact_b ON relations(fact_b_id);

CREATE TABLE IF NOT EXISTS quarantined (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    fact TEXT NOT NULL,
    reason TEXT NOT NULL,
    score REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quarantined_doc_id ON quarantined(doc_id);
"""


def _fact_row(fact: Fact) -> dict[str, Any]:
    return {
        "id": fact.id,
        "doc_id": fact.doc_id,
        "block_id": fact.block_id,
        "subject_canonical_id": fact.subject.canonical_id,
        "subject_surface_form": fact.subject.surface_form,
        "subject_type": fact.subject.type,
        "measure_canonical_id": fact.measure.canonical_id,
        "measure_surface_form": fact.measure.surface_form,
        "fact_type": fact.fact_type,
        "value_raw": fact.value.raw,
        "value_number": fact.value.number,
        "value_unit": fact.value.unit,
        "value_currency": fact.value.currency,
        "value_scale": fact.value.scale,
        "value_direction": fact.value.direction,
        "qualifiers": json.dumps(fact.qualifiers, default=str),
        "validity_start": fact.validity_interval.start.isoformat() if fact.validity_interval.start else None,
        "validity_end": fact.validity_interval.end.isoformat() if fact.validity_interval.end else None,
        "evidence": json.dumps(fact.evidence.model_dump(mode="json")),
        "confidence_extraction": fact.confidence.extraction,
        "confidence_normalization": fact.confidence.normalization,
        "provenance": json.dumps(fact.provenance.model_dump(mode="json")),
        "claim_key": compute_claim_key(fact),
        "loose_key": compute_loose_key(fact),
    }


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        id=row["id"],
        doc_id=row["doc_id"],
        block_id=row["block_id"],
        subject=Entity(
            canonical_id=row["subject_canonical_id"],
            surface_form=row["subject_surface_form"],
            type=row["subject_type"],
        ),
        measure=Measure(canonical_id=row["measure_canonical_id"], surface_form=row["measure_surface_form"]),
        fact_type=row["fact_type"],
        value=Value(
            raw=row["value_raw"],
            number=row["value_number"],
            unit=row["value_unit"],
            currency=row["value_currency"],
            scale=row["value_scale"],
            direction=row["value_direction"],
        ),
        qualifiers=json.loads(row["qualifiers"]),
        validity_interval=ValidityInterval(
            start=date.fromisoformat(row["validity_start"]) if row["validity_start"] else None,
            end=date.fromisoformat(row["validity_end"]) if row["validity_end"] else None,
        ),
        evidence=Evidence.model_validate(json.loads(row["evidence"])),
        confidence=Confidence(extraction=row["confidence_extraction"], normalization=row["confidence_normalization"]),
        provenance=Provenance.model_validate(json.loads(row["provenance"])),
    )


def _relation_row(relation: Relation) -> dict[str, Any]:
    fact_a_id, fact_b_id = sorted((relation.fact_a_id, relation.fact_b_id))
    return {
        "id": relation.id,
        "fact_a_id": fact_a_id,
        "fact_b_id": fact_b_id,
        "type": relation.type.value,
        "reason_code": relation.reason_code.value,
        "explanation": relation.explanation,
        "delta": json.dumps(relation.delta.model_dump(mode="json")) if relation.delta else None,
        "confidence": relation.confidence,
        "adjudicator": relation.adjudicator,
    }


def _row_to_relation(row: sqlite3.Row) -> Relation:
    return Relation(
        id=row["id"],
        fact_a_id=row["fact_a_id"],
        fact_b_id=row["fact_b_id"],
        type=RelationType(row["type"]),
        reason_code=ReasonCode(row["reason_code"]),
        explanation=row["explanation"],
        delta=Delta.model_validate(json.loads(row["delta"])) if row["delta"] else None,
        confidence=row["confidence"],
        adjudicator=row["adjudicator"],
    )


def _row_to_quarantine(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "doc_id": row["doc_id"],
        "fact": Fact.model_validate_json(row["fact"]),
        "reason": row["reason"],
        "score": row["score"],
    }


class Repo:
    def __init__(self, db_path: str | Path = "data/store.db"):
        path = Path(db_path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Repo:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- documents --

    def has_document(self, doc_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return row is not None

    def add_document(self, doc_id: str, pdf_path: str, doc_context: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO documents (doc_id, pdf_path, doc_context, ingested_at) VALUES (?, ?, ?, ?)",
            (doc_id, pdf_path, json.dumps(doc_context, default=str), datetime.now().isoformat()),
        )
        self.conn.commit()

    def get_document_context(self, doc_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT doc_context FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return json.loads(row["doc_context"]) if row else None

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        return {
            "doc_id": row["doc_id"],
            "pdf_path": row["pdf_path"],
            "doc_context": json.loads(row["doc_context"]),
            "ingested_at": row["ingested_at"],
        }

    def list_documents(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC").fetchall()
        return [
            {
                "doc_id": r["doc_id"],
                "pdf_path": r["pdf_path"],
                "doc_context": json.loads(r["doc_context"]),
                "ingested_at": r["ingested_at"],
            }
            for r in rows
        ]

    # -- facts --

    def add_fact(self, fact: Fact) -> None:
        """Never updates an existing row — a Fact, once persisted, is
        immutable (see module docstring). Re-adding the same fact.id is a
        no-op, which is what makes re-running ingest on an already-ingested
        document safe."""
        row = _fact_row(fact)
        columns = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        self.conn.execute(f"INSERT OR IGNORE INTO facts ({columns}) VALUES ({placeholders})", row)
        self.conn.commit()

    def get_fact(self, fact_id: str) -> Fact | None:
        row = self.conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return _row_to_fact(row) if row else None

    def facts_by_doc(self, doc_id: str) -> list[Fact]:
        rows = self.conn.execute("SELECT * FROM facts WHERE doc_id = ?", (doc_id,)).fetchall()
        return [_row_to_fact(r) for r in rows]

    def query_facts(
        self,
        doc_id: str | None = None,
        subject: str | None = None,
        measure: str | None = None,
        limit: int = 500,
    ) -> list[Fact]:
        """subject/measure match against canonical_id OR surface_form,
        case-insensitively, substring — filters are for a UI, not an exact
        key lookup (that's facts_sharing_claim_key/loose_key)."""
        clauses: list[str] = []
        params: list[Any] = []
        if doc_id:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if subject:
            clauses.append(
                "(LOWER(subject_canonical_id) LIKE ? OR LOWER(subject_surface_form) LIKE ?)"
            )
            needle = f"%{subject.lower()}%"
            params.extend([needle, needle])
        if measure:
            clauses.append(
                "(LOWER(measure_canonical_id) LIKE ? OR LOWER(measure_surface_form) LIKE ?)"
            )
            needle = f"%{measure.lower()}%"
            params.extend([needle, needle])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(f"SELECT * FROM facts {where} LIMIT ?", (*params, limit)).fetchall()
        return [_row_to_fact(r) for r in rows]

    def facts_sharing_claim_key(self, claim_key: str, exclude_fact_id: str | None = None) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE claim_key = ? AND id != ?", (claim_key, exclude_fact_id or "")
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def facts_sharing_loose_key(self, loose_key: str, exclude_fact_id: str | None = None) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE loose_key = ? AND id != ?", (loose_key, exclude_fact_id or "")
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def candidates_for_fact(self, fact: Fact, max_group_size: int = 50) -> list[tuple[Fact, str]]:
        """Facts ALREADY in the store (excluding `fact` itself, which the
        caller should have already inserted) sharing fact's claim_key
        (exact) or loose_key (loose, and not already counted as exact) —
        both indexed lookups, never a table scan. Mirrors
        link/candidates.py's exact-before-loose, no-duplicate-pairs
        contract, scoped to one fact against the store instead of all
        pairs within an in-memory batch."""
        exact = self.facts_sharing_claim_key(compute_claim_key(fact), exclude_fact_id=fact.id)[:max_group_size]
        exact_ids = {f.id for f in exact}
        loose_all = self.facts_sharing_loose_key(compute_loose_key(fact), exclude_fact_id=fact.id)
        loose = [f for f in loose_all if f.id not in exact_ids][:max_group_size]
        return [(f, "exact") for f in exact] + [(f, "loose") for f in loose]

    # -- quarantine --

    def add_quarantine(self, doc_id: str, fact: Fact, reason: str, score: float) -> None:
        self.conn.execute(
            "INSERT INTO quarantined (id, doc_id, fact_id, fact, reason, score) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), doc_id, fact.id, fact.model_dump_json(), reason, score),
        )
        self.conn.commit()

    def quarantined_by_doc(self, doc_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM quarantined WHERE doc_id = ?", (doc_id,)).fetchall()
        return [_row_to_quarantine(r) for r in rows]

    def all_quarantined(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM quarantined ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        return [_row_to_quarantine(r) for r in rows]

    # -- relations --

    def relation_exists(self, fact_a_id: str, fact_b_id: str) -> bool:
        a, b = sorted((fact_a_id, fact_b_id))
        row = self.conn.execute(
            "SELECT 1 FROM relations WHERE fact_a_id = ? AND fact_b_id = ?", (a, b)
        ).fetchone()
        return row is not None

    def add_relation(self, relation: Relation) -> None:
        row = _relation_row(relation)
        columns = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        self.conn.execute(f"INSERT OR IGNORE INTO relations ({columns}) VALUES ({placeholders})", row)
        self.conn.commit()

    def relations_by_doc(self, doc_id: str) -> list[Relation]:
        fact_ids = {f.id for f in self.facts_by_doc(doc_id)}
        if not fact_ids:
            return []
        placeholders = ", ".join("?" * len(fact_ids))
        rows = self.conn.execute(
            f"SELECT * FROM relations WHERE fact_a_id IN ({placeholders}) OR fact_b_id IN ({placeholders})",
            (*fact_ids, *fact_ids),
        ).fetchall()
        return [_row_to_relation(r) for r in rows]

    def get_relation(self, relation_id: str) -> Relation | None:
        row = self.conn.execute("SELECT * FROM relations WHERE id = ?", (relation_id,)).fetchone()
        return _row_to_relation(row) if row else None

    def query_relations(
        self, type: str | None = None, reason_code: str | None = None, limit: int = 500
    ) -> list[Relation]:
        clauses: list[str] = []
        params: list[Any] = []
        if type:
            clauses.append("type = ?")
            params.append(type)
        if reason_code:
            clauses.append("reason_code = ?")
            params.append(reason_code)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(f"SELECT * FROM relations {where} LIMIT ?", (*params, limit)).fetchall()
        return [_row_to_relation(r) for r in rows]
