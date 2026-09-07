from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from apps.api.deps import get_repo
from src.fkl.store.models import Fact, Relation
from src.fkl.store.repo import Repo

router = APIRouter()


class RelationDetail(BaseModel):
    """GET /relations/{id}: both facts fully materialized (not just their
    ids) — the whole point is showing both evidence quotes side by side
    without a second round trip per fact."""

    relation: Relation
    fact_a: Fact
    fact_b: Fact


@router.get("/relations")
def list_relations(
    type: str | None = None,
    reason_code: str | None = None,
    repo: Repo = Depends(get_repo),
) -> list[Relation]:
    return repo.query_relations(type=type, reason_code=reason_code)


@router.get("/relations/{relation_id}")
def get_relation(relation_id: str, repo: Repo = Depends(get_repo)) -> RelationDetail:
    relation = repo.get_relation(relation_id)
    if relation is None:
        raise HTTPException(404, f"relation {relation_id!r} not found")
    fact_a = repo.get_fact(relation.fact_a_id)
    fact_b = repo.get_fact(relation.fact_b_id)
    if fact_a is None or fact_b is None:
        raise HTTPException(500, "relation references a fact that no longer exists")
    return RelationDetail(relation=relation, fact_a=fact_a, fact_b=fact_b)
