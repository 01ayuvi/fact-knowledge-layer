from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from apps.api.deps import get_repo
from src.fkl.store.models import Fact
from src.fkl.store.repo import Repo

router = APIRouter()


@router.get("/facts")
def list_facts(
    doc_id: str | None = None,
    subject: str | None = None,
    measure: str | None = None,
    repo: Repo = Depends(get_repo),
) -> list[Fact]:
    return repo.query_facts(doc_id=doc_id, subject=subject, measure=measure)


@router.get("/facts/{fact_id}")
def get_fact(fact_id: str, repo: Repo = Depends(get_repo)) -> Fact:
    fact = repo.get_fact(fact_id)
    if fact is None:
        raise HTTPException(404, f"fact {fact_id!r} not found")
    return fact
