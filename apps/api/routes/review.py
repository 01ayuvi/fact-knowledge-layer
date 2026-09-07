from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from apps.api.deps import get_repo
from src.fkl.store.repo import Repo

router = APIRouter()


@router.get("/review")
def review_queue(repo: Repo = Depends(get_repo)) -> list[dict[str, Any]]:
    """The quarantine queue: every fact the grounding gate rejected, with
    its reason and score, across all documents — see
    docs/SUPERJOIN_BUILD_PLAN.md D5 (the "self-surfacing failure ledger")."""
    return repo.all_quarantined()
