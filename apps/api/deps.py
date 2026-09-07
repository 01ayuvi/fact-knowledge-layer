"""A fresh Repo (sqlite3 connection) per request. sqlite3 connections are
tied to the thread that created them, and FastAPI/Starlette can run sync
endpoint functions on different threadpool threads across requests — a
single shared connection would break. Opening/closing SQLite is cheap
enough that this isn't a real cost at this scale.

DB_PATH is a module attribute, not a value captured in a closure —
routes/documents.py's background ingest thread reads apps.api.deps.DB_PATH
directly (not `from ... import DB_PATH`) so that overriding it (tests do
this to point at an isolated temp DB) actually takes effect there too, not
just for the Depends(get_repo)-injected connections used by every other
route.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from src.fkl.store.repo import Repo

DB_PATH: str | Path = "data/store.db"


def get_repo() -> Iterator[Repo]:
    repo = Repo(DB_PATH)
    try:
        yield repo
    finally:
        repo.close()
