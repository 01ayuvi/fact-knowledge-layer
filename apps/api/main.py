"""FastAPI app: fact-knowledge-layer API + the static frontend it serves.

Run: uvicorn apps.api.main:app --reload
(see README.md "Web UI" section)

Route registration order matters here: the API routers are included
first, then the frontend is mounted as a StaticFiles catch-all at "/" —
Starlette matches routes in registration order, so /documents, /facts,
etc. are always resolved by the API routers, never shadowed by the
static mount, even though it's also rooted at "/".
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from apps.api.routes import documents, facts, relations, review

app = FastAPI(title="Fact Knowledge Layer")

app.include_router(documents.router)
app.include_router(facts.router)
app.include_router(relations.router)
app.include_router(review.router)

app.mount("/", StaticFiles(directory="apps/web", html=True), name="web")
