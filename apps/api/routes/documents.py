"""POST /documents streams ingest_document()'s progress back as SSE.
EventSource can't do POST-with-a-file-body, so the frontend uses fetch()
and reads the streamed response body itself (see apps/web/static/app.js) —
this is one endpoint doing both the upload and the progress stream, per
the brief, not a two-step upload-then-poll design.

The ingest itself runs on a background thread with its OWN Repo (SQLite
connections aren't thread-safe) — this endpoint's request-handling thread
only relays queued progress events, it never touches the repo directly
during ingest.
"""

from __future__ import annotations

import dataclasses
import json
import queue
import threading
from pathlib import Path
from typing import Any

import pymupdf
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from apps.api import deps
from apps.api.deps import get_repo
from src.fkl.pipeline import ingest_document
from src.fkl.store.repo import Repo

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
PAGE_RENDER_DPI = 150


def _json_safe(data: dict[str, Any]) -> dict[str, Any]:
    """Progress event payloads are plain JSON-safe dicts except "result",
    which carries pipeline.py's IngestResult dataclass."""
    out = dict(data)
    if "result" in out and dataclasses.is_dataclass(out["result"]):
        out["result"] = dataclasses.asdict(out["result"])
    return out


@router.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> StreamingResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf files are accepted")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)

    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def run_ingest() -> None:
        # ingest_document() already notifies an "error" stage itself before
        # re-raising (see pipeline.py) — this flag stops that from also
        # being duplicated by the except below, which previously pushed a
        # second, identical error event for every ingest failure (a user
        # watching the log would see "Error: ..." twice for one failure).
        # The except stays as a fallback for anything that fails OUTSIDE
        # ingest_document's own try/except, e.g. Repo(deps.DB_PATH) itself.
        error_notified = False

        def on_progress(stage: str, data: dict[str, Any]) -> None:
            nonlocal error_notified
            if stage == "error":
                error_notified = True
            event_queue.put({"stage": stage, **_json_safe(data)})

        try:
            repo = Repo(deps.DB_PATH)
        except Exception as exc:  # noqa: BLE001 — must reach the client as an SSE error event
            event_queue.put({"stage": "error", "message": str(exc)})
            event_queue.put(None)
            return

        try:
            ingest_document(str(dest), repo=repo, on_progress=on_progress)
        except Exception as exc:  # noqa: BLE001 — must reach the client as an SSE error event
            if not error_notified:
                event_queue.put({"stage": "error", "message": str(exc)})
        finally:
            repo.close()
            event_queue.put(None)  # sentinel: stream complete

    threading.Thread(target=run_ingest, daemon=True).start()

    def event_stream():
        while True:
            item = event_queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/documents")
def list_documents(repo: Repo = Depends(get_repo)) -> list[dict[str, Any]]:
    return repo.list_documents()


@router.get("/documents/{doc_id}/page/{n}.png")
def get_page_image(doc_id: str, n: int, repo: Repo = Depends(get_repo)) -> Response:
    doc = repo.get_document(doc_id)
    if doc is None:
        raise HTTPException(404, f"document {doc_id!r} not found")

    pdf_doc = pymupdf.open(doc["pdf_path"])
    try:
        if not (1 <= n <= len(pdf_doc)):
            raise HTTPException(404, f"page {n} out of range (document has {len(pdf_doc)} pages)")
        page = pdf_doc[n - 1]
        page_rect = page.rect
        png_bytes = page.get_pixmap(dpi=PAGE_RENDER_DPI).tobytes("png")
    finally:
        pdf_doc.close()

    # Point-dimensions of the PDF page, not the rendered PNG's pixel size —
    # the frontend uses these (not a hardcoded DPI) to scale evidence.bbox
    # (in PDF points) onto however large the <img> is actually displayed.
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Page-Width-Pt": str(page_rect.width),
            "X-Page-Height-Pt": str(page_rect.height),
        },
    )
