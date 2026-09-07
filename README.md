# Fact Knowledge Layer

Superjoin VIT 2026 assignment. In progress.

Every fact passes a fuzzy grounding gate before storage. Facts whose evidence
quote cannot be located in its source block are quarantined with a score,
never silently dropped.

## Run

```
.venv/Scripts/pip install -r requirements.txt   # first time only, if you haven't already
.venv/Scripts/python -m uvicorn apps.api.main:app --reload
```

Open http://127.0.0.1:8000 — upload a PDF, watch it ingest live (progress
streams in via SSE), then browse the extracted facts, reconciled relations,
and the grounding-gate review queue. `GET /docs` has the interactive API
reference (FastAPI's auto-generated OpenAPI UI).

Needs `GOOGLE_API_KEY` (or `GOOGLE_API_KEYS`, comma-separated) and
`GROQ_API_KEY` in `.env` — see `src/fkl/extract/providers.py`.
