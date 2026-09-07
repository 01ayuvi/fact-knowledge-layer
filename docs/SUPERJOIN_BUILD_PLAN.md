# Superjoin VIT 2026 — Fact Knowledge Layer: Build Plan

**Deadline:** 09 Sep 2026, 11:00 IST · **Now:** 07 Sep 2026 · **Working window:** ~2 days
**Datasets given:** `delhivery/` (prospectus 2022, AR FY24, Q4 FY24 deck) and `india-macroeconomy/` (Economic Survey 2024-25, RBI Annual Report 2024-25, IMF Article IV 2025)

---

## 1. What the brief is actually grading

Read the rubric literally. It says *"A graph database or visualization alone is not the solution. The interesting part is how facts are discovered, grounded, compared, and explained."*

That sentence is the whole assignment. Four verbs, four subsystems:

| Verb | Subsystem | What "good" looks like |
|---|---|---|
| **Discovered** | Extraction | Facts come from prose **and tables**. No hard-coded schema. |
| **Grounded** | Evidence | Page + character offset + **bounding box**, verified by string match back into the PDF. |
| **Compared** | Linking + reconciliation | Deterministic normalization first, LLM only for judgement calls. |
| **Explained** | Reason codes | Machine-readable *why*, not just LLM prose. |

### The dataset is telling you what to build

Both datasets were chosen deliberately. Look at what they contain:

- **Same measure, different vintage.** Economic Survey, RBI AR and IMF Article IV all publish India's FY25 real GDP growth — at different cut-off dates, with different revisions. → Case 1 and Case 2.
- **Same measure, different scope/unit.** Delhivery revenue appears as *revenue from operations* (consolidated) in the AR, *total income* in the prospectus, and as a chart label in the earnings deck. ₹ crore vs ₹ million vs US$ billion. Fiscal year vs calendar year. → Case 3.
- **Entity state changes over time.** Directors listed in the 2022 prospectus vs the FY24 board composition. An "active director" that is later resigned. → Case 2, and the brief literally hints at it.
- **Table-heavy appendices.** RBI macroeconomic appendix tables, IMF statistical tables, Delhivery consolidated financial statements. Most submissions will extract nothing usable from these.

**Design consequence:** the majority of "contradictions" in this corpus are *not* real contradictions. They are period, scale, unit, scope, or vintage mismatches. A system that just flags "these two numbers differ" scores badly. A system that names *which dimension* explains the difference scores extremely well — and that is exactly Case 3.

### What ~450 of the 500 will submit

LangChain/LlamaIndex → naive 1000-char chunking → OpenAI → "extract facts as JSON" → Neo4j → Streamlit with a force-directed graph → cosine similarity > 0.85 = "corroborates", < 0.5 = "contradicts" → four cases hand-picked and hard-narrated in the README.

Everything in Section 3 is chosen to be visibly not that.

---

## 2. Differentiators (ranked by marks-per-hour)

Build in this order. Each one is independently demoable in the 3-minute video.

**D1 — The grounding gate (highest ROI, ~2 hrs).**
Every extracted fact must carry a verbatim `evidence_quote`. Before it is stored, the quote is fuzzy-matched back into the page's extracted text. No match → fact is rejected or quarantined, never silently stored. State in the README: *"0 facts in this store are ungrounded, by construction."* This is a one-line claim no other submission can make.

**D2 — Document-level context inheritance (~2 hrs).**
Before extracting any fact, run one pass per document to establish: publisher, document type, as-of/publication date, reporting period, currency, default scale convention (crore/lakh/million), consolidation basis, geography. Every fact in that document inherits this as default qualifiers, overridable locally. This is the single decision that makes Case 3 work at all — otherwise "₹8,142" and "₹81.42" look like a contradiction instead of a scale convention difference.

**D3 — Deterministic reconciliation with reason codes (~4 hrs).**
The comparator is a typed decision procedure, not a prompt. It emits a code from a closed vocabulary:

```
CORROBORATES              values agree within tolerance after normalization
SCALE_MISMATCH            crore vs million vs absolute
UNIT_MISMATCH             ₹ vs US$ vs % of GDP
PERIOD_DISJOINT           FY24 vs FY25, Q4 vs full year
PERIOD_BASIS_MISMATCH     Indian FY (Apr–Mar) vs calendar year
SCOPE_MISMATCH            consolidated vs standalone; revenue vs revenue-from-operations
GEOGRAPHY_MISMATCH        India vs global
MODALITY_MISMATCH         actual vs projected vs estimated
VINTAGE_RESTATEMENT       later document restates an earlier figure
CONTRADICTS               same key, same qualifiers, values disagree
UNRESOLVED                sent to human review queue
```

The LLM writes the *sentence*; the code writes the *verdict*. Say this explicitly in the README — it is the strongest engineering signal in the whole submission.

**D4 — Table facts (~3 hrs).**
Extract from tables with row header + column header + section breadcrumb as context, keeping cell bounding boxes. RBI appendix tables and IMF statistical tables are where the highest-quality numeric facts live and where almost every competitor will produce garbage or nothing.

**D5 — Self-surfacing failure ledger (~1.5 hrs).**
A `review_queue` table fed automatically by: grounding-gate rejections, low parser confidence, ambiguous period parses, rule/LLM disagreement, orphan units. Case 4 asks you to find a failure. Everyone will write a paragraph. You will show a **screen** the system generated, with counts, and one worked example. Far stronger.

**D6 — Evaluation harness (~2 hrs).**
Hand-label ~60 facts and ~25 relation pairs across both datasets as a gold set. Report grounding precision, normalization accuracy, relation precision/recall. A submission with a number attached to its claims beats one with adjectives. Put the table in the README.

**D7 — Offline / zero-key mode (~1 hr).**
Commit a `data/cache/` of pre-computed extraction artifacts so `make demo` reproduces the entire knowledge layer with **no API key at all**. The brief explicitly worries about this ("include enough sample output... without needing your account"). Reviewers who can actually run it will rank you higher than reviewers who watched a video.

**D8 — Hosted demo URL (~1 hr).**
Deploy read-only to Fly.io / Railway / Render + Vercel. A clickable link at the top of the README beats every local setup guide.

**Brownie points, cheap to get:**
- *Incremental ingest* — falls out of the architecture for free. Only new facts get candidate-linked; nothing is rebuilt. Demo it: add a 4th PDF live, show the timer and "0 facts re-extracted".
- *Dynamic schema* — the `measure_registry` and `qualifier_registry` grow as new documents arrive. Show a live counter: "knows 143 measures, 27 qualifier dimensions across 6 documents."
- *Large PDFs* — page-parallel async extraction with a semaphore, content-hash caching, batched blocks. Report wall-clock for a 100-page PDF.
- *Many PDFs* — candidate linking is retrieval-bounded, not O(n²). Say so, and show the complexity note in `ARCHITECTURE.md`.

---

## 3. The stack

Deliberately overkill where it is visible, boring where it is not.

### Pipeline

```
PDF
 └─ Ingest        PyMuPDF          → page images, word-level text + bboxes, content hash
 └─ Layout        PyMuPDF + rules  → blocks: heading / para / table / caption / footnote
    (tables)      Docling or LlamaParse/Reducto on table-classified pages only
 └─ DocContext    1 LLM call/doc   → publisher, type, as-of date, currency, scale, basis
 └─ Extract       LLM, structured  → Fact[] per block-window, with verbatim quote
 └─ GATE          pure Python      → quote must match page text; else quarantine
 └─ Normalize     pure Python      → numbers, scales, units, periods, entities  (unit-tested)
 └─ Link          BM25 + vectors   → claim_key exact + hybrid retrieval + rerank → candidate pairs
 └─ Reconcile     rules → LLM      → Relation{type, reason_code, confidence, explanation}
 └─ Store         Postgres+pgvector
 └─ Serve         FastAPI (SSE)  →  Next.js + pdf.js highlight overlay
```

### Choices and why

| Layer | Choice | Why (and the trade-off to state in the README) |
|---|---|---|
| PDF primitives | **PyMuPDF** | Fast, local, free, gives word-level bboxes — the ground truth for D1. Trade-off: weak on complex tables, hence the second parser. |
| Table pages | **Docling** (self-host) or **LlamaParse**/**Reducto** (API) | Only invoked on pages classified as table-heavy — keeps cost and latency down. Reducto/LlamaParse return bboxes + citations natively. |
| Extraction LLM | **Gemini current Flash tier** | 1M context, native PDF/vision input, structured-output mode, generous free tier. Verify the model string and free-tier limits on the pricing page — the lineup has moved fast in 2026. |
| Adjudication LLM | **Claude Sonnet + Citations API** | Returns cited source spans natively. Perfect fit for "link every fact to evidence". Use for semantic/relational facts and contested pairs only. |
| Structured output | **Pydantic** + provider structured-output mode (or **BAML** / **Instructor**) | Schema-constrained generation; no regex JSON repair. |
| Embeddings | Voyage / OpenAI `text-embedding-3-large` / **bge-m3** local | For measure and entity canonicalization, not for deciding contradictions. |
| Rerank | **Cohere Rerank** or **Voyage rerank** | Cuts candidate pairs before the expensive comparator. Explicitly a cost/latency decision. |
| Store | **Postgres 16 + pgvector** | One container gives JSONB (dynamic qualifiers), full-text (BM25-ish), and vectors. Avoids the Neo4j trap the brief warns about. Graph is a *view*, not the substrate. |
| Queue | **Redis + arq** (or asyncio semaphore if time-pressed) | Page-parallel extraction, SSE progress to the UI. |
| API | **FastAPI + Pydantic v2**, SSE | Auto OpenAPI docs at `/docs` — reviewers can poke it without the UI. |
| UI | **Next.js + Tailwind + shadcn/ui**, **pdf.js** highlight overlay, **Cytoscape.js** for the (secondary) graph view | The PDF-highlight-on-click is the money shot of the video. |
| Tracing | **Langfuse** free tier | One screenshot in the README = visible engineering maturity. |
| Tests | pytest for the normalizer; gold-set eval script | Normalization is pure functions — cheap to test, high credibility. |

**Explicitly reject and say why in `DECISIONS.md`:** Neo4j as primary store (brief warns against graph-as-solution); naive fixed-size chunking (destroys table and section context); embedding-similarity-as-contradiction-detector (unprincipled); LangChain (an abstraction tax on a pipeline this specific).

---

## 4. The fact schema

This is the intellectual core — put it in `docs/SCHEMA.md` and walk through it in the video.

```jsonc
Fact {
  id, doc_id, block_id,
  subject:   { canonical_id, surface_form, type },        // Delhivery Ltd / India
  measure:   { canonical_id, surface_form },              // "revenue from operations"
  fact_type: "numeric" | "temporal" | "categorical" | "relational" | "assertion",
  value:     { raw, number, unit, currency, scale, direction },
  qualifiers:{                                            // JSONB — grows dynamically
     period_start, period_end, period_label, period_basis, // FY_IN | CY | Q
     consolidation, scope, geography, segment, modality,   // actual|projected|estimated|restated
     source_stated_as
  },
  evidence:  { page, char_start, char_end, bbox[], quote, section_path, table_ref },
  confidence:{ extraction, normalization },
  provenance:{ model, prompt_version, extracted_at, parser }
}

Relation {
  id, fact_a_id, fact_b_id,
  type: "CORROBORATES" | "CONTRADICTS" | "RECONCILED_BY_CONTEXT" | "REFINES" | "SUPERSEDES",
  reason_code,            // closed vocabulary — see D3
  explanation,            // LLM prose, grounded in both quotes
  delta: { absolute, relative, after_normalization },
  confidence,
  adjudicator: "rule" | "llm" | "rule+llm"
}
```

`claim_key = hash(subject.canonical_id, measure.canonical_id, period_norm, consolidation, geography)`
Facts sharing a key are compared directly. Facts with near keys are retrieved as candidates. This is what makes the system generalize to unseen PDFs and scale past two datasets.

**Dynamic schema story:** `qualifiers` is an open JSONB bag. A `qualifier_registry` records every key ever observed, its value domain, and whether the comparator has learned to treat it as an *explanatory dimension*. New document types introduce new qualifier keys without a migration. That is the brownie point, delivered honestly.

---

## 5. The four required cases — pre-plan them now

Do not leave these to chance on the last morning. Go find the candidate pairs today, while building.

| Case | Likely source | What to show on screen |
|---|---|---|
| **1. Corroborated, expressed differently** | India FY25 real GDP growth in Economic Survey vs RBI AR vs IMF — one says "6.4 per cent", another a table cell, another "6½ percent" | Both quotes highlighted in their PDFs, normalized values equal, `CORROBORATES`, tolerance shown |
| **2. Genuine/likely contradiction** | A director active in the 2022 prospectus but resigned per the FY24 AR; or two irreconcilable figures for the same normalized key | Both spans, `CONTRADICTS`, reason code, confidence, plus *why* the system ruled out the benign explanations |
| **3. Apparent contradiction, explained** | Delhivery revenue ₹ crore (AR) vs ₹ million (prospectus) vs a Q4-only figure in the deck; or IMF CY vs Indian FY GDP growth | `RECONCILED_BY_CONTEXT` + `SCALE_MISMATCH` / `PERIOD_BASIS_MISMATCH`, showing the normalization trace that resolved it |
| **4. Failure you found** | Multi-page table with repeated headers → row misattribution; or a footnote-qualified number extracted without its footnote | The auto-generated review-queue screen, one worked failure, root cause, the mitigation you shipped, the fix you'd build next |

Write these up in `docs/FOUR_CASES.md` with screenshots and permalinks into the UI (`/facts/{id}`, `/relations/{id}`). Link that file from the README. **This file is the single most-read artifact in your submission after the video.**

---

## 6. Repository structure

```
fact-knowledge-layer/
├── README.md                      ← the graded document. Sections in the brief's exact order.
├── LICENSE
├── Makefile                       ← make setup / make demo / make eval / make test
├── docker-compose.yml             ← postgres+pgvector, redis, api, web — one command
├── .env.example                   ← no secrets, ever
├── .github/workflows/ci.yml       ← lint + tests. Cheap green badge.
│
├── docs/
│   ├── ARCHITECTURE.md            ← the pipeline diagram + data flow
│   ├── SCHEMA.md                  ← Fact / Relation / claim_key, with rationale
│   ├── RECONCILIATION.md          ← the decision procedure + reason-code vocabulary
│   ├── FOUR_CASES.md              ← ★ the four required cases, with evidence screenshots
│   ├── DECISIONS.md               ← ADR-style: what was rejected and why
│   ├── EVALUATION.md              ← gold set, metrics, results table
│   ├── LIMITATIONS.md
│   └── assets/                    ← diagrams, screenshots, gifs
│
├── apps/
│   ├── api/                       ← FastAPI
│   │   ├── main.py
│   │   ├── routes/                ← documents.py, facts.py, relations.py, review.py, stream.py
│   │   ├── deps.py
│   │   └── schemas.py
│   └── web/                       ← Next.js
│       ├── app/
│       │   ├── page.tsx                 ← upload + corpus overview
│       │   ├── documents/[id]/page.tsx  ← PDF viewer + highlight overlay
│       │   ├── facts/page.tsx           ← filterable fact table
│       │   ├── relations/[id]/page.tsx  ← side-by-side evidence + reasoning trace
│       │   └── review/page.tsx          ← the failure ledger
│       └── components/            ← PdfHighlighter, FactCard, RelationDiff, ReasonBadge
│
├── src/fkl/                       ← the library. Importable, testable, UI-agnostic.
│   ├── ingest/                    ← pdf.py, layout.py, tables.py, hashing.py
│   ├── context/                   ← doc_context.py  (document-level defaults)
│   ├── extract/                   ← prompts/, extractor.py, grounding_gate.py
│   ├── normalize/                 ← numbers.py, units.py, periods.py, entities.py, measures.py
│   ├── link/                      ← claim_key.py, candidates.py, rerank.py
│   ├── reconcile/                 ← rules.py, reason_codes.py, adjudicator.py
│   ├── registry/                  ← measure_registry.py, qualifier_registry.py
│   ├── store/                     ← models.py, repo.py, migrations/
│   ├── eval/                      ← gold.py, metrics.py, run_eval.py
│   └── config.py
│
├── data/
│   ├── samples/                   ← the six provided PDFs
│   └── cache/                     ← ★ committed artifacts → `make demo` runs with NO API key
│
├── evals/
│   ├── gold_facts.jsonl           ← ~60 hand-labelled facts
│   └── gold_relations.jsonl       ← ~25 hand-labelled pairs
│
├── scripts/
│   ├── ingest_samples.py
│   ├── export_case_studies.py
│   └── seed_demo.sh
│
└── tests/
    ├── test_normalize_periods.py  ← FY24 / FY2023-24 / Q4FY24 / CY2024 / Apr–Mar
    ├── test_normalize_units.py    ← crore / lakh / mn / bn / % of GDP
    ├── test_grounding_gate.py
    └── test_reconcile_rules.py    ← table-driven: pair in, reason code out
```

**Git hygiene the brief explicitly asks for ("use git meaningfully"):**
- Conventional commits (`feat(reconcile): add period-basis reason code`).
- Branch per subsystem, squash-merged via PR — even solo. A visible PR history reads as engineering discipline.
- Tag `v0.1-ingestion`, `v0.2-extraction`, `v0.3-reconciliation`, `v1.0-submission`.
- No `.env`, no keys, no `node_modules`, no 100-page PDFs committed twice.

---

## 7. Timeline

Two working days. Ship-blockers first, features last. Freeze early.

### Sun 07 Sep — foundations
| Time | Task |
|---|---|
| Now – +2h | Repo scaffold, docker-compose up, Postgres+pgvector reachable, FastAPI `/health`, CI green |
| +2h – +5h | Ingest: PyMuPDF → blocks + bboxes + section breadcrumbs; content hashing; store documents/blocks |
| +5h – +7h | Document-context pass (D2) + normalizer (`periods.py`, `units.py`, `numbers.py`) **with tests** |
| +7h – +10h | Extraction prompt + structured output + **grounding gate (D1)**. First facts landing in Postgres. |
| End of day | **Checkpoint: facts exist and every one is grounded.** Manually eyeball 20. |

### Mon 08 Sep — the interesting half
| Time | Task |
|---|---|
| Morning | claim_key + candidate linking (BM25 + vectors + rerank); table extraction for table-heavy pages (D4) |
| Midday | Reconciliation rules + reason codes (D3); LLM adjudicator for semantic pairs; review queue (D5) |
| Afternoon | UI: upload → SSE progress → fact table → **PDF highlight overlay** → relation detail view |
| **17:00 hard stop** | **FEATURE FREEZE.** Anything not working now is a "Next Step". |
| Evening | Hunt and lock the four cases; write `FOUR_CASES.md`; run gold-set eval (D6); write README |
| Late | Deploy hosted demo (D8); build `data/cache/` offline mode (D7) |

### Tue 09 Sep — ship
| Time | Task |
|---|---|
| 06:30 – 07:30 | Clean-clone test: `git clone && make demo` on a fresh machine. Fix only what breaks setup. |
| 07:30 – 09:00 | Record the video. Script it. Two or three takes. ≤3:00, hard limit. |
| 09:00 – 10:00 | README polish, links, screenshots, tag `v1.0-submission` |
| 10:00 – 10:30 | Submit the form |
| 10:30 – 11:00 | Buffer. Something will go wrong here. |

---

## 8. Video script (3 minutes, timed)

The video is likely the *only* thing every reviewer consumes fully. Script it to the second.

| Time | Beat |
|---|---|
| 0:00–0:20 | One sentence on the model: facts + qualifiers + verified evidence spans + typed relations. Show the architecture diagram for 5 seconds. |
| 0:20–0:45 | Upload a **fresh, unseen PDF** live. SSE progress bar. Say the page count and elapsed time out loud. (Proves no hard-coding — the brief tests this.) |
| 0:45–1:10 | **Case 1** — click the fact, both source PDFs highlight the exact spans, `CORROBORATES` badge. |
| 1:10–1:35 | **Case 3** — the apparent contradiction, then the normalization trace resolving it, `RECONCILED_BY_CONTEXT / SCALE_MISMATCH`. This is your best 25 seconds; do not rush it. |
| 1:35–2:00 | **Case 2** — the genuine contradiction, both spans, why benign explanations were ruled out. |
| 2:00–2:20 | **Case 4** — the review queue, generated by the system, one worked failure. |
| 2:20–2:45 | Incremental ingest: add a 4th document, show the timer, show the registry counter growing. |
| 2:45–3:00 | Eval table on screen. One line on the biggest limitation. Stop. |

No intro music. No talking-head opener. No "hi my name is". Start on the product.

---

## 9. API and service shortlist

Everything here has a free tier or trivial cost at this scale. Get keys **today** — quota approval delays have killed better projects than this one.

### Tier 1 — use these
| Service | Role | Why it stands out |
|---|---|---|
| **Google Gemini (current Flash tier)** | Bulk fact extraction | 1M context, native PDF+vision input, structured outputs, real free tier. Check `ai.google.dev/gemini-api/docs/pricing` for the current model string — the lineup changed repeatedly through 2026. |
| **Anthropic Claude + Citations API** | Semantic/relational facts, contested-pair adjudication | Returns **cited source spans natively**. Almost nobody in a 500-person cohort will know this endpoint exists, and it maps 1:1 onto "link every fact to evidence". |
| **PyMuPDF** | bbox ground truth, page rendering | Free, local, fast. The backbone of the grounding gate. |

### Tier 2 — the visible differentiators
| Service | Role | Notes |
|---|---|---|
| **Docling** (open source) | Table + layout parsing | Self-hosted, no key, strong on financial tables. Zero-cost credibility. |
| **LlamaParse** / **Reducto** / **Chunkr** | Table pages, API route | Reducto returns bounding boxes and citations per extracted value; LlamaParse has a usable free daily page quota. Invoke only on table-classified pages. |
| **Cohere Rerank** or **Voyage rerank** | Candidate pair pruning | Turns O(n²) comparison into a bounded top-K problem. A real scalability argument, not a hand-wave. |
| **Voyage / OpenAI embeddings**, or **bge-m3** local | Measure + entity canonicalization | Local model = runs without keys, supports offline mode. |
| **Langfuse** (free cloud) | LLM tracing, cost, latency | One screenshot in the README signals production thinking. |

### Tier 3 — polish that buys goodwill
| Service | Role |
|---|---|
| **Fly.io / Railway / Render** + **Vercel** | Live hosted demo URL at the top of the README |
| **OpenRouter** | Provider fallback so a reviewer with *any* key can run it |
| **promptfoo** or plain pytest | Prompt regression tests against the gold set |
| **GitHub Actions** | Green CI badge; runs the normalizer tests |

### Deliberately avoided
Neo4j/AuraDB as the primary store (the brief warns against graph-as-answer — expose a graph *view* over Postgres instead), Pinecone/Weaviate (pgvector is enough and one less container), LangChain agents (unnecessary indirection over a fixed pipeline), any paid service without a free tier the reviewer can reproduce.

---

## 10. README structure (the brief dictates it — follow exactly)

```
# Fact Knowledge Layer
[hosted demo] [3-min video] [architecture diagram]
One-paragraph pitch + one screenshot of the highlight-on-click view

## Setup and Run Instructions        ← docker compose up; make demo (works with NO API key)
## Video Demo                        ← link, ≤3:00
## Approach                          ← architecture, fact schema, reconciliation procedure,
                                        key trade-offs, AI tools used (name them honestly)
## The Four Required Cases           ← summary table + link to docs/FOUR_CASES.md
## Evaluation                        ← the metrics table. Numbers, not adjectives.
## Limitations and Next Steps
## Additional Notes                  ← incremental ingest, dynamic schema, perf numbers
```

Put the hosted demo link, the video link and one screenshot **above the fold**. A reviewer grading 500 repos decides in fifteen seconds whether to keep reading.

---

## 11. Risk register

| Risk | Mitigation |
|---|---|
| Extraction quality eats the whole weekend | Timebox to Sun evening. A smaller, grounded fact set beats a large noisy one — the brief says this outright. |
| No genuine contradiction found in the corpus | Start hunting **today**, in parallel with building. If nothing clean emerges, the director-status change across Delhivery documents is a reliable fallback. |
| API rate limits mid-demo | Cache aggressively by content hash; record the video against the cache; keep a second provider key. |
| Video runs over 3:00 | Script it, rehearse once, cut the architecture explanation first. Over-length may simply not be watched. |
| Reviewer cannot run it | `make demo` offline mode + hosted URL. Test from a clean clone Tuesday morning. |
| Scope creep | Feature freeze 17:00 Monday is non-negotiable. Everything after that is docs, video, and eval. |

---

## 12. The one-line pitch

> Not a graph of extracted numbers — a **claim ledger**: every fact carries a verified evidence span and a qualifier bundle (period, unit, scope, basis, vintage), and every relationship between facts is decided by a typed procedure that names *which dimension* explains the difference, with the language model writing the explanation rather than the verdict.

Lead the README with that sentence.
