# The Four Required Cases — real records from `data/store.db`

Every fact and relation below is a live, persisted record — pulled directly from the seeded database, not reconstructed from the dossier. Relation ids are given so each case can be looked up directly in the UI (`GET /relations/{id}`, or the Relations tab).

---

## Case 1 — Corroborated across documents, expressed differently

**Verdict: `CORROBORATES` / reason code `SCALE_NORMALIZED`, with a caveat**

**Relation id:** `e3147902-557e-440d-81f9-f49773add821`

| | Fact A | Fact B |
|---|---|---|
| Document | `02-delhivery-annual-report-fy24-excerpt` (Annual Report FY24), p22 | `03-delhivery-q4-fy24-earnings-presentation` (Q4 FY24 deck), p17 |
| Subject | Delhivery Limited | Delhivery Limited |
| Measure | Revenue from Operations | Revenue for services (A) |
| Value | 81,415.38 (scale: million, INR) | 8,142 (scale: crore, INR) |
| Qualifiers | `column_header`: "Consolidated – FY ended March 31, 2024", `consolidation`: "consolidated", `period_label`: "FY24" | `column_header`: "FY24", `period_label`: "FY24" |
| Evidence quote | `"81,415.38"` | `"8,142"` |

**Explanation:** *"81,415,380,000.00 vs 81,420,000,000.00 — 0.0057% apart, within the defended rounding tolerance of 0.10%."*

**Caveat:** *"Linked via measure aliasing ('Revenue from Operations' ~ 'Revenue for services (A)') — not identical wording. The definitions may differ (e.g. one may exclude a component the other includes); this agreement is contingent, not guaranteed to hold in general."*

This is the dossier's own flagged trap (§1): 81,415.38 mn ÷ 10 = 8,141.54 Cr vs the stated 8,142 Cr — a 0.006% delta. The two measures are worded differently ("Revenue from Operations" vs "Revenue for services") and the deck's own footnote states revenue from services *excludes* revenue from traded goods — a genuinely narrower definition. They agree here only because FY24 traded-goods revenue was ≈0. `normalize/measures.py` links them as candidates via an explicit alias table (never by silently treating the wording as identical); `reconcile/rules.py` forces the reason code to `SCALE_NORMALIZED` and attaches the caveat above whenever a match was produced *through* aliasing rather than identical wording, so this agreement is never presented as unconditional.

---

## Case 2 — Genuine or likely contradiction

**Verdict: `CONTRADICTS` / reason code `VALUES_DIVERGE`**

**Relation id:** `1e4743a1-1fc8-4b6b-b474-861a847cd052`

| | Fact A | Fact B |
|---|---|---|
| Document | `02-delhivery-annual-report-fy24-excerpt`, p24 | `02-delhivery-annual-report-fy24-excerpt`, p24 |
| Subject | Delhivery Limited Board | Delhivery Limited Board |
| Measure | Board meeting held on | Board meeting held on |
| Value | August 04, 2023 | August 24, 2023 |
| Qualifiers | `period_label`: "FY2023-24" | `period_label`: "FY2023-24" |
| Evidence quote | `"meeting held on August 04, 2023"` | `"meeting held on August 24, 2023"` |

**Explanation:** *"Fact A's evidence quote says the board meeting was held on August 04, 2023, while Fact B's evidence quote says it was held on August 24, 2023; the two different dates conflict, so the statements contradict each other."*

Only 2 `CONTRADICTS` relations remain in the store after the Case 4 fix (down from 107). This is the stronger of the two mechanically — a clean value divergence between two high-confidence (0.99) temporal facts with unambiguous, distinct evidence quotes. (The other, `authorised share capital` = ₹1,342,535,980 vs `"no change"`, compares a number against a qualitative assertion rather than two disagreeing numbers — a value-comparability artifact, not really a contradiction: "no change during FY24" is consistent with the capital having *been* ₹1,342,535,980 all year.)

**Honest caveat on this one, found while preparing this document:** both facts come from the *same source block* (`p24:b2`, identical bounding box) and carry no qualifier distinguishing which meeting is which — the annual report's board-composition-changes section evidently lists multiple meeting dates in one passage, and the extractor captured two of them as separate facts sharing one generic measure name. That is almost certainly **two different, real board meetings**, not a contradiction — the same claim-key-over-collapse failure mode as Case 4 (§ below), just without a `date` qualifier this time (the date *is* the value here, not a separate field, so the Case 4 fix doesn't reach it). Flagging this transparently rather than presenting it as a clean "genuine contradiction": the system correctly executed its rules given what it extracted, but what it extracted needs a per-meeting qualifier (e.g. an ordinal, or the specific agenda item) to be a comparable claim at all. This is a real, additional instance of the limitation Case 4 fixes one instance of — see "Next steps" in the README.

---

## Case 3 — Apparent contradiction explained by context

**Verdict: `RECONCILED_BY_CONTEXT` / reason code `SCOPE_MISMATCH`**

**Relation id:** `1dcc69a7-9f56-41aa-95fd-f456dd0d497b`

| | Fact A | Fact B |
|---|---|---|
| Document | `02-delhivery-annual-report-fy24-excerpt`, p22 | `02-delhivery-annual-report-fy24-excerpt`, p22 |
| Subject | Delhivery Limited | Delhivery Limited |
| Measure | Revenue from Operations | Revenue from Operations |
| Value | 74,540.82 (scale: million, INR) | 81,415.38 (scale: million, INR) |
| Qualifiers | `column_header`: "Standalone – FY ended March 31, 2024", `consolidation`: "standalone", `period_label`: "FY24" | `column_header`: "Consolidated – FY ended March 31, 2024", `consolidation`: "consolidated", `period_label`: "FY24" |
| Evidence quote | `"74,540.82"` | `"81,415.38"` |

**Explanation:** not yet written (`explanation_pending: true`) — both providers were quota-exhausted when this pair was reconciled. The verdict and reason code above came from deterministic rules alone (`reconcile/rules.py`'s consolidation check), independent of any LLM call; only the natural-language narration is pending. It can be backfilled by re-running reconciliation once quota is available — `relation_exists()` will leave every other already-decided relation untouched.

Same-page, same period, same measure, same document — the two figures differ by ~9% only because one column is Standalone and the other Consolidated (docs/CASE_DOSSIER.md §3(a)). `reconcile/rules.py` checks the `consolidation` qualifier before it ever compares values, so this is caught as a scope difference, not miscategorized as a numeric disagreement.

---

## Case 4 — Extraction/reasoning failure found and handled: claim-key over-collapse

**What broke:** facts carrying a specific `date` qualifier alongside a coarser `period_label` were keyed on the fiscal year alone. A monthly "Employee Stock Options Exercised" table — 12 distinct rows, one per month (April 2023 through January 2024), each a genuinely different event — all shared `period_label: "FY2023-24"` with no period-level distinction between them. `link/claim_key.py`'s period resolution only consulted `period_label`/`period_start`/`period_end`, never the `date` qualifier the extractor had already correctly captured per row. Every pair of months therefore collapsed onto the same exact claim key and got value-compared against each other as if they were repeated measurements of one thing.

**Root cause**, confirmed directly in code (`src/fkl/link/claim_key.py`, before the fix):

```python
def claim_key_components(fact: Fact) -> ClaimKeyComponents:
    period_start, period_end = fact_period(fact)   # only ever read period_label/period_start/period_end
    ...
```

**Before / after**, measured on the same 240 candidate pairs in the Delhivery annual report (first 30 pages):

| Type / reason code | Before | After |
|---|---|---|
| `CONTRADICTS` / `VALUES_DIVERGE` | **107** | **2** |
| `RECONCILED_BY_CONTEXT` / `PERIOD_DISJOINT` | 8 | **113** |
| `RECONCILED_BY_CONTEXT` / `SCOPE_MISMATCH` | 94 | 94 |
| `RECONCILED_BY_CONTEXT` / `UNRESOLVED` | 29 | 28 |
| `CORROBORATES` / `VALUES_MATCH` | 1 | 2 |
| `REFINES` / `VALUES_MATCH` | 1 | 1 |
| **Total relations** | 240 | 240 |

Exactly the 105 false contradictions moved to `PERIOD_DISJOINT`, where they belong: different dates, not disagreeing values. Example pair, before the fix:

```
FACT A: measure='Employee Stock Options Exercised' value='158,855' qualifiers={"date": "2023-04-06", "period_label": "FY2023-24"}
  evidence: "April 06, 2023 Employee Stock Options Exercised 158,855"
FACT B: measure='Employee Stock Options Exercised' value='1,941,454' qualifiers={"date": "2023-06-08", "period_label": "FY2023-24"}
  evidence: "June 08, 2023 Employee Stock Options Exercised 1,941,454"
```
→ was `CONTRADICTS`/`VALUES_DIVERGE` (April's figure compared against June's, as if disagreeing about one number). After the fix, this pair resolves to `RECONCILED_BY_CONTEXT`/`PERIOD_DISJOINT`: different months, not a disagreement.

**The fix** (`src/fkl/link/claim_key.py`'s `fact_period()`): a `date` qualifier, when present, is folded into period resolution as a same-day `(date, date)` interval and takes precedence over a coarser `period_label` — a specific date is strictly more precise than a fiscal-year label, so it wins. Existing facts' `claim_key`/`loose_key` columns were backfilled (568 facts checked, 18 changed) since they're computed once at persistence time, not recomputed on read.

**How this was found:** by checking store counts against the extraction phase's reported total rather than trusting either number alone — 107 `CONTRADICTS` relations in a single ~500-fact document was implausibly high on its face, which prompted pulling concrete examples rather than accepting the aggregate count. See `tests/test_link_claim_key.py::test_date_qualifier_takes_precedence_over_period_label` for the regression test.
