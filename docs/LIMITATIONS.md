# Limitations

## Page-type router (`src/fkl/ingest/pdf.py`)

**Table false positives on regression equations.** `02-rbi-annual-report-2024-25-excerpt.pdf`
pages 55 and 87 classify as `table` but contain no table — they contain econometric
regression equations (e.g. `+ β2 (NonBank * Spreads,t) + Γ1' Xi,t–1 + Γ2' At–1 + ϵi,t`
on p55, `β5*Natural Endowmentsit + β6*i.country + β7*i.time` on p87). The numeric-cell
regex (`NUMERIC_RE`) matches subscript numbers glued to Greek-letter variable names
(`β2`, `Γ1`, `β5`, `β6`, `β7`), and their layout spacing happens to clear
`TABLE_MIN_GAP`, so `_line_is_table_row` reads them as 3+ widely-spaced numeric cells.

Left unfixed for now — tightening the regex risked being tuned to make this one case
disappear rather than fixing the underlying issue (the heuristic has no notion of
"this token is a variable with a subscript," only "this token matches a numeric
pattern"). Belongs in the `review_queue` (D5) once it exists: route low-confidence or
contested page-type classifications there instead of silently trusting them, and this
becomes a visible, explainable false positive rather than a silent one.

## Fact extraction (`src/fkl/extract/extractor.py`)

**`measure_surface_form` and `value_raw` collapse to the same string for
qualitative/narrative facts.** Measured on the real Delhivery annual report
(FY24, first 30 pages, 249 persisted facts): 41 facts (16%) have
`measure_surface_form == value_raw` verbatim. Breakdown by `fact_type`:
`categorical` 24, `assertion` 14, `numeric` 2, `relational` 1. `subject_surface_form`
is never involved — it's always a distinct real entity (e.g. "Delhivery Limited",
a named facility) — the collapse is specifically between measure and value.

Examples: a fact about Delhivery's mission statement ("We are enabling every
business to thrive and succeed in a demanding environment.") has that entire
sentence duplicated into both `measure_surface_form` and `value_raw`; a `numeric`
fact about a facility ("196 docking stations") has the count re-stated in
`value_raw` instead of being isolated to `"196"`.

Root cause: the extraction schema requires both `measure_surface_form` and
`value_raw` on every fact, but a pure qualitative assertion or categorical claim
drawn from narrative/marketing prose has no discrete value separate from the
claim itself — there's nothing else for the model to put in `value_raw`, so it
echoes the same text back. This is a schema/prompt gap (the measure/value split
implicitly assumes a numeric-fact shape), not a stray extraction bug — it recurs
proportionally to how much of a document is narrative prose vs. clean tabular
metrics.

Left unfixed for now: the fix is a prompt change (e.g. `value_raw` should be
null/omitted, or reduced to just the bare number, for non-numeric assertions),
and every extraction batch is cache-keyed on the prompt (`PROMPT_VERSION`) — any
prompt edit invalidates the whole cache and forces a full, costly re-run across
every document already ingested. Deferred until there's a natural reason to bump
the prompt version anyway.

## `validity_interval` is never populated by extraction (`src/fkl/extract/extractor.py`)

`reconcile/rules.py`'s SUPERSEDES/`TEMPORAL_STATE_CHANGE` branch — the dossier's
§2 case: a Non-Executive Independent Director role stated as current in the 2022
prospectus, then "(resigned w.e.f. February 11, 2023)" in the FY24 annual report
— depends entirely on `Fact.validity_interval` holding real, non-overlapping
date bounds. `extractor.py` never sets it: `grep -rn validity_interval
src/fkl/extract/extractor.py` returns nothing. Every extracted Fact keeps the
model's default `(None, None)`, which `_intervals_overlap()` treats as
unbounded — i.e. maximally overlapping — so the SUPERSEDES check can never
fire on real extracted data, regardless of how correct the underlying dates
are. `tests/test_reconcile_rules.py::test_dossier_case2_morparia_supersedes_critical`
only proves the *rules* branch is correct; it constructs `validity_start`/
`validity_end` by hand, which is not something anything upstream of it does.

Confirmed live on the real Morparia pair (prospectus p88 + annual report p91,
both narrowly extracted and persisted to `data/store.db` for this check):
`reconcile()` on the two actual extracted facts returns
`RECONCILED_BY_CONTEXT`/`UNRESOLVED`, not `SUPERSEDES` — and not even
`CONTRADICTS`, because a second, compounding gap hits first: the two facts'
`measure_surface_form` differ enough ("Non-Executive Independent Director of
our Company" vs "...resigned w.e.f. February 11, 2023", the resignation
clause folded into the measure itself by the model) that `claim_key.py`'s
`measure_key` never matches between them — they would never even become
linking *candidates* in the real pipeline, the recall gap already noted
above under "Anticipated limitations" in the README, now confirmed to
actually occur rather than just being theoretically possible.

Left unfixed for now: closing this needs two real changes, not a config
tweak — (1) deriving `validity_interval` from qualifiers the extractor
already captures in some cases (e.g. `effective_date`, "w.e.f."/"resigned"
language) or a dedicated normalizer pass, and (2) improving measure-surface-
form recall (embeddings, or a broader alias table) so differently-phrased
mentions of the same fact become comparable at all. Both are bigger than a
targeted fix and were deferred rather than rushed under time pressure; the
video's genuine-contradiction case uses a different, real pair instead (see
`docs/FOUR_CASES.md` §2).
