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
