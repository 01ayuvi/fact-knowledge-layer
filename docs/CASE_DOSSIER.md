# Case Dossier — verified against the actual starter PDFs

Every quote below was pulled from the provided files with PyMuPDF. Page numbers are **excerpt page indices** (1-based within the supplied PDF), not the printed page numbers. Verify the printed numbers before putting them in your UI.

---

## 0. Corpus facts you need before writing code

| File | Pages | Chars | Chars/page | Text layer? |
|---|---|---|---|---|
| `01-delhivery-prospectus-2022-excerpt.pdf` | 100 | 329,348 | 3,293 | yes |
| `02-delhivery-annual-report-fy24-excerpt.pdf` | 100 | 611,268 | 6,112 | yes |
| `03-delhivery-q4-fy24-earnings-presentation.pdf` | 27 | 21,973 | 813 | yes (but see §5) |
| `01-india-economic-survey-2024-25-excerpt.pdf` | 89 | 203,118 | 2,282 | yes |
| `02-rbi-annual-report-2024-25-excerpt.pdf` | 100 | 273,826 | 2,738 | yes |
| `03-imf-india-2025-article-iv-excerpt.pdf` | 95 | 260,204 | 2,738 | yes |

**Total ≈ 1.70M characters ≈ 425K tokens.** Two consequences:

1. **No OCR needed.** All six have clean embedded text. Delete OCR from the plan — that is 3–4 hours saved. PyMuPDF alone gives you text + word bboxes for the grounding gate.
2. **The whole corpus fits in budget.** At ~425K tokens of input, full extraction over all six documents costs cents on a Flash-tier model. You can afford to re-run the entire pipeline many times during development. Cache by content hash anyway, but do not architect around token scarcity.

---

## 1. CORROBORATED, EXPRESSED DIFFERENTLY ✅ found

**Delhivery FY24 revenue — same number, three different renderings.**

| Source | Excerpt page | Stated as | Normalized |
|---|---|---|---|
| Annual Report FY24 | p22 | `Revenue from Operations ... 81,415.38` (₹ in Million, Consolidated, FY ended March 31, 2024) | ₹81,415.38 mn |
| Q4 FY24 earnings deck | p6 | `₹8,142 Cr FY24 revenue from services` | ₹8,142 Cr = ₹81,420 mn |
| Q4 FY24 earnings deck | p14 | `Revenue from customers ... FY24 8,142` | ₹8,142 Cr |

81,415.38 mn ÷ 10 = **8,141.54 Cr** vs stated **8,142 Cr** → delta 0.006%.

Why this is the best Case 1 in the corpus:
- Requires **scale normalization** (₹ million ↔ ₹ crore) — a pure-Python win, not an LLM guess.
- Requires **rounding tolerance** — you must justify your tolerance band, which is a real engineering decision to write up.
- Requires **measure canonicalization** — "Revenue from Operations" vs "revenue from services" vs "Revenue from customers" are three surface forms.
- Crosses document types (statutory filing ↔ investor deck).

**Handle this honestly, because it is a trap.** The deck footnote reads: *"Revenue from services excludes revenue from traded goods."* That is a genuinely narrower measure than Revenue from Operations, so the two should not be identical — they coincide only because FY24 traded-goods revenue was ≈0. A naive system says "match, done." Your system should say `CORROBORATES` **with a caveat flag** noting the measure definitions differ and the agreement is contingent. Saying that out loud in the video is worth more than the match itself.

**Second instance, same measure, prior year:** AR FY23 consolidated = 72,253.01 mn = 7,225.3 Cr; deck p14 table = 7,225 Cr. Exact.

**Macro instance (cross-publisher, cross-institution):**
- RBI Annual Report p8: *"real gross domestic product (GDP) growth moderated to 6.5 per cent in 2024-25"*
- IMF Article IV p10: *"India's real GDP grew by 6.5 percent in FY2024/25"*

Same value, different period **label** (`2024-25` vs `FY2024/25`), different publisher, different phrasing. Proving these are the same period is what your `periods.py` normalizer is for.

---

## 2. GENUINE CONTRADICTION ✅ found — and it is subtler than it looks

**Kalpana Jaisingh Morparia, director status.**

- Prospectus 2022, p88: *"Kalpana Jaisingh Morparia is a Non-Executive Independent Director of our Company."* — present tense, asserted as-of the prospectus date.
- Annual Report FY24, p91: *"Ms. Kalpana Jaisingh Morparia Non Executive - Independent Director (resigned w.e.f. February 11, 2023)"*

This is exactly the pattern the brief hints at. **But the correct verdict is not `CONTRADICTS`.**

A weak system flags a contradiction. A strong system recognises that both assertions are true over **non-overlapping validity intervals** and emits `SUPERSEDES` with reason code `TEMPORAL_STATE_CHANGE` — the fact has a validity interval `[prospectus_date, 2023-02-11)` and the later document closes it. The system only escalates to `CONTRADICTS` when the validity intervals genuinely overlap.

**Say this distinction on camera.** Emitting `SUPERSEDES` where everyone else emits `CONTRADICTS` is the clearest possible signal that you modelled time properly.

Corroborating instances of the same pattern in the same document (AR p91): Munish Ravinder Varma (resigned 29 Jun 2022), Agus Tandiono (resigned 8 Apr 2022) — both listed as active in the 2022 prospectus.

**For a true `CONTRADICTS`,** use the FY23 revenue rounding split in §5, or hunt the RBI vs IMF fiscal-deficit and CAD tables where vintages align. Have one genuine contradiction ready, but lead the demo with the `SUPERSEDES` reasoning — it is the more impressive artifact.

---

## 3. APPARENT CONTRADICTION EXPLAINED BY CONTEXT ✅ four candidates, all strong

Pick two. The first is the cleanest; the fourth is the most impressive.

**(a) `SCOPE_MISMATCH` — consolidation basis.** Annual Report p22, one table:
```
Particulars              Standalone FY24    Consolidated FY24
Revenue from Operations      74,540.82          81,415.38
```
Same measure, same period, same document, same units. Differs **only** by the consolidation qualifier. ₹7,454 Cr vs ₹8,142 Cr looks like a 9% contradiction until you read the column header. This is why your extractor must carry column headers into fact context, and why fixed-size chunking loses.

**(b) `MODALITY_MISMATCH` / `VINTAGE_RESTATEMENT` — estimate vs actual.**
- Economic Survey p4: *"As per the first advance estimates of national accounts, India's real GDP is estimated to grow by 6.4 per cent in FY25."*
- RBI AR p8 and IMF p10: **6.5 per cent** for the same year.

The explanation is sitting **inside the evidence span itself** — "first advance estimates". Published Jan 2025 vs later actuals. Demo this by highlighting the qualifying clause in the source PDF as the reason. Visually excellent.

Related, from RBI p69 and p96–97 footnotes: *"GDP for 2024-25 is as per second advance estimates."* Same measure, a third vintage.

**(c) `MEASURE_MISMATCH` — GDP vs GVA.** RBI AR p23: GDP at constant prices grew **6.5%** in 2024-25. RBI AR p26: real GVA at basic prices expanded **6.4%** in 2024-25. Same document, same period, same units, 10bps apart. Not a contradiction — GDP at market prices and GVA at basic prices are different measures separated by net indirect taxes. Requires real measure canonicalization to get right.

**(d) `PERIOD_BASIS_MISMATCH` — Indian fiscal quarter vs calendar quarter.** Both in the IMF report:
- p3: *"real GDP expanded by 7.8 percent in the first quarter of FY2025/26"*
- p10: *"real GDP growth of 7.8 percent in 2025Q2"*

**Same number, two incompatible-looking quarter labels, same document.** Q1 of Indian FY2025/26 (Apr–Jun 2025) *is* calendar 2025Q2. A system that resolves this proves its period normalizer genuinely understands the Apr–Mar fiscal convention rather than pattern-matching strings. This is your single best 20 seconds of video.

**(e) Two-dimensional resolution — the showpiece.** AR p41: *"During FY24, none of the Independent Directors resigned before expiry of his/her term."* AR p39: *"Post March 31, 2024, Mr. Sandeep Kumar Barasia resigned from the office of Executive Director & Chief Business Officer, with effect from July 01, 2024."*

Reads as a flat contradiction. Resolved by **two qualifiers simultaneously**: scope (Independent Director vs Executive Director) *and* period (during FY24 vs post FY24). Reason code `SCOPE_MISMATCH + PERIOD_DISJOINT`. If your reconciler handles compound explanations, show this one.

---

## 4. Also note: the macro corpus has vintage layering everywhere

The three macro documents were published at different times over the same fiscal year, so the same measure appears at three vintages: Economic Survey (Jan 2025, first advance estimates) → RBI AR (May 2025, provisional/second advance) → IMF Article IV (Nov 2025, actuals). Model **publication date as a first-class document attribute** and `VINTAGE_RESTATEMENT` becomes a general rule rather than a special case. Do this and the "generalizes beyond the starter documents" criterion is satisfied by construction.

---

## 5. EXTRACTION FAILURE ✅ found two, and one is architecturally important

**(a) Chart labels lose their number bindings — this affects your architecture.**

The earnings deck is chart-heavy. Raw text extraction of p9 yields:
```
59% 63% 62% 24% 16% 19% 4% 6% 7% 8% 11% 10% 7,054 7,224 8,142 FY22 FY23 FY24
Express Parcel PTL TL SCS Cross Border Revenue from services(1,2) (₹ Cr)
```
The numbers and their category and year labels are all present but the **binding between them is destroyed** by PDF reading order. Which segment is 24%? Which year is 7,054? A text-only pipeline either drops these or, worse, hallucinates the pairing.

**Architectural consequence:** the text layer is insufficient for the deck. Route slide/chart pages to a **multimodal model on the rendered page image** (PyMuPDF `page.get_pixmap()` at 150–200 DPI → vision model), and keep text-layer extraction for the prose-heavy filings. Detect which route to take with a cheap heuristic — chars-per-page below ~1,200 combined with high image-area coverage signals a slide.

This is a genuine, defensible engineering decision discovered from the data, and describing *how you discovered it* is precisely what "be honest about what works and what does not" is asking for. Most submissions will silently produce garbage here.

**(b) Same figure, two values, inside one document.** Deck p9 chart shows FY23 revenue as **7,224** Cr; deck p14 table shows FY23 revenue from customers as **7,225** Cr. AR consolidated FY23 = 72,253.01 mn = **7,225.3** Cr.

A rounding artifact, not a real disagreement — but it is a live test of your tolerance policy. Show the system flagging it, classifying it as `ROUNDING_ARTIFACT` below your tolerance threshold, and routing it to the review queue rather than declaring a contradiction. Then explain what tolerance you chose and why.

**(c) Footnote-carried qualifiers — the failure you should admit you only partly solved.** Throughout the corpus, a number's meaning is fixed by a footnote elsewhere on the page:
- `(2) FY22 numbers are on pro forma basis` (deck p9)
- `(1) Revenue from services excludes revenue from traded goods` (deck p6)
- `$ : GDP for 2024-25 is as per second advance estimates` (RBI p96)
- `(3) Revenue from Cross Border Services in FY22 included freight revenue of Rs 46 Cr from ... traded goods` (deck p10)

Detecting the superscript marker, locating the matching footnote, and attaching it as a qualifier is real work. Implement the easy case (same-page footnote markers via regex + position) and be candid that cross-page and table-footnote cases remain unhandled. **This is your Case 4 write-up** — a failure class you identified from the data, partially mitigated, with a specific named next step.

---

## 6. Revised priorities

Based on what is actually in these files:

**Cut from the plan:** OCR (unnecessary — all six have text layers). Heavy table-parsing APIs are optional too; the AR financial tables extract cleanly enough with PyMuPDF's block layout. Try free/local first and only reach for Reducto/LlamaParse if the RBI appendix tables actually fail.

**Add to the plan:**
1. **Page-type router** — prose vs table vs slide/chart, choosing text-layer or vision extraction per page. ~1 hour, and it is the difference between the deck working and not working.
2. **`validity_interval` on every fact**, with `SUPERSEDES` in the relation vocabulary. Required for Case 2 to be handled correctly rather than crudely.
3. **`vintage` / `modality` qualifier** with document publication date as the anchor. Unlocks Case 3(b) and generalizes.
4. **Same-page footnote resolution.** Partial credit is fine and honest.
5. **`ROUNDING_ARTIFACT` reason code** with an explicit, defended tolerance policy.

**Demo running order** (fits 3 minutes, each beat is now backed by verified evidence):

| Beat | Case | Why it lands |
|---|---|---|
| Delhivery FY24 revenue: 81,415.38 mn ↔ 8,142 Cr | 1 | Scale normalization visible, cross-document, caveat shown |
| Standalone 74,540.82 vs consolidated 81,415.38 | 3 | One table, one qualifier, instantly legible to a reviewer |
| IMF "Q1 FY2025/26" ↔ "2025Q2", both 7.8% | 3 | Proves the period model is real |
| Morparia active → resigned 11 Feb 2023 | 2 | `SUPERSEDES`, not `CONTRADICTS` — the sophistication beat |
| Chart-label binding failure + review queue | 4 | System-generated, with the architectural fix you shipped |

Lock these five into `docs/FOUR_CASES.md` today. You now have the whole demo scripted before you have written a line of pipeline code, which means Monday's feature freeze is realistic instead of aspirational.
