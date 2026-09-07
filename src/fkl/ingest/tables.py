"""Row/column reconstruction for page_type == "table" pages.

Block.text (from pdf.py) is a flattened string — it throws away the
per-span x/y geometry a table needs. This module re-opens the source PDF
page and rebuilds that structure: which spans form a row (reusing pdf.py's
_block_rows y-clustering) and which column each cell belongs to (by x-range
overlap against header spans found above the data).

Deliberately scoped to single-page, non-merged-cell tables with headers
directly above the data — that covers AR page 22 and similar simple
financial-statement tables, not multi-page or rowspan/colspan layouts.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pymupdf

from src.fkl.ingest.pdf import NUMERIC_RE, _block_rows, _has_table_row

HEADER_SEARCH_MARGIN = 60.0
CAPTION_SEARCH_MARGIN = 50.0
CAPTION_HEADING_MAX_CHARS = 80
X_OVERLAP_MIN_FRACTION = 0.3
COLUMN_MATCH_PAD = 8.0
SCALE_ANNOTATION_RE = re.compile(
    r"\(\s*[₹$]?\s*(?:in\s+)?(million|crore|lakh|thousand|billion|%|percent)", re.IGNORECASE
)


def _flat_block_text(b: dict) -> str:
    return " ".join(
        span["text"].strip() for line in b.get("lines", []) for span in line.get("spans", [])
    ).strip()


def _looks_like_caption_heading(text: str) -> bool:
    """Same short/unpunctuated heuristic as extractor.py's _looks_like_heading
    (duplicated locally to avoid a circular import between tables.py and
    extractor.py)."""
    stripped = text.strip()
    return (
        bool(stripped)
        and len(stripped) <= CAPTION_HEADING_MAX_CHARS
        and not stripped.endswith((".", ",", ";"))
    )


@dataclass
class TableCell:
    text: str
    bbox: tuple[float, float, float, float]
    source_block_no: int


@dataclass
class TableRow:
    label: TableCell | None
    values: list[TableCell]


@dataclass
class ReconstructedTable:
    page_no: int
    caption: str
    rows: list[TableRow]
    header_cells: list[TableCell]
    header_matches: list[tuple[TableCell, float, float]]

    def column_header_for(self, cell: TableCell) -> str:
        return _header_text_for(cell, self.header_matches)


def _row_from_spans(spans: list[dict], block_no: int) -> list[TableCell]:
    return [
        TableCell(text=s["text"].strip(), bbox=tuple(s["bbox"]), source_block_no=block_no)
        for s in sorted(spans, key=lambda s: s["bbox"][0])
        if s["text"].strip()
    ]


def _x_overlap_fraction(bbox: tuple[float, float, float, float], x0: float, x1: float) -> float:
    ox0, ox1 = max(bbox[0], x0), min(bbox[2], x1)
    if ox1 <= ox0:
        return 0.0
    width = bbox[2] - bbox[0]
    return (ox1 - ox0) / width if width > 0 else 0.0


def _widen_ranges_within_row(row: list[TableCell]) -> list[tuple[TableCell, float, float]]:
    """A header cell's own bbox is often narrower than the column(s) it
    labels — e.g. "Standalone – FY ended" sits visually above just the first
    of the two Standalone date columns, but conceptually spans both. Extend
    each cell's matching range to the midpoint gap to its row-neighbors (open
    to +/-inf at the row's ends) so a spanning header matches every column
    beneath it, not just the one its text happens to sit above."""
    sorted_cells = sorted(row, key=lambda c: c.bbox[0])
    if len(sorted_cells) == 1:
        # No siblings to bound against (e.g. a lone "Particulars" row-label
        # header) — widening to +/-inf here would make it match every value
        # cell on the page. Keep its own tight bbox instead.
        c = sorted_cells[0]
        return [(c, c.bbox[0], c.bbox[2])]
    out: list[tuple[TableCell, float, float]] = []
    for i, cell in enumerate(sorted_cells):
        left = -math.inf if i == 0 else (sorted_cells[i - 1].bbox[2] + cell.bbox[0]) / 2
        right = (
            math.inf
            if i == len(sorted_cells) - 1
            else (cell.bbox[2] + sorted_cells[i + 1].bbox[0]) / 2
        )
        out.append((cell, left, right))
    return out


def _header_text_for(
    value_cell: TableCell, header_matches: list[tuple[TableCell, float, float]]
) -> str:
    x_center = (value_cell.bbox[0] + value_cell.bbox[2]) / 2
    matches = [
        cell
        for cell, left, right in header_matches
        if left - COLUMN_MATCH_PAD <= x_center <= right + COLUMN_MATCH_PAD
    ]
    matches.sort(key=lambda h: h.bbox[1])
    return " ".join(h.text for h in matches).strip()


def reconstruct_table(pdf_path: str, page_no: int) -> ReconstructedTable | None:
    """page_no is 1-based, matching pdf.py's Block.page_no."""
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_no - 1]
        page_dict = page.get_text("dict")
    finally:
        doc.close()

    # (true_block_no, block_dict) pairs — true_block_no matches pdf.py's
    # Block.block_no (enumerate() over the full page_dict["blocks"], images
    # included) so a TableCell can be traced back to the right Block/block_id.
    indexed_text_blocks = [
        (i, b) for i, b in enumerate(page_dict["blocks"]) if b.get("type") == 0
    ]
    data_blocks = [(i, b) for i, b in indexed_text_blocks if _has_table_row([b])]
    if not data_blocks:
        return None

    data_x0 = min(b["bbox"][0] for _, b in data_blocks)
    data_x1 = max(b["bbox"][2] for _, b in data_blocks)
    data_top_y = min(b["bbox"][1] for _, b in data_blocks)
    data_block_ids = {i for i, _ in data_blocks}

    def x_overlaps_table(b: dict) -> bool:
        return _x_overlap_fraction(tuple(b["bbox"]), data_x0, data_x1) > X_OVERLAP_MIN_FRACTION

    header_blocks = sorted(
        (
            (i, b)
            for i, b in indexed_text_blocks
            if i not in data_block_ids
            and b["bbox"][3] <= data_top_y + 1
            and b["bbox"][1] >= data_top_y - HEADER_SEARCH_MARGIN
            and x_overlaps_table(b)
            and _looks_like_caption_heading(_flat_block_text(b))
            and not SCALE_ANNOTATION_RE.search(_flat_block_text(b))
        ),
        key=lambda ib: ib[1]["bbox"][1],
    )
    header_block_ids = {i for i, _ in header_blocks}

    header_cells: list[TableCell] = []
    header_matches: list[tuple[TableCell, float, float]] = []
    for block_no, b in header_blocks:
        for row_spans in _block_rows(b):
            row_cells = _row_from_spans(row_spans, block_no)
            header_cells.extend(row_cells)
            header_matches.extend(_widen_ranges_within_row(row_cells))

    # Caption: nearest preceding heading-like block, plus any scale/currency
    # annotation (e.g. "(₹ in Million)") sitting above the data region —
    # restricted to blocks that horizontally overlap the table itself, so a
    # neighboring column's text (e.g. a two-page-spread's other half) can't
    # leak in.
    header_top_y = min((c.bbox[1] for c in header_cells), default=data_top_y)
    caption_parts: list[str] = []
    heading_candidates = sorted(
        (
            (i, b)
            for i, b in indexed_text_blocks
            if i not in data_block_ids
            and i not in header_block_ids
            and b["bbox"][3] <= header_top_y + 1
            and b["bbox"][1] >= header_top_y - CAPTION_SEARCH_MARGIN
            and x_overlaps_table(b)
        ),
        key=lambda ib: ib[1]["bbox"][1],
    )
    for _, b in heading_candidates:
        text = _flat_block_text(b)
        if not text:
            continue
        if _looks_like_caption_heading(text) or SCALE_ANNOTATION_RE.search(text):
            caption_parts.append(text)
    caption = " — ".join(caption_parts)

    rows: list[TableRow] = []
    for block_no, b in sorted(data_blocks, key=lambda ib: ib[1]["bbox"][1]):
        for row_spans in _block_rows(b):
            cells = _row_from_spans(row_spans, block_no)
            label_cells = [c for c in cells if not NUMERIC_RE.match(c.text)]
            value_cells = [c for c in cells if NUMERIC_RE.match(c.text)]
            if not value_cells:
                continue
            label = min(label_cells, key=lambda c: c.bbox[0]) if label_cells else None
            rows.append(TableRow(label=label, values=value_cells))

    return ReconstructedTable(
        page_no=page_no,
        caption=caption,
        rows=rows,
        header_cells=header_cells,
        header_matches=header_matches,
    )
