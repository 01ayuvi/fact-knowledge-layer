"""Extract text blocks and page-type classification from a PDF.

Pure PyMuPDF, no LLM calls. Downstream stages (doc-context, extraction) build on
the Block list this module produces.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass

import pymupdf

MIN_BLOCK_CHARS = 40
SLIDE_MAX_CHARS = 1200
SLIDE_MIN_IMAGE_COVERAGE = 0.30
TABLE_MIN_GAP = 15.0
TABLE_MIN_NUMERIC_CELLS = 3
ROW_Y_TOLERANCE = 2.0

NUMERIC_RE = re.compile(r"^[+\-]?[₹$]?\(?[\d,]+(\.\d+)?\)?%?$")


@dataclass
class Block:
    page_no: int
    block_no: int
    text: str
    bbox: tuple[float, float, float, float]
    page_type: str


def _block_text(block: dict) -> str:
    lines = []
    for line in block.get("lines", []):
        span_text = "".join(span["text"] for span in line.get("spans", []))
        if span_text.strip():
            lines.append(span_text)
    return " ".join(lines).strip()


def _raster_image_coverage(page: pymupdf.Page, page_area: float) -> float:
    covered = 0.0
    for img in page.get_image_info():
        x0, y0, x1, y1 = img["bbox"]
        covered += max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return covered / page_area


def _vector_drawing_coverage(page: pymupdf.Page, page_area: float) -> float:
    covered = 0.0
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        covered += max(0.0, rect.width) * max(0.0, rect.height)
    return covered / page_area


def _visual_coverage(page: pymupdf.Page) -> float:
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return 0.0
    return max(
        _raster_image_coverage(page, page_area),
        _vector_drawing_coverage(page, page_area),
    )


def _line_is_table_row(spans: list[dict]) -> bool:
    numeric = sorted(
        (s for s in spans if NUMERIC_RE.match(s["text"].strip())),
        key=lambda s: s["bbox"][0],
    )
    if len(numeric) < TABLE_MIN_NUMERIC_CELLS:
        return False
    return all(
        numeric[i + 1]["bbox"][0] - numeric[i]["bbox"][2] >= TABLE_MIN_GAP
        for i in range(len(numeric) - 1)
    )


def _block_rows(block: dict) -> list[list[dict]]:
    """Group a block's spans into visual rows by y-coordinate.

    PyMuPDF splits a table row into several single-span "line" dict entries
    when cells are far apart horizontally (e.g. wide column gaps), even
    though they share the same y-range. Re-clustering by y recovers the row.
    """
    spans = [s for line in block.get("lines", []) for s in line.get("spans", [])]
    spans.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
    rows: list[list[dict]] = []
    for span in spans:
        y0 = span["bbox"][1]
        if rows and abs(y0 - rows[-1][0]["bbox"][1]) <= ROW_Y_TOLERANCE:
            rows[-1].append(span)
        else:
            rows.append([span])
    return rows


def _has_table_row(text_blocks: list[dict]) -> bool:
    return any(
        _line_is_table_row(row) for block in text_blocks for row in _block_rows(block)
    )


def _classify_page(page: pymupdf.Page, page_dict: dict, char_count: int) -> str:
    if char_count < SLIDE_MAX_CHARS and _visual_coverage(page) > SLIDE_MIN_IMAGE_COVERAGE:
        return "slide"
    text_blocks = [b for b in page_dict["blocks"] if b.get("type") == 0]
    if _has_table_row(text_blocks):
        return "table"
    return "prose"


def extract_blocks(pdf_path: str) -> list[Block]:
    doc = pymupdf.open(pdf_path)
    blocks: list[Block] = []
    try:
        for page_index, page in enumerate(doc):
            page_no = page_index + 1
            page_dict = page.get_text("dict")
            char_count = len(page.get_text("text"))
            page_type = _classify_page(page, page_dict, char_count)
            for block_no, raw_block in enumerate(page_dict["blocks"]):
                if raw_block.get("type") != 0:
                    continue
                text = _block_text(raw_block)
                if len(text) < MIN_BLOCK_CHARS:
                    continue
                blocks.append(
                    Block(
                        page_no=page_no,
                        block_no=block_no,
                        text=text,
                        bbox=tuple(raw_block["bbox"]),
                        page_type=page_type,
                    )
                )
    finally:
        doc.close()
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract blocks from a PDF.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    args = parser.parse_args()

    blocks = extract_blocks(args.pdf_path)

    per_page_count: Counter[int] = Counter()
    per_page_type: dict[int, str] = {}
    for block in blocks:
        per_page_count[block.page_no] += 1
        per_page_type[block.page_no] = block.page_type

    for page_no in sorted(per_page_count):
        count = per_page_count[page_no]
        page_type = per_page_type[page_no]
        print(f"page {page_no}: {count} blocks ({page_type})")

    distribution = Counter(per_page_type.values())
    print("page_type distribution:", dict(distribution))


if __name__ == "__main__":
    main()
