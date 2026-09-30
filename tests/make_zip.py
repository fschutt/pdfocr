"""Build a pages.zip from sample.pdf with synthetic OCR words (no OCR engine needed).

Usage: python tests/make_zip.py OUT.zip
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from pdf_ocr_bench.html_output import create_engine_zip, load_template, render_pdf_pages
from pdf_ocr_bench.models import BBox, OcrResult, OcrWord, PageResult

SAMPLE = Path(__file__).resolve().parent.parent / "sample.pdf"
WORDS = [("Grüße", 0.10, 0.08), ("aus", 0.25, 0.08), ("Köln", 0.33, 0.08), ("Straße", 0.10, 0.12)]


def main(out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        images = render_pdf_pages(SAMPLE, Path(tmp), dpi=72, pages=[0, 1])
        pages = [
            PageResult(
                page_num=img.page_num,
                width_px=img.width_px,
                height_px=img.height_px,
                words=[OcrWord(text=t, bbox=BBox(x=x, y=y, w=0.1, h=0.02), confidence=1.0) for t, x, y in WORDS],
                full_text=" ".join(t for t, *_ in WORDS),
                engine_name="synthetic",
                elapsed_seconds=0.0,
            )
            for img in images
        ]
        create_engine_zip(OcrResult(engine_name="synthetic", pages=pages, total_elapsed=0.0), images, out, load_template())
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
