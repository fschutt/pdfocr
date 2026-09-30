from __future__ import annotations

import json
import zipfile
from pathlib import Path

from jinja2 import Template

from ..models import OcrResult, PageImage
from .renderer import html_name, render_page_html


def create_pages_zip(
    ocr_result: OcrResult,
    output_zip: Path,
    page_template: Template,  # Jinja2
    page_width_pt: float,
    page_height_pt: float,
    page_sizes: dict[int, tuple[float, float]] | None = None,
    html_lang: str = "en",
    has_confidence: bool = True,
) -> Path:
    """Bundle the HTML pages and metadata.json into a .zip.

    The pages hold only the recognized text (no scan), so the zip stays small and the PDF
    made from it is text only.

    `page_width_pt`/`page_height_pt` are the default size; `page_sizes` overrides it per
    page (0-indexed) for PDFs with mixed page sizes.
    """
    page_sizes = page_sizes or {}
    conf = (lambda v: round(v, 4)) if has_confidence else (lambda v: None)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    pages_meta = []
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for page in ocr_result.pages:
            width_pt, height_pt = page_sizes.get(page.page_num, (page_width_pt, page_height_pt))
            html = render_page_html(page_template, page, width_pt, height_pt, html_lang)
            zf.writestr(html_name(page.page_num), html)
            pages_meta.append(
                {
                    "page_num": page.page_num,
                    "html": html_name(page.page_num),
                    "width_px": page.width_px,
                    "height_px": page.height_px,
                    "width_pt": round(width_pt, 4),
                    "height_pt": round(height_pt, 4),
                    "words": len(page.words),
                    "avg_confidence": conf(page.avg_confidence),
                    "elapsed_seconds": round(page.elapsed_seconds, 3),
                    "skipped": page.skipped,
                    "error": page.error,
                }
            )
        metadata = {
            "engine": ocr_result.engine_name,
            "page_count": len(ocr_result.pages),
            "total_words": ocr_result.total_words,
            "elapsed_seconds": round(ocr_result.total_elapsed, 3),
            "avg_confidence": conf(ocr_result.avg_confidence),
            "pages": pages_meta,
        }
        zf.writestr("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))
    return output_zip


def create_engine_zip(
    ocr_result: OcrResult,
    images: list[PageImage],
    output_zip: Path,
    page_template: Template,
    html_lang: str = "en",
    has_confidence: bool = True,
) -> Path:
    """Convenience wrapper: take the page sizes from the rendered page images."""
    first = images[0]
    return create_pages_zip(
        ocr_result,
        output_zip,
        page_template,
        first.width_pt,
        first.height_pt,
        page_sizes={img.page_num: (img.width_pt, img.height_pt) for img in images},
        html_lang=html_lang,
        has_confidence=has_confidence,
    )
