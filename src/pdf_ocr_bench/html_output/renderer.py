"""Render source PDF pages to PNG (once, shared by all engines) and OCR results to HTML."""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import median
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

from ..engines.base import group_lines
from ..log import get_logger
from ..models import OcrWord, PageImage, PageResult

TEMPLATE_DIR = Path(__file__).parent
PT_PER_MM = 72.0 / 25.4
# font-size fills the box height; 0.8 accounts for line-height vs cap-height
FONT_FILL = 0.8
# average glyph advance of a sans-serif font, in em
AVG_ADVANCE_EM = 0.6

log = get_logger("Pipeline")


def image_name(page_num: int) -> str:
    return f"page_{page_num + 1:03d}.png"


def html_name(page_num: int) -> str:
    return f"page_{page_num + 1:03d}.html"


def render_pdf_pages(pdf_path: Path, out_dir: Path, dpi: int, pages: list[int]) -> list[PageImage]:
    """Rasterize the selected pages (0-indexed) with pypdfium2."""
    import pypdfium2 as pdfium

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Rendering pages at {dpi} DPI...")
    start = time.perf_counter()
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        images = [_render_page(pdf, n, out_dir, dpi) for n in pages]
    finally:
        pdf.close()
    log.info(f"Rendered {len(images)} pages in {time.perf_counter() - start:.1f}s")
    return images


def _render_page(pdf, page_num: int, out_dir: Path, dpi: int) -> PageImage:
    page = pdf[page_num]
    try:
        width_pt, height_pt = page.get_size()
        bitmap = page.render(scale=dpi / 72.0, rotation=0)
        pil = bitmap.to_pil().convert("RGB")
    finally:
        page.close()
    path = out_dir / image_name(page_num)
    pil.save(path, format="PNG", optimize=False, dpi=(dpi, dpi))
    log.debug(f"page {page_num + 1}: {pil.width}x{pil.height}px, {width_pt:.1f}x{height_pt:.1f}pt -> {path}")
    return PageImage(
        page_num=page_num,
        path=path,
        width_px=pil.width,
        height_px=pil.height,
        width_pt=width_pt,
        height_pt=height_pt,
        dpi=dpi,
    )


def page_count(pdf_path: Path) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def load_template() -> Template:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("page.html.j2")


@dataclass(frozen=True)
class WordStyle:
    font_size_pt: float
    top: float  # normalized, centers the text on the line
    suffix: str  # " " between words of a line, so copy-paste and PDF extraction keep spaces


def word_styles(words: list[OcrWord], page_width_pt: float, page_height_pt: float) -> dict[int, WordStyle]:
    """Per-word font size and trailing separator, keyed by id(word).

    Size = line height * page height * FONT_FILL. Word boxes are ink extents ("a" is shorter
    than "Äg"), so the tallest box of the line gives one size per line. It is then capped so
    the word fits its box width: a word wider than its box overlaps the next one and PDF text
    extractors merge them.

    Vertical placement centers the span's `line-height: 1` box on the line's median box
    center. Engines disagree on what a box is (Tesseract: tight ink, RapidOCR/Paddle: padded
    detection boxes) and words with/without descenders shift the bottom, but the center of a
    line is stable across all of them.
    """
    styles: dict[int, WordStyle] = {}
    for line in group_lines(words):
        line_size = max(w.bbox.h for w in line) * page_height_pt * FONT_FILL
        center = median(w.bbox.y + w.bbox.h / 2 for w in line)
        for i, word in enumerate(line):
            fit = word.bbox.w * page_width_pt / (AVG_ADVANCE_EM * max(len(word.text), 1))
            size = max(1.0, min(line_size, fit))
            styles[id(word)] = WordStyle(
                font_size_pt=size,
                top=max(0.0, center - size / 2 / page_height_pt),
                suffix=" " if i < len(line) - 1 else "",
            )
    return styles


def render_page_html(
    template: Template,
    page: PageResult,
    width_pt: float,
    height_pt: float,
    html_lang: str = "en",
) -> str:
    return template.render(
        page_num=page.page_num,
        engine_name=page.engine_name,
        html_lang=html_lang,
        image_name=image_name(page.page_num),
        width_pt=width_pt,
        height_pt=height_pt,
        width_mm=width_pt / PT_PER_MM,
        height_mm=height_pt / PT_PER_MM,
        words=page.words,
        style=lambda word, styles=word_styles(page.words, width_pt, height_pt): styles[id(word)],
    )
