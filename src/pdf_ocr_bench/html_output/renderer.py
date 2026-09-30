"""Render source PDF pages to PNG (once, shared by all engines) and OCR results to HTML."""

from __future__ import annotations

import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

from ..log import get_logger
from ..models import PageImage, PageResult
from .layout import FONT_FAMILY, word_styles

TEMPLATE_DIR = Path(__file__).parent
PT_PER_MM = 72.0 / 25.4

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
        font_family=FONT_FAMILY,
        width_pt=width_pt,
        height_pt=height_pt,
        width_mm=width_pt / PT_PER_MM,
        height_mm=height_pt / PT_PER_MM,
        words=page.words,
        style=lambda word, styles=word_styles(page.words, width_pt, height_pt): styles[id(word)],
    )
