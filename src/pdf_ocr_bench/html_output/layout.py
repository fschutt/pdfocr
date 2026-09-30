"""Place OCR words as visible text: one font size per line, measured with real font metrics.

The page is set in Helvetica (`font-family: Helvetica, Arial, sans-serif`). printpdf embeds its
own Helvetica for that family, and browsers use Helvetica or the metric-compatible Arial /
Liberation Sans. PyMuPDF's built-in Helvetica has the same advance widths, so a word measured
here renders at the measured width.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from statistics import median

from ..engines.base import group_lines
from ..models import OcrWord

FONT_FAMILY = "Helvetica, Arial, sans-serif"

# printpdf's Helvetica: ascender 0.770 em, cap height 0.718 em. With `line-height: 1` the
# baseline sits one ascender below the top of the span, so the middle of the cap-height band
# (where a word's ink is centered) is this far below the top.
TEXT_CENTER_EM = 0.770 - 0.718 / 2
# Guard for broken boxes: a line's text is never taller than 1.5x its tallest box.
MAX_SIZE_PER_BOX_HEIGHT = 1.5
# A word squeezed to avoid its right neighbour keeps at least half the line's size.
MIN_SQUEEZE = 0.5
# Advance used for characters Helvetica lacks (the renderer falls back to another font).
FALLBACK_ADVANCE_EM = 0.6
WIDE_ADVANCE_EM = 1.0


@dataclass(frozen=True)
class WordStyle:
    font_size_pt: float
    top: float  # normalized top of the span
    suffix: str  # " " between words of a line, so copy-paste and PDF extraction keep spaces


@lru_cache(maxsize=1)
def _helvetica():
    import pymupdf

    return pymupdf.Font("helv")


def _advance_em(ch: str) -> float:
    if unicodedata.combining(ch):
        return 0.0
    font = _helvetica()
    if font.has_glyph(ord(ch)):
        return font.glyph_advance(ord(ch))
    return WIDE_ADVANCE_EM if unicodedata.east_asian_width(ch) in ("W", "F") else FALLBACK_ADVANCE_EM


def text_width_em(text: str) -> float:
    """Rendered width of `text` at font size 1."""
    return sum(_advance_em(ch) for ch in unicodedata.normalize("NFC", text))


def word_styles(words: list[OcrWord], page_width_pt: float, page_height_pt: float) -> dict[int, WordStyle]:
    """Per-word font size, vertical position and trailing separator, keyed by id(word).

    Size: every word's box width divided by its measured width gives the size at which it
    would exactly span its box; the line takes the median of those, so one badly boxed word
    cannot resize the line. A word is only made smaller than its line if it would otherwise
    run into the next word.

    Position: words keep their OCR x. Vertically every span is centered on the median center
    of the line's boxes, which is stable across engines (Tesseract returns tight ink boxes,
    RapidOCR/PaddleOCR padded detection boxes) and across words with or without descenders.
    """
    styles: dict[int, WordStyle] = {}
    for line in group_lines(words):
        widths = [text_width_em(w.text) for w in line]
        fits = [w.bbox.w * page_width_pt / em for w, em in zip(line, widths) if em > 0]
        line_height_pt = max(w.bbox.h for w in line) * page_height_pt
        line_size = min(median(fits) if fits else line_height_pt, MAX_SIZE_PER_BOX_HEIGHT * line_height_pt)
        center = median(w.bbox.y + w.bbox.h / 2 for w in line)
        for i, (word, em) in enumerate(zip(line, widths)):
            last = i == len(line) - 1
            suffix = "" if last else " "
            right_edge = 1.0 if last else line[i + 1].bbox.x
            room_pt = (right_edge - word.bbox.x) * page_width_pt
            span_em = em + text_width_em(suffix)
            fit = room_pt / span_em if span_em > 0 else line_size
            size = max(1.0, min(line_size, max(fit, MIN_SQUEEZE * line_size)))
            styles[id(word)] = WordStyle(
                font_size_pt=size,
                top=max(0.0, center - TEXT_CENTER_EM * size / page_height_pt),
                suffix=suffix,
            )
    return styles
