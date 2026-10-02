"""Page layout from pixels, and the structure → HTML steps of `reconstruct`, on synthetic pages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pdf_ocr_bench.llm_structure import to_items  # noqa: E402
from pdf_ocr_bench.page_layout import Box, PageLayout, Zone, analyse  # noqa: E402
from pdf_ocr_bench.reconstruct import Item, Line, build_blocks, blocks_html, fit_size, heuristic_structure, wrap_lines  # noqa: E402

LH = 24  # glyph height of the synthetic pages


def _text_lines(page: np.ndarray, x0: int, x1: int, y0: int, n: int, pitch: int = 40, rng=None) -> None:
    """`n` lines of ragged "words" (black boxes a glyph high) between x0 and x1."""
    rng = rng or np.random.default_rng(1)
    for i in range(n):
        y, x = y0 + i * pitch, x0
        while x < x1 - 40:
            w = int(rng.integers(30, 110))
            w = min(w, x1 - x)
            for gx in range(x, x + w - 6, 14):  # glyphs, a few px apart
                page[y:y + LH, gx:gx + 10] = 0
            x += w + int(rng.integers(12, 20))  # word space, narrower than any gutter


def synthetic_page(tmp_path: Path) -> Path:
    page = np.full((2400, 1600), 255, np.uint8)
    rng = np.random.default_rng(7)
    _text_lines(page, 500, 1100, 60, 1, rng=rng)  # running head, across the gutter
    _text_lines(page, 200, 760, 200, 45, rng=rng)  # column 1
    _text_lines(page, 820, 1380, 200, 45, rng=rng)  # column 2 (60 px gutter)
    for y in (300, 900, 1500):  # marginal notes, 15 px from column 2
        _text_lines(page, 1395, 1560, y, 2, rng=rng)
    page[2050:2350, 300:1300] = 0  # a picture: one big dark shape
    page[2080:2320, 330:1270] = 255
    page[2100:2300, 350:1250:7] = 0  # its hatching
    path = tmp_path / "page.png"
    cv2.imwrite(str(path), page)
    return path


def test_layout_finds_head_columns_notes_and_picture(tmp_path):
    layout = analyse(synthetic_page(tmp_path))
    roles = sorted(z.role for z in layout.zones)
    assert roles.count("column") == 2, layout.zones
    assert "header" in roles and "picture" in roles
    assert [z.side for z in layout.of("notes")] == ["right"]
    col1, col2 = sorted(layout.of("column"), key=lambda z: z.box.x0)
    assert col1.box.x1 <= 790 and col2.box.x0 >= 800  # cut in the gutter
    assert abs(layout.line_height - LH) <= 2


def test_wrap_and_fit_with_real_metrics():
    import pymupdf

    font = pymupdf.Font("tiro")
    text = " ".join(["word"] * 60)
    one = wrap_lines(text, 10_000, 12, font)
    assert one == 1
    narrow = wrap_lines(text, 200, 12, font)
    assert narrow > 5
    size = fit_size(text, 200, 4, 12, font)
    assert size < 12 and wrap_lines(text, 200, size, font) <= 4
    assert fit_size("short", 200, 1, 12, font) == 12


def _lines() -> list[Line]:
    rows = [("L1", "First line of the para-", 100), ("L2", "graph goes on here.", 140), ("L3", "Second paragraph.", 220)]
    return [Line(i, t, Box(100, y, 900, y + 30), [], "column0") for i, t, y in rows]


def _layout() -> PageLayout:
    layout = PageLayout(width=1000, height=1000, line_height=30)
    layout.zones = [Zone("column", Box(100, 100, 900, 300), 0)]
    return layout


def test_heuristic_joins_hyphens_and_splits_at_gaps():
    items = heuristic_structure(_lines(), _layout())
    assert [i.text for i in items] == ["First line of the paragraph goes on here.", "Second paragraph."]
    assert items[0].lines == ["L1", "L2"]


def test_blocks_stop_at_the_next_block_and_render_as_regions():
    items = heuristic_structure(_lines(), _layout())
    blocks = build_blocks(items, _lines(), _layout(), {}, 1000, 1000)
    assert len(blocks) == 2
    assert blocks[0].limit == pytest.approx(blocks[1].y)  # the first paragraph may reach the second
    assert blocks[1].limit == 1000  # the last: the page's end
    page = blocks_html(blocks, 1000, 1000)
    assert page.count('<div class="region"') == 2 and "Times" in page


def test_answer_is_checked_against_the_ocr_lines():
    layout, lines = _layout(), _lines()
    answer = {"items": [
        {"zone": "column0", "kind": "paragraph", "lines": ["L1", "L2", "L9"], "text": "First line of the paragraph goes on here."},
        {"zone": "nowhere", "kind": "paragraph", "lines": ["L3"], "text": "Second paragraph."},
        {"zone": "column0", "kind": "noise", "lines": [], "text": "x"},
    ]}
    items = to_items(answer, lines, layout)
    assert [i.lines for i in items] == [["L1", "L2"], ["L3"]]  # unknown L9 dropped
    assert items[1].zone == "column0"  # an unknown zone: the zone its lines are in
    lost = {"items": [{"zone": "column0", "kind": "paragraph", "lines": ["L3"], "text": "Second paragraph."}]}
    assert to_items(lost, lines, layout) is None  # most of the page's text left out
