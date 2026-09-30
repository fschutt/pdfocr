"""macOS Vision engine, run for real on macOS (the CI macOS job, or `make test` on a Mac)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS Vision is macOS only")

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    from pdf_ocr_bench.html_output.renderer import render_pdf_pages

    return render_pdf_pages(ROOT / "sample.pdf", tmp_path_factory.mktemp("img"), dpi=200, pages=[0, 1])


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("Vision")
    from pdf_ocr_bench.engines import ENGINES
    from pdf_ocr_bench.languages import parse_languages

    cls = ENGINES["macos_vision"]
    route = cls.preflight(cls.route(parse_languages("deu+eng")))
    assert route.ok, route.unsupported
    eng = cls(route, timeout=120)
    eng.prepare()
    return eng


def test_reads_english_with_word_boxes(engine, pages):
    result = engine.run(pages[0])
    texts = [w.text for w in result.words]
    assert "Quarterly" in texts and "Digitization" in texts
    quarterly = next(w for w in result.words if w.text == "Quarterly")
    # the title is the first line, top left: y grows downwards after the flip from Vision's origin
    assert quarterly.bbox.y < 0.15 and quarterly.bbox.x < 0.35
    assert all(0 <= w.bbox.x <= w.bbox.x1 <= 1 and 0 <= w.bbox.y <= w.bbox.y1 <= 1 for w in result.words)
    assert all(0 < w.confidence <= 1 for w in result.words)


def test_reads_german_umlauts(engine, pages):
    text = engine.run(pages[1]).full_text
    for expected in ("Größere", "Bestände", "Übersicht", "Straße", "Grüßen"):
        assert expected in text, (expected, text)
