"""The book's own typeface (pdf_ocr_bench.typeface): letters from the scans, metrics, outlines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdf_ocr_bench import typeface as T  # noqa: E402


def test_the_models_s_tells_a_long_s_from_the_f_vision_read():
    assert T.printed("fhould", "should") == "ſhould"
    assert T.printed("fucceffors", "successors") == "ſucceſſors"
    assert T.printed("firft", "first") == "firſt"
    assert T.printed("of", "of") == "of"
    assert T.printed("Golpel", "Gospel") is None  # misread: no letter of it is taken


def test_a_word_is_cut_into_as_many_glyphs_as_it_has_boxes():
    assert T.spell("ſtill", 4) == ["ſt", "i", "l", "l"]
    assert T.spell("Chriſtians", 9) == ["C", "h", "r", "i", "ſt", "i", "a", "n", "s"]
    assert T.spell("the", 3) == ["t", "h", "e"]
    assert T.spell("the", 2) is None  # no ligature in it
    assert T.spell("the", 4) is None


def test_side_bearings_come_from_the_gaps_inside_words():
    # three letters with known bearings (em); words of them, gaps = right(a) + left(b)
    left, right, ink = {"o": 0.02, "n": 0.03, "e": 0.01}, {"o": 0.02, "n": 0.03, "e": 0.04}, {"o": 0.4, "n": 0.45, "e": 0.35}
    rng = np.random.default_rng(1)
    glyphs = []
    words = ["one", "neon", "noon", "eon", "none", "onee", "ene", "nee"]
    for w_no, word in enumerate(words * 20):
        x = 0.0
        for pos, c in enumerate(word):
            x0 = x + left[c]
            glyphs.append((c, False, x0, x0 + ink[c], 0.0, 0.45, 0, w_no, pos, 0))
            x = x0 + ink[c] + right[c] + rng.normal(0, 0.002)
    got_left, got_right, kerns = T.side_bearings(glyphs, least=5)
    for c in "one":
        assert got_left[c] + got_right[c] == pytest.approx(left[c] + right[c], abs=0.005)
    assert got_left["o"] == pytest.approx(got_right["o"], abs=0.005)  # a round letter: alike
    assert not kerns


def test_a_traced_glyph_becomes_a_font(tmp_path):
    canvas = np.zeros(T.CANVAS, dtype=np.float32)
    # an "o": a ring standing on the baseline, 0.45 em tall
    yy, xx = np.mgrid[: T.CANVAS[0], : T.CANVAS[1]]
    r = np.hypot((yy - (T.BASE_ROW - 36)) / 36, (xx - T.MID_COL) / 30)
    canvas[(r <= 1.0) & (r >= 0.6)] = 1.0
    contours = T.trace(canvas > 0.5)
    assert len(contours) == 2  # outside and the hole
    shapes = {("roman", "o"): (canvas, 100)}
    metrics = {"roman": {"glyphs": {"o": {"width": 60 / T.EM_PX, "bottom": 0.0, "top": 0.45, "count": 100,
                                          "left": 0.02, "right": 0.02}},
                         "kerns": {}, "spaces": {"space": 0.25}}}
    advances = T.build_font("roman", shapes, metrics, "Test", tmp_path / "Test-Regular.ttf", svg_dir=tmp_path / "svg")
    assert advances["o"] == pytest.approx(0.02 + 60 / T.EM_PX + 0.02, abs=0.002)
    assert advances[" "] == pytest.approx(0.25)
    from fontTools.ttLib import TTFont

    font = TTFont(tmp_path / "Test-Regular.ttf")
    assert font.getBestCmap()[ord("o")] == "o"
    glyph = font["glyf"]["o"]
    assert glyph.numberOfContours == 2
    assert font["hmtx"]["o"][0] == round(1000 * (0.04 + 60 / T.EM_PX))
    assert (tmp_path / "svg" / "roman-006F.svg").exists()
