from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from pdf_ocr_bench.preprocess import FILTERS, FilterError, apply_filters, binarize, denoise, parse_filters, skew_angle


def text_page(size=(1200, 900)) -> Image.Image:
    """A white page with lines of dark text-like blocks."""
    img = Image.new("L", size, 250)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(28)
    for i, y in enumerate(range(80, size[1] - 80, 60)):
        draw.text((80, y), f"Line {i} with some words to make a text row", fill=20, font=font)
    return img


def test_parse_filters_keeps_order_and_rejects_unknown():
    assert parse_filters("grayscale, deskew ,denoise") == ["grayscale", "deskew", "denoise"]
    assert parse_filters("") == [] and parse_filters(None) == []
    with pytest.raises(FilterError, match="unknown filter\\(s\\): blur. Available: grayscale"):
        parse_filters("grayscale,blur")


def test_binarize_gives_pure_black_and_white():
    out = binarize(text_page())
    assert out.mode == "L" and set(np.unique(np.asarray(out))) == {0, 255}


def test_denoise_removes_isolated_specks():
    img = Image.new("L", (100, 100), 255)
    img.putpixel((50, 50), 0)
    img.putpixel((20, 70), 0)
    assert np.asarray(denoise(img)).min() == 255


@pytest.mark.parametrize("angle", [-3.0, -1.2, 0.0, 2.5])
def test_skew_angle_recovers_the_rotation(angle):
    rotated = text_page().rotate(angle, resample=Image.BICUBIC, fillcolor=250)
    assert skew_angle(rotated) == pytest.approx(-angle, abs=0.25)


def test_apply_filters_in_place_keeps_the_page_size(tmp_path):
    path = tmp_path / "page.png"
    text_page().rotate(2.0, fillcolor=250).convert("RGB").save(path)
    apply_filters(path, ["grayscale", "deskew", "denoise", "autocontrast", "sharpen", "binarize"])
    with Image.open(path) as out:
        assert out.size == (1200, 900) and out.mode == "L"


def test_every_filter_has_a_description():
    assert all(callable(fn) and doc for fn, doc in FILTERS.values())
