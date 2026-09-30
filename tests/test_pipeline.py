from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

import pymupdf
import pytest

from pdf_ocr_bench import pipeline
from pdf_ocr_bench.engines import ENGINES, OcrEngine, select_engines
from pdf_ocr_bench.engines.base import split_block, split_line, words_to_text
from pdf_ocr_bench.engines.paddleocr_engine import merge_fragments
from pdf_ocr_bench.evaluation import metrics
from pdf_ocr_bench.html_output.renderer import load_template, render_page_html, word_styles
from pdf_ocr_bench.lang_map import get_lang, html_lang
from pdf_ocr_bench.models import BBox, OcrWord, PageImage, PageResult
from pdf_ocr_bench.pipeline import PipelineConfig, parse_page_range

ROOT = Path(__file__).resolve().parent.parent


def word(text: str, x: float, y: float, w: float = 0.1, h: float = 0.02, conf: float = 0.9) -> OcrWord:
    return OcrWord(text=text, bbox=BBox(x=x, y=y, w=w, h=h), confidence=conf)


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    """Two-page image-only PDF (small, so tests stay fast)."""
    path = tmp_path / "in.pdf"
    doc = pymupdf.open()
    for text in ("Hello OCR world", "Grüße aus Köln"):
        page = doc.new_page(width=300, height=200)
        page.insert_text((30, 100), text, fontsize=24)
        pix = page.get_pixmap(dpi=100)
        img = doc.new_page(width=300, height=200)
        img.insert_image(img.rect, pixmap=pix)
        doc.delete_page(doc.page_count - 2)
    doc.save(path)
    return path


# --- small pure helpers ------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "count", "expected"),
    [
        (None, 3, [0, 1, 2]),
        ("", 3, [0, 1, 2]),
        ("all", 2, [0, 1]),
        ("1-3", 10, [0, 1, 2]),
        ("1,3,7-10", 12, [0, 2, 6, 7, 8, 9]),
        ("2-99", 4, [1, 2, 3]),
        ("3,1,3", 5, [0, 2]),
    ],
)
def test_parse_page_range(spec, count, expected):
    assert parse_page_range(spec, count) == expected


@pytest.mark.parametrize("spec", ["0", "5-2", "9", "a-b"])
def test_parse_page_range_rejects(spec):
    with pytest.raises(ValueError):
        parse_page_range(spec, 4)


def test_lang_mapping():
    assert get_lang("deu", "paddleocr") == "german"
    assert get_lang("deu_frak", "easyocr") == "de"
    assert get_lang("eng+deu", "surya") == "en"
    assert get_lang("chi_sim", "rapidocr") == "ch"
    assert get_lang("xyz", "doctr") == "en"
    assert get_lang("deu_frak+eng", "tesseract") == "deu_frak+eng"
    assert get_lang("frk", "ocrmypdf_rapid") == "deu"
    assert html_lang("chi_tra") == "zh-Hant"


def test_bbox_normalization():
    b = BBox.from_points([[100, 50], [300, 50], [300, 90], [100, 90]], 1000, 500)
    assert (b.x, b.y, b.w, b.h) == pytest.approx((0.1, 0.1, 0.2, 0.08))
    clamped = BBox.from_pixels(-10, -5, 1100, 600, 1000, 500)
    assert (clamped.x, clamped.y, clamped.x1, clamped.y1) == (0, 0, 1, 1)


def test_split_line_is_proportional():
    words = split_line("ab cdef", BBox(x=0.1, y=0.2, w=0.7, h=0.05), 0.8)
    assert [w.text for w in words] == ["ab", "cdef"]
    assert words[0].bbox.x == pytest.approx(0.1)
    assert words[0].bbox.w == pytest.approx(0.2)
    assert words[1].bbox.x == pytest.approx(0.4)
    assert all(w.confidence == 0.8 for w in words)


def test_split_block_stacks_lines():
    words = split_block(["one two", "", "three"], BBox(x=0, y=0, w=1, h=0.2), 1.0)
    assert [w.text for w in words] == ["one", "two", "three"]
    assert words[2].bbox.y == pytest.approx(0.1)


def test_merge_paddle_fragments():
    frags = ["Gr", "üß", "e", " ", "w", "ö", "rld", ", ", "OCR", "."]
    regions = [[(i, 0), (i + 1, 1)] for i in range(len(frags))]
    merged = merge_fragments(frags, regions)
    assert [t for t, _ in merged] == ["Grüße", "wörld,", "OCR."]
    assert merged[0][1] == [(0, 0), (1, 1), (1, 0), (2, 1), (2, 0), (3, 1)]


def test_words_to_text_orders_lines_and_words():
    words = [word("world", 0.4, 0.1), word("Hello", 0.1, 0.105), word("second", 0.1, 0.3)]
    assert words_to_text(words) == "Hello world\nsecond"


def test_metrics():
    assert metrics.cer("abc", "abc") == 0.0
    assert metrics.cer("", "") == 0.0
    assert metrics.cer("", "x") == 1.0
    assert metrics.cer("abcd", "abce") == pytest.approx(0.25)
    assert metrics.cer("a", "a very long hallucination") == 1.0  # capped
    assert metrics.symmetric(metrics.cer, "abcd", "abc") == pytest.approx((0.25 + 1 / 3) / 2)
    assert metrics.agreement("Hello  world", "Hello world") == 1.0
    a = [word("x", 0.1, 0.1), word("y", 0.5, 0.1)]
    assert metrics.bbox_iou(a, a) == pytest.approx(1.0)
    assert metrics.bbox_iou(a, [word("z", 0.8, 0.8)]) == 0.0
    assert metrics.bbox_iou([], []) == 1.0


def test_select_engines():
    all_cpu = [c.name for c in select_engines("all")]
    assert "olmocr" not in all_cpu and "tesseract" in all_cpu and len(all_cpu) == 8
    assert "olmocr" in [c.name for c in select_engines("all", include_gpu=True)]
    assert [c.name for c in select_engines("rapidocr, tesseract,rapidocr")] == ["rapidocr", "tesseract"]
    assert [c.name for c in select_engines("ocrmypdf-rapid")] == ["ocrmypdf_rapid"]
    assert ENGINES["olmocr"].requires_gpu
    with pytest.raises(ValueError, match="unknown engine"):
        select_engines("tesseract,nope")


# --- HTML output ------------------------------------------------------------------


def _page(words: list[OcrWord]) -> PageResult:
    return PageResult(
        page_num=0, width_px=1000, height_px=1400, words=words, full_text="", engine_name="t", elapsed_seconds=0
    )


def test_page_html():
    html = render_page_html(load_template(), _page([word("a<b&c", 0.1, 0.2), word("ß", 0.3, 0.2)]), 595.28, 841.89, "de")
    assert '<html lang="de">' in html
    assert '<img src="page_001.png"' in html
    assert "width: 595.28pt" in html and "height: 841.89pt" in html
    assert 'name="pdf.options.pageWidth" content="210.0016"' in html
    assert "a&lt;b&amp;c </span>" in html  # escaped, with the inter-word space
    assert ">ß</span>" in html  # last word of the line: no trailing space
    assert "left: 10.0000%" in html
    style = html[html.index("<style>") : html.index("</style>")]
    assert "color: transparent" in style and ".word::selection" in style and ".page.debug .word" in style


def test_word_styles_share_line_size_and_fit_width():
    tall = word("Äg", 0.1, 0.1, w=0.02, h=0.03)
    short = word("a", 0.2, 0.105, w=0.2, h=0.02)
    long = word("x" * 40, 0.1, 0.5, w=0.1, h=0.03)
    styles = word_styles([tall, short, long], 600, 800)
    # same line -> height-derived size of the tallest box (0.03 * 800 * 0.8), unless width-capped
    assert styles[id(short)].font_size_pt == pytest.approx(19.2)
    assert styles[id(tall)].font_size_pt == pytest.approx(0.02 * 600 / (0.6 * 2))  # 2 glyphs in 12pt
    assert styles[id(long)].font_size_pt == pytest.approx(0.1 * 600 / (0.6 * 40))
    assert styles[id(tall)].suffix == " " and styles[id(short)].suffix == "" and styles[id(long)].suffix == ""


# --- pipeline with fake engines --------------------------------------------------


class FakeEngine(OcrEngine):
    name = "fake_a"
    display_name = "FakeA"
    texts = ("Hello", "OCR", "world")

    def ocr_page(self, image: PageImage, lang: str) -> list[OcrWord]:
        return [word(t, 0.1 + 0.2 * i, 0.4, conf=0.8) for i, t in enumerate(self.texts)]


class FakeEngineB(FakeEngine):
    name = "fake_b"
    display_name = "FakeB"
    texts = ("Hello", "0CR", "world")


class FakeEngineC(FakeEngine):
    name = "fake_c"
    display_name = "FakeC"
    texts = ("Hallo", "OCR", "world")


class BrokenEngine(FakeEngine):
    name = "broken"
    display_name = "Broken"

    def prepare(self) -> None:
        raise ImportError("backend not installed")


class SlowEngine(FakeEngine):
    name = "slow"
    display_name = "Slow"

    def ocr_page(self, image, lang):
        if image.page_num == 0:
            time.sleep(3)
        return super().ocr_page(image, lang)


class CrashEngine(FakeEngine):
    name = "crash"
    display_name = "Crash"

    def ocr_page(self, image, lang):
        raise RuntimeError("boom")


def _run(monkeypatch, tmp_path, pdf, engines, **kw):
    monkeypatch.setattr(pipeline, "select_engines", lambda spec, include_gpu=False: engines)
    cfg = PipelineConfig(input_pdf=pdf, output_dir=tmp_path / "results", dpi=72, **kw)
    return pipeline.run(cfg), tmp_path / "results"


def test_pipeline_end_to_end(monkeypatch, tmp_path, pdf):
    report, out = _run(monkeypatch, tmp_path, pdf, [FakeEngine, FakeEngineB, FakeEngineC, BrokenEngine, CrashEngine])

    assert sorted(p.name for p in (out / "images").iterdir()) == ["page_001.png", "page_002.png"]
    with zipfile.ZipFile(out / "fake_a" / "pages.zip") as zf:
        assert sorted(zf.namelist()) == [
            "metadata.json", "page_001.html", "page_001.png", "page_002.html", "page_002.png",
        ]  # fmt: skip
        meta = json.loads(zf.read("metadata.json"))
        assert meta["engine"] == "fake_a" and meta["page_count"] == 2 and meta["total_words"] == 6
        assert meta["avg_confidence"] == pytest.approx(0.8)
        assert meta["pages"][0]["width_pt"] == pytest.approx(300)
        assert "Hello </span>" in zf.read("page_002.html").decode()

    by_name = {e.name: e for e in report.engines}
    assert by_name["fake_a"].success and by_name["fake_a"].total_words == 6
    assert not by_name["broken"].success and "backend not installed" in by_name["broken"].error
    assert not by_name["crash"].success and "boom" in by_name["crash"].error
    assert not (out / "broken").exists() and not (out / "crash").exists()

    # fake_a agrees with both others; b and c each differ from each other in two places
    assert report.best_engine == "fake_a"
    assert [r.name for r in report.ranking][0] == "fake_a"
    assert report.cer_matrix["fake_a"]["fake_a"] == 0.0
    assert report.cer_matrix["fake_b"]["fake_c"] == report.cer_matrix["fake_c"]["fake_b"] > 0

    saved = json.loads((out / "report.json").read_text())
    assert saved["best_engine"] == "fake_a" and len(saved["engines"]) == 5


def test_page_timeout_skips_page(monkeypatch, tmp_path, pdf):
    report, out = _run(monkeypatch, tmp_path, pdf, [SlowEngine], timeout_per_page=1)
    (slow,) = report.engines
    assert slow.success and slow.pages_skipped == [1] and slow.pages_processed == 1
    with zipfile.ZipFile(out / "slow" / "pages.zip") as zf:
        pages = json.loads(zf.read("metadata.json"))["pages"]
    assert pages[0]["skipped"] and "timeout" in pages[0]["error"] and pages[1]["words"] == 3


def test_page_selection(monkeypatch, tmp_path, pdf):
    report, out = _run(monkeypatch, tmp_path, pdf, [FakeEngine], pages="2")
    assert report.pages == [2]
    with zipfile.ZipFile(out / "fake_a" / "pages.zip") as zf:
        assert "page_002.html" in zf.namelist() and "page_001.html" not in zf.namelist()


# --- real engine (only if Tesseract is installed) --------------------------------


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_tesseract_on_sample(tmp_path):
    pytest.importorskip("pytesseract")
    cfg = PipelineConfig(input_pdf=ROOT / "sample.pdf", output_dir=tmp_path, engines="tesseract", dpi=150, pages="1")
    report = pipeline.run(cfg)
    (tess,) = report.engines
    assert tess.success and tess.total_words > 50
    with zipfile.ZipFile(tmp_path / "tesseract" / "pages.zip") as zf:
        html = zf.read("page_001.html").decode()
    assert "Quarterly" in html and "Digitization" in html
