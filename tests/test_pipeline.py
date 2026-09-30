from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

import pymupdf
import pytest

from pdf_ocr_bench import pipeline, plan
from pdf_ocr_bench.engines import ENGINES, OcrEngine, Route, select_engines
from pdf_ocr_bench.engines.base import split_block, split_line, words_to_text
from pdf_ocr_bench.engines.paddleocr_engine import line_words
from pdf_ocr_bench.evaluation import metrics
from pdf_ocr_bench.html_output.layout import text_width_em, word_styles
from pdf_ocr_bench.html_output.renderer import load_template, render_page_html
from pdf_ocr_bench.models import BBox, OcrWord, PageImage, PageResult
from pdf_ocr_bench.pipeline import PipelineConfig, parse_page_range
from pdf_ocr_bench.plan import InputError

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


def test_paddle_line_words_follow_the_line_text():
    # PaddleOCR fragments split at character-class changes and may hold spaces (", Ü")
    text = "Grüße wörld, Übergröße."
    frags = ["Gr", "üß", "e ", "w", "ö", "rld", ", Ü", "bergr", "öß", "e", "."]
    regions, x = [], 0
    for frag in frags:
        regions.append([(x, 0), (x + 10 * len(frag), 0), (x + 10 * len(frag), 20), (x, 20)])
        x += 10 * len(frag)
    words = line_words(text, frags, regions)
    assert [w for w, _ in words] == ["Grüße", "wörld,", "Übergröße."]
    # one character = 10 px: "Grüße" spans x 0..50, "wörld," 60..120, "Übergröße." 130..230
    assert [box[0] for _, box in words] == [0, 60, 130] and [box[2] for _, box in words] == [50, 120, 230]
    assert line_words("something else", frags, regions) is None  # fragments do not spell the text


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
    assert "olmocr" not in all_cpu and "tesseract" in all_cpu and "macos_vision" in all_cpu and len(all_cpu) == 9
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
    assert "<img" not in html  # text only: the scan never goes into the page
    assert "width: 595.28pt" in html and "height: 841.89pt" in html
    assert 'name="pdf.options.pageWidth" content="210.0016"' in html
    assert "a&lt;b&amp;c </span>" in html  # escaped, with the inter-word space
    assert ">ß</span>" in html  # last word of the line: no trailing space
    assert "left: 10.0000%" in html
    style = html[html.index("<style>") : html.index("</style>")]
    assert "color: #000;" in style and "transparent" not in style


def test_text_width_uses_helvetica_metrics():
    assert text_width_em("Hallo") == pytest.approx(0.722 + 0.556 + 0.222 + 0.222 + 0.556, abs=1e-3)
    assert text_width_em("Grüße") == pytest.approx(0.778 + 0.333 + 0.556 + 0.611 + 0.556, abs=1e-3)
    assert text_width_em("u\u0308") == text_width_em("ü")  # decomposed umlaut measured as one glyph
    assert text_width_em("漢字") == pytest.approx(2.0)  # no Helvetica glyph: wide fallback


def _boxed(text: str, x: float, y: float, size_pt: float, page_w: float = 600, h: float = 0.02) -> OcrWord:
    """A word whose box is exactly as wide as `text` set at `size_pt`."""
    return word(text, x, y, w=text_width_em(text) * size_pt / page_w, h=h)


def test_word_styles_one_size_per_line():
    a, b, c = _boxed("Hello", 0.1, 0.1, 12), _boxed("big", 0.3, 0.1, 12), _boxed("world", 0.5, 0.1, 12)
    styles = word_styles([a, b, c], 600, 800)
    assert [styles[id(w)].font_size_pt for w in (a, b, c)] == pytest.approx([12, 12, 12])
    assert [styles[id(w)].suffix for w in (a, b, c)] == [" ", " ", ""]
    # one badly boxed word does not resize the line (median)
    a2, b2, c2 = _boxed("Hello", 0.1, 0.1, 12), _boxed("big", 0.3, 0.1, 30), _boxed("world", 0.5, 0.1, 12)
    styles = word_styles([a2, b2, c2], 600, 800)
    assert styles[id(a2)].font_size_pt == pytest.approx(12) and styles[id(b2)].font_size_pt == pytest.approx(12)


def test_word_styles_squeeze_only_before_a_neighbour():
    first = _boxed("Danach", 0.1, 0.1, 12)
    # the next word starts right where "Danach" ends: no room for the space at 12pt
    nxt = _boxed("vergleichen", first.bbox.x1, 0.1, 12)
    styles = word_styles([first, nxt], 600, 800)
    room_pt = (nxt.bbox.x - first.bbox.x) * 600
    assert styles[id(first)].font_size_pt == pytest.approx(room_pt / text_width_em("Danach "))
    assert styles[id(nxt)].font_size_pt == pytest.approx(12)  # last word: only the page edge limits it
    # squeezing never goes below half the line size
    crowded = [word("Danach", 0.1, 0.1, w=0.1), word("x", 0.101, 0.1, w=0.1), word("y", 0.3, 0.1, w=0.1)]
    styles = word_styles(crowded, 600, 800)
    line = max(styles[id(w)].font_size_pt for w in crowded)
    assert styles[id(crowded[0])].font_size_pt == pytest.approx(line / 2)


def test_word_styles_height_guard_and_vertical_center():
    wide = word("i", 0.1, 0.1, w=0.5, h=0.01)  # box far wider than one "i": width fit is huge
    styles = word_styles([wide], 600, 800)
    assert styles[id(wide)].font_size_pt == pytest.approx(1.5 * 0.01 * 800)
    w = _boxed("Hello", 0.1, 0.5, 12)
    st = word_styles([w], 600, 800)[id(w)]
    center_pt = (w.bbox.y + w.bbox.h / 2) * 800
    assert st.top * 800 + (0.770 - 0.718 / 2) * st.font_size_pt == pytest.approx(center_pt)


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


def _run(monkeypatch, tmp_path, pdf, engine_classes, **kw):
    monkeypatch.setattr(plan, "select_engines", lambda spec, include_gpu=False: engine_classes)
    cfg = PipelineConfig(input_pdf=pdf, output_dir=tmp_path / "results", dpi=72, **kw)
    return pipeline.run(cfg), tmp_path / "results"


def test_pipeline_end_to_end(monkeypatch, tmp_path, pdf):
    report, out = _run(monkeypatch, tmp_path, pdf, [FakeEngine, FakeEngineB, FakeEngineC, BrokenEngine, CrashEngine])

    assert sorted(p.name for p in (out / "images").iterdir()) == ["page_001.png", "page_002.png"]
    with zipfile.ZipFile(out / "fake_a" / "pages.zip") as zf:
        # the shared page images stay in results/images; zips hold only text
        assert sorted(zf.namelist()) == ["metadata.json", "page_001.html", "page_002.html"]
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


class LatinOnlyEngine(FakeEngine):
    name = "latin_only"
    display_name = "LatinOnly"

    @classmethod
    def route(cls, languages):
        other = [lang.code for lang in languages if lang.script != "Latin"]
        return Route(unsupported=f"no model for {', '.join(other)}") if other else Route(lang="latin", detail="latin")


class JoinedWordsEngine(FakeEngine):
    name = "joined"
    display_name = "Joined"

    def ocr_page(self, image, lang):
        return [word("Hello world", 0.1, 0.4, w=0.4)]


def test_engine_that_cannot_read_the_language(monkeypatch, tmp_path, pdf):
    # --engines all: skipped, and the report says why
    report, out = _run(monkeypatch, tmp_path, pdf, [FakeEngine, LatinOnlyEngine], lang="chi_sim")
    by_name = {e.name: e for e in report.engines}
    assert by_name["fake_a"].success
    assert not by_name["latin_only"].success and by_name["latin_only"].error == "skipped: no model for chi_sim"
    assert not (out / "latin_only").exists()
    # asked for by name: rejected before anything is rendered
    with pytest.raises(InputError, match="latin_only: no model for chi_sim"):
        _run(monkeypatch, tmp_path / "x", pdf, [FakeEngine, LatinOnlyEngine], lang="chi_sim", engines="fake_a,latin_only")
    assert not (tmp_path / "x" / "results" / "images").exists()


def test_bad_input_is_rejected_before_rendering(monkeypatch, tmp_path, pdf):
    for kw, message in [
        ({"lang": "deu+xyz"}, "unknown language code"),
        ({"preprocess": "grayscale,blur"}, "unknown filter"),
        ({"pages": "1-x"}, "invalid page range"),
        ({"pages": "9"}, "selects no pages"),
    ]:
        with pytest.raises(InputError, match=message):
            _run(monkeypatch, tmp_path, pdf, [FakeEngine], **kw)
    assert not (tmp_path / "results" / "images").exists()


def test_words_never_contain_whitespace(monkeypatch, tmp_path, pdf):
    report, out = _run(monkeypatch, tmp_path, pdf, [JoinedWordsEngine])
    assert report.engines[0].total_words == 4  # 2 pages x "Hello", "world"
    html = zipfile.ZipFile(out / "joined" / "pages.zip").read("page_001.html").decode()
    assert ">Hello </span>" in html and ">world</span>" in html


def test_preprocessing_runs_before_the_engines(monkeypatch, tmp_path, pdf):
    from PIL import Image

    report, out = _run(monkeypatch, tmp_path, pdf, [FakeEngine], preprocess="grayscale,binarize")
    assert report.preprocess == ["grayscale", "binarize"]
    with Image.open(out / "images" / "page_001.png") as img:
        assert img.mode == "L" and set(img.getdata()) <= {0, 255}


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
