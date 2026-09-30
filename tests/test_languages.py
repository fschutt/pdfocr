from __future__ import annotations

import pytest

from pdf_ocr_bench.engines import ENGINES
from pdf_ocr_bench.engines import tesseract as tess
from pdf_ocr_bench.languages import LANGUAGES, LanguageError, html_lang, parse_languages, tesseract_packages
from pdf_ocr_bench import plan as plan_module
from pdf_ocr_bench.plan import InputError, check_page_spec, make_plan


def route(engine: str, spec: str):
    return ENGINES[engine].route(parse_languages(spec))


def test_parse_languages_keeps_order_and_dedupes():
    assert [l.code for l in parse_languages("deu+eng+deu")] == ["deu", "eng"]
    assert html_lang(parse_languages("chi_tra+eng")) == "zh-Hant"


@pytest.mark.parametrize("spec", ["", "+", "deu+", "deu++eng"])
def test_parse_languages_rejects_malformed(spec):
    with pytest.raises(LanguageError, match="expected codes joined by"):
        parse_languages(spec)


def test_parse_languages_rejects_unknown_codes_and_lists_the_supported_ones():
    with pytest.raises(LanguageError, match=r"unknown language code\(s\): german, xx\. Supported: eng, deu"):
        parse_languages("german+deu+xx")


def test_tesseract_packages():
    assert tesseract_packages(parse_languages("deu_frak+eng")) == ["tesseract-ocr-frk", "tesseract-ocr-script-frak", "tesseract-ocr-eng"]
    assert tesseract_packages(parse_languages("chi_sim")) == ["tesseract-ocr-chi-sim"]


@pytest.mark.parametrize(
    ("spec", "family"),
    [("eng", "en"), ("deu+eng", "latin"), ("eng+deu", "latin"), ("deu_frak", "latin"), ("chi_sim+eng", "ch"),
     ("rus+ukr+eng", "eslav"), ("jpn", "japan"), ("kor+eng", "korean")],
)  # fmt: skip
def test_rapidocr_routes_to_one_recognizer(spec, family):
    r = route("rapidocr", spec)
    assert r.ok and r.lang == family


def test_rapidocr_rejects_languages_without_a_shared_recognizer():
    assert "no single RapidOCR recognizer covers kor+rus" in route("rapidocr", "kor+rus").unsupported
    assert not route("rapidocr", "deu+chi_sim").ok


def test_paddleocr_routes_by_script():
    assert route("paddleocr", "deu+eng").lang == "de"
    assert route("paddleocr", "eng").lang == "en"
    assert route("paddleocr", "chi_sim+deu").lang == "ch"  # one multilingual PP-OCRv6 model
    assert route("paddleocr", "ukr+eng").lang == "uk"
    assert not route("paddleocr", "ara+rus").ok


def test_easyocr_follows_its_script_groups():
    assert route("easyocr", "deu+eng").lang == ("de", "en")
    assert route("easyocr", "chi_sim+eng").lang == ("ch_sim", "en")
    assert route("easyocr", "rus+ukr+eng").lang == ("ru", "uk", "en")
    assert "only combines" in route("easyocr", "chi_sim+chi_tra").unsupported
    assert "cannot combine Cyrillic and Latin" in route("easyocr", "rus+deu").unsupported


def test_doctr_needs_the_multilingual_model_beyond_english_and_french():
    assert route("doctr", "eng+fra").lang == "builtin"
    assert route("doctr", "deu").lang == "multilingual"
    assert "Latin-script" in route("doctr", "rus").unsupported


def test_macos_vision_routes_to_bcp47_tags_and_is_macos_only(monkeypatch):
    assert route("macos_vision", "deu+eng").lang == ("de-DE", "en-US")
    assert "no recognizer for lat" in route("macos_vision", "lat").unsupported
    monkeypatch.setattr("sys.platform", "linux")
    assert ENGINES["macos_vision"].preflight(route("macos_vision", "eng")).unsupported == "macOS only (Apple Vision framework)"


def test_ocrmypdf_rapid_passes_one_code_and_the_recognizer_version():
    assert route("ocrmypdf_rapid", "deu_frak+eng").lang == ("deu", "PP-OCRv5", "mobile")
    assert route("ocrmypdf_rapid", "eng").lang == ("eng", "PP-OCRv6", "small")
    assert route("ocrmypdf_rapid", "lat").lang[0] == "ita"


def test_vlm_engines_need_no_language():
    assert route("surya", "kor+rus").ok and route("olmocr", "ara").ok


def test_tesseract_preflight_uses_installed_models(monkeypatch):
    monkeypatch.setattr(tess.shutil, "which", lambda _: "/usr/bin/tesseract")
    monkeypatch.setattr(tess, "installed_models", lambda: frozenset({"eng", "deu", "Fraktur"}))
    ok = tess.tesseract_preflight(route("tesseract", "deu_frak+eng"))
    assert ok.lang == "Fraktur+eng" and "deu_frak uses Fraktur" in ok.detail
    missing = tess.tesseract_preflight(route("tesseract", "chi_sim"))
    assert missing.unsupported == "no Tesseract model for chi_sim (chi_sim); install tesseract-ocr-chi-sim"
    monkeypatch.setattr(tess.shutil, "which", lambda _: None)
    assert "not installed" in tess.tesseract_preflight(route("tesseract", "eng")).unsupported


def test_plan_rejects_named_engines_and_skips_under_all(monkeypatch):
    with pytest.raises(InputError, match="doctr: docTR only has Latin-script recognizers"):
        make_plan("tesseract,doctr", "chi_sim", preflight=False)
    p = make_plan("all", "kor+rus", preflight=False)
    assert {e.engine.name for e in p.skipped} == {"rapidocr", "paddleocr", "easyocr", "doctr", "ocrmypdf_rapid"}
    assert {e.engine.name for e in p.runnable} >= {"tesseract", "surya"}
    # under `all`, it is still an error when nothing at all can run
    monkeypatch.setattr(plan_module, "select_engines", lambda spec, include_gpu=False: [ENGINES["doctr"], ENGINES["easyocr"]])
    with pytest.raises(InputError, match=r"no engine can read --lang kor\+rus"):
        make_plan("all", "kor+rus", preflight=False)


def test_check_page_spec():
    for ok in (None, "", "all", "3", "1-5", "1,3,7-10", " 2 - 4 , 9 "):
        check_page_spec(ok)
    for bad in ("1-x", "1,,2", "-3", "a"):
        with pytest.raises(InputError):
            check_page_spec(bad)


def test_every_language_is_complete():
    for lang in LANGUAGES.values():
        assert lang.tesseract_models and lang.tesseract_packages and lang.tag
