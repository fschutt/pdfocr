from __future__ import annotations

import pytest

from pdf_ocr_bench.engines import ENGINES
from pdf_ocr_bench.engines.base import Option, Route
from pdf_ocr_bench.plan import InputError, make_plan, parse_engine_options


def test_option_parse_by_type():
    assert Option(True, "").parse("no") is False and Option(False, "").parse("YES") is True
    assert Option(3, "", choices=(1, 3, 6)).parse("6") == 6
    assert Option(0.5, "", minimum=0.0, maximum=1.0).parse("0.25") == 0.25
    assert Option("x", "").parse(" http://host:8000/v1 ") == "http://host:8000/v1"
    for option, raw, message in [
        (Option(True, ""), "maybe", "true or false"),
        (Option(3, "", choices=(1, 3, 6)), "2", "one of 1, 3, 6"),
        (Option(3, ""), "three", "expected int"),
        (Option(0.5, "", minimum=0.0, maximum=1.0), "1.5", "0.0..1.0"),
    ]:
        with pytest.raises(ValueError, match=message):
            option.parse(raw)


def test_parse_engine_options_validates_everything():
    selected = [ENGINES["tesseract"], ENGINES["macos_vision"]]
    assert parse_engine_options(["macos-vision.level=fast", "tesseract.psm=6"], selected) == {
        "macos_vision": {"level": "fast"},
        "tesseract": {"psm": 6},
    }
    for raw, message in [
        ("tesseract.psm", "expected ENGINE.KEY=VALUE"),
        ("psm=6", "expected ENGINE.KEY=VALUE"),
        ("nope.psm=6", "unknown engine 'nope'"),
        ("easyocr.decoder=greedy", "easyocr is not among the selected engines"),
        ("tesseract.oem=1", "tesseract has no option 'oem' \\(its options: psm\\)"),
        ("macos_vision.level=slow", "expected one of accurate, fast"),
    ]:
        with pytest.raises(InputError, match=message):
            parse_engine_options([raw], selected)


def test_engines_get_defaults_plus_overrides():
    engine = ENGINES["tesseract"](Route(lang="eng"), options={"psm": 6})
    assert engine.opts == {"psm": 6}
    assert ENGINES["rapidocr"](Route(lang="en")).opts == {"min_score": 0.5, "text_orientation": True}
    with pytest.raises(ValueError, match="unknown option"):
        ENGINES["surya"](Route(), options={"level": "fast"})


def test_plan_carries_options():
    plan = make_plan("tesseract,rapidocr", "deu", preflight=False, engine_options=["rapidocr.min_score=0.8"])
    by_name = {p.engine.name: p for p in plan.runnable}
    assert by_name["rapidocr"].options == {"min_score": 0.8} and by_name["tesseract"].options == {}


def test_every_engine_documents_its_model_and_options():
    for cls in ENGINES.values():
        assert cls.model, cls.name
        for key, option in cls.options.items():
            assert option.help and option.domain(), (cls.name, key)
