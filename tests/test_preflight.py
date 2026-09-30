from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from pdf_ocr_bench import plan
from pdf_ocr_bench.engines import ENGINES, OcrEngine, Route
from pdf_ocr_bench.plan import InputError, make_plan

ROOT = Path(__file__).resolve().parent.parent


class Installed(OcrEngine):
    name = "installed"
    display_name = "Installed"
    modules = ("json",)
    extra = "installed"

    def ocr_page(self, image, lang):
        return []


class Uninstalled(Installed):
    name = "uninstalled"
    display_name = "Uninstalled"
    modules = ("json", "no_such_module_for_pdf_ocr_bench")
    extra = "uninstalled"


def test_missing_packages_are_named_with_their_extra():
    route = Uninstalled.preflight(Route(lang="en"))
    assert route.unsupported == "not installed (no_such_module_for_pdf_ocr_bench): pip install -e '.[uninstalled]'"
    assert Installed.preflight(Route(lang="en")) == Route(lang="en")
    assert Uninstalled.preflight(Route(unsupported="no model")).unsupported == "no model"


def test_uninstalled_engines_are_skipped_under_all_and_rejected_by_name(monkeypatch):
    monkeypatch.setattr(plan, "select_engines", lambda spec, gpu: [Installed, Uninstalled])
    result = make_plan("all", "eng")
    assert [p.engine for p in result.runnable] == [Installed]
    assert [p.engine for p in result.skipped] == [Uninstalled]
    with pytest.raises(InputError, match=r"uninstalled: not installed .*pip install -e '\.\[uninstalled\]'"):
        make_plan("installed,uninstalled", "eng")
    # without preflight (the workflow's validate job) nothing is checked against the machine
    assert len(make_plan("installed,uninstalled", "eng", preflight=False).runnable) == 2


def test_every_engine_names_a_real_extra():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    for cls in ENGINES.values():
        assert cls.modules and cls.extra in extras, cls.name


def test_surya_needs_llama_server_unless_a_server_is_given(monkeypatch):
    pytest.importorskip("surya")
    from surya.settings import settings

    monkeypatch.setattr(settings, "SURYA_INFERENCE_BACKEND", "llamacpp")
    monkeypatch.setattr(settings, "SURYA_INFERENCE_URL", None)
    monkeypatch.setattr(settings, "LLAMA_CPP_BINARY", "no-such-llama-server")
    surya = ENGINES["surya"]
    assert "no-such-llama-server is not installed" in surya.preflight(surya.route([])).unsupported
    monkeypatch.setattr(settings, "SURYA_INFERENCE_URL", "http://127.0.0.1:8000/v1")
    assert surya.preflight(surya.route([])).ok
