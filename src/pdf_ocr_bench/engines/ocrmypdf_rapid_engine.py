from __future__ import annotations

from importlib import resources

from ..languages import Language
from .base import Route
from .ocrmypdf_engine import OcrmypdfEngine
from .rapidocr_engine import FAMILIES, rapidocr_family

# ocrmypdf-rapidocr takes one Tesseract code and maps it to a RapidOCR recognizer. It knows no
# Fraktur or Latin-language code; any Latin-script code selects the same LATIN recognizer.
PLUGIN_CODE = {"deu_frak": "deu", "frk": "deu", "lat": "ita"}
# The plugin maps Russian/Ukrainian to its CYRILLIC recognizer (it has no East Slavic one).
PLUGIN_FAMILY = {"eslav": "cyrillic"}


class OcrmypdfRapidEngine(OcrmypdfEngine):
    name = "ocrmypdf_rapid"
    display_name = "ocrmypdf+rapid"
    model = "ocrmypdf with the ocrmypdf-rapidocr plugin (RapidOCR PP-OCR models)"
    options = {}

    @classmethod
    def route(cls, languages: list[Language]) -> Route:
        family = rapidocr_family(languages)
        if isinstance(family, Route):
            return family
        main = next((lang for lang in languages if lang.code != "eng"), languages[0])
        code = PLUGIN_CODE.get(main.code, main.code)
        _, version, model_type = FAMILIES[family]
        rec = PLUGIN_FAMILY.get(family, family)
        return Route(lang=(code, version, model_type), detail=f"-l {code}: {rec} ({version} {model_type})")

    @classmethod
    def preflight(cls, route: Route) -> Route:
        return route

    def prepare(self) -> None:
        import ocrmypdf_rapidocr  # noqa: F401 - fail early if missing

        super().prepare()
        self._config = self._write_config()

    def ocr_language(self) -> str:
        return self.lang[0]

    def plugin_args(self) -> list[str]:
        args = ["--plugin", "ocrmypdf_rapidocr"]
        return args if self._config is None else [*args, "--rapidocr-config-path", str(self._config)]

    def _write_config(self):
        """RapidOCR's default recognizer is PP-OCRv6 small, which only has ch/en/japan/chinese_cht.

        The plugin only overrides `lang_type`, so other scripts need a config selecting their
        PP-OCRv5 model. RapidOCR reads a config file instead of its defaults, not on top of
        them, so the file is its own default config with the recognizer version changed.
        """
        import yaml

        _, version, model_type = self.lang
        default = yaml.safe_load(resources.files("rapidocr").joinpath("config.yaml").read_text(encoding="utf-8"))
        if (default["Rec"]["ocr_version"], default["Rec"]["model_type"]) == (version, model_type):
            return None
        default["Rec"]["ocr_version"] = version
        default["Rec"]["model_type"] = model_type
        path = self._tmp / "rapidocr.yaml"
        path.write_text(yaml.safe_dump(default, sort_keys=False), encoding="utf-8")
        return path
