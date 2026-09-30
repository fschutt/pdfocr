from __future__ import annotations

from .ocrmypdf_engine import OcrmypdfEngine


class OcrmypdfRapidEngine(OcrmypdfEngine):
    name = "ocrmypdf_rapid"
    display_name = "ocrmypdf+rapid"
    plugin_args = ("--plugin", "ocrmypdf_rapidocr")

    def prepare(self) -> None:
        import ocrmypdf_rapidocr  # noqa: F401 - fail early if missing

        super().prepare()
