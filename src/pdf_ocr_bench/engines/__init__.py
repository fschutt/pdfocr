"""Engine registry. Order here is the run order."""

from __future__ import annotations

from .base import OcrEngine, PageTimeout, Route
from .doctr_engine import DoctrEngine
from .easyocr_engine import EasyOcrEngine
from .macos_vision_engine import MacOSVisionEngine
from .ocrmypdf_engine import OcrmypdfEngine
from .ocrmypdf_rapid_engine import OcrmypdfRapidEngine
from .olmocr_engine import OlmOcrEngine
from .paddleocr_engine import PaddleOcrEngine
from .rapidocr_engine import RapidOcrEngine
from .surya_engine import SuryaEngine
from .tesseract import TesseractEngine

ENGINES: dict[str, type[OcrEngine]] = {
    cls.name: cls
    for cls in (
        TesseractEngine,
        RapidOcrEngine,
        PaddleOcrEngine,
        EasyOcrEngine,
        DoctrEngine,
        SuryaEngine,
        OcrmypdfEngine,
        OcrmypdfRapidEngine,
        MacOSVisionEngine,
        OlmOcrEngine,
    )
}


def _names(spec: str) -> list[str]:
    return [n.strip().lower().replace("-", "_") for n in (spec or "").split(",") if n.strip()]


def is_all(spec: str) -> bool:
    """`all` selects whatever can run; an explicit list asks for exactly those engines."""
    return _names(spec) in ([], ["all"])


def select_engines(spec: str, include_gpu: bool = False) -> list[type[OcrEngine]]:
    """`all` = every CPU engine (+ GPU engines with `include_gpu`); otherwise a comma list.

    Naming a GPU engine explicitly runs it regardless of `include_gpu`.
    """
    names = _names(spec)
    if is_all(spec):
        return [cls for cls in ENGINES.values() if include_gpu or not cls.requires_gpu]
    unknown = [n for n in names if n not in ENGINES]
    if unknown:
        raise ValueError(f"unknown engine(s): {', '.join(unknown)} (available: {', '.join(ENGINES)})")
    return [ENGINES[n] for n in dict.fromkeys(names)]


__all__ = ["ENGINES", "OcrEngine", "PageTimeout", "Route", "is_all", "select_engines"]
