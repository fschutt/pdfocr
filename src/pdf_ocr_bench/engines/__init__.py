"""Engine registry. Order here is the run order."""

from __future__ import annotations

from .base import OcrEngine, PageTimeout
from .doctr_engine import DoctrEngine
from .easyocr_engine import EasyOcrEngine
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
        OlmOcrEngine,
    )
}


def select_engines(spec: str, include_gpu: bool = False) -> list[type[OcrEngine]]:
    """`all` = every CPU engine (+ GPU engines with `include_gpu`); otherwise a comma list.

    Naming a GPU engine explicitly runs it regardless of `include_gpu`.
    """
    names = [n.strip().lower().replace("-", "_") for n in spec.split(",") if n.strip()]
    if not names or names == ["all"]:
        return [cls for cls in ENGINES.values() if include_gpu or not cls.requires_gpu]
    unknown = [n for n in names if n not in ENGINES]
    if unknown:
        raise ValueError(f"unknown engine(s): {', '.join(unknown)} (available: {', '.join(ENGINES)})")
    return [ENGINES[n] for n in dict.fromkeys(names)]


__all__ = ["ENGINES", "OcrEngine", "PageTimeout", "select_engines"]
