from __future__ import annotations

from ..models import BBox, OcrWord, PageImage
from .base import OcrEngine

# The default pretrained recognizer covers Latin-script languages (en, fr, de, es, it, pt, ...).
_LATIN = {"en", "fr", "de", "la", "es", "it", "pt", "nl"}
_MULTILINGUAL_HUB = "Felix92/doctr-torch-parseq-multilingual-v1"


class DoctrEngine(OcrEngine):
    name = "doctr"
    display_name = "docTR"

    def prepare(self) -> None:
        from doctr.models import ocr_predictor

        self._model = ocr_predictor(pretrained=True, **self._reco_kwargs())

    def _reco_kwargs(self) -> dict:
        if self.lang in _LATIN:
            return {}
        try:
            from doctr.models import from_hub

            self.log.info(f"lang '{self.lang}': loading multilingual recognizer {_MULTILINGUAL_HUB}")
            return {"reco_arch": from_hub(_MULTILINGUAL_HUB)}
        except Exception as exc:  # noqa: BLE001 - offline / hub unavailable
            self.log.warning(f"lang '{self.lang}' unsupported ({exc}), using default Latin model")
            return {}

    def ocr_page(self, image: PageImage, lang: str) -> list[OcrWord]:
        from doctr.io import DocumentFile

        doc = self._model(DocumentFile.from_images([str(image.path)]))
        return [
            OcrWord(text=word.value, bbox=BBox.from_corners(x0, y0, x1, y1), confidence=float(word.confidence))
            for page in doc.pages
            for block in page.blocks
            for line in block.lines
            for word in line.words
            for (x0, y0), (x1, y1) in [_corners(word.geometry)]
        ]


def _corners(geometry) -> tuple[tuple[float, float], tuple[float, float]]:
    """((x0, y0), (x1, y1)) for straight pages, 4-point polygon for rotated ones."""
    xs = [float(p[0]) for p in geometry]
    ys = [float(p[1]) for p in geometry]
    return (min(xs), min(ys)), (max(xs), max(ys))
