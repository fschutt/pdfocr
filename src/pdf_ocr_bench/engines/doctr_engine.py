from __future__ import annotations

from ..languages import Language
from ..models import BBox, OcrWord, PageImage
from .base import OcrEngine, Option, Route, unsupported

# docTR's built-in recognizers use the `french` vocabulary: English and French, but no ä/ö/ß.
# Other Latin-script languages need the multilingual PARSeq model from the Hugging Face hub,
# whose vocabulary is Latin-script only.
BUILTIN_TAGS = {"en", "fr"}
DET_ARCHS = (
    "fast_base", "fast_small", "fast_tiny", "db_resnet50", "db_resnet34", "db_mobilenet_v3_large",
    "linknet_resnet18", "linknet_resnet34", "linknet_resnet50",
)  # fmt: skip
MULTILINGUAL_HUB = "Felix92/doctr-torch-parseq-multilingual-v1"


class DoctrEngine(OcrEngine):
    name = "doctr"
    display_name = "docTR"
    model = "docTR (PyTorch): text detector + CRNN (en/fr) or multilingual PARSeq from the HF hub"
    options = {
        "det_arch": Option("fast_base", "text detection model", choices=DET_ARCHS),
        "straight_pages": Option(True, "assume pages are not rotated (faster, axis-aligned boxes)"),
    }

    @classmethod
    def route(cls, languages: list[Language]) -> Route:
        other = [lang.code for lang in languages if lang.script != "Latin"]
        if other:
            return unsupported(f"docTR only has Latin-script recognizers (not {', '.join(other)})")
        if all(lang.tag in BUILTIN_TAGS for lang in languages):
            return Route(lang="builtin", detail="built-in (french vocab)")
        return Route(lang="multilingual", detail="multilingual PARSeq (HF hub)")

    def prepare(self) -> None:
        from doctr.models import from_hub, ocr_predictor

        kwargs = {"det_arch": self.opts["det_arch"], "assume_straight_pages": self.opts["straight_pages"], "pretrained": True}
        if self.lang == "multilingual":
            kwargs["reco_arch"] = from_hub(MULTILINGUAL_HUB)
        self._model = ocr_predictor(**kwargs)

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
