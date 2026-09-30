from __future__ import annotations

from ..languages import Language
from ..models import BBox, OcrWord, PageImage
from .base import OcrEngine, Option, Route, split_line, unsupported

# Recognizer family -> (Rec.lang_type, Rec.ocr_version, Rec.model_type) that RapidOCR 3.9 ships.
# PP-OCRv6 "small" only takes ch/en/japan/chinese_cht; the other scripts are PP-OCRv5 mobile.
# Every family also reads English, so English combines with any of them.
FAMILIES: dict[str, tuple[str, str, str]] = {
    "en": ("en", "PP-OCRv6", "small"),
    "ch": ("ch", "PP-OCRv6", "small"),
    "chinese_cht": ("chinese_cht", "PP-OCRv6", "small"),
    "japan": ("japan", "PP-OCRv6", "small"),
    "latin": ("latin", "PP-OCRv5", "mobile"),
    "eslav": ("eslav", "PP-OCRv5", "mobile"),
    "arabic": ("arabic", "PP-OCRv5", "mobile"),
    "korean": ("korean", "PP-OCRv5", "mobile"),
}


def rapidocr_family(languages: list[Language]) -> str | Route:
    """The one recognizer family covering all `languages`, or an unsupported Route."""
    families = list(dict.fromkeys(lang.rapidocr for lang in languages if lang.rapidocr != "en")) or ["en"]
    if len(families) > 1:
        return unsupported(f"no single RapidOCR recognizer covers {'+'.join(l.code for l in languages)} ({', '.join(families)})")
    return families[0]


def rec_params(family: str) -> dict:
    from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

    lang_type, version, model_type = FAMILIES[family]
    return {"Rec.lang_type": LangRec(lang_type), "Rec.ocr_version": OCRVersion(version), "Rec.model_type": ModelType(model_type)}


class RapidOcrEngine(OcrEngine):
    name = "rapidocr"
    display_name = "RapidOCR"
    model = "RapidOCR (PP-OCR on ONNX Runtime): PP-OCRv6 small for en/zh/ja, PP-OCRv5 mobile for other scripts"
    options = {
        "min_score": Option(0.5, "drop text lines recognized with a lower score", minimum=0.0, maximum=1.0),
        "text_orientation": Option(True, "detect and turn upside-down (180°) text lines"),
    }

    @classmethod
    def route(cls, languages: list[Language]) -> Route:
        family = rapidocr_family(languages)
        if isinstance(family, Route):
            return family
        lang_type, version, model_type = FAMILIES[family]
        return Route(lang=family, detail=f"{lang_type} ({version} {model_type})")

    def prepare(self) -> None:
        from rapidocr import RapidOCR

        params = rec_params(self.lang) | {
            "Global.text_score": self.opts["min_score"],
            "Global.use_cls": self.opts["text_orientation"],
        }
        self._engine = RapidOCR(params=params)

    def ocr_page(self, image: PageImage, lang: str) -> list[OcrWord]:
        out = self._engine(str(image.path), return_word_box=True)
        if out.txts is None or out.boxes is None:
            return []
        lines = zip(out.boxes, out.txts, out.scores)
        word_lines = out.word_results or [()] * len(out.txts)
        return [
            word
            for (box, text, score), words in zip(lines, word_lines)
            for word in self._line_words(box, text, float(score), words, image)
        ]

    @staticmethod
    def _line_words(box, text: str, score: float, words, image: PageImage) -> list[OcrWord]:
        if not words:
            return split_line(text, BBox.from_points(box, image.width_px, image.height_px), score)
        return [
            OcrWord(text=w_text, bbox=BBox.from_points(w_box, image.width_px, image.height_px), confidence=float(w_score))
            for w_text, w_score, w_box in words
        ]
