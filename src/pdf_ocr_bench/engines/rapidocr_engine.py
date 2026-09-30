from __future__ import annotations

from ..models import BBox, OcrWord, PageImage
from .base import OcrEngine, split_line


class RapidOcrEngine(OcrEngine):
    name = "rapidocr"
    display_name = "RapidOCR"

    def prepare(self) -> None:
        from rapidocr import RapidOCR

        try:
            self._engine = RapidOCR(params=rec_params(self.lang))
        except Exception as exc:  # noqa: BLE001 - unsupported lang/model download
            self.log.warning(f"lang '{self.lang}' unavailable ({exc}), using default model")
            self._engine = RapidOCR()

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


def rec_params(lang: str) -> dict:
    """The default PP-OCRv6 "small" recognizer only covers ch/en; other scripts ship as PP-OCRv5 mobile."""
    from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

    if lang in ("ch", "en"):
        return {"Rec.lang_type": LangRec(lang)}
    return {"Rec.lang_type": LangRec(lang), "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE}
