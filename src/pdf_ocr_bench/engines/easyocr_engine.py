from __future__ import annotations

from ..models import BBox, OcrWord, PageImage
from .base import OcrEngine, split_line

# EasyOCR only combines these scripts with English.
_WITH_ENGLISH = {"ch_sim", "ch_tra", "ja", "ko", "ar", "ru", "de", "fr", "la"}


class EasyOcrEngine(OcrEngine):
    name = "easyocr"
    display_name = "EasyOCR"

    def prepare(self) -> None:
        import easyocr

        langs = [self.lang, "en"] if self.lang in _WITH_ENGLISH else [self.lang]
        try:
            self._reader = easyocr.Reader(langs, gpu=False, verbose=False)
        except Exception as exc:  # noqa: BLE001 - unsupported lang
            self.log.warning(f"lang {langs} unavailable ({exc}), using ['en']")
            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def ocr_page(self, image: PageImage, lang: str) -> list[OcrWord]:
        detections = self._reader.readtext(str(image.path), detail=1, paragraph=False)
        return [
            word
            for points, text, conf in detections
            for word in split_line(text, BBox.from_points(points, image.width_px, image.height_px), float(conf))
        ]
