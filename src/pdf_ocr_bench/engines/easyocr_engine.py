from __future__ import annotations

from ..languages import Language
from ..models import BBox, OcrWord, PageImage
from .base import OcrEngine, Option, Route, split_line, unsupported

# EasyOCR has one recognizer per script group. Latin languages mix freely; a Cyrillic or
# Arabic language combines with others of its script and English; Chinese, Japanese and
# Korean each only combine with English (easyocr.Reader raises otherwise).
ENGLISH_ONLY_PARTNER = {"Han", "Japanese", "Hangul"}


class EasyOcrEngine(OcrEngine):
    name = "easyocr"
    display_name = "EasyOCR"
    model = "EasyOCR (PyTorch): CRAFT text detector + one CRNN recognizer per script group"
    options = {
        "decoder": Option("greedy", "CTC decoding: greedy (fast) or beam search", choices=("greedy", "beamsearch", "wordbeamsearch")),
    }

    @classmethod
    def route(cls, languages: list[Language]) -> Route:
        codes = tuple(dict.fromkeys(lang.easyocr for lang in languages))
        main = [lang for lang in languages if lang.code != "eng"]
        scripts = {lang.script for lang in main}
        if len(scripts) > 1:
            return unsupported(f"EasyOCR cannot combine {' and '.join(sorted(scripts))} scripts in one reader")
        if scripts & ENGLISH_ONLY_PARTNER and len(main) > 1:
            return unsupported(f"EasyOCR only combines {main[0].name} with English")
        return Route(lang=codes, detail=", ".join(codes))

    def prepare(self) -> None:
        import easyocr

        self._reader = easyocr.Reader(list(self.lang), gpu=False, verbose=False)

    def ocr_page(self, image: PageImage, lang: tuple[str, ...]) -> list[OcrWord]:
        detections = self._reader.readtext(str(image.path), detail=1, paragraph=False, decoder=self.opts["decoder"])
        return [
            word
            for points, text, conf in detections
            for word in split_line(text, BBox.from_points(points, image.width_px, image.height_px), float(conf))
        ]
