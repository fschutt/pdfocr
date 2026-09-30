from __future__ import annotations

import os
import re

from ..languages import Language
from ..models import BBox, OcrWord, PageImage
from .base import OcrEngine, Option, Route, split_line, unsupported

# PaddleOCR 3.7 picks the recognizer from `lang`: one multilingual PP-OCRv6 model reads English,
# Chinese, Japanese and every Latin-script language; the other scripts have PP-OCRv5 models.
# Every model also reads English.
MODEL_BY_SCRIPT = {
    "Latin": "PP-OCRv6 multilingual",
    "Han": "PP-OCRv6 multilingual",
    "Japanese": "PP-OCRv6 multilingual",
    "Cyrillic": "PP-OCRv5 eslav",
    "Arabic": "PP-OCRv5 arabic",
    "Hangul": "PP-OCRv5 korean",
}


class PaddleOcrEngine(OcrEngine):
    name = "paddleocr"
    display_name = "PaddleOCR"
    model = "PaddleOCR 3.x (PaddlePaddle): PP-OCRv6 medium multilingual, or the PP-OCRv5 model of the script"
    options = {
        "min_score": Option(0.0, "drop text lines recognized with a lower score", minimum=0.0, maximum=1.0),
        "textline_orientation": Option(True, "detect and turn upside-down (180°) text lines"),
    }

    @classmethod
    def route(cls, languages: list[Language]) -> Route:
        main = [lang for lang in languages if lang.code != "eng"] or languages
        models = list(dict.fromkeys(MODEL_BY_SCRIPT[lang.script] for lang in main))
        if len(models) > 1:
            return unsupported(f"no single PaddleOCR recognizer covers {'+'.join(l.code for l in languages)} ({', '.join(models)})")
        return Route(lang=main[0].paddle, detail=f"{main[0].paddle} ({models[0]})")

    def prepare(self) -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        # enable_mkldnn=False: PaddlePaddle 3.x oneDNN kernels crash on CPU
        # ("ConvertPirAttribute2RuntimeAttribute not support").
        self._ocr = PaddleOCR(
            lang=self.lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=self.opts["textline_orientation"],
            text_rec_score_thresh=self.opts["min_score"],
            return_word_box=True,
            enable_mkldnn=False,
        )

    def ocr_page(self, image: PageImage, lang: str) -> list[OcrWord]:
        results = self._ocr.predict(str(image.path))
        if not results:
            return []
        r = results[0]
        lines = zip(r["rec_texts"], r["rec_scores"], r["rec_polys"])
        fragments = r.get("text_word") or [None] * len(r["rec_texts"])
        regions = r.get("text_word_region") or [None] * len(r["rec_texts"])
        return [
            word
            for (text, score, poly), frags, regs in zip(lines, fragments, regions)
            for word in self._line_words(text, float(score), poly, frags, regs, image)
        ]

    @staticmethod
    def _line_words(text, score, poly, frags, regs, image: PageImage) -> list[OcrWord]:
        words = line_words(text, frags, regs) if frags and regs and len(frags) == len(regs) else None
        if words is None:
            return split_line(text, BBox.from_points(poly, image.width_px, image.height_px), score)
        return [
            OcrWord(text=word, bbox=BBox.from_pixels(*box, image.width_px, image.height_px), confidence=score)
            for word, box in words
        ]


def line_words(text: str, frags: list[str], regions: list) -> list[tuple[str, tuple[float, float, float, float]]] | None:
    """Words of a recognized line with pixel boxes `(x0, y0, x1, y1)`.

    PaddleOCR's `text_word` fragments split at character-class changes ("Gr", "üß", "e") and may
    contain spaces (", Ü"), so they are not words. The recognized line text decides the word
    boundaries; the fragment boxes, spread evenly over their characters, give the geometry.
    Returns None when the fragments do not spell the line text.
    """
    char_boxes = [
        box
        for frag, region in zip(frags, regions)
        for i, ch in enumerate(frag)
        if not ch.isspace()
        for box in [_char_box(region, i, len(frag))]
    ]
    if len(char_boxes) != sum(not ch.isspace() for ch in text):
        return None
    words, k = [], 0
    for m in re.finditer(r"\S+", text):
        boxes = char_boxes[k : k + len(m.group())]
        k += len(m.group())
        words.append((m.group(), (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))))
    return words


def _char_box(region, index: int, length: int) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in region]
    ys = [float(p[1]) for p in region]
    x0, x1 = min(xs), max(xs)
    step = (x1 - x0) / max(length, 1)
    return (x0 + index * step, min(ys), x0 + (index + 1) * step, max(ys))
