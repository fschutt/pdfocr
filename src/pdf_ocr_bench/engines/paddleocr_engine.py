from __future__ import annotations

import os

from ..models import BBox, OcrWord, PageImage
from .base import OcrEngine, split_line


class PaddleOcrEngine(OcrEngine):
    name = "paddleocr"
    display_name = "PaddleOCR"

    def prepare(self) -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        # enable_mkldnn=False: PaddlePaddle 3.x oneDNN kernels crash on CPU
        # ("ConvertPirAttribute2RuntimeAttribute not support").
        opts = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            return_word_box=True,
            enable_mkldnn=False,
        )
        try:
            self._ocr = PaddleOCR(lang=self.lang, **opts)
        except Exception as exc:  # noqa: BLE001 - unsupported lang
            self.log.warning(f"lang '{self.lang}' unavailable ({exc}), using 'en'")
            self._ocr = PaddleOCR(lang="en", **opts)

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
        if not frags or not regs or len(frags) != len(regs):
            return split_line(text, BBox.from_points(poly, image.width_px, image.height_px), score)
        return [
            OcrWord(text=word, bbox=BBox.from_points(points, image.width_px, image.height_px), confidence=score)
            for word, points in merge_fragments(frags, regs)
        ]


def merge_fragments(frags: list[str], regions: list) -> list[tuple[str, list]]:
    """PaddleOCR splits words by character class ("Gr", "üß", "e"); rejoin them at whitespace."""
    words: list[tuple[str, list]] = []
    text, points = "", []
    for frag, region in zip(frags, regions):
        if frag.strip():
            text += frag.strip()
            points.extend(region)
        if frag != frag.rstrip() or not frag.strip():
            if text:
                words.append((text, points))
            text, points = "", []
    if text:
        words.append((text, points))
    return words
