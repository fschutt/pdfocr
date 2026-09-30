from __future__ import annotations

import html
import re

from ..languages import Language
from ..models import BBox, OcrWord, PageImage
from .base import OcrEngine, Route, split_block

_LINE_BREAK = re.compile(r"<br\s*/?>|</(?:p|div|li|tr|h[1-6])>", re.I)
_TAG = re.compile(r"<[^>]+>")


class SuryaEngine(OcrEngine):
    """Surya 2: a VLM served by llama.cpp (CPU) or vLLM (GPU), spawned by SuryaInferenceManager.

    Full-page mode returns layout blocks (label, html, polygon, bbox, confidence) — no word
    boxes — so each block's text is spread over its box line by line, proportional to length.
    """

    name = "surya"
    display_name = "Surya"
    model = "Surya 2 VLM (datalab-to/surya-ocr-2), served by llama.cpp (CPU) or vLLM (GPU)"

    @classmethod
    def route(cls, languages: list[Language]) -> Route:
        return Route(detail="automatic (VLM)")

    def prepare(self) -> None:
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor

        self._manager = SuryaInferenceManager()
        self._rec = RecognitionPredictor(self._manager)
        self._rec.disable_tqdm = True
        # spawns llama-server (CPU) / vLLM (GPU) now, so a missing backend fails setup once
        self._manager.start()

    def close(self) -> None:
        manager = getattr(self, "_manager", None)
        if manager is None:
            return
        manager.stop()

    def ocr_page(self, image: PageImage, lang: str) -> list[OcrWord]:
        from PIL import Image

        with Image.open(image.path) as img:
            page = self._rec([img.convert("RGB")])[0]
        width, height = page.image_bbox[2] or image.width_px, page.image_bbox[3] or image.height_px
        return [
            word
            for block in page.blocks
            if not block.skipped and not block.error and block.html
            for word in split_block(
                html_to_lines(block.html),
                BBox.from_pixels(*block.bbox, width, height),
                float(block.confidence or 0.0),
            )
        ]


def html_to_lines(fragment: str) -> list[str]:
    text = _TAG.sub(" ", _LINE_BREAK.sub("\n", fragment))
    return [" ".join(html.unescape(line).split()) for line in text.splitlines() if line.strip()]
