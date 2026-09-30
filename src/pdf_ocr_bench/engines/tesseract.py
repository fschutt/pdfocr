from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache

from ..languages import Language
from ..models import BBox, OcrWord, PageImage
from .base import OcrEngine, Option, PageTimeout, Route, unsupported


@lru_cache(maxsize=1)
def installed_models() -> frozenset[str]:
    """`tesseract --list-langs`; unlike pytesseract.get_languages it keeps script models (Fraktur)."""
    out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, check=True).stdout
    return frozenset(line.strip() for line in out.splitlines()[1:] if line.strip())


PSM = Option(
    3,
    "page segmentation mode: 3 automatic, 4 one column, 6 one uniform block, 11 sparse text, 13 raw line",
    choices=(1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
)


def tesseract_route(languages: list[Language]) -> Route:
    """First-choice model per language; `tesseract_preflight` swaps in what is installed."""
    code = "+".join(dict.fromkeys(lang.tesseract_models[0] for lang in languages))
    return Route(lang=tuple(languages), detail=code)


def tesseract_preflight(route: Route) -> Route:
    """Every language must resolve to an installed model; `-l` joins them with '+'."""
    if not route.ok:
        return route
    if shutil.which("tesseract") is None:
        return unsupported("the tesseract binary is not installed")
    available = installed_models()
    models = []
    for lang in route.lang:
        model = next((m for m in lang.tesseract_models if m in available), None)
        if model is None:
            wanted = " or ".join(lang.tesseract_models)
            return unsupported(f"no Tesseract model for {lang.code} ({wanted}); install {' '.join(lang.tesseract_packages)}")
        models.append(model)
    code = "+".join(dict.fromkeys(models))
    swapped = [f"{lang.code} uses {m}" for lang, m in zip(route.lang, models) if lang.code != m]
    return Route(lang=code, detail=code + (f" ({', '.join(swapped)})" if swapped else ""))


class TesseractEngine(OcrEngine):
    name = "tesseract"
    display_name = "Tesseract"
    handles_timeout = True
    model = "Tesseract 5 LSTM models, one per --lang code (tesseract --list-langs)"
    modules = ("pytesseract",)
    extra = "tesseract"
    options = {"psm": PSM}

    @classmethod
    def route(cls, languages: list[Language]) -> Route:
        return tesseract_route(languages)

    @classmethod
    def preflight(cls, route: Route) -> Route:
        return tesseract_preflight(super().preflight(route))

    def prepare(self) -> None:
        import pytesseract

        self._tess = pytesseract

    def ocr_page(self, image: PageImage, lang: str) -> list[OcrWord]:
        try:
            data = self._tess.image_to_data(
                str(image.path),
                lang=lang,
                config=f"--psm {self.opts['psm']}",
                output_type=self._tess.Output.DICT,
                timeout=self.timeout or 0,
            )
        except RuntimeError as exc:
            if "timeout" in str(exc).lower():
                raise PageTimeout(f"exceeded {self.timeout:.0f}s") from exc
            raise
        rows = zip(data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"])
        return [
            OcrWord(
                text=text.strip(),
                bbox=BBox.from_pixels(left, top, left + w, top + h, image.width_px, image.height_px),
                confidence=max(0.0, float(conf)) / 100.0,
            )
            for text, conf, left, top, w, h in rows
            if text.strip() and float(conf) >= 0
        ]
