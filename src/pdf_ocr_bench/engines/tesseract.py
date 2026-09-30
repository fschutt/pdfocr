from __future__ import annotations

import subprocess

from ..models import BBox, OcrWord, PageImage
from .base import OcrEngine, PageTimeout


# Current tessdata ships no deu_frak; frk (German Fraktur) and the Fraktur script model do.
SUBSTITUTES = {
    "deu_frak": ("frk", "script/Fraktur", "Fraktur", "deu"),
    "frk": ("deu_frak", "script/Fraktur", "Fraktur", "deu"),
}


def resolve_lang(lang: str, available: set[str]) -> str | None:
    return next((l for l in (lang, *SUBSTITUTES.get(lang, ())) if l in available), None)


def installed_langs(cmd: str = "tesseract") -> set[str]:
    """`tesseract --list-langs`; unlike pytesseract.get_languages it keeps script models (Fraktur, script/Latin)."""
    out = subprocess.run([cmd, "--list-langs"], capture_output=True, text=True, check=True).stdout
    return {line.strip() for line in out.splitlines()[1:] if line.strip()}


class TesseractEngine(OcrEngine):
    name = "tesseract"
    display_name = "Tesseract"
    handles_timeout = True

    def prepare(self) -> None:
        import pytesseract

        self._tess = pytesseract
        available = installed_langs(pytesseract.pytesseract.tesseract_cmd)
        resolved = [resolve_lang(l, available) for l in self.lang.split("+")]
        for want, got in zip(self.lang.split("+"), resolved):
            if got is None:
                self.log.warning(f"language '{want}' not installed (have {sorted(available)}), dropping it")
            elif got != want:
                self.log.warning(f"language '{want}' not installed, using '{got}'")
        self.lang = "+".join(dict.fromkeys(l for l in resolved if l)) or "eng"

    def ocr_page(self, image: PageImage, lang: str) -> list[OcrWord]:
        try:
            data = self._tess.image_to_data(
                str(image.path),
                lang=lang,
                config="--psm 3",
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
