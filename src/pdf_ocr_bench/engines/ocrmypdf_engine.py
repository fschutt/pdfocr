from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..languages import Language
from ..models import BBox, OcrWord, PageImage
from .base import OcrEngine, PageTimeout, Route
from .tesseract import PSM, tesseract_preflight, tesseract_route


class OcrmypdfEngine(OcrEngine):
    """Run ocrmypdf on a one-page PDF built from the shared page image, read words back with PyMuPDF.

    Working page by page (instead of on the whole input) keeps page ranges, per-page
    timeouts and coordinates identical to every other engine.
    """

    name = "ocrmypdf"
    display_name = "ocrmypdf"
    handles_timeout = True
    reports_confidence = False
    model = "ocrmypdf with Tesseract; the PDF text layer is read back with PyMuPDF"
    modules = ("ocrmypdf",)
    extra = "ocrmypdf"
    options = {"psm": PSM}

    @classmethod
    def route(cls, languages: list[Language]) -> Route:
        return tesseract_route(languages)

    @classmethod
    def preflight(cls, route: Route) -> Route:
        return tesseract_preflight(super().preflight(route))

    def prepare(self) -> None:
        import ocrmypdf  # noqa: F401 - fail early if missing

        self._tmp = Path(tempfile.mkdtemp(prefix=f"{self.name}_"))

    def plugin_args(self) -> list[str]:
        return ["--tesseract-pagesegmode", str(self.opts["psm"])]

    def ocr_language(self) -> str:
        return self.lang

    def close(self) -> None:
        tmp = getattr(self, "_tmp", None)
        if tmp is None:
            return
        shutil.rmtree(tmp, ignore_errors=True)

    def ocr_page(self, image: PageImage, lang: str) -> list[OcrWord]:
        src = self._tmp / f"in_{image.page_num:04d}.pdf"
        dst = self._tmp / f"out_{image.page_num:04d}.pdf"
        image_to_pdf(image, src)
        self._run(src, dst, image.dpi)
        return pdf_words(dst)

    def _run(self, src: Path, dst: Path, dpi: int) -> None:
        cmd = [
            sys.executable, "-m", "ocrmypdf",
            *self.plugin_args(),
            "--force-ocr",
            "--output-type", "pdf",
            "--optimize", "0",
            "--image-dpi", str(dpi),
            "-l", self.ocr_language(),
            str(src), str(dst),
        ]  # fmt: skip
        self.log.debug(" ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise PageTimeout(f"exceeded {self.timeout:.0f}s") from exc
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
            raise RuntimeError(f"ocrmypdf exited {proc.returncode}: {tail}")


def image_to_pdf(image: PageImage, out: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    try:
        page = doc.new_page(width=image.width_pt, height=image.height_pt)
        page.insert_image(page.rect, filename=str(image.path))
        doc.save(out)
    finally:
        doc.close()


def pdf_words(path: Path) -> list[OcrWord]:
    """Words of the first page, normalized by the page rect. The text layer has no confidence."""
    import pymupdf

    doc = pymupdf.open(path)
    try:
        page = doc[0]
        w, h = page.rect.width, page.rect.height
        return [
            OcrWord(text=text, bbox=BBox.from_pixels(x0, y0, x1, y1, w, h), confidence=1.0)
            for x0, y0, x1, y1, text, *_ in page.get_text("words")
        ]
    finally:
        doc.close()
