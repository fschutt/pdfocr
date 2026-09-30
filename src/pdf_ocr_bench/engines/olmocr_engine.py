from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..models import BBox, OcrWord, PageImage
from .base import OcrEngine, PageTimeout, split_block
from .ocrmypdf_engine import image_to_pdf

DEFAULT_MODEL = "allenai/olmOCR-7B-0725"


class OlmOcrEngine(OcrEngine):
    """olmOCR (a 7B VLM) via `python -m olmocr.pipeline ... --markdown`.

    Works on CPU but takes minutes per page, hence `requires_gpu`. The model returns plain
    text without positions, so lines are spread evenly over the page's ink bounding box:
    good enough for search/copy-paste, not for exact word overlay.

    Env: OLMOCR_MODEL (default allenai/olmOCR-7B-0725, e.g. the -FP8 variant),
    OLMOCR_SERVER (an existing vLLM-compatible endpoint, skips spawning one).
    """

    name = "olmocr"
    display_name = "olmOCR"
    requires_gpu = True
    handles_timeout = True
    reports_confidence = False

    def prepare(self) -> None:
        import olmocr  # noqa: F401 - fail early if missing

        self._tmp = Path(tempfile.mkdtemp(prefix="olmocr_"))
        self._model = os.environ.get("OLMOCR_MODEL", DEFAULT_MODEL)
        self.log.info(f"model {self._model} — expect several minutes per page on CPU")

    def close(self) -> None:
        tmp = getattr(self, "_tmp", None)
        if tmp is None:
            return
        shutil.rmtree(tmp, ignore_errors=True)

    def ocr_page(self, image: PageImage, lang: str) -> list[OcrWord]:
        workspace = self._tmp / f"ws_{image.page_num:04d}"
        pdf = self._tmp / f"page_{image.page_num + 1:04d}.pdf"
        image_to_pdf(image, pdf)
        self._run(workspace, pdf)
        markdown = next(workspace.glob("markdown/**/*.md"), None)
        if markdown is None:
            raise RuntimeError("olmOCR produced no markdown output")
        lines = [strip_markdown(l) for l in markdown.read_text(encoding="utf-8").splitlines()]
        return split_block(lines, ink_bbox(image.path), confidence=1.0)

    def _run(self, workspace: Path, pdf: Path) -> None:
        cmd = [sys.executable, "-m", "olmocr.pipeline", str(workspace), "--markdown", "--pdfs", str(pdf), "--model", self._model]
        server = os.environ.get("OLMOCR_SERVER")
        if server:
            cmd += ["--server", server]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise PageTimeout(f"exceeded {self.timeout:.0f}s") from exc
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
            raise RuntimeError(f"olmocr exited {proc.returncode}: {tail}")


def strip_markdown(line: str) -> str:
    return line.lstrip("#>*-+ ").replace("**", "").replace("__", "").strip()


def ink_bbox(path: Path, threshold: int = 200) -> BBox:
    """Bounding box of the dark pixels (the text area), falling back to the full page."""
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        gray = ImageOps.invert(img.convert("L")).point(lambda v: 255 if v > 255 - threshold else 0)
        box = gray.getbbox()
        if box is None:
            return BBox(x=0, y=0, w=1, h=1)
        return BBox.from_pixels(*box, img.width, img.height)
