from __future__ import annotations

import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Iterable, TypeVar

from ..lang_map import get_lang
from ..log import get_logger
from ..models import BBox, OcrWord, PageImage, PageResult

T = TypeVar("T")


class PageTimeout(Exception):
    pass


class OcrEngine(ABC):
    """One OCR backend. Subclasses implement `ocr_page`; everything else is shared."""

    name: ClassVar[str]
    display_name: ClassVar[str]
    requires_gpu: ClassVar[bool] = False
    # Subprocess-based engines enforce the timeout themselves (and can kill the child).
    handles_timeout: ClassVar[bool] = False
    # False for engines whose output carries no confidence (PDF text layers, VLM text).
    reports_confidence: ClassVar[bool] = True

    def __init__(self, lang: str = "eng", timeout: float | None = None):
        self.tesseract_lang = lang
        self.lang = get_lang(lang, self.name)
        self.timeout = timeout or None
        self.log = get_logger(self.display_name)

    def prepare(self) -> None:
        """Import the backend and load models. Called once before the first page."""

    def close(self) -> None:
        """Release resources (servers, temp dirs)."""

    @abstractmethod
    def ocr_page(self, image: PageImage, lang: str) -> list[OcrWord]:
        ...

    def run(self, image: PageImage) -> PageResult:
        start = time.perf_counter()
        call = lambda: self.ocr_page(image, self.lang)  # noqa: E731
        words = call() if self.handles_timeout else run_with_timeout(call, self.timeout)
        words = [w for w in words if w.text.strip() and w.bbox.w > 0 and w.bbox.h > 0]
        return PageResult(
            page_num=image.page_num,
            width_px=image.width_px,
            height_px=image.height_px,
            words=words,
            full_text=words_to_text(words),
            engine_name=self.name,
            elapsed_seconds=time.perf_counter() - start,
        )


def run_with_timeout(fn: Callable[[], T], timeout: float | None) -> T:
    """Run `fn` in a daemon thread. On timeout the thread is abandoned (Python cannot kill it)."""
    if not timeout:
        return fn()
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise PageTimeout(f"exceeded {timeout:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def group_lines(words: Iterable[OcrWord]) -> list[list[OcrWord]]:
    """Group words into lines by vertical overlap; lines top to bottom, words left to right."""
    lines: list[list[OcrWord]] = []
    for word in sorted(words, key=lambda w: (w.bbox.y + w.bbox.h / 2, w.bbox.x)):
        line = next((l for l in reversed(lines[-3:]) if _same_line(l[-1].bbox, word.bbox)), None)
        if line is None:
            lines.append([word])
            continue
        line.append(word)
    lines.sort(key=lambda l: min(w.bbox.y for w in l))
    return [sorted(l, key=lambda w: w.bbox.x) for l in lines]


def words_to_text(words: Iterable[OcrWord]) -> str:
    return "\n".join(" ".join(w.text for w in line) for line in group_lines(words))


def _same_line(a: BBox, b: BBox) -> bool:
    overlap = min(a.y1, b.y1) - max(a.y, b.y)
    return overlap > 0.5 * min(a.h, b.h)


def split_line(text: str, bbox: BBox, confidence: float) -> list[OcrWord]:
    """Split a line-level detection into word boxes, proportional to character count."""
    tokens = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    if not tokens:
        return []
    if len(tokens) == 1:
        return [OcrWord(text=tokens[0][2], bbox=bbox, confidence=confidence)]
    span = max(len(text.rstrip()), 1)
    char_w = bbox.w / span
    return [
        OcrWord(
            text=tok,
            bbox=BBox(x=bbox.x + start * char_w, y=bbox.y, w=(end - start) * char_w, h=bbox.h),
            confidence=confidence,
        )
        for start, end, tok in tokens
    ]


def split_block(lines: list[str], bbox: BBox, confidence: float) -> list[OcrWord]:
    """Distribute text lines evenly over a block box, then split each into words."""
    lines = [l for l in lines if l.strip()]
    if not lines:
        return []
    line_h = bbox.h / len(lines)
    longest = max(len(l.strip()) for l in lines)
    return [
        word
        for i, line in enumerate(lines)
        for word in split_line(
            line.strip(),
            BBox(x=bbox.x, y=bbox.y + i * line_h, w=bbox.w * len(line.strip()) / longest, h=line_h),
            confidence,
        )
    ]
